# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run any Kimi-K3 XPU decoder layer with real checkpoint weights.

Layer indices are zero-based. The default ``--layer-index 3`` therefore runs
the fourth transformer layer. Activations can be synthetic or loaded from a
``torch.save`` file containing ``hidden_states`` and, when attn-res is enabled,
``prefix_sum`` and ``residual`` tensors. Set ``--benchmark-iters`` to measure
steady-state, host-observed decoder-layer forward latency. Set
``--profile-output`` to export a trace for Perfetto.
"""

import argparse
import json
import os
import statistics
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any

import torch

from kimi_k3_xpu_moe_weight_probe import (
    ProbeError,
    default_dtype,
    load_checkpoint_config,
    load_moe_weights,
    load_quant_config,
    load_text_config,
    load_weight_map,
    read_tensor,
)
from vllm.config import CacheConfig, ModelConfig, VllmConfig, set_current_vllm_config
from vllm.distributed import init_distributed_environment, initialize_model_parallel
from vllm.forward_context import set_forward_context
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.models.kimi_k3.xpu.kda import KimiK3DeltaAttention
from vllm.models.kimi_k3.xpu.linear import KimiDecoderLayer
from vllm.utils.math_utils import cdiv
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.kv_cache_interface import MambaSpec
from vllm.v1.worker.workspace import init_workspace_manager


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--layer-index", type=int, default=3)
    parser.add_argument("--num-tokens", type=int, default=1)
    parser.add_argument(
        "--warmup-iters",
        type=int,
        default=5,
        help="Warmup forwards before benchmarking.",
    )
    parser.add_argument(
        "--benchmark-iters",
        type=int,
        default=0,
        help="Timed forwards; zero disables benchmarking.",
    )
    parser.add_argument(
        "--profile-output",
        type=Path,
        help="Write a Perfetto-compatible PyTorch profiler trace.",
    )
    parser.add_argument(
        "--num-experts",
        type=int,
        help="Load only experts [0, N) for a smaller development run.",
    )
    parser.add_argument(
        "--input-state",
        type=Path,
        help="Optional torch file with hidden_states, prefix_sum, and residual.",
    )
    parser.add_argument("--save-output", type=Path)
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("/tmp/kimi_k3_xpu_layer_probe.json"),
    )
    return parser.parse_args()


def initialize_single_rank() -> None:
    file_descriptor, init_file = tempfile.mkstemp(prefix="kimi_layer_probe_")
    os.close(file_descriptor)
    init_distributed_environment(
        world_size=1,
        rank=0,
        distributed_init_method=f"file://{init_file}",
        local_rank=0,
        backend="gloo",
    )
    initialize_model_parallel(1, 1)


def make_model_config(
    source_config: Any,
    checkpoint_dir: Path,
    max_model_len: int,
) -> ModelConfig:
    config_dir = Path(tempfile.mkdtemp(prefix="kimi_linear_config_"))
    config_dict = source_config.to_dict()
    config_dict.update(
        architectures=["KimiLinearForCausalLM"],
        model_type="kimi_linear",
    )
    (config_dir / "config.json").write_text(
        json.dumps(config_dict),
        encoding="utf-8",
    )
    return ModelConfig(
        model=str(config_dir),
        model_weights=str(checkpoint_dir),
        dtype=torch.bfloat16,
        max_model_len=max_model_len,
        enforce_eager=True,
    )


def make_common_metadata(
    num_tokens: int,
    block_size: int,
    device: torch.device,
) -> CommonAttentionMetadata:
    query_start_loc = torch.tensor(
        [0, num_tokens], dtype=torch.int32, device=device
    )
    seq_lens = torch.tensor([num_tokens], dtype=torch.int32, device=device)
    seq_lens_cpu = seq_lens.cpu()
    num_blocks = cdiv(num_tokens, block_size)
    return CommonAttentionMetadata(
        query_start_loc=query_start_loc,
        query_start_loc_cpu=query_start_loc.cpu(),
        seq_lens=seq_lens,
        seq_lens_cpu_upper_bound=seq_lens_cpu,
        _seq_lens_cpu=seq_lens_cpu,
        _num_computed_tokens_cpu=torch.zeros(1, dtype=torch.int32),
        num_reqs=1,
        num_actual_tokens=num_tokens,
        max_query_len=num_tokens,
        max_seq_len=num_tokens,
        block_table_tensor=torch.arange(
            num_blocks, dtype=torch.int32, device=device
        ).view(1, num_blocks),
        slot_mapping=torch.arange(num_tokens, dtype=torch.int64, device=device),
        causal=True,
    )


def bind_mla_cache_and_metadata(
    layer: KimiDecoderLayer,
    vllm_config: VllmConfig,
    num_tokens: int,
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    mla = layer.self_attn.mla_attn.mla_attn
    layer_name = mla.layer_name
    backend = mla.get_attn_backend()
    cache_spec = mla.get_kv_cache_spec(vllm_config)
    builder = backend.get_builder_cls()(
        kv_cache_spec=cache_spec,
        layer_names=[layer_name],
        vllm_config=vllm_config,
        device=device,
    )
    common = make_common_metadata(
        num_tokens,
        vllm_config.cache_config.block_size,
        device,
    )
    metadata = builder.build(
        common_prefix_len=0,
        common_attn_metadata=common,
    )
    num_cache_blocks = cdiv(num_tokens, cache_spec.block_size)
    cache_shape = backend.get_kv_cache_shape(
        num_cache_blocks,
        cache_spec.block_size,
        cache_spec.num_kv_heads,
        cache_spec.head_size,
    )
    mla.kv_cache = torch.zeros(cache_shape, dtype=cache_spec.dtype, device=device)
    return {layer_name: metadata}, {layer_name: common.slot_mapping}


def bind_kda_cache_and_metadata(
    layer: KimiDecoderLayer,
    vllm_config: VllmConfig,
    num_tokens: int,
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    kda = layer.self_attn
    if not isinstance(kda, KimiK3DeltaAttention):
        raise ProbeError("Selected layer does not use XPU KDA")
    layer_name = kda.prefix
    backend = kda.get_attn_backend()
    cache_spec = kda.get_kv_cache_spec(vllm_config)
    if not isinstance(cache_spec, MambaSpec):
        raise ProbeError("KDA layer did not produce a Mamba cache spec")
    builder = backend.get_builder_cls()(
        kv_cache_spec=cache_spec,
        layer_names=[layer_name],
        vllm_config=vllm_config,
        device=device,
    )
    common = make_common_metadata(num_tokens, cache_spec.block_size, device)
    metadata = builder.build(
        common_prefix_len=0,
        common_attn_metadata=common,
    )
    num_cache_blocks = cache_spec.max_num_blocks_per_req(
        vllm_config,
        num_tokens,
    )
    raw_cache = torch.zeros(
        num_cache_blocks,
        1,
        1,
        cache_spec.page_size_bytes,
        dtype=torch.uint8,
        device=device,
    )
    kda.bind_kv_cache(raw_cache)
    return {layer_name: metadata}, {layer_name: common.slot_mapping}


def bind_attention_cache_and_metadata(
    layer: KimiDecoderLayer,
    vllm_config: VllmConfig,
    num_tokens: int,
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    if isinstance(layer.self_attn, KimiK3DeltaAttention):
        return bind_kda_cache_and_metadata(
            layer,
            vllm_config,
            num_tokens,
            device,
        )
    return bind_mla_cache_and_metadata(layer, vllm_config, num_tokens, device)


def load_direct_parameter(
    parameters: dict[str, torch.nn.Parameter],
    target_name: str,
    tensor: torch.Tensor,
    shard_id: int | None = None,
) -> None:
    parameter = parameters[target_name]
    loader = getattr(parameter, "weight_loader", default_weight_loader)
    if shard_id is None:
        loader(parameter, tensor)
    else:
        loader(parameter, tensor, shard_id)


def load_layer_weights(
    layer: KimiDecoderLayer,
    weight_map: dict[str, Path],
    checkpoint_prefix: str,
    source_config: Any,
    expert_ids: list[int],
) -> tuple[set[str], list[dict[str, Any]]]:
    parameters = dict(layer.named_parameters())
    loaded: set[str] = set()
    stacked_mapping = (
        ("self_attn.in_proj_qkvgfab.weight", "self_attn.q_proj.weight", 0),
        ("self_attn.in_proj_qkvgfab.weight", "self_attn.k_proj.weight", 1),
        ("self_attn.in_proj_qkvgfab.weight", "self_attn.v_proj.weight", 2),
        ("self_attn.in_proj_qkvgfab.weight", "self_attn.g_proj.weight", 3),
        ("self_attn.in_proj_qkvgfab.weight", "self_attn.f_a_proj.weight", 4),
        ("self_attn.in_proj_qkvgfab.weight", "self_attn.b_proj.weight", 5),
        ("self_attn.conv1d.weight", "self_attn.q_conv1d.weight", 0),
        ("self_attn.conv1d.weight", "self_attn.k_conv1d.weight", 1),
        ("self_attn.conv1d.weight", "self_attn.v_conv1d.weight", 2),
        ("self_attn.fused_qkv_a_proj.weight", "self_attn.q_a_proj.weight", 0),
        (
            "self_attn.fused_qkv_a_proj.weight",
            "self_attn.kv_a_proj_with_mqa.weight",
            1,
        ),
        ("mlp.gate_up_proj.weight", "mlp.gate_proj.weight", 0),
        ("mlp.gate_up_proj.weight", "mlp.up_proj.weight", 1),
    )
    moe_prefix = f"{checkpoint_prefix}.block_sparse_moe"
    for source_name in weight_map:
        if not source_name.startswith(f"{checkpoint_prefix}."):
            continue
        if source_name.startswith(f"{moe_prefix}.experts."):
            continue
        relative_name = source_name.removeprefix(f"{checkpoint_prefix}.")
        if relative_name.startswith("block_sparse_moe."):
            continue
        for target_name, source_suffix, shard_id in stacked_mapping:
            if relative_name == source_suffix and target_name in parameters:
                load_direct_parameter(
                    parameters,
                    target_name,
                    read_tensor(weight_map, source_name),
                    shard_id,
                )
                loaded.add(target_name)
                break
        else:
            if relative_name not in parameters:
                continue
            load_direct_parameter(
                parameters,
                relative_name,
                read_tensor(weight_map, source_name),
            )
            loaded.add(relative_name)

    moe_records: list[dict[str, Any]] = []
    if layer.is_moe_layer:
        moe_records = load_moe_weights(
            layer.block_sparse_moe,
            weight_map,
            moe_prefix,
            source_config,
            expert_ids,
        )
        loaded.update(
            f"block_sparse_moe.{record['target']}" for record in moe_records
        )
    return loaded, moe_records


def make_input_state(
    args: argparse.Namespace,
    layer: KimiDecoderLayer,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    if args.input_state is not None:
        state = torch.load(args.input_state, map_location=device, weights_only=True)
        hidden_states = state["hidden_states"]
        expected_shape = (args.num_tokens, layer.hidden_size)
        if tuple(hidden_states.shape) != expected_shape:
            raise ProbeError(
                f"Input hidden_states shape {tuple(hidden_states.shape)} does not "
                f"match {expected_shape}"
            )
        prefix_sum = state.get("prefix_sum")
        residual = state.get("residual")
        if layer.use_attn_res and (prefix_sum is None or residual is None):
            raise ProbeError(
                "Attn-res layer input requires prefix_sum and residual tensors"
            )
        return (
            hidden_states,
            prefix_sum,
            residual,
        )
    generator = torch.Generator(device=device).manual_seed(17)
    shape = (args.num_tokens, layer.hidden_size)
    hidden_states = torch.randn(
        shape, dtype=torch.bfloat16, device=device, generator=generator
    )
    prefix_sum = None
    residual = None
    if layer.use_attn_res:
        prefix_sum = torch.randn(
            shape, dtype=torch.bfloat16, device=device, generator=generator
        )
        num_blocks = layer.prev_valid_blocks + int(layer.is_block_write_layer)
        residual = torch.randn(
            args.num_tokens,
            num_blocks,
            layer.hidden_size,
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
        )
    return hidden_states, prefix_sum, residual


def capture_attention_cache(layer: KimiDecoderLayer) -> tuple[torch.Tensor, ...]:
    if isinstance(layer.self_attn, KimiK3DeltaAttention):
        return tuple(state.clone() for state in layer.self_attn.kv_cache)
    mla = layer.self_attn.mla_attn.mla_attn
    return (mla.kv_cache.clone(),)


def restore_attention_cache(
    layer: KimiDecoderLayer,
    cache_state: tuple[torch.Tensor, ...],
) -> None:
    if isinstance(layer.self_attn, KimiK3DeltaAttention):
        for state, initial_state in zip(layer.self_attn.kv_cache, cache_state):
            state.copy_(initial_state)
        return
    mla = layer.self_attn.mla_attn.mla_attn
    mla.kv_cache.copy_(cache_state[0])


def percentile(samples: list[float], fraction: float) -> float:
    ordered = sorted(samples)
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def benchmark_layer_forward(
    layer: KimiDecoderLayer,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    residual: torch.Tensor | None,
    prefix_sum: torch.Tensor | None,
    metadata: dict[str, Any],
    slot_mapping: dict[str, torch.Tensor],
    vllm_config: VllmConfig,
    cache_state: tuple[torch.Tensor, ...],
    warmup_iters: int,
    benchmark_iters: int,
) -> dict[str, Any]:
    def make_iteration_inputs() -> tuple[
        torch.Tensor, torch.Tensor | None, torch.Tensor | None
    ]:
        restore_attention_cache(layer, cache_state)
        iteration_hidden_states = hidden_states.clone()
        iteration_residual = None if residual is None else residual.clone()
        iteration_prefix_sum = None if prefix_sum is None else prefix_sum.clone()
        return iteration_hidden_states, iteration_residual, iteration_prefix_sum

    with torch.inference_mode():
        with set_forward_context(
            metadata,
            vllm_config,
            num_tokens=hidden_states.size(0),
            slot_mapping=slot_mapping,
        ):
            for _ in range(warmup_iters):
                iteration_inputs = make_iteration_inputs()
                layer(positions, *iteration_inputs)
            torch.xpu.synchronize()

            latencies_ms: list[float] = []
            for _ in range(benchmark_iters):
                iteration_inputs = make_iteration_inputs()
                torch.xpu.synchronize()
                start_ns = time.perf_counter_ns()
                layer(positions, *iteration_inputs)
                torch.xpu.synchronize()
                latencies_ms.append(
                    (time.perf_counter_ns() - start_ns) / 1_000_000
                )

    median_ms = statistics.median(latencies_ms)
    return {
        "warmup_iters": warmup_iters,
        "benchmark_iters": benchmark_iters,
        "latency_mean_ms": statistics.fmean(latencies_ms),
        "latency_min_ms": min(latencies_ms),
        "latency_max_ms": max(latencies_ms),
        "latency_p50_ms": median_ms,
        "latency_median_ms": median_ms,
        "latency_p90_ms": percentile(latencies_ms, 0.90),
        "latency_p99_ms": percentile(latencies_ms, 0.99),
        "tokens_per_second_median": hidden_states.size(0) * 1000 / median_ms,
        "attention_mode": (
            "cold_decode" if hidden_states.size(0) == 1 else "prefill"
        ),
        "cache_reset_between_iters": True,
        "input_reset_between_iters": True,
        "timing_method": "synchronized_host_wall_clock",
        "timing_scope": "KimiDecoderLayer.forward and device completion",
    }


def profile_layer_forward(
    layer: KimiDecoderLayer,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    residual: torch.Tensor | None,
    prefix_sum: torch.Tensor | None,
    metadata: dict[str, Any],
    slot_mapping: dict[str, torch.Tensor],
    vllm_config: VllmConfig,
    cache_state: tuple[torch.Tensor, ...],
    output_path: Path,
) -> None:
    restore_attention_cache(layer, cache_state)
    iteration_hidden_states = hidden_states.clone()
    iteration_residual = None if residual is None else residual.clone()
    iteration_prefix_sum = None if prefix_sum is None else prefix_sum.clone()
    torch.xpu.synchronize()

    with torch.inference_mode():
        with set_forward_context(
            metadata,
            vllm_config,
            num_tokens=hidden_states.size(0),
            slot_mapping=slot_mapping,
        ):
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.XPU,
                ],
                record_shapes=True,
                profile_memory=True,
                with_stack=False,
            ) as profiler:
                with torch.profiler.record_function(
                    "kimi_decoder_layer_forward"
                ):
                    layer(
                        positions,
                        iteration_hidden_states,
                        iteration_residual,
                        iteration_prefix_sum,
                    )
                torch.xpu.synchronize()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    profiler.export_chrome_trace(str(output_path))


def main() -> int:
    args = parse_args()
    report: dict[str, Any] = {
        "status": "failed",
        "checkpoint_dir": str(args.checkpoint_dir),
        "layer_index": args.layer_index,
    }
    try:
        if args.layer_index < 0:
            raise ProbeError("--layer-index must be non-negative")
        if args.num_tokens < 1:
            raise ProbeError("--num-tokens must be positive")
        if args.warmup_iters < 0 or args.benchmark_iters < 0:
            raise ProbeError("Benchmark iteration counts must be non-negative")
        raw_config = load_checkpoint_config(args.checkpoint_dir)
        source_config = load_text_config(raw_config)
        source_num_experts = source_config.num_experts
        if source_num_experts is None:
            if args.num_experts is not None:
                raise ProbeError("--num-experts requires a MoE checkpoint")
            num_experts = 0
            expert_ids: list[int] = []
        else:
            num_experts = args.num_experts or source_num_experts
            if not 16 <= num_experts <= source_num_experts:
                raise ProbeError(
                    "--num-experts must be between TopK 16 and source count"
                )
            expert_ids = list(range(num_experts))
        config_dict = source_config.to_dict()
        if source_num_experts is not None:
            config_dict.update(
                num_experts=num_experts,
                num_experts_per_token=min(
                    source_config.num_experts_per_token or num_experts,
                    num_experts,
                ),
                num_expert_group=(
                    1
                    if num_experts < source_num_experts
                    else source_config.num_expert_group
                ),
                topk_group=(
                    1
                    if num_experts < source_num_experts
                    else source_config.topk_group
                ),
            )
        config = type(source_config)(**config_dict)
        weight_map = load_weight_map(args.checkpoint_dir)
        checkpoint_prefix = f"language_model.model.layers.{args.layer_index}"
        if not any(name.startswith(f"{checkpoint_prefix}.") for name in weight_map):
            raise ProbeError(f"Checkpoint has no {checkpoint_prefix} tensors")

        device = torch.device("xpu:0")
        torch.xpu.set_device(device)
        model_config = make_model_config(
            source_config,
            args.checkpoint_dir,
            args.num_tokens,
        )
        cache_config = CacheConfig(
            block_size=16,
            cache_dtype="auto",
            mamba_block_size=model_config.max_model_len,
        )
        vllm_config = VllmConfig(
            model_config=model_config,
            cache_config=cache_config,
            quant_config=load_quant_config(raw_config),
        )
        with set_current_vllm_config(vllm_config):
            initialize_single_rank()
            init_workspace_manager(device)
            with default_dtype(torch.bfloat16):
                layer = KimiDecoderLayer(
                    config,
                    vllm_config,
                    prefix=f"model.layers.{args.layer_index}",
                ).to(device)
            layer.eval()
            loaded, moe_records = load_layer_weights(
                layer,
                weight_map,
                checkpoint_prefix,
                source_config,
                expert_ids,
            )
            missing = sorted(set(dict(layer.named_parameters())) - loaded)
            if missing:
                raise ProbeError(f"Unloaded layer parameters: {missing}")
            if layer.is_moe_layer:
                routed_experts = layer.block_sparse_moe.experts.routed_experts
                routed_experts.quant_method.process_weights_after_loading(
                    routed_experts
                )
                moe_runner = layer.block_sparse_moe.experts
                quant_method = moe_runner.routed_experts.quant_method
                moe_kernel = quant_method.moe_kernel
                situ_config = {
                    "source_beta": source_config.activation_situ_beta,
                    "source_linear_beta": source_config.activation_situ_linear_beta,
                    "runner_beta": moe_runner.moe_config.activation_situ_beta,
                    "runner_linear_beta": (
                        moe_runner.moe_config.activation_situ_linear_beta
                    ),
                    "kernel_beta": moe_kernel.moe_config.activation_situ_beta,
                    "kernel_linear_beta": (
                        moe_kernel.moe_config.activation_situ_linear_beta
                    ),
                }
                report["situ_config"] = situ_config
                if (
                    config.hidden_act == "situ"
                    and situ_config["kernel_beta"] is None
                ):
                    raise ProbeError("SITU beta was lost before MXFP4 kernel creation")
            is_kda = isinstance(layer.self_attn, KimiK3DeltaAttention)
            if not is_kda:
                layer.self_attn.mla_attn.mla_attn.process_weights_after_loading(
                    torch.bfloat16
                )
            metadata, slot_mapping = bind_attention_cache_and_metadata(
                layer,
                vllm_config,
                args.num_tokens,
                device,
            )
            hidden_states, prefix_sum, residual = make_input_state(args, layer, device)
            positions = torch.zeros(args.num_tokens, dtype=torch.int64, device=device)
            initial_cache_state = capture_attention_cache(layer)
            with torch.inference_mode():
                with set_forward_context(
                    metadata,
                    vllm_config,
                    num_tokens=args.num_tokens,
                    slot_mapping=slot_mapping,
                ):
                    output, output_prefix_sum, output_residual = layer(
                        positions,
                        hidden_states.clone(),
                        None if residual is None else residual.clone(),
                        None if prefix_sum is None else prefix_sum.clone(),
                    )
            torch.xpu.synchronize()
            if not bool(torch.isfinite(output).all()):
                raise ProbeError("Layer output contains non-finite values")
            output_state = {
                "hidden_states": output.cpu(),
                "prefix_sum": (
                    None if output_prefix_sum is None else output_prefix_sum.cpu()
                ),
                "residual": (
                    None if output_residual is None else output_residual.cpu()
                ),
            }
            if is_kda:
                conv_state, recurrent_state = layer.self_attn.kv_cache
                output_state.update(
                    conv_state=conv_state.cpu(),
                    recurrent_state=recurrent_state.cpu(),
                )
            benchmark_result = None
            if args.benchmark_iters > 0:
                benchmark_result = benchmark_layer_forward(
                    layer=layer,
                    positions=positions,
                    hidden_states=hidden_states,
                    residual=residual,
                    prefix_sum=prefix_sum,
                    metadata=metadata,
                    slot_mapping=slot_mapping,
                    vllm_config=vllm_config,
                    cache_state=initial_cache_state,
                    warmup_iters=args.warmup_iters,
                    benchmark_iters=args.benchmark_iters,
                )
            if args.profile_output is not None:
                profile_layer_forward(
                    layer=layer,
                    positions=positions,
                    hidden_states=hidden_states,
                    residual=residual,
                    prefix_sum=prefix_sum,
                    metadata=metadata,
                    slot_mapping=slot_mapping,
                    vllm_config=vllm_config,
                    cache_state=initial_cache_state,
                    output_path=args.profile_output,
                )
            if args.save_output is not None:
                args.save_output.parent.mkdir(parents=True, exist_ok=True)
                torch.save(output_state, args.save_output)
            report.update(
                status="passed",
                ordinal_layer=args.layer_index + 1,
                checkpoint_prefix=checkpoint_prefix,
                num_experts=num_experts,
                attention_type="kda" if is_kda else "mla",
                mlp_type="moe" if layer.is_moe_layer else "dense",
                loaded_parameters=len(loaded),
                loaded_moe_tensors=len(moe_records),
                input_state="file" if args.input_state else "synthetic",
                output_shape=list(output.shape),
                output_dtype=str(output.dtype),
                output_all_finite=True,
                output_max_abs=float(output.abs().max()),
            )
            if benchmark_result is not None:
                report["benchmark"] = benchmark_result
            if args.profile_output is not None:
                report["profile"] = {
                    "output": str(args.profile_output),
                    "format": "chrome_trace_json",
                    "activities": ["cpu", "xpu"],
                    "forward_iters": 1,
                    "perfetto_url": "https://ui.perfetto.dev/",
                }
            if is_kda:
                report.update(
                    conv_state_shape=list(conv_state.shape),
                    conv_state_dtype=str(conv_state.dtype),
                    recurrent_state_shape=list(recurrent_state.shape),
                    recurrent_state_dtype=str(recurrent_state.dtype),
                    conv_state_max_abs=float(conv_state.abs().max()),
                    recurrent_state_max_abs=float(recurrent_state.abs().max()),
                )
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        report["traceback"] = traceback.format_exc()
    finally:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())