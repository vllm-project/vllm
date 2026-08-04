# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run one Kimi-K3 XPU decoder layer with real checkpoint weights.

Layer indices are zero-based. The default ``--layer-index 3`` therefore runs
the fourth transformer layer. Activations can be synthetic or loaded from a
``torch.save`` file containing ``hidden_states``, ``prefix_sum``, and
``residual`` tensors.
"""

import argparse
import json
import os
import sys
import tempfile
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
from vllm.models.kimi_k3.xpu.linear import KimiDecoderLayer
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.worker.workspace import init_workspace_manager


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--layer-index", type=int, default=3)
    parser.add_argument("--num-tokens", type=int, default=1)
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
        max_model_len=128,
        enforce_eager=True,
    )


def make_common_metadata(
    num_tokens: int,
    block_size: int,
    device: torch.device,
) -> CommonAttentionMetadata:
    query_start_loc = torch.arange(
        num_tokens + 1, dtype=torch.int32, device=device
    )
    seq_lens = torch.ones(num_tokens, dtype=torch.int32, device=device)
    seq_lens_cpu = seq_lens.cpu()
    return CommonAttentionMetadata(
        query_start_loc=query_start_loc,
        query_start_loc_cpu=query_start_loc.cpu(),
        seq_lens=seq_lens,
        seq_lens_cpu_upper_bound=seq_lens_cpu,
        _seq_lens_cpu=seq_lens_cpu,
        _num_computed_tokens_cpu=torch.zeros(num_tokens, dtype=torch.int32),
        num_reqs=num_tokens,
        num_actual_tokens=num_tokens,
        max_query_len=1,
        max_seq_len=1,
        block_table_tensor=torch.arange(
            num_tokens, dtype=torch.int32, device=device
        ).view(num_tokens, 1),
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
    cache_shape = backend.get_kv_cache_shape(
        num_tokens,
        cache_spec.block_size,
        cache_spec.num_kv_heads,
        cache_spec.head_size,
    )
    mla.kv_cache = torch.zeros(cache_shape, dtype=cache_spec.dtype, device=device)
    return {layer_name: metadata}, {layer_name: common.slot_mapping}


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
        ("self_attn.fused_qkv_a_proj.weight", "self_attn.q_a_proj.weight", 0),
        (
            "self_attn.fused_qkv_a_proj.weight",
            "self_attn.kv_a_proj_with_mqa.weight",
            1,
        ),
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
            if relative_name == source_suffix:
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

    moe_records = load_moe_weights(
        layer.block_sparse_moe,
        weight_map,
        moe_prefix,
        source_config,
        expert_ids,
    )
    loaded.update(f"block_sparse_moe.{record['target']}" for record in moe_records)
    return loaded, moe_records


def make_input_state(
    args: argparse.Namespace,
    layer: KimiDecoderLayer,
    device: torch.device,
) -> tuple[torch.Tensor | None, torch.Tensor, torch.Tensor]:
    if args.input_state is not None:
        state = torch.load(args.input_state, map_location=device, weights_only=True)
        return (
            state["hidden_states"],
            state["prefix_sum"],
            state["residual"],
        )
    generator = torch.Generator(device=device).manual_seed(17)
    shape = (args.num_tokens, layer.hidden_size)
    hidden_states = torch.randn(
        shape, dtype=torch.bfloat16, device=device, generator=generator
    )
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


def main() -> int:
    args = parse_args()
    report: dict[str, Any] = {
        "status": "failed",
        "checkpoint_dir": str(args.checkpoint_dir),
        "layer_index": args.layer_index,
    }
    try:
        if args.layer_index < 0 or args.num_tokens < 1:
            raise ProbeError("Layer index must be non-negative and tokens positive")
        raw_config = load_checkpoint_config(args.checkpoint_dir)
        source_config = load_text_config(raw_config)
        num_experts = args.num_experts or source_config.num_experts
        if not 16 <= num_experts <= source_config.num_experts:
            raise ProbeError("--num-experts must be between TopK 16 and source count")
        expert_ids = list(range(num_experts))
        config_dict = source_config.to_dict()
        config_dict.update(
            num_experts=num_experts,
            num_expert_group=(
                1
                if num_experts < source_config.num_experts
                else source_config.num_expert_group
            ),
            topk_group=(
                1
                if num_experts < source_config.num_experts
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
        model_config = make_model_config(source_config, args.checkpoint_dir)
        cache_config = CacheConfig(block_size=16, cache_dtype="auto")
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
            if not layer.is_moe_layer:
                raise ProbeError("Selected layer is not an MLA + MoE layer")
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
            if config.hidden_act == "situ" and situ_config["kernel_beta"] is None:
                raise ProbeError("SITU beta was lost before MXFP4 kernel creation")
            layer.self_attn.mla_attn.mla_attn.process_weights_after_loading(
                torch.bfloat16
            )
            metadata, slot_mapping = bind_mla_cache_and_metadata(
                layer,
                vllm_config,
                args.num_tokens,
                device,
            )
            hidden_states, prefix_sum, residual = make_input_state(args, layer, device)
            positions = torch.zeros(args.num_tokens, dtype=torch.int64, device=device)
            with set_forward_context(
                metadata,
                vllm_config,
                num_tokens=args.num_tokens,
                slot_mapping=slot_mapping,
            ):
                output, output_prefix_sum, output_residual = layer(
                    positions,
                    hidden_states,
                    residual,
                    prefix_sum,
                )
            torch.xpu.synchronize()
            if not bool(torch.isfinite(output).all()):
                raise ProbeError("Layer output contains non-finite values")
            output_state = {
                "hidden_states": output.cpu(),
                "prefix_sum": output_prefix_sum.cpu(),
                "residual": output_residual.cpu(),
            }
            if args.save_output is not None:
                args.save_output.parent.mkdir(parents=True, exist_ok=True)
                torch.save(output_state, args.save_output)
            report.update(
                status="passed",
                ordinal_layer=args.layer_index + 1,
                checkpoint_prefix=checkpoint_prefix,
                num_experts=num_experts,
                loaded_parameters=len(loaded),
                loaded_moe_tensors=len(moe_records),
                input_state="file" if args.input_state else "synthetic",
                output_shape=list(output.shape),
                output_dtype=str(output.dtype),
                output_all_finite=True,
                output_max_abs=float(output.abs().max()),
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