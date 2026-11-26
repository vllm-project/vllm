# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measure complete unquantized MoNE layers against ordinary MoE.

Run inside the prepared vLLM environment; use torchrun for TP=2. Timings
include routing, remapping, novice accumulation and output reduction. The
no-accumulation ablation is a diagnostic with deliberately incomplete output.
"""

import argparse
import contextlib
import gc
import json
import os
import random
import statistics
import tempfile
import time
from dataclasses import replace
from pathlib import Path
from types import MethodType

import torch
import torch.distributed as dist
from transformers import Qwen3MoeConfig

from vllm.config import (
    KernelConfig,
    ModelConfig,
    ParallelConfig,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.distributed import (
    destroy_distributed_environment,
    destroy_model_parallel,
    graph_capture,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.forward_context import get_forward_context, set_forward_context
from vllm.model_executor.layers.fused_moe import FusedMoEFactory
from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
from vllm.model_executor.layers.fused_moe.routed_experts_capturer import (
    RoutedExpertsSink,
)
from vllm.v1.worker.workspace import init_workspace_manager


def make_config(directory, retained, args, tp_size):
    directory.mkdir(parents=True, exist_ok=True)
    config = Qwen3MoeConfig(
        architectures=["Qwen3MoeForCausalLM"],
        vocab_size=32,
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        moe_intermediate_size=args.intermediate_size,
        num_hidden_layers=1,
        num_attention_heads=16,
        num_key_value_heads=4,
        num_experts=args.num_experts,
        num_experts_per_tok=args.top_k,
        max_position_embeddings=8192,
    )
    if retained < args.num_experts:
        config.quantization_config = {
            "quant_method": "compressed-tensors",
            "config_groups": {},
            "format": "fakequant",
            "quantization_status": "compressed",
            "moe_sparsity_config": {"method": "novice", "num_experts": [retained]},
        }
    config.save_pretrained(directory)
    return VllmConfig(
        model_config=ModelConfig(
            model=str(directory),
            skip_tokenizer_init=True,
            dtype="bfloat16",
            max_model_len=8192,
            enforce_eager=True,
        ),
        parallel_config=ParallelConfig(tensor_parallel_size=tp_size),
        kernel_config=KernelConfig(moe_backend=args.backend),
    )


def make_layer(config, args):
    with set_current_vllm_config(config):
        layer = FusedMoEFactory(
            num_experts=args.num_experts,
            top_k=args.top_k,
            hidden_size=args.hidden_size,
            intermediate_size=args.intermediate_size,
            params_dtype=torch.bfloat16,
            prefix="model.layers.0.mlp.experts",
            renormalize=True,
            router_logits_dtype=torch.float32,
        ).cuda()
    assert layer._quant_method.is_monolithic == (args.backend == "flashinfer_trtllm")
    return layer


def inputs_for_case(tokens, retained, fraction, args):
    generator = torch.Generator().manual_seed(1234 + tokens)
    k, experts = args.top_k, args.num_experts
    novice_slots = torch.rand(tokens * k, generator=generator).argsort()[
        : round(tokens * k * fraction)
    ]
    novice_mask = torch.zeros(tokens * k, dtype=torch.bool)
    novice_mask[novice_slots] = True
    novice_mask = novice_mask.view(tokens, k)
    retained_ids = torch.rand(tokens, retained, generator=generator).argsort(dim=1)[
        :, :k
    ]
    novice_ids = (
        torch.rand(tokens, experts - retained, generator=generator).argsort(dim=1)[
            :, :k
        ]
        + retained
    )
    assert retained_ids.shape[1] == novice_ids.shape[1] == k
    ids = torch.where(novice_mask, novice_ids, retained_ids)
    logits = torch.rand(tokens, experts, generator=generator) - 8
    scores = torch.rand(tokens, k, generator=generator) + 2
    logits.scatter_(1, ids, scores)
    hidden = torch.randn(tokens, args.hidden_size, generator=generator).to(
        torch.bfloat16
    )
    return hidden.cuda(), logits.cuda(), ids, novice_mask.float().mean().item()


@contextlib.contextmanager
def without_accumulation(substitution):
    original = substitution.compute_output
    substitution.compute_output = lambda hidden, weights, ids: None
    try:
        yield
    finally:
        substitution.compute_output = original


@contextlib.contextmanager
def native_without_novices(routed_experts, routing_sink):
    """Keep identical compact dispatch, omitting novice output for diagnostics.

    This is valid inference only for the no-novice-selection profile. Using
    the standard forward preserves the runner, native kernel and TP reduction.
    """
    had_override = "forward_monolithic" in routed_experts.__dict__
    original = routed_experts.forward_monolithic
    original_sink = routed_experts.routing_sink
    routed_experts.forward_monolithic = MethodType(
        RoutedExperts.forward_monolithic, routed_experts
    )
    routed_experts.routing_sink = routing_sink
    try:
        yield
    finally:
        routed_experts.routing_sink = original_sink
        if had_override:
            routed_experts.forward_monolithic = original
        else:
            del routed_experts.forward_monolithic


def capture(function):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(4):
            function()
    torch.cuda.current_stream().wait_stream(stream)
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with (
        graph_capture(
            device=torch.device("cuda", torch.accelerator.current_device_index())
        ) as ctx,
        torch.cuda.graph(graph, stream=ctx.stream),
    ):
        output = function()
    graph.replay()
    torch.accelerator.synchronize()
    return graph, output


def time_call(function, iterations, cache_flush=None):
    dist.barrier()
    torch.accelerator.synchronize()
    if cache_flush is not None:
        events = [
            (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
            for _ in range(iterations)
        ]
        for started, finished in events:
            cache_flush.zero_()
            started.record()
            function()
            finished.record()
        events[-1][1].synchronize()
        microseconds = (
            sum(start.elapsed_time(end) for start, end in events) * 1000 / iterations
        )
        elapsed = torch.tensor(microseconds).cuda()
        dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
        return elapsed.item()
    started, finished = (
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True),
    )
    started.record()
    for _ in range(iterations):
        function()
    finished.record()
    finished.synchronize()
    elapsed = torch.tensor(started.elapsed_time(finished) * 1000 / iterations).cuda()
    dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
    return elapsed.item()


def measure_pair(
    full,
    compact,
    full_config,
    compact_config,
    args,
    retained,
    rank,
    reference=None,
    reference_config=None,
):
    substitution = compact.routed_experts.expert_substitution
    capture_sink = (
        RoutedExpertsSink(
            replace(compact.moe_config, max_num_tokens=max(args.tokens)),
            lambda ids: None,
        )
        if args.monolithic_diagnostics and rank == 0
        else None
    )

    def diagnostic_context(name):
        if name == "remap_only_diagnostic":
            return without_accumulation(substitution)
        if name in ("retained_native_diagnostic", "capture_only_diagnostic"):
            sink = capture_sink if name == "capture_only_diagnostic" else None
            return native_without_novices(compact.routed_experts, sink)
        return contextlib.nullcontext()

    cache_flush = (
        torch.empty(256 * 1024**2, dtype=torch.int8, device="cuda")
        if args.cache_mode == "cold"
        else None
    )
    rows = []
    profiles = {"none": 0.0, "rare": 0.08, "uniform": 1 - retained / args.num_experts}
    if args.all_novices:
        profiles["all"] = 1.0
    for tokens in args.tokens:
        for profile, fraction in profiles.items():
            hidden, logits, expected_ids, actual_fraction = inputs_for_case(
                tokens, retained, fraction, args
            )

            def invoke(layer, config, hidden=hidden, logits=logits):
                with set_current_vllm_config(config), set_forward_context(None, config):
                    get_forward_context().all_moe_layers = None
                    return layer(hidden, logits)

            with set_current_vllm_config(compact_config):
                _, actual_ids = compact.router.select_experts(
                    hidden_states=hidden,
                    router_logits=logits,
                    topk_indices_dtype=torch.int32,
                )
            assert torch.equal(
                actual_ids.cpu().sort(dim=1).values, expected_ids.sort(dim=1).values
            )
            actual_full = invoke(full, full_config)
            actual_compact = invoke(compact, compact_config)
            assert (
                torch.isfinite(actual_full).all()
                and torch.isfinite(actual_compact).all()
            )
            if profile == "none":
                torch.testing.assert_close(
                    actual_compact, actual_full, atol=0.01, rtol=0.03
                )
            graphs = {}
            outputs = []
            variants = [
                ("ordinary", full, full_config),
                ("mone", compact, compact_config),
            ]
            if not compact._quant_method.is_monolithic:
                variants.append(("remap_only_diagnostic", compact, compact_config))
            elif args.monolithic_diagnostics:
                variants.extend(
                    [
                        ("retained_native_diagnostic", compact, compact_config),
                        ("capture_only_diagnostic", compact, compact_config),
                    ]
                )
            if reference is not None:
                variants.append(("mone_reference", reference, reference_config))
                reference_output = invoke(reference, reference_config)
                torch.testing.assert_close(
                    actual_compact, reference_output, atol=0.01, rtol=0.03
                )
            for name, layer, config in variants:
                with diagnostic_context(name):
                    function = lambda layer=layer, config=config: invoke(layer, config)
                    if profile == "none" and name in (
                        "retained_native_diagnostic",
                        "capture_only_diagnostic",
                    ):
                        torch.testing.assert_close(
                            function(), actual_compact, atol=0, rtol=0
                        )
                    if args.mode == "graph":
                        graph, output = capture(function)
                        graphs[name] = graph
                        outputs.append(output)
                    else:
                        for _ in range(4):
                            function()
                    if name == "capture_only_diagnostic" and rank == 0:
                        assert capture_sink is not None
                        torch.testing.assert_close(
                            capture_sink.buffer[:tokens].cpu().long().sort(-1).values,
                            expected_ids.sort(-1).values,
                        )
                if args.mode != "graph":
                    graphs[name] = function
            measurements = {name: [] for name in graphs}
            order_rng = random.Random(2026)
            for _ in range(args.repeats):
                order = list(graphs)
                order_rng.shuffle(order)
                for name in order:
                    context = (
                        diagnostic_context(name)
                        if args.mode != "graph"
                        else contextlib.nullcontext()
                    )
                    with context:
                        function = (
                            graphs[name].replay
                            if args.mode == "graph"
                            else graphs[name]
                        )
                        measurements[name].append(
                            time_call(function, args.iterations, cache_flush)
                        )
            medians = {
                name: statistics.median(values) for name, values in measurements.items()
            }
            row = {
                "backend": args.backend,
                "mode": args.mode,
                "cache_mode": args.cache_mode,
                "tp": dist.get_world_size(),
                "tokens": tokens,
                "experts": args.num_experts,
                "retained": retained,
                "hidden": args.hidden_size,
                "intermediate": args.intermediate_size,
                "top_k": args.top_k,
                "profile": profile,
                "novice_route_fraction": actual_fraction,
                "median_us": medians,
                "samples_us": measurements,
                "mone_over_ordinary": medians["mone"] / medians["ordinary"],
                "novice_accumulation_delta_us": (
                    medians["mone"] - medians["remap_only_diagnostic"]
                    if "remap_only_diagnostic" in medians
                    else None
                ),
                "reference_backend": args.reference_backend,
                "monolithic_overhead_us": (
                    medians["mone"] - medians["retained_native_diagnostic"]
                    if args.monolithic_diagnostics
                    else None
                ),
                "routing_capture_delta_us": (
                    medians["capture_only_diagnostic"]
                    - medians["retained_native_diagnostic"]
                    if args.monolithic_diagnostics
                    else None
                ),
                "novice_processing_delta_us": (
                    medians["mone"] - medians["capture_only_diagnostic"]
                    if args.monolithic_diagnostics
                    else None
                ),
                "ordinary_expert_bytes_per_rank": sum(
                    parameter.numel() * parameter.element_size()
                    for parameter in full.routed_experts.parameters()
                ),
                "mone_expert_bytes_per_rank": sum(
                    parameter.numel() * parameter.element_size()
                    for parameter in compact.routed_experts.parameters()
                ),
            }
            rows.append(row)
            if rank == 0:
                print(json.dumps(row), flush=True)
                with args.output.open("a") as output:
                    output.write(json.dumps(row) + "\n")
            del graphs, outputs
            gc.collect()
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend",
        choices=["triton", "flashinfer_cutlass", "flashinfer_trtllm"],
        required=True,
    )
    parser.add_argument("--reference-backend", choices=["triton", "flashinfer_cutlass"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--tokens", type=int, nargs="+", default=[1, 4, 16, 64, 256, 1024, 4096]
    )
    parser.add_argument("--retained", type=int, nargs="+", default=[96])
    parser.add_argument("--num-experts", type=int, default=128)
    parser.add_argument("--hidden-size", type=int, default=2048)
    parser.add_argument("--intermediate-size", type=int, default=768)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--mode", choices=["graph", "eager"], default="graph")
    parser.add_argument("--cache-mode", choices=["warm", "cold"], default="warm")
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--all-novices", action="store_true")
    parser.add_argument(
        "--monolithic-diagnostics",
        action="store_true",
        help="Time compact dispatch without novice output and with capture only",
    )
    args = parser.parse_args()
    if args.monolithic_diagnostics and args.backend != "flashinfer_trtllm":
        parser.error("--monolithic-diagnostics requires --backend flashinfer_trtllm")
    rank, local_rank, world = (
        int(os.environ.get(key, default))
        for key, default in (("RANK", "0"), ("LOCAL_RANK", "0"), ("WORLD_SIZE", "1"))
    )
    torch.accelerator.set_device_index(local_rank)
    torch.manual_seed(42 + rank)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="mone-perf-") as directory:
        root = Path(directory)
        full_config = make_config(root / "full", args.num_experts, args, world)
        init_method = "env://" if world > 1 else f"file://{root / 'dist'}"
        with set_current_vllm_config(full_config):
            init_distributed_environment(
                world, rank, init_method, local_rank, backend="nccl"
            )
            initialize_model_parallel(world, 1)
            init_workspace_manager(torch.device("cuda", local_rank))
        if rank == 0:
            print(
                f"GPU: {torch.cuda.get_device_name()}; TP={world}; "
                f"started={time.time()}",
                flush=True,
            )
        with torch.inference_mode():
            for retained in args.retained:
                compact_config = make_config(
                    root / f"compact-{retained}", retained, args, world
                )
                full, compact = (
                    make_layer(full_config, args),
                    make_layer(compact_config, args),
                )
                reference = reference_config = None
                if args.reference_backend:
                    reference_args = argparse.Namespace(**vars(args))
                    reference_args.backend = args.reference_backend
                    reference_config = make_config(
                        root / f"reference-{retained}", retained, reference_args, world
                    )
                    reference = make_layer(reference_config, reference_args)
                with torch.no_grad():
                    full.routed_experts.w13_weight.normal_(std=0.01)
                    full.routed_experts.w2_weight.normal_(std=0.01)
                    compact.routed_experts.w13_weight.copy_(
                        full.routed_experts.w13_weight[:retained]
                    )
                    compact.routed_experts.w2_weight.copy_(
                        full.routed_experts.w2_weight[:retained]
                    )
                    compact.routed_experts.expert_substitution.novices.normal_(std=0.1)
                    if reference is not None:
                        reference.routed_experts.w13_weight.copy_(
                            compact.routed_experts.w13_weight
                        )
                        reference.routed_experts.w2_weight.copy_(
                            compact.routed_experts.w2_weight
                        )
                        reference.routed_experts.expert_substitution.novices.copy_(
                            compact.routed_experts.expert_substitution.novices
                        )
                layers = [(full, full_config), (compact, compact_config)]
                if reference is not None:
                    layers.append((reference, reference_config))
                for layer, config in layers:
                    with set_current_vllm_config(config):
                        layer._quant_method.process_weights_after_loading(
                            layer.routed_experts
                        )
                assert full.routed_experts.w13_weight.shape[0] == args.num_experts
                assert compact.routed_experts.w13_weight.shape[0] == retained
                measure_pair(
                    full,
                    compact,
                    full_config,
                    compact_config,
                    args,
                    retained,
                    rank,
                    reference,
                    reference_config,
                )
                for _, config in layers:
                    config.compilation_config.static_forward_context.clear()
                    config.compilation_config.static_all_moe_layers.clear()
                del full, compact, reference, layers
                gc.collect()
                torch.accelerator.empty_cache()
        destroy_model_parallel()
        destroy_distributed_environment()


if __name__ == "__main__":
    main()
