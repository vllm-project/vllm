# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark the Kimi K3 latent-MoE tail and its up-projection kernels.

The ``up-projection`` subcommand isolates the TP-local dynamic and static-M
skinny GEMMs. It rotates weights through a working set larger than L2 to model
successive model layers.

The ``whole-tail`` subcommand measures the distributed operator. Its reference
path includes two AllReduces, RMSNorm, the replicated up-projection, and the
final add. CUDA-event samples report the slowest rank so cross-rank skew is
included.

Examples:

.. code-block:: console

    .venv/bin/python \
      benchmarks/kernels/benchmark_kimi_k3_latent_moe_tail.py up-projection

    torchrun --nproc-per-node=8 \
      benchmarks/kernels/benchmark_kimi_k3_latent_moe_tail.py whole-tail

    torchrun --nproc-per-node=8 \
      benchmarks/kernels/benchmark_kimi_k3_latent_moe_tail.py whole-tail \
      --backend fused --routed-input both

For multi-node runs, launch one ``torchrun`` agent per node and use a shared
rendezvous endpoint.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
from collections.abc import Callable, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import cutlass
import cutlass.utils as utils
import torch
import torch.distributed as dist
import torch.nn.functional as F
from cuda.bindings import driver as cuda

from vllm.distributed import get_tp_group
from vllm.distributed.parallel_state import (
    init_distributed_environment,
    initialize_model_parallel,
    set_custom_all_reduce,
)
from vllm.model_executor.layers.fused_moe.moe_output import UnfinalizedMoEOutput
from vllm.model_executor.warmup.cutedsl_warmup import cutedsl_warmup
from vllm.models.kimi_k3.nvidia.ops import latent_moe_tail
from vllm.models.kimi_k3.nvidia.ops.cute_dsl.latent_moe_tail import (
    fused_add_multicast_gemm,
    fused_add_multicast_skinny_gemm,
)

HIDDEN_SIZE = 7168
LATENT_SIZE = 3584
RMS_EPS = 0.1
MAX_NUM_TOKENS = 16
MMA_TILER_MN = (64, 32)
CLUSTER_SHAPE_MN = (1, 8)
B_PRIME_STAGES = 2


def parse_up_projection_config(
    value: str,
) -> fused_add_multicast_skinny_gemm.SkinnyConfig:
    try:
        values = [int(part) for part in value.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "config must be BLOCK,OUTPUTS,K_UNROLL[,VECTOR_WIDTH[,PREFETCH_B]]"
        ) from error
    if len(values) in (3, 4):
        return fused_add_multicast_skinny_gemm.SkinnyConfig(*values)
    if len(values) == 5 and values[4] in (0, 1):
        return fused_add_multicast_skinny_gemm.SkinnyConfig(
            *values[:4],
            prefetch_b_before_pdl=bool(values[4]),
        )
    raise argparse.ArgumentTypeError(
        "config must be BLOCK,OUTPUTS,K_UNROLL"
        "[,VECTOR_WIDTH[,PREFETCH_B]], where PREFETCH_B is 0 or 1"
    )


def parse_tail_skinny_config(
    value: str,
) -> tuple[int, fused_add_multicast_skinny_gemm.SkinnyConfig]:
    try:
        values = [int(part) for part in value.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "config must be M,BLOCK,OUTPUTS,K_UNROLL[,VECTOR_WIDTH[,PREFETCH_B]]"
        ) from error
    if len(values) == 4:
        num_tokens, *config = values
        return num_tokens, fused_add_multicast_skinny_gemm.SkinnyConfig(*config)
    if len(values) == 5:
        num_tokens, *config = values
        return num_tokens, fused_add_multicast_skinny_gemm.SkinnyConfig(*config)
    if len(values) == 6 and values[5] in (0, 1):
        num_tokens, block, outputs, unroll, vector_width, prefetch = values
        return num_tokens, fused_add_multicast_skinny_gemm.SkinnyConfig(
            block,
            outputs,
            unroll,
            vector_width,
            bool(prefetch),
        )
    raise argparse.ArgumentTypeError(
        "config must be M,BLOCK,OUTPUTS,K_UNROLL"
        "[,VECTOR_WIDTH[,PREFETCH_B]], where PREFETCH_B is 0 or 1"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="scope", required=True)

    up_projection = subparsers.add_parser(
        "up-projection",
        help="Benchmark the isolated TP-local up-projection kernels.",
    )
    up_projection.add_argument(
        "--backend",
        choices=("dynamic", "skinny", "both"),
        default="both",
    )
    up_projection.add_argument("--tp-size", type=int, default=16)
    up_projection.add_argument(
        "--num-tokens",
        type=int,
        nargs="+",
        default=[*range(1, 9), 16],
    )
    up_projection.add_argument(
        "--skinny-config",
        type=parse_up_projection_config,
        action="append",
        help="Benchmark a static-M config for every selected token count.",
    )
    up_projection.add_argument("--cache-multiplier", type=float, default=2.0)
    up_projection.add_argument("--max-weights", type=int, default=64)
    up_projection.add_argument("--warmup-replays", type=int, default=10)
    up_projection.add_argument("--samples", type=int, default=31)
    up_projection.add_argument("--output", type=Path)

    whole_tail = subparsers.add_parser(
        "whole-tail",
        help="Benchmark the distributed latent-MoE tail operator.",
    )
    whole_tail.add_argument(
        "--backend",
        choices=("reference", "fused", "both"),
        default="both",
    )
    whole_tail.add_argument(
        "--num-tokens",
        type=int,
        nargs="+",
        default=[1, 5, 8, 16],
    )
    whole_tail.add_argument(
        "--routed-input",
        choices=("finalized", "deferred", "both"),
        default="finalized",
        help="Select whether the fused tail performs the top-k finalize step.",
    )
    whole_tail.add_argument(
        "--top-k",
        type=int,
        default=8,
        help="Expert count used by the synthetic deferred-finalize input.",
    )
    whole_tail.add_argument("--warmup-replays", type=int, default=20)
    whole_tail.add_argument("--samples", type=int, default=51)
    whole_tail.add_argument(
        "--skinny-max-num-tokens",
        type=int,
        nargs="+",
        help="Override the fused operator's static-M cutoff; use 0 for dynamic-only.",
    )
    whole_tail.add_argument(
        "--skinny-config",
        type=parse_tail_skinny_config,
        action="append",
        help="Override one static-M config for tuning.",
    )
    whole_tail.add_argument("--output", type=Path)
    return parser.parse_args()


def percentile(samples: Sequence[float], fraction: float) -> float:
    ordered = sorted(samples)
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    upper_weight = position - lower
    return ordered[lower] * (1.0 - upper_weight) + ordered[upper] * upper_weight


def summarize(samples_us: Sequence[float]) -> dict[str, Any]:
    mean_us = statistics.mean(samples_us)
    return {
        "median_us": statistics.median(samples_us),
        "p10_us": percentile(samples_us, 0.1),
        "p90_us": percentile(samples_us, 0.9),
        "mean_us": mean_us,
        "cv_pct": statistics.pstdev(samples_us) / mean_us * 100.0,
        "samples_us": list(samples_us),
    }


def rotating_weight_count(
    shard_size: int,
    cache_multiplier: float,
    limit: int,
) -> int:
    properties = torch.cuda.get_device_properties(
        torch.accelerator.current_device_index()
    )
    weight_bytes = shard_size * LATENT_SIZE * 2
    target_bytes = math.ceil(properties.L2_cache_size * cache_multiplier)
    return max(2, min(limit, math.ceil(target_bytes / weight_bytes)))


def capture_up_projection_graph(
    launches: Sequence[Callable[[], None]],
) -> torch.cuda.CUDAGraph:
    for launch in launches:
        launch()
    torch.accelerator.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for launch in launches:
            launch()
    torch.accelerator.synchronize()
    return graph


def benchmark_up_projection_graph(
    graph: torch.cuda.CUDAGraph,
    *,
    operations_per_replay: int,
    warmup_replays: int,
    samples: int,
) -> dict[str, Any]:
    for _ in range(warmup_replays):
        graph.replay()
    torch.accelerator.synchronize()

    samples_us = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(samples):
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        samples_us.append(start.elapsed_time(end) * 1000.0 / operations_per_replay)
    return summarize(samples_us)


class DynamicKernel:
    def __init__(
        self,
        shard_size: int,
        mailbox: torch.Tensor,
        shared_shard: torch.Tensor,
    ) -> None:
        self.shard_size = shard_size
        self.mailbox = mailbox
        self.mailbox_c = fused_add_multicast_gemm._as_cute(mailbox)
        compile_latent = torch.empty(
            (1, MAX_NUM_TOKENS, LATENT_SIZE),
            dtype=torch.bfloat16,
            device=mailbox.device,
        )
        compile_weight = torch.empty(
            (1, shard_size, LATENT_SIZE),
            dtype=torch.bfloat16,
            device=mailbox.device,
        )
        cluster_size = math.prod(CLUSTER_SHAPE_MN)
        max_active_clusters = utils.HardwareInfo().get_max_active_clusters(cluster_size)
        self.compiled = fused_add_multicast_gemm.compile_kernel(
            (MAX_NUM_TOKENS, shard_size, LATENT_SIZE, 1),
            fused_add_multicast_gemm._as_cute(
                compile_latent,
                dynamic_m=True,
            ),
            fused_add_multicast_gemm._as_cute(compile_weight),
            self.mailbox_c,
            fused_add_multicast_gemm._as_cute(shared_shard),
            HIDDEN_SIZE,
            shard_size,
            MMA_TILER_MN,
            CLUSTER_SHAPE_MN,
            max_active_clusters,
            B_PRIME_STAGES,
        )

    def launch(
        self,
        latent: torch.Tensor,
        weight: torch.Tensor,
        shared_shard: torch.Tensor,
    ) -> None:
        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        self.compiled(
            fused_add_multicast_gemm._as_cute(
                latent.unsqueeze(0),
                dynamic_m=True,
            ),
            fused_add_multicast_gemm._as_cute(weight.unsqueeze(0)),
            self.mailbox_c,
            fused_add_multicast_gemm._as_cute(shared_shard),
            cutlass.Int64(latent.shape[0]),
            cutlass.Int64(self.mailbox.data_ptr()),
            stream,
        )


class SkinnyKernel:
    def __init__(
        self,
        num_tokens: int,
        shard_size: int,
        config: fused_add_multicast_skinny_gemm.SkinnyConfig,
    ) -> None:
        self.compiled = fused_add_multicast_skinny_gemm.compile_kernel(
            num_rows=num_tokens,
            latent_dim=LATENT_SIZE,
            hidden_dim=HIDDEN_SIZE,
            shard_dim=shard_size,
            config=config,
        )

    def launch(
        self,
        latent: torch.Tensor,
        weight: torch.Tensor,
        shared_shard: torch.Tensor,
        mailbox: torch.Tensor,
    ) -> None:
        self.compiled(
            fused_add_multicast_skinny_gemm._as_cute(latent),
            fused_add_multicast_skinny_gemm._as_cute(weight),
            fused_add_multicast_skinny_gemm._as_cute(shared_shard),
            cutlass.Int64(mailbox.data_ptr()),
            cuda.CUstream(torch.cuda.current_stream().cuda_stream),
        )


def check_up_projection_output(
    actual: torch.Tensor,
    latent: torch.Tensor,
    weight: torch.Tensor,
    shared_shard: torch.Tensor,
) -> None:
    gemm = F.linear(latent.float(), weight.float()).to(torch.bfloat16)
    expected = (gemm.float() + shared_shard.float()).to(torch.bfloat16)
    torch.testing.assert_close(actual, expected, atol=8e-2, rtol=3e-2)


def make_up_projection_launches(
    launch: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], None],
    latent: torch.Tensor,
    weights: Sequence[torch.Tensor],
    shared_shard: torch.Tensor,
) -> list[Callable[[], None]]:
    return [
        lambda weight=weight: launch(latent, weight, shared_shard) for weight in weights
    ]


def benchmark_up_projection(args: argparse.Namespace) -> None:
    if args.tp_size <= 0 or HIDDEN_SIZE % args.tp_size:
        raise ValueError("TP size must be positive and divide the hidden size")
    if any(not 1 <= num_tokens <= MAX_NUM_TOKENS for num_tokens in args.num_tokens):
        raise ValueError("--num-tokens values must be in [1, 16]")
    if args.cache_multiplier <= 0 or args.max_weights <= 0:
        raise ValueError("cache multiplier and max weights must be positive")
    if args.warmup_replays < 0 or args.samples <= 0:
        raise ValueError("warmup replays must be nonnegative and samples positive")

    torch.accelerator.set_device_index(0)
    device = torch.device("cuda", 0)
    if torch.cuda.get_device_capability(device)[0] != 10:
        raise RuntimeError("Kimi K3 latent-MoE tail requires SM100")

    shard_size = HIDDEN_SIZE // args.tp_size
    weight_count = rotating_weight_count(
        shard_size,
        args.cache_multiplier,
        args.max_weights,
    )
    torch.manual_seed(20260726)
    weights = [
        torch.randn(
            (shard_size, LATENT_SIZE),
            dtype=torch.bfloat16,
            device=device,
        )
        / LATENT_SIZE**0.5
        for _ in range(weight_count)
    ]
    mailbox = torch.empty(
        (1, MAX_NUM_TOKENS, HIDDEN_SIZE),
        dtype=torch.bfloat16,
        device=device,
    )
    shared = torch.randn(
        (MAX_NUM_TOKENS, HIDDEN_SIZE),
        dtype=torch.bfloat16,
        device=device,
    )
    shared_shard = shared[:, :shard_size]
    use_dynamic = args.backend in ("dynamic", "both")
    use_skinny = args.backend in ("skinny", "both")
    dynamic_kernel = (
        DynamicKernel(shard_size, mailbox, shared_shard) if use_dynamic else None
    )

    results = []
    for num_tokens in args.num_tokens:
        latent = torch.randn(
            (num_tokens, LATENT_SIZE),
            dtype=torch.bfloat16,
            device=device,
        )
        result: dict[str, Any] = {"num_tokens": num_tokens}
        if dynamic_kernel is not None:
            launches = make_up_projection_launches(
                dynamic_kernel.launch,
                latent,
                weights,
                shared_shard,
            )
            graph = capture_up_projection_graph(launches)
            result["dynamic"] = benchmark_up_projection_graph(
                graph,
                operations_per_replay=len(launches),
                warmup_replays=args.warmup_replays,
                samples=args.samples,
            )
            check_up_projection_output(
                mailbox[0, :num_tokens, :shard_size],
                latent,
                weights[-1],
                shared_shard[:num_tokens],
            )
        if use_skinny:
            configs = args.skinny_config or [
                fused_add_multicast_skinny_gemm.config_for_m(
                    num_tokens,
                    shard_size,
                )
            ]
            skinny_results = []
            for config in configs:
                skinny_kernel = SkinnyKernel(num_tokens, shard_size, config)

                def launch_skinny(
                    latent: torch.Tensor,
                    weight: torch.Tensor,
                    shared_shard: torch.Tensor,
                    *,
                    skinny_kernel: SkinnyKernel = skinny_kernel,
                    num_tokens: int = num_tokens,
                ) -> None:
                    skinny_kernel.launch(
                        latent,
                        weight,
                        shared_shard[:num_tokens],
                        mailbox,
                    )

                launches = make_up_projection_launches(
                    launch_skinny,
                    latent,
                    weights,
                    shared_shard,
                )
                graph = capture_up_projection_graph(launches)
                timing = benchmark_up_projection_graph(
                    graph,
                    operations_per_replay=len(launches),
                    warmup_replays=args.warmup_replays,
                    samples=args.samples,
                )
                check_up_projection_output(
                    mailbox[0, :num_tokens, :shard_size],
                    latent,
                    weights[-1],
                    shared_shard[:num_tokens],
                )
                skinny_results.append(
                    {
                        "config": asdict(config),
                        **timing,
                    }
                )
            result["skinny"] = skinny_results
        results.append(result)

    properties = torch.cuda.get_device_properties(device)
    report = {
        "scope": "up-projection",
        "device": properties.name,
        "compute_capability": list(torch.cuda.get_device_capability(device)),
        "tp_size": args.tp_size,
        "shard_size": shard_size,
        "weight_count": weight_count,
        "cache_multiplier": args.cache_multiplier,
        "warmup_replays": args.warmup_replays,
        "samples": args.samples,
        "results": results,
    }
    rendered = json.dumps(report, indent=2)
    print(rendered, flush=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


def capture_tail_graph(
    operation: Callable[[], torch.Tensor],
    cpu_group: dist.ProcessGroup,
) -> tuple[torch.cuda.CUDAGraph, torch.Tensor]:
    for _ in range(3):
        dist.barrier(group=cpu_group)
        output = operation()
    torch.accelerator.synchronize()

    dist.barrier(group=cpu_group)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = operation()
    torch.accelerator.synchronize()
    return graph, output


def benchmark_tail_graph(
    graph: torch.cuda.CUDAGraph,
    *,
    warmup_replays: int,
    samples: int,
    device_group: dist.ProcessGroup,
    cpu_group: dist.ProcessGroup,
) -> dict[str, Any]:
    for _ in range(warmup_replays):
        graph.replay()
    torch.accelerator.synchronize()

    dist.barrier(group=cpu_group)
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(samples + 1)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(samples + 1)]
    for start, end in zip(starts, ends):
        start.record()
        graph.replay()
        end.record()
    torch.accelerator.synchronize()

    samples_us = torch.tensor(
        [start.elapsed_time(end) * 1000.0 for start, end in zip(starts, ends)],
        dtype=torch.float64,
        device=torch.accelerator.current_device_index(),
    )
    dist.all_reduce(samples_us, op=dist.ReduceOp.MAX, group=device_group)
    return summarize(samples_us[1:].tolist())


def make_inputs(
    num_tokens: int,
    rank: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(20260726 + 100 * num_tokens + rank)
    routed = torch.randn(
        (num_tokens, LATENT_SIZE),
        dtype=torch.bfloat16,
        device=device,
    ).mul_(0.01)
    shared = torch.randn(
        (num_tokens, HIDDEN_SIZE),
        dtype=torch.bfloat16,
        device=device,
    )
    return routed, shared


def make_deferred_input(
    routed: torch.Tensor,
    top_k: int,
) -> UnfinalizedMoEOutput:
    num_tokens = routed.shape[0]
    gemm2_permuted = (
        routed[:, None, :]
        .expand(num_tokens, top_k, LATENT_SIZE)
        .contiguous()
        .view(num_tokens * top_k, LATENT_SIZE)
    )
    expert_weights = torch.zeros(
        num_tokens,
        top_k,
        dtype=routed.dtype,
        device=routed.device,
    )
    expert_weights[:, 0] = 1
    expanded_idx = torch.arange(
        num_tokens * top_k,
        dtype=torch.int32,
        device=routed.device,
    ).view(num_tokens, top_k)
    return UnfinalizedMoEOutput(
        gemm2_permuted=gemm2_permuted,
        expert_weights=expert_weights,
        expanded_idx_to_permuted_idx=expanded_idx,
    )


def make_reference(
    routed: torch.Tensor,
    shared: torch.Tensor,
    rms_weight: torch.Tensor,
    up_weight: torch.Tensor,
    device_group: dist.ProcessGroup,
) -> Callable[[], torch.Tensor]:
    routed_workspace = torch.empty_like(routed)
    shared_workspace = torch.empty_like(shared)

    def reference() -> torch.Tensor:
        routed_workspace.copy_(routed)
        dist.all_reduce(routed_workspace, group=device_group)
        normalized = F.rms_norm(
            routed_workspace,
            (LATENT_SIZE,),
            rms_weight,
            RMS_EPS,
        )
        projected = F.linear(normalized, up_weight)
        shared_workspace.copy_(shared)
        dist.all_reduce(shared_workspace, group=device_group)
        return projected.add(shared_workspace)

    return reference


def check_fused_output(
    fused_output: torch.Tensor,
    reference: Callable[[], torch.Tensor],
    cpu_group: dist.ProcessGroup,
) -> None:
    dist.barrier(group=cpu_group)
    expected = reference()
    torch.testing.assert_close(fused_output, expected, atol=8e-2, rtol=3e-2)


def benchmark_whole_tail(args: argparse.Namespace) -> None:
    if any(not 1 <= num_tokens <= 16 for num_tokens in args.num_tokens):
        raise ValueError("--num-tokens values must be in [1, 16]")
    if args.warmup_replays < 0 or args.samples <= 0:
        raise ValueError("warmup replays must be nonnegative and samples positive")
    if args.top_k <= 0:
        raise ValueError("--top-k must be positive")
    if args.skinny_max_num_tokens is not None and any(
        not 0 <= cutoff <= 8 for cutoff in args.skinny_max_num_tokens
    ):
        raise ValueError("--skinny-max-num-tokens must be in [0, 8]")
    skinny_configs = dict(args.skinny_config or ())
    if len(skinny_configs) != len(args.skinny_config or ()):
        raise ValueError("--skinny-config must not repeat an M value")
    if any(not 1 <= num_tokens <= 8 for num_tokens in skinny_configs):
        raise ValueError("--skinny-config M values must be in [1, 8]")
    if not {"RANK", "WORLD_SIZE", "LOCAL_RANK"} <= os.environ.keys():
        raise RuntimeError("launch this benchmark with torchrun")

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local_rank)
    torch.accelerator.set_device_index(device)
    init_distributed_environment()
    if world_size > 8:
        set_custom_all_reduce(False)
    initialize_model_parallel(tensor_model_parallel_size=world_size)
    device_group = get_tp_group().device_group
    cpu_group = dist.new_group(backend="gloo")

    if torch.cuda.get_device_capability(device)[0] != 10:
        raise RuntimeError("Kimi K3 latent-MoE tail requires SM100")

    torch.manual_seed(20260726)
    rms_weight = 1 + 0.1 * torch.randn(
        LATENT_SIZE,
        dtype=torch.bfloat16,
        device=device,
    )
    up_weight = (
        torch.randn(
            (HIDDEN_SIZE, LATENT_SIZE),
            dtype=torch.bfloat16,
            device=device,
        )
        / LATENT_SIZE**0.5
    )

    use_reference = args.backend in ("reference", "both")
    use_fused = args.backend in ("fused", "both")
    routed_input_modes = (
        ("finalized", "deferred")
        if args.routed_input == "both"
        else (args.routed_input,)
    )
    fused_ops = []
    if use_fused:
        production_config_for_m = fused_add_multicast_skinny_gemm.config_for_m

        def config_for_m(
            num_rows: int,
            shard_dim: int = 896,
        ) -> fused_add_multicast_skinny_gemm.SkinnyConfig:
            config = skinny_configs.get(num_rows)
            if config is not None:
                return config
            return production_config_for_m(num_rows, shard_dim)

        fused_add_multicast_skinny_gemm.config_for_m = config_for_m
        cutoffs = args.skinny_max_num_tokens or [latent_moe_tail._SKINNY_MAX_NUM_TOKENS]
        for cutoff in cutoffs:
            latent_moe_tail._SKINNY_MAX_NUM_TOKENS = cutoff
            latent_moe_tail.KimiK3LatentMoETailOp._instances.clear()
            for routed_input_mode in routed_input_modes:
                fused_ops.append(
                    (
                        cutoff,
                        routed_input_mode,
                        latent_moe_tail.KimiK3LatentMoETailOp.initialize(
                            hidden_size=HIDDEN_SIZE,
                            latent_size=LATENT_SIZE,
                            dtype=torch.bfloat16,
                            device=device,
                            rms_eps=RMS_EPS,
                            experts_per_token=(
                                args.top_k if routed_input_mode == "deferred" else 0
                            ),
                        ),
                    )
                )
        cutedsl_warmup()

    results = []
    for num_tokens in args.num_tokens:
        routed, shared = make_inputs(num_tokens, rank, device)
        routed_inputs: dict[str, torch.Tensor | UnfinalizedMoEOutput] = {
            "finalized": routed
        }
        if "deferred" in routed_input_modes:
            routed_inputs["deferred"] = make_deferred_input(routed, args.top_k)
        reference = make_reference(
            routed,
            shared,
            rms_weight,
            up_weight,
            device_group,
        )
        result: dict[str, Any] = {"num_tokens": num_tokens}
        if use_reference:
            reference_graph, _ = capture_tail_graph(reference, cpu_group)
            result["reference"] = benchmark_tail_graph(
                reference_graph,
                warmup_replays=args.warmup_replays,
                samples=args.samples,
                device_group=device_group,
                cpu_group=cpu_group,
            )
        for cutoff, routed_input_mode, fused_op in fused_ops:
            routed_input = routed_inputs[routed_input_mode]

            def fused(
                routed_input: torch.Tensor | UnfinalizedMoEOutput = routed_input,
                shared: torch.Tensor = shared,
                fused_op: latent_moe_tail.KimiK3LatentMoETailOp = fused_op,
            ) -> torch.Tensor:
                return fused_op(routed_input, shared, rms_weight, up_weight)

            fused_graph, fused_output = capture_tail_graph(fused, cpu_group)
            fused_key_parts = ["fused"]
            if routed_input_mode != "finalized" or len(routed_input_modes) > 1:
                fused_key_parts.append(routed_input_mode)
            if len(cutoffs) > 1:
                fused_key_parts.extend(("skinny", "max", str(cutoff)))
            fused_key = "_".join(fused_key_parts)
            result[fused_key] = benchmark_tail_graph(
                fused_graph,
                warmup_replays=args.warmup_replays,
                samples=args.samples,
                device_group=device_group,
                cpu_group=cpu_group,
            )
            check_fused_output(fused_output, reference, cpu_group)
            if "reference" in result:
                speedup = (
                    result["reference"]["median_us"] / result[fused_key]["median_us"]
                )
                if fused_key == "fused":
                    result["speedup"] = speedup
                else:
                    result[f"{fused_key}_speedup"] = speedup
        results.append(result)

    properties = torch.cuda.get_device_properties(device)
    report = {
        "scope": "whole-tail",
        "device": properties.name,
        "compute_capability": list(torch.cuda.get_device_capability(device)),
        "world_size": world_size,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "warmup_replays": args.warmup_replays,
        "samples": args.samples,
        "routed_input": args.routed_input,
        "top_k": args.top_k,
        "skinny_max_num_tokens": sorted({cutoff for cutoff, _, _ in fused_ops}),
        "skinny_configs": {
            str(num_tokens): asdict(config)
            for num_tokens, config in skinny_configs.items()
        },
        "timing_scope": {
            "reference": (
                "two input copies, two AllReduces, RMSNorm, full replicated "
                "up-projection GEMM, and final add"
            ),
            "fused": (
                "routed AllReduce/RMSNorm plus shared ReduceScatter, sharded "
                "up-projection/multicast, and Lamport copy"
            ),
        },
        "results": results,
    }
    if rank == 0:
        rendered = json.dumps(report, indent=2)
        print(rendered, flush=True)
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered + "\n", encoding="utf-8")

    dist.barrier(group=cpu_group)


def main() -> None:
    args = parse_args()
    if args.scope == "up-projection":
        benchmark_up_projection(args)
        return

    from vllm.config import VllmConfig, set_current_vllm_config

    with set_current_vllm_config(VllmConfig()):
        benchmark_whole_tail(args)


if __name__ == "__main__":
    main()
