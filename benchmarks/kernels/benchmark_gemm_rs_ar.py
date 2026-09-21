# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark the SM100 GEMM-RS/AR kernel against the model's unfused path.

The baseline is what a model runs without the fusion: the projection's own
linear kernel followed by vLLM's collective (``sp_reduce_scatter`` for RS,
``tensor_model_parallel_all_reduce`` for AR), with the collective backend
chosen by vLLM's dispatch exactly as in serving. The same two kernels are also
timed bare (the GEMM alone, the collective alone on a precomputed partial);
their max is the floor a perfectly overlapped fused kernel could reach.

All ranks must belong to one NVLink domain. Run a TP8 sweep over the Kimi-K3
projections with:

    torchrun --nproc-per-node=8 -- benchmarks/kernels/benchmark_gemm_rs_ar.py

or the DeepSeek-V4.1 MXFP8 ``wo_b`` projection at TP4 with:

    torchrun --nproc-per-node=4 -- benchmarks/kernels/benchmark_gemm_rs_ar.py \
        --model deepseek_v41 --backend flashinfer_cutedsl
"""

import argparse
import json
import os
import statistics
from collections.abc import Callable

import torch
import torch.distributed as dist

from vllm.config import ModelConfig, VllmConfig, set_current_vllm_config
from vllm.distributed import (
    cleanup_dist_env_and_memory,
    tensor_model_parallel_all_reduce,
)
from vllm.distributed.parallel_state import (
    get_tp_group,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.model_executor.kernels.linear.cute_dsl.gemm_rs_ar import GemmRsAr
from vllm.model_executor.layers.linear import RowParallelLinear
from vllm.models.common.ops.sequence_parallel import sp_reduce_scatter

# Per model: (hidden size N, global input widths K). Kimi-K3 lists the
# shared-expert down-proj and attention O-proj; DeepSeek-V4.1 lists the
# ``wo_b`` output projection (o_groups * o_lora_rank).
_MODEL_PROJECTIONS = {
    "kimi_k3": (7168, (6144, 12288)),
    "deepseek_v41": (5120, (8192,)),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("rs", "ar"),
        default="rs",
        help="Collective mode to benchmark.",
    )
    parser.add_argument(
        "--m",
        type=int,
        nargs="+",
        default=[128, 512, 2048, 8192, 32768],
        help="Global token counts to benchmark.",
    )
    parser.add_argument(
        "--model",
        choices=tuple(_MODEL_PROJECTIONS),
        default="kimi_k3",
        help="Model whose projection shapes set the default --n and --k.",
    )
    parser.add_argument(
        "--backend",
        choices=("bf16", "flashinfer_cutlass", "flashinfer_cutedsl"),
        default="bf16",
        help="Weight dtype/layout: BF16 or an online-MXFP8 FlashInfer kernel.",
    )
    parser.add_argument(
        "--k",
        type=int,
        nargs="+",
        help=(
            "Per-rank input dimensions. By default, derive the selected "
            "model's projection dimensions from the TP world size."
        ),
    )
    parser.add_argument(
        "--n",
        type=int,
        help="Output width. Defaults to the selected model's hidden size.",
    )
    parser.add_argument(
        "--num-workspaces",
        type=int,
        default=10,
        help="Pointer-distinct inputs and CUDA graphs to rotate.",
    )
    parser.add_argument("--warmup-replays", type=int, default=5)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--json", type=str, help="Also write the results here.")
    return parser.parse_args()


def make_projection(
    gemm_rs_ar: GemmRsAr,
    weight: torch.Tensor,
    world_size: int,
    backend: str,
) -> RowParallelLinear:
    """Wrap a BF16 [N, K] weight the way the model would hand it to GEMM-RS.

    For the MXFP8 backends this quantizes online through the FlashInfer
    kernel's post-load step, so the fused kernel and the unfused baseline see
    the production weight and scale layouts.
    """
    N, K = weight.shape
    quant_config = None
    if backend != "bf16":
        from vllm.config.quantization import QuantizationConfigArgs
        from vllm.model_executor.layers.quantization.online.base import (
            OnlineQuantizationConfig,
        )

        quant_config = OnlineQuantizationConfig(QuantizationConfigArgs(linear="mxfp8"))
    with torch.device(weight.device):
        linear = RowParallelLinear(
            K * world_size,
            N,
            bias=False,
            params_dtype=weight.dtype,
            quant_config=quant_config,
            # The baseline reduces the local partial itself, like the models.
            reduce_results=False,
            return_bias=False,
        )
    # Select the fused path before online quantization replaces the weights.
    assert gemm_rs_ar.can_run(linear)
    linear.weight = torch.nn.Parameter(weight, requires_grad=False)
    linear.quant_method.process_weights_after_loading(linear)
    return linear


def capture_graph(
    op: Callable[[], torch.Tensor],
    stream: torch.cuda.Stream,
    cpu_group: dist.ProcessGroup,
) -> torch.cuda.CUDAGraph:
    stream.wait_stream(torch.cuda.current_stream())
    dist.barrier(group=cpu_group)
    with torch.cuda.stream(stream):
        for _ in range(3):
            op()
    stream.synchronize()
    dist.barrier(group=cpu_group)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        op()
    torch.cuda.current_stream().wait_stream(stream)
    dist.barrier(group=cpu_group)
    return graph


def benchmark_graphs(
    candidate_graphs: dict[str, list[torch.cuda.CUDAGraph]],
    warmup_replays: int,
    samples: int,
    device_group: dist.ProcessGroup,
    device_barrier: Callable[[], None],
) -> dict[str, float]:
    """Median over samples of the slowest rank's replay time, in us."""
    names = list(candidate_graphs)
    for round_index in range(warmup_replays):
        for offset in range(len(names)):
            name = names[(round_index + offset) % len(names)]
            graphs = candidate_graphs[name]
            device_barrier()
            graphs[round_index % len(graphs)].replay()
    torch.accelerator.synchronize()

    timings: dict[str, list[float]] = {name: [] for name in names}
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for sample_index in range(samples):
        for offset in range(len(names)):
            name = names[(sample_index + offset) % len(names)]
            graphs = candidate_graphs[name]
            device_barrier()
            start.record()
            graphs[sample_index % len(graphs)].replay()
            end.record()
            end.synchronize()

            elapsed = torch.tensor(
                start.elapsed_time(end) * 1000,
                dtype=torch.float64,
                device=torch.accelerator.current_device_index(),
            )
            dist.all_reduce(elapsed, op=dist.ReduceOp.MAX, group=device_group)
            timings[name].append(elapsed.item())
    return {name: statistics.median(values) for name, values in timings.items()}


def benchmark_shape(
    gemm_rs_ar: GemmRsAr,
    mode: str,
    backend: str,
    M: int,
    N: int,
    K: int,
    num_workspaces: int,
    warmup_replays: int,
    samples: int,
    device_group: dist.ProcessGroup,
    cpu_group: dist.ProcessGroup,
    device_barrier: Callable[[], None],
) -> dict[str, float | int | str]:
    all_reduce = mode == "ar"
    world_size = dist.get_world_size(device_group)
    rank = dist.get_rank(device_group)
    device = torch.device("cuda", torch.accelerator.current_device_index())
    local_M = (M + world_size - 1) // world_size

    rng = torch.Generator(device=device)
    rng.manual_seed(1000 + rank * 10 + M + K)
    inputs = [
        torch.randn(M, K, dtype=torch.bfloat16, device=device, generator=rng)
        for _ in range(num_workspaces)
    ]
    projections = [
        make_projection(
            gemm_rs_ar,
            torch.randn(N, K, dtype=torch.bfloat16, device=device, generator=rng),
            world_size,
            backend,
        )
        for _ in range(num_workspaces)
    ]

    def collective(partial: torch.Tensor) -> torch.Tensor:
        if all_reduce:
            return tensor_model_parallel_all_reduce(partial)
        return sp_reduce_scatter(partial)

    def gemm(x: torch.Tensor, linear: RowParallelLinear) -> torch.Tensor:
        return linear(x)

    def unfused(x: torch.Tensor, linear: RowParallelLinear) -> torch.Tensor:
        return collective(linear(x))

    def fused(x: torch.Tensor, linear: RowParallelLinear) -> torch.Tensor:
        return gemm_rs_ar.apply(x, linear)

    # The bare collective runs on the GEMM's own output so it sees the same
    # message size and layout the unfused path hands to the collective.
    partials = [linear(x) for x, linear in zip(inputs, projections)]

    expected = unfused(inputs[0], projections[0])
    actual = fused(inputs[0], projections[0])
    torch.accelerator.synchronize(device)
    rows = M if all_reduce else min(max(M - rank * local_M, 0), local_M)
    torch.testing.assert_close(
        actual[:rows], expected[:rows], rtol=5e-2, atol=4.0, msg=f"M={M}, K={K}"
    )

    candidates: dict[str, list[Callable[[], torch.Tensor]]] = {
        "gemm": [lambda x=x, w=w: gemm(x, w) for x, w in zip(inputs, projections)],
        "collective": [lambda p=p: collective(p) for p in partials],
        "vllm": [lambda x=x, w=w: unfused(x, w) for x, w in zip(inputs, projections)],
        "gemm_rs_ar": [
            lambda x=x, w=w: fused(x, w) for x, w in zip(inputs, projections)
        ],
    }

    candidate_graphs = {}
    keepalive: list[object] = []
    for name, ops in candidates.items():
        stream = torch.cuda.Stream()
        candidate_graphs[name] = [capture_graph(op, stream, cpu_group) for op in ops]
        keepalive.append(stream)

    times = benchmark_graphs(
        candidate_graphs, warmup_replays, samples, device_group, device_barrier
    )
    floor = max(times["gemm"], times["collective"])
    return {
        "mode": mode.upper(),
        "backend": backend,
        "M": M,
        "N": N,
        "K": K,
        "gemm_us": times["gemm"],
        "collective_us": times["collective"],
        "floor_us": floor,
        "vllm_us": times["vllm"],
        "gemm_rs_ar_us": times["gemm_rs_ar"],
        "latency_change": (times["gemm_rs_ar"] - times["vllm"]) / times["vllm"],
        "floor_gap": (times["gemm_rs_ar"] - floor) / floor,
    }


def print_results(results: list[dict[str, float | int | str]]) -> None:
    collective = results[0]["mode"]
    backend = results[0]["backend"]
    print(f"### GEMM-{collective} vs vLLM unfused path ({backend} weights)")
    print(
        f"| M | N | K | GEMM (us) | {collective} (us) | floor (us) | vLLM (us) | "
        f"GEMM-{collective} (us) | vs vLLM | vs floor |"
    )
    print("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in results:
        print(
            f"| {r['M']} | {r['N']} | {r['K']} | {r['gemm_us']:.1f} | "
            f"{r['collective_us']:.1f} | {r['floor_us']:.1f} | {r['vllm_us']:.1f} | "
            f"{r['gemm_rs_ar_us']:.1f} | {r['latency_change']:+.1%} | "
            f"{r['floor_gap']:+.1%} |"
        )


def main() -> None:
    args = parse_args()
    default_n, default_k = _MODEL_PROJECTIONS[args.model]
    if args.n is None:
        args.n = default_n
    assert args.m and min(args.m) > 0
    assert args.n % 128 == 0
    assert args.num_workspaces > 0
    assert args.warmup_replays >= 0
    assert args.samples > 0

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.accelerator.set_device_index(local_rank)
    init_distributed_environment()
    world_size = dist.get_world_size()
    if args.k is None:
        assert all(K % world_size == 0 for K in default_k)
        K_values = [K // world_size for K in default_k]
    else:
        K_values = args.k
    k_alignment = 64 if args.backend == "bf16" else 128
    assert all(K % k_alignment == 0 for K in K_values)

    config = VllmConfig()
    config.model_config = ModelConfig(dtype="bfloat16")
    # Pin the online-MXFP8 kernel the projections quantize through.
    config.kernel_config.linear_backend = (
        "auto" if args.backend == "bf16" else args.backend
    )
    with set_current_vllm_config(config):
        # Builds the TP communicator as in serving, so the baseline's
        # collective is whatever vLLM's dispatch selects (FlashInfer all-reduce
        # first, then NCCL symmetric memory, custom all-reduce, pynccl).
        initialize_model_parallel(tensor_model_parallel_size=world_size)

    tp_group = get_tp_group()
    pynccl_comm = tp_group.device_communicator.pynccl_comm
    assert pynccl_comm is not None
    sync_input = torch.zeros(1, device=torch.accelerator.current_device_index())
    sync_output = torch.empty_like(sync_input)

    def device_barrier() -> None:
        # Order the timed launch after a device-side rank rendezvous without
        # including the rendezvous itself in the measured event interval.
        pynccl_comm.all_reduce(sync_input, sync_output)

    gemm_rs_ar = GemmRsAr(
        max_M=max(args.m),
        N=args.n,
        all_reduce=args.mode == "ar",
    )
    with set_current_vllm_config(config):
        results = [
            benchmark_shape(
                gemm_rs_ar,
                args.mode,
                args.backend,
                M,
                args.n,
                K,
                args.num_workspaces,
                args.warmup_replays,
                args.samples,
                tp_group.device_group,
                tp_group.cpu_group,
                device_barrier,
            )
            for K in K_values
            for M in args.m
        ]
    del gemm_rs_ar

    if tp_group.rank_in_group == 0:
        print_results(results)
        if args.json:
            with open(args.json, "w") as f:
                json.dump(results, f, indent=2)

    dist.barrier(group=tp_group.cpu_group)
    cleanup_dist_env_and_memory()


if __name__ == "__main__":
    main()
