# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark the SP/AsyncTP comm+GEMM kernels in
``vllm.model_executor.kernels.sequence_parallel`` against each other and
against doing the GEMM and the collective separately.

Two region-boundary directions are benchmarked, matching the two linear
kinds ``mark_sp_region`` wraps:

- ``qkv``/``gate_up`` (``--proj``): a ``ColumnParallelLinear`` *entering* an
  SP region -- all-gather the token-sharded input, then GEMM.
- ``o``/``down``: a ``RowParallelLinear`` *exiting* an SP region -- GEMM,
  then reduce-scatter the output back to a token shard.

For each, five candidates are timed:

- ``gemm``: the GEMM alone, on the same (already-gathered) input the fused/
  unfused paths feed it -- the compute floor.
- ``collective``: the all-gather/reduce-scatter alone, on the same message
  size -- the comm floor.
- ``unfused``: what ``--enable-sequence-parallel`` runs without
  ``--enable-async-tp`` -- a plain collective, then (or after) the layer's
  own GEMM.
- ``fused``: what ``--enable-async-tp`` swaps in instead -- a
  symmetric-memory kernel that overlaps the collective with the GEMM,
  chosen by ``init_fused_comm_kernel`` exactly as in serving.
- ``baseline``: plain TP with neither SP nor AsyncTP enabled. For
  ``qkv``/``gate_up`` this is identical to ``gemm`` (the input is already
  full/replicated, so there is no collective at this boundary at all). For
  ``o``/``down`` this is the GEMM followed by a full
  ``tensor_model_parallel_all_reduce`` of the ``[M, N]`` partial sums --
  what ``RowParallelLinear(reduce_results=True)`` actually runs -- not the
  smaller reduce-scatter ``collective``/``unfused``/``fused`` use.

Run a TP2 sweep over the Qwen3-32B projections with:

    torchrun --nproc-per-node=2 -- \\
        benchmarks/kernels/benchmark_sequence_parallel.py --proj qkv

Runs on NVIDIA and AMD (ROCm) unchanged -- both expose CUDA-graph capture
through ``torch.cuda``.
"""

import argparse
import json
import os
import statistics
from collections.abc import Callable
from contextlib import contextmanager

import torch
import torch.distributed as dist

from vllm.config import ModelConfig, VllmConfig, set_current_vllm_config
from vllm.distributed import cleanup_dist_env_and_memory
from vllm.distributed.communication_op import tensor_model_parallel_all_reduce
from vllm.distributed.parallel_state import (
    get_tp_group,
    graph_capture,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.model_executor.kernels.sequence_parallel import (
    LinearCommKernel,
    init_fused_comm_kernel,
    init_unfused_comm_kernel,
)
from vllm.model_executor.layers.linear import ColumnParallelLinear, RowParallelLinear
from vllm.models.common.ops.sequence_parallel import sp_all_gather, sp_reduce_scatter
from vllm.platforms import current_platform

# Per model: (N, K) for each projection, where N is the global output width
# and K the global input width. qkv/gate_up are ColumnParallelLinear
# (AllGather+GEMM); o/down are RowParallelLinear (GEMM+ReduceScatter).
# Approximate public dense configs -- close enough for shape-accurate
# latency, not a source of truth for the real checkpoints.
_MODEL_PROJECTIONS: dict[str, dict[str, tuple[int, int]]] = {
    "qwen3_0_6b": {
        "qkv": (16 * 128 + 2 * 8 * 128, 1024),
        "o": (1024, 16 * 128),
        "gate_up": (2 * 3072, 1024),
        "down": (1024, 3072),
    },
    "qwen3_1_7b": {
        "qkv": (16 * 128 + 2 * 8 * 128, 2048),
        "o": (2048, 16 * 128),
        "gate_up": (2 * 6144, 2048),
        "down": (2048, 6144),
    },
    "qwen3_32b": {
        "qkv": (64 * 128 + 2 * 8 * 128, 5120),
        "o": (5120, 64 * 128),
        "gate_up": (2 * 25600, 5120),
        "down": (5120, 25600),
    },
}

# qkv/gate_up all-gather their (token-sharded) input then GEMM; o/down GEMM
# then reduce-scatter their output.
_AG_PROJECTIONS = {"qkv", "gate_up"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--proj",
        choices=("qkv", "gate_up", "o", "down"),
        default="gate_up",
        help="Which SP region boundary to benchmark.",
    )
    parser.add_argument(
        "--model",
        choices=tuple(_MODEL_PROJECTIONS),
        default="qwen3_32b",
        help="Model whose projection shapes (BF16) to benchmark.",
    )
    parser.add_argument(
        "--m",
        type=int,
        nargs="+",
        default=[128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768],
        help="Global token counts to benchmark.",
    )
    parser.add_argument("--n", type=int, help="Output width override.")
    parser.add_argument("--k", type=int, help="Input width override.")
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


def make_linear(
    proj: str,
    n: int,
    k: int,
    world_size: int,
    device: torch.device,
) -> ColumnParallelLinear | RowParallelLinear:
    """Build the unquantized (BF16) ``LinearBase`` the real model would wire
    ``mark_sp_region`` around for ``proj``."""
    with torch.device(device):
        if proj in _AG_PROJECTIONS:
            linear = ColumnParallelLinear(
                k,
                n,
                bias=False,
                gather_output=False,
                params_dtype=torch.bfloat16,
                return_bias=False,
            )
        else:
            linear = RowParallelLinear(
                k,
                n,
                bias=False,
                params_dtype=torch.bfloat16,
                # The baseline reduce-scatters the local partial itself, as
                # the unfused SP path does.
                reduce_results=False,
                return_bias=False,
            )

    weight = torch.randn(
        linear.weight.shape, dtype=torch.bfloat16, device=linear.weight.device
    )
    linear.weight = torch.nn.Parameter(weight, requires_grad=False)
    linear.quant_method.process_weights_after_loading(linear)
    return linear


def pick_fused_kernel(
    linear: ColumnParallelLinear | RowParallelLinear,
) -> LinearCommKernel | None:
    """The kernel ``--enable-async-tp`` would swap in."""
    return init_fused_comm_kernel(linear)


@contextmanager
def accelerator_graph_apis():
    """On NVIDIA/AMD, ``torch.cuda`` already *is* the graph/stream/event API
    (ROCm's build hipifies it in place)."""
    if not current_platform.is_xpu():
        yield
        return

    from functools import partial

    original = {
        name: getattr(torch.cuda, name)
        for name in (
            "Stream",
            "default_stream",
            "current_stream",
            "stream",
            "set_stream",
            "Event",
            "graph",
            "CUDAGraph",
            "graph_pool_handle",
            "is_current_stream_capturing",
        )
    }
    torch.cuda.Stream = torch.xpu.Stream
    torch.cuda.default_stream = partial(torch.xpu.current_stream)
    torch.cuda.current_stream = partial(torch.xpu.current_stream)
    torch.cuda.stream = partial(torch.xpu.stream)
    torch.cuda.set_stream = partial(torch.xpu.set_stream)

    def _xpu_event(*args, blocking=None, **kwargs):
        return torch.xpu.Event(*args, **kwargs)

    torch.cuda.Event = _xpu_event
    torch.cuda.graph = partial(torch.xpu.graph)
    torch.cuda.CUDAGraph = torch.xpu.XPUGraph
    torch.cuda.graph_pool_handle = partial(torch.xpu.graph_pool_handle)
    torch.cuda.is_current_stream_capturing = torch.xpu.is_current_stream_capturing
    try:
        yield
    finally:
        for name, value in original.items():
            setattr(torch.cuda, name, value)


def capture_graph(
    op: Callable[[], torch.Tensor],
    stream: torch.cuda.Stream,
    cpu_group: dist.ProcessGroup,
    device: torch.device,
) -> torch.cuda.CUDAGraph:
    stream.wait_stream(torch.cuda.current_stream())
    dist.barrier(group=cpu_group)
    with torch.cuda.stream(stream):
        for _ in range(3):
            op()
    stream.synchronize()
    dist.barrier(group=cpu_group)

    graph = torch.cuda.CUDAGraph()
    # tensor_model_parallel_all_reduce (used by the "baseline" candidate) may
    # dispatch to an IPC-backed custom all-reduce; graph_capture() is what
    # makes that graph-safe, matching how real serving captures CUDA graphs.
    with graph_capture(device=device), torch.cuda.graph(graph, stream=stream):
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
    proj: str,
    M: int,
    N: int,
    K: int,
    num_workspaces: int,
    warmup_replays: int,
    samples: int,
    device_group: dist.ProcessGroup,
    cpu_group: dist.ProcessGroup,
    device_barrier: Callable[[], None],
) -> dict[str, float | int | str] | None:
    is_ag = proj in _AG_PROJECTIONS
    world_size = dist.get_world_size(device_group)
    rank = dist.get_rank(device_group)
    device = torch.device(
        torch.accelerator.current_accelerator().type,
        torch.accelerator.current_device_index(),
    )
    local_M = (M + world_size - 1) // world_size

    rng = torch.Generator(device=device)
    rng.manual_seed(1000 + rank * 10 + M + K)
    linears = [
        make_linear(proj, N, K, world_size, device) for _ in range(num_workspaces)
    ]

    fused = pick_fused_kernel(linears[0])
    if fused is None:
        print(f"[M={M}, N={N}, K={K}] no fused kernel supports this shape; skipping")
        return None
    unfused_kernels = [init_unfused_comm_kernel(linear) for linear in linears]
    fused_kernels = [fused] + [pick_fused_kernel(linear) for linear in linears[1:]]
    assert all(k is not None for k in fused_kernels), (
        "fused-kernel support should only depend on shape, which is "
        "identical across workspaces"
    )

    if is_ag:
        # Token-sharded input: [local_M, K] per rank, all-gathered to
        # [M, K] before the GEMM.
        locals_ = [
            torch.randn(local_M, K, dtype=torch.bfloat16, device=device, generator=rng)
            for _ in range(num_workspaces)
        ]
        gathered = [sp_all_gather(x) for x in locals_]

        def bare_gemm(linear=None, x=None) -> torch.Tensor:
            return linear.quant_method.apply(linear, x)

        def bare_collective(x: torch.Tensor) -> torch.Tensor:
            return sp_all_gather(x)

        candidates: dict[str, list[Callable[[], torch.Tensor]]] = {
            "gemm": [
                lambda ln=ln, x=x: bare_gemm(ln, x) for ln, x in zip(linears, gathered)
            ],
            "collective": [lambda x=x: bare_collective(x) for x in locals_],
            "unfused": [
                lambda k=k, x=x: k.apply(x) for k, x in zip(unfused_kernels, locals_)
            ],
            "fused": [
                lambda k=k, x=x: k.apply(x) for k, x in zip(fused_kernels, locals_)
            ],
            # Plain TP (no SP): the input entering this projection is
            # already full/replicated across ranks, so there is no
            # collective here at all -- this is just the GEMM alone,
            # identical to "gemm" above.
            "baseline": [
                lambda ln=ln, x=x: bare_gemm(ln, x) for ln, x in zip(linears, gathered)
            ],
        }
        correctness_input = locals_[0]
        # Both paths all-gather the full token dim first, so every rank's
        # output already covers every row -- nothing to slice off.
        correctness_rows = slice(None)
    else:
        # Full input [M, K_local] per rank; the GEMM output [M, N] is
        # reduce-scattered to [local_M, N].
        local_K = K // world_size
        fulls = [
            torch.randn(M, local_K, dtype=torch.bfloat16, device=device, generator=rng)
            for _ in range(num_workspaces)
        ]
        partials = [
            linear.quant_method.apply(linear, x) for linear, x in zip(linears, fulls)
        ]

        def bare_gemm(linear=None, x=None) -> torch.Tensor:
            return linear.quant_method.apply(linear, x)

        def bare_collective(x: torch.Tensor) -> torch.Tensor:
            return sp_reduce_scatter(x)

        def bare_gemm_allreduce(linear=None, x=None) -> torch.Tensor:
            # Plain TP (no SP): GEMM on the local K-shard, then a full
            # all-reduce of the [M, N] partial sums (not a reduce-scatter)
            # -- what RowParallelLinear(reduce_results=True) does.
            return tensor_model_parallel_all_reduce(bare_gemm(linear, x))

        candidates = {
            "gemm": [
                lambda ln=ln, x=x: bare_gemm(ln, x) for ln, x in zip(linears, fulls)
            ],
            "collective": [lambda p=p: bare_collective(p) for p in partials],
            "unfused": [
                lambda k=k, x=x: k.apply(x) for k, x in zip(unfused_kernels, fulls)
            ],
            "fused": [
                lambda k=k, x=x: k.apply(x) for k, x in zip(fused_kernels, fulls)
            ],
            "baseline": [
                lambda ln=ln, x=x: bare_gemm_allreduce(ln, x)
                for ln, x in zip(linears, fulls)
            ],
        }
        correctness_input = fulls[0]
        correctness_rows = slice(0, local_M)

    expected = unfused_kernels[0].apply(correctness_input)
    actual = fused_kernels[0].apply(correctness_input)
    torch.accelerator.synchronize(device)
    torch.testing.assert_close(
        actual[correctness_rows],
        expected[correctness_rows],
        rtol=5e-2,
        atol=4.0,
        msg=f"proj={proj}, M={M}, N={N}, K={K}",
    )

    candidate_graphs = {}
    keepalive: list[object] = []
    for name, ops in candidates.items():
        stream = torch.cuda.Stream()
        candidate_graphs[name] = [
            capture_graph(op, stream, cpu_group, device) for op in ops
        ]
        keepalive.append(stream)

    times = benchmark_graphs(
        candidate_graphs, warmup_replays, samples, device_group, device_barrier
    )
    floor = max(times["gemm"], times["collective"])
    return {
        "proj": proj,
        "fused_kernel": type(fused_kernels[0]).__name__,
        "M": M,
        "N": N,
        "K": K,
        "gemm_us": times["gemm"],
        "collective_us": times["collective"],
        "floor_us": floor,
        "unfused_us": times["unfused"],
        "fused_us": times["fused"],
        "fused_vs_unfused": (times["fused"] - times["unfused"]) / times["unfused"],
        "fused_vs_floor": (times["fused"] - floor) / floor,
        "baseline_us": times["baseline"],
        "unfused_vs_baseline": (
            (times["unfused"] - times["baseline"]) / times["baseline"]
        ),
        "fused_vs_baseline": (times["fused"] - times["baseline"]) / times["baseline"],
    }


def print_results(results: list[dict[str, float | int | str]]) -> None:
    proj = results[0]["proj"]
    kernel = results[0]["fused_kernel"]
    print(f"### {proj} SP kernels (bf16, fused={kernel})")
    print(
        "| M | N | K | GEMM (us) | collective (us) | floor (us) | "
        "unfused (us) | fused (us) | fused vs unfused | fused vs floor | "
        "baseline (us) | unfused vs baseline | fused vs baseline |"
    )
    print("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in results:
        print(
            f"| {r['M']} | {r['N']} | {r['K']} | {r['gemm_us']:.1f} | "
            f"{r['collective_us']:.1f} | {r['floor_us']:.1f} | "
            f"{r['unfused_us']:.1f} | {r['fused_us']:.1f} | "
            f"{r['fused_vs_unfused']:+.1%} | {r['fused_vs_floor']:+.1%} | "
            f"{r['baseline_us']:.1f} | {r['unfused_vs_baseline']:+.1%} | "
            f"{r['fused_vs_baseline']:+.1%} |"
        )


def make_device_barrier(tp_group) -> Callable[[], None]:
    """A cheap device-side rendezvous to order each timed replay after every
    rank reaches it, without including the rendezvous itself in the timed
    interval (it runs before ``start.record()``).

    Prefers the direct pynccl/RCCL call bypassing ``torch.distributed``
    dispatch overhead, same as vLLM uses internally (NVIDIA/AMD); falls
    back to a plain ``dist.all_reduce`` for backends without one.
    """
    device = torch.accelerator.current_device_index()
    sync_input = torch.zeros(1, device=device)
    pynccl_comm = getattr(tp_group.device_communicator, "pynccl_comm", None)
    if pynccl_comm is not None:
        sync_output = torch.empty_like(sync_input)

        def device_barrier() -> None:
            pynccl_comm.all_reduce(sync_input, sync_output)

        return device_barrier

    def device_barrier() -> None:
        dist.all_reduce(sync_input, group=tp_group.device_group)

    return device_barrier


def main() -> None:
    args = parse_args()
    default_n, default_k = _MODEL_PROJECTIONS[args.model][args.proj]
    n = args.n or default_n
    k = args.k or default_k
    assert args.m and min(args.m) > 0
    k_alignment = 64
    assert n % k_alignment == 0 and k % k_alignment == 0
    assert args.num_workspaces > 0
    assert args.warmup_replays >= 0
    assert args.samples > 0

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.accelerator.set_device_index(local_rank)
    init_distributed_environment(backend=current_platform.dist_backend)
    world_size = dist.get_world_size()

    config = VllmConfig()
    config.model_config = ModelConfig(dtype="bfloat16")
    with set_current_vllm_config(config):
        # Builds the TP communicator as in serving, and enables the
        # symmetric-memory group the fused kernels rely on.
        initialize_model_parallel(tensor_model_parallel_size=world_size)

    tp_group = get_tp_group()
    device_barrier = make_device_barrier(tp_group)

    with set_current_vllm_config(config), accelerator_graph_apis():
        results = [
            r
            for M in args.m
            if (
                r := benchmark_shape(
                    args.proj,
                    M,
                    n,
                    k,
                    args.num_workspaces,
                    args.warmup_replays,
                    args.samples,
                    tp_group.device_group,
                    tp_group.cpu_group,
                    device_barrier,
                )
            )
            is not None
        ]

    if tp_group.rank_in_group == 0 and results:
        print_results(results)
        if args.json:
            with open(args.json, "w") as f:
                json.dump(results, f, indent=2)

    dist.barrier(group=tp_group.cpu_group)
    cleanup_dist_env_and_memory()


if __name__ == "__main__":
    main()
