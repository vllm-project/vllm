# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Benchmark the RDNA4 FlyDSL all-reduce against PyNCCL on PCIe GPUs.

The benchmark intentionally forces the NCCL Simple protocol. It measures eager
and HIP-graph execution separately and reports the slowest rank, which is the
latency visible to the collective caller.
"""

import argparse
import gc
import json
import math
import os
import socket
import statistics
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

os.environ["NCCL_PROTO"] = "Simple"
# AITER's normal package import eagerly loads FlyDSL operators from a newer
# revision than the one used by vLLM's RDNA4 kernels.  The AOT import mode keeps
# the benchmark-only HIP oracle isolated from the production FlyDSL path.
os.environ["AITER_AOT_IMPORT"] = "1"

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

KIB = 1024
MIB = 1024 * KIB


@dataclass
class Result:
    world_size: int
    execution: str
    requested_provider: str
    selected_provider: str
    size_bytes: int
    supported: bool
    latency_ms: float | None
    sample_spread_pct: float | None
    rank_spread_pct: float | None
    algorithm_bandwidth_gbps: float | None
    bus_bandwidth_gbps: float | None


def _parse_size(value: str) -> int:
    text = value.strip().lower()
    multipliers = {
        "kib": KIB,
        "kb": 1000,
        "k": KIB,
        "mib": MIB,
        "mb": 1000 * 1000,
        "m": MIB,
        "gib": 1024 * MIB,
        "gb": 1000 * 1000 * 1000,
        "g": 1024 * MIB,
        "b": 1,
    }
    for suffix in sorted(multipliers, key=len, reverse=True):
        if text.endswith(suffix):
            number = text[: -len(suffix)]
            return int(float(number) * multipliers[suffix])
    return int(text)


def _default_sizes(world_size: int) -> list[int]:
    small = [8 * KIB, 32 * KIB, 64 * KIB, 128 * KIB]
    if world_size == 2:
        boundaries = [
            64 * KIB - 16,
            64 * KIB,
            64 * KIB + 16,
            256 * KIB,
            512 * KIB,
            MIB,
        ]
    else:
        boundaries = [
            192 * KIB - 16,
            192 * KIB,
            192 * KIB + 16,
            256 * KIB,
            384 * KIB,
            512 * KIB,
            768 * KIB,
            MIB - 64,
            MIB,
            MIB + 64,
            32 * MIB - 64,
            32 * MIB,
            32 * MIB + 64,
            48 * MIB - 64,
            48 * MIB,
            48 * MIB + 64,
        ]
    powers = [2 * MIB, 4 * MIB, 8 * MIB, 16 * MIB, 64 * MIB, 128 * MIB]
    return sorted(set(small + boundaries + powers))


def _find_open_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _load_aiter_custom_allreduce():
    """Load AITER's HIP custom-all-reduce without its FlyDSL front end."""
    import importlib

    import aiter

    ar_ops = importlib.import_module("aiter.ops.custom_all_reduce")
    torch_guard = importlib.import_module("aiter.jit.utils.torch_guard")
    aiter.torch_compile_guard = torch_guard.torch_compile_guard
    op_names = (
        "all_gather_reg",
        "all_gather_unreg",
        "all_reduce",
        "allocate_meta_buffer",
        "dispose",
        "free_meta_buffer",
        "fused_allreduce_rmsnorm",
        "fused_allreduce_rmsnorm_pad",
        "fused_allreduce_rmsnorm_quant",
        "fused_allreduce_rmsnorm_quant_per_group",
        "fused_qknorm_allreduce",
        "get_graph_buffer_count",
        "get_graph_buffer_ipc_meta",
        "get_meta_buffer_ipc_handle",
        "init_custom_ar",
        "meta_size",
        "reduce_scatter",
        "register_graph_buffers",
        "register_input_buffer",
        "register_output_buffer",
    )
    for name in op_names:
        setattr(aiter, name, getattr(ar_ops, name))

    # gfx1201 requires IPC-exported uncached allocations to end on a 64-KiB
    # boundary.  vLLM's RDNA4 communicator already applies this constraint;
    # AITER's generic wrapper does not.
    raw_allocate_meta_buffer = aiter.allocate_meta_buffer

    def allocate_meta_buffer(size: int) -> int:
        alignment = 64 * KIB
        aligned_size = (size + alignment - 1) // alignment * alignment
        return raw_allocate_meta_buffer(aligned_size)

    aiter.allocate_meta_buffer = allocate_meta_buffer

    module = importlib.import_module(
        "aiter.dist.device_communicators.custom_all_reduce"
    )
    return module.CustomAllreduce


def _iterations_for(size_bytes: int, requested: int) -> int:
    if requested > 0:
        return requested
    return max(10, min(500, math.ceil(512 * MIB / size_bytes)))


def _make_runner(
    *,
    provider: str,
    execution: str,
    inp: torch.Tensor,
    out: torch.Tensor,
    rdna4,
    pynccl,
    aiter_ar,
) -> tuple[Callable[[], None] | None, str]:
    selected = provider
    if provider == "routed":
        use_rdna4 = (
            rdna4.should_use_graph(inp)
            if execution == "graph"
            else rdna4.should_use(inp)
        )
        selected = "rdna4" if use_rdna4 else "pynccl"

    if selected == "rdna4":
        supported = (
            rdna4.should_use_graph(inp)
            if execution == "graph"
            else rdna4.should_use(inp)
        )
        if not supported:
            return None, selected
    elif selected == "aiter":
        if aiter_ar is None or not aiter_ar.should_custom_ar(inp):
            return None, selected

    if execution == "eager":

        def run_eager() -> None:
            if selected == "rdna4":
                result = rdna4.custom_all_reduce(inp, out=out)
            elif selected == "aiter":
                result = aiter_ar.all_reduce(inp, out=out, registered_input=False)
            else:
                result = pynccl.all_reduce(inp, out)
            if result is None:
                raise RuntimeError(f"{selected} unexpectedly rejected the tensor")

        return run_eager, selected

    graph = torch.cuda.CUDAGraph()
    if selected == "rdna4":
        with rdna4.capture(), torch.cuda.graph(graph):
            result = rdna4.custom_all_reduce(inp, out=out)
            if result is None:
                raise RuntimeError("RDNA4 graph capture rejected a supported tensor")
    elif selected == "aiter":
        with aiter_ar.capture(), torch.cuda.graph(graph):
            result = aiter_ar.all_reduce(inp, out=out, registered_input=True)
            if result is None:
                raise RuntimeError("AITER graph capture rejected a supported tensor")
    else:
        with torch.cuda.graph(graph):
            result = pynccl.all_reduce(inp, out)
            if result is None:
                raise RuntimeError("PyNCCL graph capture failed")
    return graph.replay, selected


def _measure(
    run: Callable[[], None],
    *,
    iterations: int,
    warmup: int,
    samples: int,
) -> list[float]:
    for _ in range(warmup):
        run()
    torch.accelerator.synchronize()
    timings = []
    for _ in range(samples):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            run()
        end.record()
        end.synchronize()
        timings.append(float(start.elapsed_time(end)) / iterations)
    return timings


def _correctness(run: Callable[[], None], out: torch.Tensor, expected: float) -> None:
    run()
    torch.accelerator.synchronize()
    torch.testing.assert_close(
        out,
        torch.full_like(out, expected),
        rtol=0,
        atol=0,
    )


def _worker(rank: int, args: argparse.Namespace, port: int) -> None:
    device = torch.device(f"cuda:{rank}")
    torch.accelerator.set_device_index(rank)
    dist.init_process_group(
        backend="gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=args.world_size,
    )

    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
    from vllm.distributed.device_communicators.rdna4_all_reduce import (
        RDNA4AllReduce,
    )

    max_size = max(args.sizes)
    rdna4 = RDNA4AllReduce(dist.group.WORLD, device, max_size=max_size)
    pynccl = PyNcclCommunicator(dist.group.WORLD, device)
    CustomAllreduce = None
    if "aiter" in args.providers:
        CustomAllreduce = _load_aiter_custom_allreduce()
    if rdna4.disabled:
        raise RuntimeError("RDNA4 FlyDSL all-reduce is unavailable")
    if pynccl.disabled:
        raise RuntimeError("PyNCCL is unavailable")
    results: list[Result] = []
    expected = args.world_size * (args.world_size + 1) / 2
    try:
        for execution in args.executions:
            for size_bytes in args.sizes:
                if size_bytes <= 0 or size_bytes % torch.bfloat16.itemsize:
                    raise ValueError(f"invalid BF16 buffer size: {size_bytes}")
                numel = size_bytes // torch.bfloat16.itemsize
                inp = torch.full(
                    (numel,), rank + 1, dtype=torch.bfloat16, device=device
                )
                out = torch.empty_like(inp)
                # AITER's HIP signal state is sensitive to changing grid sizes
                # on gfx1201.  A fresh communicator per benchmark case isolates
                # the kernel oracle from that wrapper-level lifetime bug.
                aiter_ar = None
                if CustomAllreduce is not None:
                    aiter_ar = CustomAllreduce(
                        dist.group.WORLD,
                        device,
                        max_size=size_bytes,
                    )
                    if aiter_ar.disabled:
                        raise RuntimeError("AITER HIP custom all-reduce is unavailable")
                for provider in args.providers:
                    dist.barrier()
                    run, selected = _make_runner(
                        provider=provider,
                        execution=execution,
                        inp=inp,
                        out=out,
                        rdna4=rdna4,
                        pynccl=pynccl,
                        aiter_ar=aiter_ar,
                    )
                    if run is None:
                        if rank == 0:
                            results.append(
                                Result(
                                    args.world_size,
                                    execution,
                                    provider,
                                    selected,
                                    size_bytes,
                                    False,
                                    None,
                                    None,
                                    None,
                                    None,
                                    None,
                                )
                            )
                        continue

                    _correctness(run, out, expected)
                    timings = _measure(
                        run,
                        iterations=_iterations_for(size_bytes, args.iterations),
                        warmup=args.warmup,
                        samples=args.samples,
                    )
                    inp.fill_(rank + 2)
                    changed_expected = args.world_size * (args.world_size + 3) / 2
                    _correctness(run, out, changed_expected)
                    inp.fill_(rank + 1)
                    gathered: list[list[float] | None] = [None] * args.world_size
                    dist.all_gather_object(gathered, timings)
                    if rank == 0:
                        complete = [item for item in gathered if item is not None]
                        slowest_rank_samples = [
                            max(rank_samples[index] for rank_samples in complete)
                            for index in range(args.samples)
                        ]
                        latency_ms = statistics.median(slowest_rank_samples)
                        sample_spread = (
                            100
                            * (max(slowest_rank_samples) - min(slowest_rank_samples))
                            / latency_ms
                        )
                        per_rank_medians = [
                            statistics.median(item) for item in complete
                        ]
                        spread = (
                            100
                            * (max(per_rank_medians) - min(per_rank_medians))
                            / latency_ms
                        )
                        algorithm_bw = size_bytes / (latency_ms * 1e6)
                        bus_bw = (
                            algorithm_bw * 2 * (args.world_size - 1) / args.world_size
                        )
                        results.append(
                            Result(
                                args.world_size,
                                execution,
                                provider,
                                selected,
                                size_bytes,
                                True,
                                latency_ms,
                                sample_spread,
                                spread,
                                algorithm_bw,
                                bus_bw,
                            )
                        )
                    del run
                    gc.collect()
                if aiter_ar is not None:
                    aiter_ar.close()
                    del aiter_ar
                    gc.collect()
                del inp, out
        if rank == 0:
            payload = {
                "schema": "rdna4-all-reduce-baseline-v1",
                "nccl_proto": os.environ["NCCL_PROTO"],
                "device": torch.cuda.get_device_name(0),
                "results": [asdict(result) for result in results],
            }
            rendered = json.dumps(payload, indent=2, sort_keys=True)
            if args.output:
                output_path = Path(args.output)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text(rendered + "\n", encoding="utf-8")
            print(rendered)
            for result in results:
                if result.supported:
                    print(
                        "GEAK_RESULT_LATENCY_MS="
                        f"{result.latency_ms:.9f} "
                        f"case=tp{result.world_size}:{result.execution}:"
                        f"{result.requested_provider}:{result.size_bytes}B "
                        f"selected={result.selected_provider} "
                        f"bus_bandwidth_gbps={result.bus_bandwidth_gbps:.6f}"
                    )
                else:
                    print(
                        f"GEAK_UNSUPPORTED case=tp{result.world_size}:"
                        f"{result.execution}:{result.requested_provider}:"
                        f"{result.size_bytes}B"
                    )
    finally:
        torch.accelerator.synchronize()
        dist.barrier()
        rdna4.close()
        pynccl.destroy()
        dist.destroy_process_group()


def _csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world-size", type=int, choices=(2, 4), default=4)
    parser.add_argument(
        "--sizes",
        help="comma-separated byte sizes; suffixes such as KiB and MiB are accepted",
    )
    parser.add_argument(
        "--providers",
        type=_csv,
        default=["pynccl", "rdna4", "routed"],
    )
    parser.add_argument("--executions", type=_csv, default=["eager", "graph"])
    parser.add_argument("--iterations", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--output", help="optional path for the JSON result")
    args = parser.parse_args()
    invalid_providers = set(args.providers) - {
        "aiter",
        "pynccl",
        "rdna4",
        "routed",
    }
    invalid_executions = set(args.executions) - {"eager", "graph"}
    if invalid_providers or invalid_executions:
        parser.error(
            f"invalid providers={sorted(invalid_providers)} "
            f"or executions={sorted(invalid_executions)}"
        )
    if args.world_size > torch.accelerator.device_count():
        parser.error(
            f"world size {args.world_size} requires more than "
            f"{torch.accelerator.device_count()} visible GPUs"
        )
    args.sizes = (
        [_parse_size(value) for value in _csv(args.sizes)]
        if args.sizes
        else _default_sizes(args.world_size)
    )
    return args


def main() -> None:
    args = parse_args()
    mp.start_processes(
        _worker,
        args=(args, _find_open_port()),
        nprocs=args.world_size,
        join=True,
        start_method="spawn",
    )


if __name__ == "__main__":
    main()
