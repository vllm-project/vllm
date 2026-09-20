# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Graph benchmark of the unified RDNA HIP all-reduce against RCCL.

Run with torch.distributed.run (2 or 4 ranks) and --library pointing to the
compiled csrc/rocm/rdna_custom_all_reduce.cu. Based on the standalone
benchmark_rdna{3,4}_tp{2,4}_custom_all_reduce.py harnesses.

GPU events inside the graph exclude host submission latency. A common device
all-reduce precedes the start event inside each graph to align ranks without
timing host submission skew. Report the median of
sample means, taking the slowest rank for each replay before averaging.
Timing uses zero inputs to avoid overflow in repeated in-place RCCL sums.
Changed-input correctness is checked before and after timing.
"""

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import time
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist


def gather(value, world):
    values = [None] * world
    dist.all_gather_object(values, value)
    return values


def capture(fn, cycles):
    torch.accelerator.synchronize()
    dist.barrier()
    stream = torch.cuda.Stream()
    start = torch.cuda.Event(enable_timing=True, external=True)
    end = torch.cuda.Event(enable_timing=True, external=True)
    arrival = torch.zeros(1, device="cuda")
    with torch.cuda.stream(stream):
        for _ in range(3):
            fn()
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        dist.all_reduce(arrival)
        start.record()
        for _ in range(cycles):
            fn()
        end.record()
    torch.accelerator.synchronize()
    dist.barrier()
    return graph, start, end, arrival, stream


def measure(captured, args, world):
    graph, start, end = captured[:3]
    for _ in range(args.warmup):
        graph.replay()
    torch.accelerator.synchronize()
    elapsed = []
    for _ in range(args.trials):
        graph.replay()
        end.synchronize()
        elapsed.append(start.elapsed_time(end) * 1000 / args.cycles)
    per_rank = gather(elapsed, world)
    slowest = [max(row[i] for row in per_rank) for i in range(args.trials)]
    return statistics.mean(slowest), [statistics.mean(row) for row in per_rank]


def check(inp, out, rccl, custom, args, world, rank):
    captured = capture(custom, 1)
    graph = captured[0]
    maximum = {"custom_max_abs": 0.0, "rccl_max_abs": 0.0}
    for replay in range(args.check_replays):
        torch.manual_seed(1234 + rank + replay * 100)
        inp.copy_(torch.randn_like(inp) * 0.1)
        saved = inp.clone()
        rccl.copy_(inp)
        graph.replay()
        dist.all_reduce(rccl)
        exact = inp.float()
        magnitude = exact.abs()
        dist.all_reduce(exact)
        dist.all_reduce(magnitude)
        u = torch.finfo(inp.dtype).eps / 2
        gamma = (world - 1) * u / (1 - (world - 1) * u)
        bound = gamma * magnitude + torch.finfo(inp.dtype).tiny * u * world
        for name, result in (("custom", out), ("rccl", rccl)):
            error = (result.float() - exact).abs()
            assert torch.isfinite(result).all(), name
            assert torch.all(error <= bound), (name, error.max().item())
            maximum[f"{name}_max_abs"] = max(
                maximum[f"{name}_max_abs"], error.max().item()
            )
        torch.testing.assert_close(inp, saved, rtol=0, atol=0)
        peers = [torch.empty_like(out) for _ in range(world)]
        dist.all_gather(peers, out)
        for peer in peers:
            torch.testing.assert_close(out, peer, rtol=0, atol=0)
    torch.accelerator.synchronize()
    maxima = gather(maximum, world)
    return {key: max(row[key] for row in maxima) for key in maximum}


def selected_path(arch, world, numel):
    if world == 2:
        return "chunked_push" if arch == 3 and numel > 128 * 1024 else "push"
    if numel > 256 * 1024:
        return "bulk_ring"
    if arch == 3:
        return "tagged_ring" if numel >= 128 * 1024 else "pairwise"
    return "oneshot" if numel <= 2048 else "pairwise"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--batches", default="1,2,4,8,16,32,64,128")
    parser.add_argument("--hidden-sizes", default="1024,2048,4096,5120,8192")
    parser.add_argument("--cycles", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--samples", type=int, default=9)
    parser.add_argument("--check-replays", type=int, default=3)
    args = parser.parse_args()
    batches = [int(x) for x in args.batches.split(",")]
    hidden_sizes = [int(x) for x in args.hidden_sizes.split(",")]
    assert all(0 < x <= 128 for x in batches)
    assert all(0 < x <= 8192 for x in hidden_sizes)
    assert min(args.cycles, args.trials, args.samples, args.check_replays) > 0
    world, rank = int(os.environ["WORLD_SIZE"]), int(os.environ["RANK"])
    assert world in (2, 4)
    assert not any(k.startswith("VLLM_RDNA") for k in os.environ), (
        "This benchmark labels the default algorithm policy; unset tuning overrides"
    )
    torch.accelerator.set_device_index(rank)
    dist.init_process_group(
        "nccl", device_id=torch.device(f"cuda:{rank}"), timeout=timedelta(seconds=120)
    )
    torch.ops.load_library(str(args.library.resolve()))
    ops = torch.ops._rdna_custom_ar
    props = torch.cuda.get_device_properties(rank)
    arch = 3 if props.gcnArchName.startswith("gfx11") else 4
    devices = gather({"name": props.name, "arch": props.gcnArchName}, world)
    assert len({row["arch"] for row in devices}) == 1
    stride = 128 * 8192 * 2
    shared, handle = ops.allocate_shared_buffer_and_handle(stride, world)
    handles = gather(handle, world)
    peers = set(ops.get_required_peer_ranks(rank, world))
    pointers = [
        shared if p == rank else ops.open_mem_handle(handles[p]) if p in peers else 0
        for p in range(world)
    ]
    payloads = [p + ops.meta_size() if p else 0 for p in pointers]
    rank_data = torch.empty(ops.rank_data_size(), dtype=torch.uint8, device="cuda")
    reducer = ops.init_custom_ar(pointers, rank_data, rank, stride)
    ops.register_buffer(reducer, payloads)
    metadata = {
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "rccl": torch.cuda.nccl.version(),
        "devices": devices,
        "world_size": world,
        "arch": arch,
        "args": {
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
        "environment": {
            k: v
            for k, v in os.environ.items()
            if k.startswith(("NCCL_", "RCCL_", "HIP_", "ROCR_", "HSA_"))
        },
        "source_sha256": hashlib.sha256(
            Path("csrc/rocm/rdna_custom_all_reduce.cu").read_bytes()
        ).hexdigest(),
        "library_sha256": hashlib.sha256(args.library.read_bytes()).hexdigest(),
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "timing": (
            "graph-prefix device all-reduce before start event; "
            "max across ranks per replay, sample mean, median"
        ),
        "cache": "warm, fixed pointers; no cache flush",
        "scope": "custom out-of-place vs RCCL in-place; zero inputs; no staging copies",
    }
    results = []
    if rank == 0:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        print(json.dumps(metadata), flush=True)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    for hidden in hidden_sizes:
        for batch in batches:
            assert batch * hidden % 8 == 0
            inp = torch.zeros((batch, hidden), dtype=dtype, device="cuda")
            out, rccl = torch.empty_like(inp), torch.zeros_like(inp)

            def custom(inp=inp, out=out):
                ops.all_reduce(reducer, inp, out, payloads[rank], stride)

            def baseline(rccl=rccl):
                dist.all_reduce(rccl)

            errors = check(inp, out, rccl, custom, args, world, rank)
            inp.zero_()
            rccl.zero_()
            graphs = {
                "custom": capture(custom, args.cycles),
                "rccl": capture(baseline, args.cycles),
            }
            samples = {"custom": [], "rccl": []}
            rank_samples = {"custom": [], "rccl": []}
            started = time.monotonic()
            for sample in range(args.samples):
                order = ("custom", "rccl") if sample % 2 == 0 else ("rccl", "custom")
                for provider in order:
                    value, per_rank = measure(graphs[provider], args, world)
                    samples[provider].append(value)
                    rank_samples[provider].append(per_rank)
            for value in (inp, out, rccl):
                assert torch.count_nonzero(value).item() == 0
            del graphs
            post = check(inp, out, rccl, custom, args, world, rank)
            errors = {key: max(errors[key], post[key]) for key in errors}
            custom_us, rccl_us = (
                statistics.median(samples[p]) for p in ("custom", "rccl")
            )
            nbytes = inp.numel() * inp.element_size()
            row = {
                "batch": batch,
                "hidden": hidden,
                "bytes": nbytes,
                "path": selected_path(arch, world, inp.numel()),
                "custom_us": custom_us,
                "rccl_us": rccl_us,
                "speedup": rccl_us / custom_us,
                "custom_bus_gbps": nbytes * 2 * (world - 1) / world / custom_us / 1000,
                "rccl_bus_gbps": nbytes * 2 * (world - 1) / world / rccl_us / 1000,
                "samples_us": samples,
                "rank_samples_us": rank_samples,
                "correctness": errors,
                "wall_seconds": time.monotonic() - started,
            }
            results.append(row)
            if rank == 0:
                print(
                    f"B={batch:3d} H={hidden:4d} {row['path']:12s} "
                    f"custom={custom_us:8.3f} us rccl={rccl_us:8.3f} us "
                    f"speedup={row['speedup']:.3f}x PASS",
                    flush=True,
                )
                args.output.write_text(
                    json.dumps({"metadata": metadata, "results": results}, indent=2)
                )
    torch.accelerator.synchronize()
    dist.barrier()
    ops.dispose(reducer)
    for peer in peers:
        ops.close_mem_handle(pointers[peer])
    ops.free_shared_buffer(shared)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
