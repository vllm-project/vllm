"""Microbenchmark dispatch overhead for the same trivial CUDA kernel exposed
through three different binding mechanisms:

    unstable  -> TORCH_LIBRARY (classic vLLM csrc, unboxed dispatcher)
    stable    -> STABLE_TORCH_LIBRARY (vLLM's libtorch_stable migration target,
                                       boxed dispatcher)
    tvmffi    -> TVM_FFI_DLL_EXPORT_TYPED_FUNC (no torch dispatcher at all)

We use a 1-element float32 tensor so kernel compute time is negligible and
per-call latency is dominated by Python -> binding -> launch overhead.
"""
import os
import statistics
import sys
import time

import torch
import tvm_ffi

HERE = os.path.dirname(os.path.abspath(__file__))
BUILD = os.path.join(HERE, "build")

# Load the three .so files.
torch.ops.load_library(os.path.join(BUILD, "unstable.so"))
torch.ops.load_library(os.path.join(BUILD, "stable.so"))
tvmffi_mod = tvm_ffi.load_module(os.path.join(BUILD, "tvmffi.so"))

device = torch.device("cuda:0")
dtype = torch.float32

N = 1  # 1-element so compute is ~free
in_t = torch.randn(N, device=device, dtype=dtype)
out_t = torch.empty_like(in_t)
factor = 2.0


def correctness():
    expected = in_t * factor
    for name, fn in [
        ("unstable", lambda: torch.ops.bench_unstable.scale(out_t, in_t, factor)),
        ("stable",   lambda: torch.ops.bench_stable.scale(out_t, in_t, factor)),
        ("tvmffi",   lambda: tvmffi_mod.scale(out_t, in_t, factor)),
    ]:
        out_t.zero_()
        fn()
        torch.cuda.synchronize()
        ok = torch.allclose(out_t, expected)
        print(f"  {name}: out={out_t.item():.4f} expected={expected.item():.4f} ok={ok}")


def time_calls(fn, iters):
    # Time `iters` calls; sync once at the end. Per-call latency = total / iters.
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    return (t1 - t0) / iters * 1e6  # microseconds per call


def bench_eager(iters=20000, repeats=7):
    print(f"\n=== eager mode (iters={iters} per repeat, {repeats} repeats) ===")
    print("                  median_us    min_us    stdev_us")
    cases = {
        "unstable_TORCH_LIBRARY":         lambda: torch.ops.bench_unstable.scale(out_t, in_t, factor),
        "stable_STABLE_TORCH_LIBRARY":    lambda: torch.ops.bench_stable.scale(out_t, in_t, factor),
        "tvmffi_direct":                  lambda: tvmffi_mod.scale(out_t, in_t, factor),
    }
    for name, fn in cases.items():
        # warmup
        for _ in range(2000):
            fn()
        torch.cuda.synchronize()
        # repeats
        samples = [time_calls(fn, iters) for _ in range(repeats)]
        med = statistics.median(samples)
        mn = min(samples)
        sd = statistics.stdev(samples) if len(samples) > 1 else 0.0
        print(f"  {name:<35s} {med:8.3f}  {mn:8.3f}  {sd:8.3f}")


def bench_cuda_graph(graph_iters=512, replays=10000, repeats=7):
    """Capture `graph_iters` calls into a CUDA graph; time `replays` replays."""
    print(f"\n=== CUDA graph (capture {graph_iters} calls per graph, "
          f"replay {replays} times, {repeats} repeats) ===")
    print("                  median_us_per_replay   per_op_ns   stdev_us")

    cases = {
        "unstable_TORCH_LIBRARY":         lambda: torch.ops.bench_unstable.scale(out_t, in_t, factor),
        "stable_STABLE_TORCH_LIBRARY":    lambda: torch.ops.bench_stable.scale(out_t, in_t, factor),
        "tvmffi_direct":                  lambda: tvmffi_mod.scale(out_t, in_t, factor),
    }

    for name, fn in cases.items():
        # warmup with a side stream so capture has a clean state
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(50):
                fn()
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()

        # Try to capture
        graph = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(graph):
                for _ in range(graph_iters):
                    fn()
        except Exception as e:
            print(f"  {name:<35s} CAPTURE FAILED: {type(e).__name__}: {e}")
            continue

        # Time replays
        torch.cuda.synchronize()
        samples = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            for _ in range(replays):
                graph.replay()
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            samples.append((t1 - t0) / replays * 1e6)  # us per replay
        med = statistics.median(samples)
        per_op_ns = med / graph_iters * 1000.0
        sd = statistics.stdev(samples) if len(samples) > 1 else 0.0
        print(f"  {name:<35s} {med:18.3f}     {per_op_ns:7.1f}   {sd:8.3f}")


def bench_capture_time(graph_iters=4096, repeats=5):
    """How long does *capturing* the graph take? This is dispatch overhead at
    record time — relevant if you build/rebuild graphs frequently."""
    print(f"\n=== CUDA graph capture cost (record {graph_iters} calls) ===")
    print("                  median_capture_ms   per_op_us")
    cases = {
        "unstable_TORCH_LIBRARY":         lambda: torch.ops.bench_unstable.scale(out_t, in_t, factor),
        "stable_STABLE_TORCH_LIBRARY":    lambda: torch.ops.bench_stable.scale(out_t, in_t, factor),
        "tvmffi_direct":                  lambda: tvmffi_mod.scale(out_t, in_t, factor),
    }
    for name, fn in cases.items():
        samples = []
        for _ in range(repeats):
            # warmup on side stream before capture
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(50):
                    fn()
            torch.cuda.current_stream().wait_stream(s)
            torch.cuda.synchronize()

            graph = torch.cuda.CUDAGraph()
            t0 = time.perf_counter()
            with torch.cuda.graph(graph):
                for _ in range(graph_iters):
                    fn()
            t1 = time.perf_counter()
            samples.append((t1 - t0) * 1e3)  # ms
            del graph
        med = statistics.median(samples)
        per_op_us = med * 1000.0 / graph_iters
        print(f"  {name:<35s} {med:18.3f}   {per_op_us:8.3f}")


if __name__ == "__main__":
    print("=== correctness ===")
    correctness()
    bench_eager()
    bench_capture_time()
    bench_cuda_graph()
