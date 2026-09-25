# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Graph-replay microbenchmark with rotating weight buffers (CUDA and ROCm).

Platform-agnostic: HIP implements the torch.cuda.* API, including CUDAGraph.
Pattern: capture one graph (HIP graph on ROCm) containing R*INNER calls
cycling over R rotating weight buffers, then time replays with CUDA/HIP
events. The rotating buffers keep the captured footprint above the GPU's
last-level cache, so every call reads weights cold from HBM. Prefer this over
do_bench on ROCm: do_bench's per-rep launch/event overhead inflates
small/decode-size kernels, and its default flush buffer is only as large as
the LLC — on a 256 MB LLC that evicts marginally and leaves the cache full of
dirty flush-buffer lines. The same pattern extends to multi-GPU benchmarks
(capture per rank, collectives in identical order every replay); see the
Multi-GPU section of SKILL.md.
"""

import statistics

import pandas as pd
import torch

LLC_TARGET_BYTES = 400 * 1024**2  # above 256 MB MI355X LLC and B200 L2 (126 MB)
INNER = 2  # repeat slots within one graph to amortize per-replay overhead
REPLAYS = 10  # replays per round: makes each measurement large vs overhead
ROUNDS = 5  # median across rounds: replay time is not always stable
WARMUP = 5  # warmup replays to reach steady clocks before timing
MATMUL_CASES = [
    ("compute-bound", 4096, 4096, 4096),
    ("decode-GEMM", 16, 16384, 8192),
]


def timeit_graph(calls, inner=INNER, replays=REPLAYS, rounds=ROUNDS):
    """Time a list of per-slot callables as one graph (zero launch overhead);
    weights rotate across slots so they stay HBM-cold. Returns the median over
    rounds of per-round means, ms per call."""
    for c in calls:
        c()
    torch.accelerator.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):  # capture runs on a side stream
        for i in range(len(calls) * inner):
            calls[i % len(calls)]()
    torch.accelerator.synchronize()
    events = [(torch.cuda.Event(True), torch.cuda.Event(True)) for _ in range(rounds)]
    for _ in range(WARMUP):
        g.replay()
    # No sync between warmup and the first round: queued replays keep the GPU
    # busy, so each round's launch latency stays outside its event window.
    for s, e in events:
        s.record()
        for _ in range(replays):
            g.replay()
        e.record()
    torch.accelerator.synchronize()
    samples = [s.elapsed_time(e) / (replays * len(calls) * inner) for s, e in events]
    return statistics.median(samples)  # ms per call


def main() -> None:
    if not torch.accelerator.is_available():
        raise RuntimeError("No GPU visible (torch.accelerator covers HIP on ROCm).")

    torch.set_default_device("cuda")
    torch.manual_seed(0)

    rows = []
    for name, m, n, k in MATMUL_CASES:
        a = torch.randn(m, k, dtype=torch.bfloat16)  # shared: producer-warm
        out = torch.empty(m, n, dtype=torch.bfloat16)
        b_bytes = k * n * 2
        r = max(2, -(-LLC_TARGET_BYTES // b_bytes))  # ceil
        bs = [torch.randn(k, n, dtype=torch.bfloat16) for _ in range(r)]

        def make_call(i, a=a, bs=bs, out=out):
            def call():
                torch.mm(a, bs[i], out=out)

            return call

        # correctness before timing, per slot (the graph would freeze a wrong
        # result into every replay)
        for i in range(r):
            make_call(i)()
            torch.accelerator.synchronize()
            torch.testing.assert_close(out, torch.mm(a, bs[i]), atol=1e-1, rtol=1e-1)

        us = timeit_graph([make_call(i) for i in range(r)]) * 1e3
        rows.append(
            {
                "case": name,
                "shape": f"{m}x{n}x{k}",
                "slots": r,
                "us": us,
                "tflops": 2 * m * n * k / (us * 1e6),
                "gbps": 2 * (m * k + k * n + m * n) / (us * 1e3),
            }
        )

    df = pd.DataFrame(rows)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
