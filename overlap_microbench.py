"""Microbenchmark: does issuing the shared-layer gather on a copy stream hide
its PCIe transfer behind compute? Isolates overlapped prefetch (#1) at realistic
GLM-5.2 sizes without the 700GB model / PD topology.

Compares, per simulated decoder-layer step:
  serial  = compute; then gather        (overlap OFF)
  overlap = gather on copy stream || compute; join   (overlap ON)
The gather is hisparse_gather_plan (host->device of the plan's misses); compute
is a matmul sized to a decode layer's rough GPU time. Sweeps concurrency and
miss rate; reports the hidden time = serial - overlap.
"""

import argparse
import json

import torch
import vllm  # noqa: F401 - registers torch.ops._C_cache_ops (hisparse_gather_plan)

DEV = "cuda"


def _run(num_reqs: int, top_k: int, row_width: int, miss_frac: float, compute_n: int,
         iters: int, compute_iters: int = 1) -> dict:
    host_rows = 1 << 20  # 1M-row host pool (plenty)
    host = torch.arange(host_rows * row_width, dtype=torch.float32).view(
        host_rows, row_width
    ).to(torch.bfloat16).pin_memory()
    host_valid = torch.ones(host_rows, dtype=torch.bool).pin_memory()
    stride = ((top_k + 1 + 127) // 128) * 128
    hot = torch.zeros(num_reqs * stride, row_width, dtype=torch.bfloat16, device=DEV)
    num_real = torch.tensor([num_reqs], dtype=torch.int32, device=DEV)

    n_miss = max(1, int(top_k * miss_frac))
    gi = torch.full((num_reqs, top_k), -1, dtype=torch.int32, device=DEV)
    hi = torch.full((num_reqs, top_k), -1, dtype=torch.int32, device=DEV)
    mm = torch.zeros((num_reqs, top_k), dtype=torch.int32, device=DEV)
    for r in range(num_reqs):
        # distinct host rows per (req, miss) so the gather is a real scattered H2D
        gi[r, :n_miss] = torch.arange(r * n_miss, (r + 1) * n_miss, device=DEV, dtype=torch.int32)
        hi[r, :n_miss] = torch.arange(r * stride, r * stride + n_miss, device=DEV, dtype=torch.int32)
        mm[r, :n_miss] = 1

    A = torch.randn(compute_n, compute_n, device=DEV, dtype=torch.bfloat16)
    copy = torch.cuda.Stream()

    def gather():
        torch.ops._C_cache_ops.hisparse_gather_plan(hot, host, host_valid, hot, gi, hi, mm, num_real)

    def compute():
        out = A
        for _ in range(compute_iters):
            out = A @ out
        return out

    def timed(fn, n=iters):
        for _ in range(5):
            fn()
        torch.cuda.synchronize()
        e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
        e0.record()
        for _ in range(n):
            fn()
        e1.record()
        torch.cuda.synchronize()
        return e0.elapsed_time(e1) / n  # ms

    t_gather = timed(gather)
    t_compute = timed(compute)

    def serial():
        compute()
        gather()

    def overlap():
        main = torch.cuda.current_stream()
        copy.wait_stream(main)
        with torch.cuda.stream(copy):
            gather()
        compute()
        main.wait_stream(copy)

    t_serial = timed(serial)
    t_overlap = timed(overlap)
    return {
        "num_reqs": num_reqs, "miss_frac": miss_frac, "n_miss_per_req": n_miss,
        "t_gather_ms": round(t_gather, 4), "t_compute_ms": round(t_compute, 4),
        "t_serial_ms": round(t_serial, 4), "t_overlap_ms": round(t_overlap, 4),
        "hidden_ms": round(t_serial - t_overlap, 4),
        "hidden_frac_of_gather": round((t_serial - t_overlap) / max(t_gather, 1e-9), 3),
        "speedup": round(t_serial / max(t_overlap, 1e-9), 3),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-k", type=int, default=2048)
    ap.add_argument("--row-width", type=int, default=576)  # GLM-5.2 MLA kv_lora+rope
    ap.add_argument("--compute-n", type=int, default=4096)  # matmul ~ a decode layer's GPU time
    ap.add_argument("--iters", type=int, default=50)
    args = ap.parse_args()
    print(f"# top_k={args.top_k} row_width={args.row_width}")
    # Sweep the per-step compute size: a single 4096^2 bf16 matmul ~0.17ms is a
    # LOWER bound for a decode layer; the prefetch hides behind ~3 layers of
    # attention+MoE, so realistic hideable compute is several ms. compute_iters
    # stacks matmuls to model that. Characterize the crossover (overlap wins iff
    # compute >~ gather) rather than pick one point.
    for compute_iters in (1, 8, 24):
        for num_reqs in (64, 128, 256):
            for miss_frac in (0.05, 0.2, 0.5):
                r = _run(num_reqs, args.top_k, args.row_width, miss_frac,
                         args.compute_n, args.iters, compute_iters)
                r["compute_iters"] = compute_iters
                print("MB " + json.dumps(r))


if __name__ == "__main__":
    main()
