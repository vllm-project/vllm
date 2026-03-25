# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import argparse
import time

import torch
from flashinfer import bmm_fp8 as flashinfer_bmm_fp8


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--k", type=int, default=8192)
    p.add_argument("--n", type=int, default=28672)
    p.add_argument(
        "--m-list",
        type=str,
        default="1,2,4,8,16,32,64,128,256,512,1024",
        help="Comma-separated list of token counts to benchmark.",
    )
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    p.add_argument("--backend", default="auto")
    return p.parse_args()


def rand_fp8(shape, device, dtype):
    x = torch.randn(shape, device=device, dtype=torch.float16).clamp_(-6, 6)
    return x.to(dtype).contiguous()


def rand_fp8_col_major(n, k, device, dtype):
    # Logical weight [n, k] -> stored as column-major [k, n].
    w = torch.randn((n, k), device=device, dtype=torch.float16).clamp_(-6, 6)
    return w.transpose(-2, -1).to(dtype)


def bench_one(m, k, n, out_dtype, warmup, iters, backend):
    device = torch.device(f"cuda:{torch.accelerator.current_device_index()}")
    fp8 = torch.float8_e4m3fn

    A = rand_fp8((1, m, k), device, fp8)
    B = rand_fp8_col_major(n, k, device, fp8).unsqueeze(0)
    A_scale = torch.tensor(1.0, device=device, dtype=torch.float32)
    B_scale = torch.tensor(1.0, device=device, dtype=torch.float32)
    out = torch.empty((1, m, n), device=device, dtype=out_dtype)

    for _ in range(warmup):
        flashinfer_bmm_fp8(A, B, A_scale, B_scale, out_dtype, out, backend)
    torch.accelerator.synchronize(device=device)

    times_ms = []
    for _ in range(iters):
        torch.accelerator.synchronize(device=device)
        t0 = time.perf_counter()
        flashinfer_bmm_fp8(A, B, A_scale, B_scale, out_dtype, out, backend)
        torch.accelerator.synchronize(device=device)
        times_ms.append((time.perf_counter() - t0) * 1000.0)

    times_ms.sort()
    mean_ms = sum(times_ms) / len(times_ms)
    p50_ms = times_ms[len(times_ms) // 2]
    p95_ms = times_ms[min(len(times_ms) - 1, int(len(times_ms) * 0.95))]

    sec = mean_ms / 1000.0
    tflops = (2.0 * m * n * k) / sec / 1e12
    tok_per_s = m / sec
    return mean_ms, p50_ms, p95_ms, tok_per_s, tflops


def main():
    args = parse_args()
    assert torch.accelerator.is_available(), "Accelerator is required"

    out_dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    m_list = [int(x) for x in args.m_list.split(",") if x.strip()]

    print(
        f"# flashinfer.bmm_fp8 sweep k={args.k} n={args.n} "
        f"dtype={args.dtype} backend={args.backend}"
    )
    print("m,mean_ms,p50_ms,p95_ms,tokens_per_s,tflops")

    for m in m_list:
        mean_ms, p50_ms, p95_ms, tok_per_s, tflops = bench_one(
            m=m,
            k=args.k,
            n=args.n,
            out_dtype=out_dtype,
            warmup=args.warmup,
            iters=args.iters,
            backend=args.backend,
        )
        print(
            f"{m},{mean_ms:.6f},{p50_ms:.6f},{p95_ms:.6f},{tok_per_s:.2f},{tflops:.2f}"
        )


if __name__ == "__main__":
    main()
