# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tile-shape sweep for AITER's a16w4 MXFP4 MoE GEMM.

AITER's ``get_kernel_config`` picks one tile shape for every AMD architecture.
Its constants (``block_k=256``, ``block_n=512``, ``num_stages=1``) assume
CDNA4's 160 KiB LDS; CDNA3 has 64 KiB, so the prefill tile leaves a large
amount of throughput on the table there.

This sweeps the tile shape for a given MoE geometry and reports the best
config against the stock one. Defaults describe Kimi-K3 at TP=8.

Example:
    python benchmarks/kernels/benchmark_moe_a16w4_tiles.py \
        --num-experts 896 --topk 16 --tokens 4096 \
        --hidden 3584 --intermediate 768
"""

import argparse
import itertools

import torch

from vllm.triton_utils import triton


def _import_aiter():
    import aiter.ops.triton.moe.moe_op_gemm_a16w4 as a16w4
    from aiter.ops.triton.moe.moe_routing import routing as routing_mod
    from triton_kernels.tensor import FP4, convert_layout, wrap_torch_tensor
    from triton_kernels.tensor_details.layout import StridedLayout

    return a16w4, routing_mod, (FP4, convert_layout, wrap_torch_tensor, StridedLayout)


def _raw(t):
    return t.storage.data if hasattr(t, "storage") else t


def main(args: argparse.Namespace) -> None:
    a16w4, routing_mod, (FP4, convert_layout, wrap, StridedLayout) = _import_aiter()
    stock_config = a16w4.get_kernel_config
    device = "cuda"
    e, topk, m = args.num_experts, args.topk, args.tokens
    k, n = args.hidden, args.intermediate

    torch.manual_seed(args.seed)
    x = torch.randn(m * topk, k, dtype=torch.bfloat16, device=device)
    # Packed MXFP4: two values per byte, one e8m0 scale per 32 values. Values
    # are irrelevant to timing, so build them directly to keep the footprint
    # small enough to run alongside a loaded model.
    w = torch.randint(0, 255, (e, n, k // 2), dtype=torch.uint8, device=device)
    s = torch.randint(120, 134, (e, n, k // 32), dtype=torch.uint8, device=device)
    wt = convert_layout(wrap(w.transpose(-2, -1), dtype=FP4), StridedLayout)
    st = convert_layout(wrap(s.transpose(-2, -1)), StridedLayout)

    logits = torch.randn(m, e, dtype=torch.bfloat16, device=device)
    routing_data, gather_indx, _ = routing_mod.routing(logits, topk, sm_first=False)

    def run():
        return a16w4.moe_gemm_a16w4(
            x, _raw(wt), None, _raw(st), None, None, None, routing_data,
            gather_indx=gather_indx, swizzle_mx_scale=None, apply_swiglu=False,
            unpadded_N=n, unpadded_K=k,
        )

    def bench() -> float:
        for _ in range(3):
            run()
        torch.cuda.synchronize()
        return triton.testing.do_bench(run, warmup=25, rep=100)

    baseline = bench()
    cfg = stock_config(m * topk, n, k, routing_data)
    print(
        f"stock: block_m={cfg['block_m']} block_n={cfg['block_n']} "
        f"block_k={cfg['block_k']} warps={cfg['num_warps']} "
        f"stages={cfg['num_stages']}  {baseline:.3f} ms"
    )

    def override(block_n, block_k, num_stages, num_warps):
        def picker(m_, n_, k_, rd):
            out = stock_config(m_, n_, k_, rd)
            if out["block_m"] >= args.min_block_m:
                out.update(
                    block_n=block_n, block_k=block_k,
                    num_stages=num_stages, num_warps=num_warps,
                )
            return out

        return picker

    best = (baseline, "stock")
    grid = itertools.product(args.block_n, args.block_k, args.stages, args.warps)
    for block_n, block_k, num_stages, num_warps in grid:
        a16w4.get_kernel_config = override(block_n, block_k, num_stages, num_warps)
        try:
            elapsed = bench()
        except Exception as exc:  # noqa: BLE001 - OOM on LDS is an expected outcome
            print(
                f"  block_n={block_n:3d} block_k={block_k:3d} stages={num_stages} "
                f"warps={num_warps}: skipped ({type(exc).__name__})"
            )
            continue
        finally:
            a16w4.get_kernel_config = stock_config
        label = (
            f"block_n={block_n} block_k={block_k} "
            f"stages={num_stages} warps={num_warps}"
        )
        if elapsed < best[0]:
            best = (elapsed, label)
        if elapsed < baseline:
            print(f"  {label}: {elapsed:7.3f} ms  {baseline / elapsed:.2f}x")

    print(f"best: {best[1]}  {best[0]:.3f} ms  ({baseline / best[0]:.2f}x vs stock)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-experts", type=int, default=896)
    parser.add_argument("--topk", type=int, default=16)
    parser.add_argument("--tokens", type=int, default=4096)
    parser.add_argument("--hidden", type=int, default=3584, help="GEMM K")
    parser.add_argument("--intermediate", type=int, default=768, help="GEMM N")
    parser.add_argument("--min-block-m", type=int, default=64,
                        help="only override tiles at or above this block_m")
    parser.add_argument("--block-n", type=int, nargs="+", default=[128, 256, 512])
    parser.add_argument("--block-k", type=int, nargs="+", default=[64, 128, 256])
    parser.add_argument("--stages", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--warps", type=int, nargs="+", default=[4, 8])
    parser.add_argument("--seed", type=int, default=0)
    main(parser.parse_args())
