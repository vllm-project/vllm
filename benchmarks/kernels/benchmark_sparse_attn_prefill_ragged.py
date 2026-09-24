# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark the ROCm sparse-MLA ragged prefill kernel in isolation.

Example usage:
VLLM_ROCM_USE_AITER=1 HIP_VISIBLE_DEVICES=0 \
python3 benchmarks/kernels/benchmark_sparse_attn_prefill_ragged.py \
    --queries 512 1024 2048 --topk 2048 --nope 512 --rope 64
"""

import argparse
import itertools
import json

import torch

from vllm.triton_utils import triton
from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
    _rocm_sparse_attn_prefill_ragged_triton,
    _sparse_attn_prefill_ragged_kernel,
)


def make_inputs(queries: int, topk: int, args):
    torch.manual_seed(args.seed)
    head_dim = args.nope + args.rope
    ctx = max(args.ctx, topk)
    q = torch.randn(
        queries, args.heads, head_dim, device="cuda", dtype=torch.bfloat16
    ) * (head_dim**-0.25)
    kv = torch.randn(ctx, head_dim, device="cuda", dtype=torch.bfloat16) * (
        head_dim**-0.25
    )
    counts = torch.full((queries,), min(topk, ctx), device="cuda", dtype=torch.int32)
    indptr = torch.zeros(queries + 1, device="cuda", dtype=torch.int32)
    torch.cumsum(counts, dim=0, out=indptr[1:])
    indices = torch.randint(
        0, ctx, (int(indptr[-1].item()),), device="cuda", dtype=torch.int32
    )
    attn_sink = (
        torch.randn(args.heads, device="cuda", dtype=torch.float32)
        if args.attn_sink
        else None
    )
    return q, kv, indices, indptr, attn_sink


def launch(q, kv, indices, indptr, attn_sink, out, scale, split: bool, args):
    num_queries, num_heads, head_dim = q.shape
    has_attn_sink = attn_sink is not None
    if attn_sink is None:
        attn_sink = torch.empty(1, device=q.device, dtype=torch.float32)
    block_h = 16
    if split:
        block_dv = triton.next_power_of_2(args.nope)
        block_dr = triton.next_power_of_2(args.rope)
    else:
        block_dv, block_dr = triton.next_power_of_2(head_dim), 1
    out_has_rope = split and out.shape[-1] == head_dim
    _sparse_attn_prefill_ragged_kernel[(num_queries, triton.cdiv(num_heads, block_h))](
        q,
        kv,
        indices,
        indptr,
        attn_sink,
        out,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        kv.stride(0),
        kv.stride(1),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        num_heads,
        kv.shape[0],
        float(scale),
        HAS_ATTN_SINK=has_attn_sink,
        HAS_ROPE=split,
        OUT_HAS_ROPE=out_has_rope,
        NOPE=args.nope if split else head_dim,
        ROPE=args.rope if split else 0,
        OUT_DV=args.nope if split else out.shape[-1],
        BLOCK_H=block_h,
        BLOCK_DV=block_dv,
        BLOCK_DR=block_dr,
        BLOCK_K=16 if head_dim >= 256 else 32,
        num_warps=4,
    )


def make_runner(variant: str, q, kv, indices, indptr, attn_sink, out, scale, args):
    if variant == "fused_copy":

        def run():
            scratch = torch.empty_like(q)
            launch(q, kv, indices, indptr, attn_sink, scratch, scale, False, args)
            out.copy_(scratch[..., : out.shape[-1]].to(out.dtype))

        return run

    split = variant == "split_direct"

    def run():
        launch(q, kv, indices, indptr, attn_sink, out, scale, split, args)

    return run


VARIANTS = ("fused_copy", "fused_direct", "split_direct")


def lanes(variant: str, args) -> int:
    head_dim = args.nope + args.rope
    if variant == "split_direct":
        return triton.next_power_of_2(args.nope) + triton.next_power_of_2(args.rope)
    return triton.next_power_of_2(head_dim)


def benchmark(variant: str, queries: int, topk: int, args) -> dict:
    q, kv, indices, indptr, attn_sink = make_inputs(queries, topk, args)
    scale = (args.nope + args.rope) ** -0.5
    width = args.nope + args.rope if args.wide_out else args.nope
    out = torch.empty(
        q.shape[0], q.shape[1], width, device=q.device, dtype=args.out_dtype
    )
    run = make_runner(variant, q, kv, indices, indptr, attn_sink, out, scale, args)

    run()
    torch.accelerator.synchronize()
    reference = _rocm_sparse_attn_prefill_ragged_triton(
        q=q,
        kv=kv,
        indices=indices,
        indptr=indptr,
        scale=scale,
        attn_sink=attn_sink,
        nope_head_dim=args.nope,
        rope_head_dim=args.rope,
    )
    torch.testing.assert_close(
        out.float(), reference[..., :width].float(), atol=2e-2, rtol=2e-2
    )

    ms = triton.testing.do_bench(run, warmup=args.warmup, rep=args.rep)
    return dict(
        variant=variant,
        queries=queries,
        heads=args.heads,
        topk=topk,
        nope=args.nope,
        rope=args.rope,
        out_width=width,
        lanes=lanes(variant, args),
        us=ms * 1000,
        kv_gbps=queries * topk * (args.nope + args.rope) * 2 / (ms * 1e6),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=int, nargs="+", default=[512, 1024, 2048])
    parser.add_argument("--topk", type=int, nargs="+", default=[1024, 2048])
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--nope", type=int, default=512, help="kv_lora_rank")
    parser.add_argument("--rope", type=int, default=64)
    parser.add_argument(
        "--ctx", type=int, default=32768, help="kv rows to select from; >= --topk"
    )
    parser.add_argument(
        "--wide-out",
        action="store_true",
        help="Write a head-dim-wide destination, as DeepSeek V4/V4.1 do",
    )
    parser.add_argument(
        "--variants", nargs="+", choices=list(VARIANTS), default=list(VARIANTS)
    )
    parser.add_argument("--attn-sink", action="store_true")
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--rep", type=int, default=200)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()
    args.out_dtype = torch.bfloat16

    print(
        json.dumps(
            dict(gpu=torch.cuda.get_device_name(0), args=vars(args)), default=str
        )
    )
    for variant, queries, topk in itertools.product(
        args.variants, args.queries, args.topk
    ):
        if variant == "split_direct" and args.rope <= 0:
            continue
        print(json.dumps(benchmark(variant, queries, topk, args)), flush=True)


if __name__ == "__main__":
    main()
