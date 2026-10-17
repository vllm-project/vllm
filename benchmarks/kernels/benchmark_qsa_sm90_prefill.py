# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark Qwen4Exp QSA sparse prefill attention: Triton vs the SM90 kernel.

Each case is one prefill chunk of `rows` queries ending at `context` tokens.
Every row selects 512 random compressed blocks (2048 tokens) plus its causal
tail through the production expand kernel, over a paged BF16 cache with the
per-rank head split of TP1/TP2/TP4/TP8. Both providers run through
`qsa_sparse_paged_attention`, toggled by VLLM_QSA_SM90_NATIVE.
"""

import itertools
import os

import torch

from vllm.models.qwen4_exp.nvidia.ops import qsa as qsa_ops
from vllm.models.qwen4_exp.nvidia.ops import qsa_indexer as qsa_indexer_ops
from vllm.triton_utils import triton
from vllm.utils.argparse_utils import FlexibleArgumentParser

HEAD_DIM = 256
TOKEN_TOPK = 2048
COMPRESS_RATIO = 4
# (query heads, KV heads, page size) per rank for Qwen3.8-Flash-Next.
HEAD_SPLITS = {
    "tp1": (24, 2, 1568),
    "tp2": (12, 1, 1568),
    "tp4": (6, 1, 784),
    "tp8": (3, 1, 392),
}


def make_inputs(rows: int, context: int, split: str, seed: int = 0):
    num_query_heads, num_kv_heads, page_size = HEAD_SPLITS[split]
    gen = torch.Generator(device="cuda").manual_seed(seed)
    num_pages = triton.cdiv(context, page_size)
    kv_cache = torch.randn(
        num_pages,
        num_kv_heads,
        page_size,
        2 * HEAD_DIM,
        device="cuda",
        dtype=torch.bfloat16,
        generator=gen,
    )
    k_cache, v_cache = kv_cache.transpose(1, 2).split(HEAD_DIM, dim=-1)
    block_table = torch.randperm(num_pages, device="cuda", generator=gen)
    block_table = block_table.to(torch.int32).view(1, num_pages)
    q = torch.randn(
        rows,
        num_query_heads,
        HEAD_DIM,
        device="cuda",
        dtype=torch.bfloat16,
        generator=gen,
    )
    output_gate = torch.randn(q.shape, device="cuda", generator=gen).to(q.dtype)
    token_to_req = torch.zeros(rows, device="cuda", dtype=torch.int32)

    positions = torch.arange(context - rows, context, device="cuda")
    visible_blocks = ((positions + 1) // COMPRESS_RATIO).to(torch.int32)
    block_topk = TOKEN_TOPK // COMPRESS_RATIO
    max_blocks = int(visible_blocks.max())
    block_indices = torch.full((rows, block_topk), -1, device="cuda", dtype=torch.int32)
    for start in range(0, rows, 2048):
        end = min(start + 2048, rows)
        scores = torch.rand(end - start, max_blocks, device="cuda", generator=gen)
        columns = torch.arange(max_blocks, device="cuda")
        scores[columns >= visible_blocks[start:end, None]] = -1.0
        k = min(block_topk, max_blocks)
        values, chosen = scores.topk(k, dim=1)
        block_indices[start:end, :k] = torch.where(values >= 0, chosen, -1).to(
            torch.int32
        )
    indices = torch.empty(
        rows, TOKEN_TOPK + COMPRESS_RATIO, device="cuda", dtype=torch.int32
    )
    qsa_indexer_ops.expand_qsa_block_indices(
        block_indices, positions, visible_blocks, COMPRESS_RATIO, TOKEN_TOPK, indices
    )
    return q, k_cache, v_cache, indices, block_table, token_to_req, output_gate


def run(inputs, native: bool) -> torch.Tensor:
    q, k_cache, v_cache, indices, block_table, token_to_req, output_gate = inputs
    os.environ["VLLM_QSA_SM90_NATIVE"] = "1" if native else "0"
    return qsa_ops.qsa_sparse_paged_attention(
        q,
        k_cache,
        v_cache,
        indices,
        block_table,
        token_to_req,
        use_prefill_config=True,
        output_gate=output_gate,
    )


def bench_case(split: str, rows: int, context: int) -> str:
    prefix = f"{split:>5} {rows:>6} {context:>8}"
    inputs = make_inputs(rows, context, split)
    if rows * inputs[1].shape[2] <= qsa_ops._SM90_NATIVE_MIN_PROGRAMS:
        return f"{prefix}   below the native threshold"
    diff = (run(inputs, True).float() - run(inputs, False).float()).abs().max()
    triton_ms = triton.testing.do_bench(lambda: run(inputs, False))
    native_ms = triton.testing.do_bench(lambda: run(inputs, True))
    return (
        f"{prefix} {triton_ms:>10.3f} {native_ms:>10.3f} "
        f"{triton_ms / native_ms:>7.2f}x {diff.item():>9.1e}"
    )


def main(args) -> None:
    assert qsa_ops._is_sm90() and qsa_ops._sm90_native_built(), (
        "requires an SM90 GPU and a build with the SM90 QSA kernel"
    )
    print(
        f"{'split':>5} {'rows':>6} {'context':>8} {'triton ms':>10} "
        f"{'native ms':>10} {'speedup':>8} {'max diff':>9}"
    )
    for split, rows, context in itertools.product(
        args.splits, args.rows, args.contexts
    ):
        if rows <= context:
            print(bench_case(split, rows, context))
            torch.accelerator.empty_cache()


if __name__ == "__main__":
    parser = FlexibleArgumentParser(description=__doc__)
    parser.add_argument(
        "--splits", nargs="+", choices=list(HEAD_SPLITS), default=list(HEAD_SPLITS)
    )
    parser.add_argument(
        "--rows", nargs="+", type=int, default=[1024, 2048, 4096, 8192, 16384]
    )
    parser.add_argument("--contexts", nargs="+", type=int, default=[4096, 21000, 65536])
    main(parser.parse_args())
