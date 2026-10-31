# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare shared-KV C128 prefill against padded-head FlashMLA on Hopper.

Example:
    .venv/bin/python benchmarks/kernels/benchmark_deepseek_v4_c128_prefill.py

"""

import argparse
import json

import torch

from vllm.models.deepseek_v4.nvidia.ops.c128_prefill import c128_prefill_attention
from vllm.triton_utils import triton
from vllm.v1.attention.ops.flashmla import flash_mla_sparse_fwd


def measure(context: int, query_len: int) -> dict:
    num_heads = 8
    compressed = context // 128
    gather_len = min(context, query_len + 127)
    q = torch.randn(query_len, 64, 512, device="cuda", dtype=torch.bfloat16)
    q[:, num_heads:] = 0
    kv = torch.randn(
        1, compressed + gather_len, 512, device="cuda", dtype=torch.bfloat16
    )
    sink = torch.randn(64, device="cuda", dtype=torch.float32)
    candidate_out, reference_out = torch.empty_like(q), torch.empty_like(q)
    seq_lens = torch.tensor([context], device="cuda", dtype=torch.int32)
    gather_lens = torch.tensor([gather_len], device="cuda", dtype=torch.int32)
    pos = torch.arange(context - query_len, context, device="cuda")
    compressed_lens = (pos + 1) // 128
    swa_lens = torch.minimum(pos + 1, torch.full_like(pos, 128))
    lens = (compressed_lens + swa_lens).int()
    columns = torch.arange(triton.cdiv(compressed + 128, 128) * 128, device="cuda")
    indices = torch.where(
        columns[None, :] < compressed_lens[:, None],
        columns[None, :],
        compressed
        + pos[:, None]
        - swa_lens[:, None]
        + 1
        + columns[None, :]
        - compressed_lens[:, None]
        - (context - gather_len),
    )
    indices = torch.where(columns[None, :] < lens[:, None], indices, -1).int()

    def candidate():
        c128_prefill_attention(
            q,
            kv,
            sink,
            candidate_out,
            seq_lens,
            gather_lens,
            compressed,
            512**-0.5,
        )

    def reference():
        flash_mla_sparse_fwd(
            q,
            kv.view(-1, 1, 512),
            indices[:, None, :],
            512**-0.5,
            attn_sink=sink,
            topk_length=lens,
            out=reference_out,
        )

    candidate()
    reference()
    candidate_bits = candidate_out[:, :num_heads].view(torch.int16)
    reference_bits = reference_out[:, :num_heads].view(torch.int16)
    assert torch.equal(candidate_bits, reference_bits)
    # Also report both implementations' error against FP32 mathematics.
    oracle_errors = {"candidate": 0.0, "reference": 0.0}
    for row in sorted({0, 1, query_len // 2, query_len - 2, query_len - 1}):
        if row < 0 or row >= query_len:
            continue
        keys = kv.view(-1, 512)[indices[row, : lens[row]].long()].float()
        scores = q[row, :num_heads].float() @ keys.T * (512**-0.5)
        normalizer = torch.logaddexp(scores.logsumexp(-1), sink[:num_heads])
        oracle = torch.exp(scores - normalizer[:, None]) @ keys
        for name, output in (
            ("candidate", candidate_out),
            ("reference", reference_out),
        ):
            oracle_errors[name] = max(
                oracle_errors[name],
                (output[row, :num_heads].float() - oracle).abs().max().item(),
            )
    reference_ms = triton.testing.do_bench(reference, warmup=100, rep=300)
    candidate_ms = triton.testing.do_bench(candidate, warmup=100, rep=300)
    return {
        "context": context,
        "num_heads": num_heads,
        "query_len": query_len,
        "reference_ms": reference_ms,
        "candidate_ms": candidate_ms,
        "speedup": reference_ms / candidate_ms,
        "bitwise_mismatches": (candidate_bits != reference_bits).sum().item(),
        "max_error_to_fp32": oracle_errors,
        "max_error": (
            candidate_out[:, :num_heads].float() - reference_out[:, :num_heads].float()
        )
        .abs()
        .max()
        .item(),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--contexts", type=int, nargs="+", default=[8192, 131072, 1048576]
    )
    parser.add_argument("--query-len", type=int, default=8192)
    args = parser.parse_args()
    torch.manual_seed(123)
    for context in args.contexts:
        print(
            json.dumps(measure(context, min(args.query_len, context))),
            flush=True,
        )
