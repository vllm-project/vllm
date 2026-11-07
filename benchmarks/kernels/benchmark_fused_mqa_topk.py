# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sparse-indexer prefill: the dense path (DeepGEMM logits + top_k_per_row_prefill,
row-split under the logits budget) vs fused_mqa_topk_prefill, per chunk shape.

Each shape is one chunk of ``rows`` causal query rows, the last rows of a
request of ``keys`` keys (32 FP8 heads x 128). Random q / k / weights by
default; ``--inputs CALL.pt`` takes real indexer operands instead (FlashTopK
reproduce/ FP8 files: q_fp8, kv_fp8, kv_scales, w_eff).

Example:
    python benchmarks/kernels/benchmark_fused_mqa_topk.py \
        --rows 1024,8192 --keys 32768,131072,1048576 --topk 512,2048

"""

import argparse
import itertools
import statistics

import torch

from vllm import _custom_ops as ops
from vllm.model_executor.kernels.attention.dsa.fused_mqa_topk import (
    fused_mqa_topk_available,
    fused_mqa_topk_chunk,
    fused_mqa_topk_prefill,
)
from vllm.model_executor.layers.sparse_attn_indexer import as_fp32_logits
from vllm.utils.deep_gemm import (
    fp8_fp4_mqa_logits,
    mqa_logits_mx_inputs,
    mqa_logits_use_mx_scales,
)
from vllm.v1.attention.backends.mla.indexer import dense_logits_rows


def _operands(rows, keys, inputs, seed=0):
    if inputs is not None:
        q, k, s, w = inputs
        if keys > k.shape[0]:
            raise SystemExit(f"--inputs holds {k.shape[0]} keys, fewer than {keys}")
        q, w = q[keys - rows : keys], w[keys - rows : keys]
        return q.contiguous(), k[:keys].contiguous(), s[:keys].contiguous(), w
    g = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randn(rows, 32, 128, generator=g, device="cuda").to(torch.float8_e4m3fn)
    k = torch.randn(keys, 128, generator=g, device="cuda").to(torch.float8_e4m3fn)
    s = torch.ones(keys, device="cuda")
    w = torch.randn(rows, 32, generator=g, device="cuda") / 32
    return q, k, s, w


def _dense(q, k, scales, weights, starts, ends, out, max_seqlen_k):
    k_scale = scales
    if mqa_logits_use_mx_scales():
        exp = torch.log2(scales).round().to(torch.int64) + 127
        k_scale = (exp * 0x01010101).to(torch.int32)
    step = dense_logits_rows(max_seqlen_k)
    lens = ends - starts
    for i in range(0, q.shape[0], step):
        j = min(q.shape[0], i + step)
        q_scale, w = mqa_logits_mx_inputs(q[i:j], None, weights[i:j], torch.float32)
        logits = as_fp32_logits(
            fp8_fp4_mqa_logits(
                (q[i:j], q_scale),
                (k, k_scale),
                w,
                starts[i:j],
                ends[i:j],
                max_seqlen_k,
            ),
            lens[i:j],
        )
        ops.top_k_per_row_prefill(
            logits,
            torch.zeros_like(lens[i:j]),
            lens[i:j],
            out[i:j],
            j - i,
            logits.stride(0),
            logits.stride(1),
            out.shape[1],
        )


def _time(fn, iters, warmup):
    for _ in range(warmup):
        fn()
    torch.accelerator.synchronize()
    times = []
    for _ in range(iters):
        start, end = torch.cuda.Event(True), torch.cuda.Event(True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
    return statistics.median(times)


def _shape(rows, keys, topk, inputs, iters, warmup):
    """(dense ms, fused ms) of one chunk: the last ``rows`` rows of ``keys``."""
    q, k, s, w = _operands(rows, keys, inputs)
    starts = torch.zeros(rows, dtype=torch.int32, device="cuda")
    ends = torch.arange(keys - rows + 1, keys + 1, device="cuda").int()
    out = torch.empty(rows, topk, dtype=torch.int32, device="cuda")
    dense_ms = _time(lambda: _dense(q, k, s, w, starts, ends, out, keys), iters, warmup)
    fused_ms = _time(
        lambda: fused_mqa_topk_prefill(q, k, s, w, starts, ends, out), iters, warmup
    )
    return dense_ms, fused_ms


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rows", default="1024,4096,16384")
    p.add_argument("--keys", default="16384,32768,65536,131072,262144,1048576")
    p.add_argument("--topk", default="512,2048")
    p.add_argument("--inputs", help="a FlashTopK reproduce/ FP8 call file")
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--warmup", type=int, default=2)
    a = p.parse_args()
    if not fused_mqa_topk_available():
        raise SystemExit("no fused_mqa_topk kernel on this GPU")
    inputs = None
    if a.inputs:
        d = torch.load(a.inputs, map_location="cpu", mmap=True, weights_only=False)
        inputs = tuple(d[n].cuda() for n in ("q_fp8", "kv_fp8", "kv_scales", "w_eff"))
    print(f"{torch.cuda.get_device_name()}, inputs: {a.inputs or 'random'}")
    print("rows,keys,topk,dense_ms,fused_ms,speedup,gate")
    for rows, keys, topk in itertools.product(
        *(map(int, x.split(",")) for x in (a.rows, a.keys, a.topk))
    ):
        if rows > keys:
            continue
        dense_ms, fused_ms = _shape(rows, keys, topk, inputs, a.iters, a.warmup)
        gate = fused_mqa_topk_chunk(keys, keys, topk, rows, dense_logits_rows(keys))
        print(
            f"{rows},{keys},{topk},{dense_ms:.3f},{fused_ms:.3f},"
            f"{dense_ms / fused_ms:.2f},{'fused' if gate else 'dense'}",
            flush=True,
        )
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
