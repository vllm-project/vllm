# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TRITON_ATTN_DIFFKV vs TRITON_FLASHINFER_DIFFKV prefill on one layer.

Defaults are a MiMo-V2 full-attention layer at TP2 (32 query heads, 2 KV
heads, K/V head sizes 192/128, BF16): one request prefilling ``--chunk``
tokens on top of ``--prefix`` cached tokens. Both kernels get preallocated
inputs, passed through the timer so it can rotate them for cold-L2 timing.
FlashInfer's per-step ``plan()`` (host time, shared by every layer in a step)
is reported separately.
"""

import argparse
import time
from functools import partial

import numpy as np
import torch
from flashinfer import BatchPrefillWithPagedKVCacheWrapper

from vllm.utils.math_utils import cdiv, next_power_of_2
from vllm.v1.attention.backends.triton_flashinfer_diffkv import (
    plan_paged_prefill,
    run_paged_prefill,
)
from vllm.v1.attention.ops.triton_unified_attention_diffkv import (
    unified_attention_diffkv,
)
from vllm.v1.utils import CpuGpuBuffer

NUM_SEGMENTS = 16


def _timer(method: str):
    if method == "cupti":
        from flashinfer.testing import bench_gpu_time_with_cupti

        return partial(
            bench_gpu_time_with_cupti, use_cuda_graph=True, cold_l2_cache=True
        )
    if method == "cudagraph":
        from flashinfer.testing import bench_gpu_time_with_cudagraph

        return partial(bench_gpu_time_with_cudagraph, cold_l2_cache=True)
    raise ValueError(f"unknown timing method: {method}")


@torch.inference_mode()
def benchmark(args, timer, chunk: int, prefix: int, block_size: int) -> None:
    hq, hkv, dqk, dv = args.num_q_heads, args.num_kv_heads, args.qk_dim, args.v_dim
    dtype, scale = torch.bfloat16, dqk**-0.5
    kv_len = prefix + chunk
    pages = cdiv(kv_len, block_size)
    num_blocks = pages + 64
    cache = torch.randn(num_blocks, hkv, block_size, dqk + dv, dtype=dtype)
    block_table = torch.randperm(num_blocks)[:pages].to(torch.int32)[None]
    query = torch.randn(chunk, hq, dqk, dtype=dtype)
    out_triton = torch.empty(chunk, hq, dv, dtype=dtype)
    out_fi = torch.empty_like(out_triton)

    cu_seqlens_q = torch.tensor([0, chunk], dtype=torch.int32)
    seqused_k = torch.tensor([kv_len], dtype=torch.int32)
    segm_output = torch.empty(
        0, hq, NUM_SEGMENTS, next_power_of_2(dv), dtype=torch.float32
    )
    segm_max = torch.empty(0, hq, NUM_SEGMENTS, dtype=torch.float32)
    segm_expsum = torch.empty(0, hq, NUM_SEGMENTS, dtype=torch.float32)

    def triton(query, cache, out):
        nhd = cache.transpose(1, 2)
        unified_attention_diffkv(
            q=query,
            k=nhd[..., :dqk],
            v=nhd[..., dqk:],
            out=out,
            cu_seqlens_q=cu_seqlens_q,
            seqused_k=seqused_k,
            softmax_scale=scale,
            causal=True,
            window_size=(-1, -1),
            block_table=block_table,
            softcap=0,
            max_seqlen_q=chunk,
            seq_threshold_3D=0,
            num_par_softmax_segments=NUM_SEGMENTS,
            softmax_segm_output=segm_output,
            softmax_segm_max=segm_max,
            softmax_segm_expsum=segm_expsum,
        )

    wrapper = BatchPrefillWithPagedKVCacheWrapper(
        torch.zeros(256 << 20, dtype=torch.uint8), "HND"
    )
    plan = partial(
        plan_paged_prefill,
        wrapper,
        qo_indptr_cpu=torch.tensor([0, chunk], dtype=torch.int32, device="cpu"),
        seq_lens_cpu=torch.tensor([kv_len], dtype=torch.int32, device="cpu"),
        block_table=block_table,
        page_size=block_size,
        paged_kv_indptr=CpuGpuBuffer(2, dtype=torch.int32, device="cuda"),
        paged_kv_indices=torch.zeros(pages, dtype=torch.int32),
        paged_kv_last_page_len=CpuGpuBuffer(1, dtype=torch.int32, device="cuda"),
        num_qo_heads=hq,
        num_kv_heads=hkv,
        head_size_qk=dqk,
        head_size_v=dv,
        sm_scale=scale,
        q_dtype=dtype,
        kv_dtype=dtype,
    )
    plan()
    plan_us = []
    for _ in range(20):
        torch.accelerator.synchronize()
        start = time.perf_counter()
        plan()
        plan_us.append((time.perf_counter() - start) * 1e6)

    def flashinfer(query, cache, out):
        run_paged_prefill(wrapper, query, cache, out, dqk, dv)

    t_triton = float(np.median(timer(triton, input_args=(query, cache, out_triton))))
    t_fi = float(np.median(timer(flashinfer, input_args=(query, cache, out_fi))))
    triton(query, cache, out_triton)
    flashinfer(query, cache, out_fi)
    visible = chunk * prefix + chunk * (chunk + 1) / 2
    tflop = 2 * hq * visible * (dqk + dv) / 1e12
    rel_l2 = (out_fi.float() - out_triton.float()).norm() / out_triton.float().norm()
    print(
        f"{block_size:>5} {chunk:>6} {prefix:>7} {t_triton * 1e3:>10.1f} "
        f"{t_fi * 1e3:>9.1f} {t_triton / t_fi:>7.2f}x {np.median(plan_us):>8.0f} "
        f"{tflop / t_triton * 1e3:>8.1f} {tflop / t_fi * 1e3:>7.1f} "
        f"{rel_l2.item():>8.1e}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-q-heads", type=int, default=32)
    parser.add_argument("--num-kv-heads", type=int, default=2)
    parser.add_argument("--qk-dim", type=int, default=192)
    parser.add_argument("--v-dim", type=int, default=128)
    parser.add_argument("--block-sizes", type=int, nargs="+", default=[16, 64])
    parser.add_argument(
        "--chunk", type=int, nargs="+", default=[17, 32, 64, 128, 512, 2048, 8192]
    )
    parser.add_argument("--prefix", type=int, nargs="+", default=[0, 8192, 32768])
    parser.add_argument("--method", choices=["cupti", "cudagraph"], default="cupti")
    args = parser.parse_args()

    torch.set_default_device("cuda")
    torch.manual_seed(0)
    timer = _timer(args.method)
    print(f"Timing method: {args.method}; plan is host time per step")
    print(
        "block  chunk  prefix  triton_us     fi_us speedup  plan_us "
        "triton_TF   fi_TF   rel_l2"
    )
    for block_size in args.block_sizes:
        for chunk in args.chunk:
            for prefix in args.prefix:
                benchmark(args, timer, chunk, prefix, block_size)
