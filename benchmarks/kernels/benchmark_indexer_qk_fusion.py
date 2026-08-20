# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused vs unfused DSA indexer QK pre-processing on ROCm.

The DSA indexer (DeepSeek-V3.2, GLM-5.x) runs five launches per layer per step:
LayerNorm(k), RoPE(q, k), per-token-group fp8 quant of q, folding the q scale
into the indexer weights, and the fp8 K quant + paged K-cache write. With
VLLM_ROCM_USE_AITER_INDEXER_QK_FUSION=1 one AITER kernel does all five.

Usage:
    python benchmarks/kernels/benchmark_indexer_qk_fusion.py
    python benchmarks/kernels/benchmark_indexer_qk_fusion.py --num-tokens 1 8 64
"""

import argparse
import functools

import torch

from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    per_token_group_quant_fp8,
)
from vllm.platforms import current_platform
from vllm.triton_utils import triton
from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
    indexer_k_quant_and_cache_triton,
)

HEAD_DIM = 128
ROPE_DIM = 64
N_HEAD = 32
MAX_POS = 65536
QUANT_BLOCK = 128
EPS = 1e-6
SCALE_FMT = "ue8m0"
WEIGHTS_SCALE = HEAD_DIM**-0.5 * N_HEAD**-0.5


def _unfused(
    q, k_raw, weights_raw, positions, cache, norm_w, norm_b, kv, slots, is_neox
):
    num_tokens = q.shape[0]
    k = torch.nn.functional.layer_norm(
        k_raw.float(), (HEAD_DIM,), norm_w.float(), norm_b.float(), EPS
    ).to(q.dtype)
    q = q.clone()
    ops.rotary_embedding(
        positions,
        q[..., :ROPE_DIM],
        k[..., :ROPE_DIM].unsqueeze(1),
        ROPE_DIM,
        cache,
        is_neox,
    )
    q_fp8, q_scale = per_token_group_quant_fp8(
        q.view(-1, HEAD_DIM), QUANT_BLOCK, column_major_scales=False, use_ue8m0=True
    )
    _ = weights_raw.float() * q_scale.view(num_tokens, N_HEAD) * WEIGHTS_SCALE
    indexer_k_quant_and_cache_triton(k, kv, slots, QUANT_BLOCK, SCALE_FMT)


def _fused(
    q,
    k_raw,
    weights_raw,
    positions,
    cache,
    norm_w,
    norm_b,
    kv,
    slots,
    q_out,
    w_out,
    is_neox,
):
    from aiter import indexer_qk_rope_quant_and_cache

    half = ROPE_DIM // 2
    indexer_qk_rope_quant_and_cache(
        q,
        q_out,
        weights_raw,
        w_out,
        k_raw,
        kv,
        slots,
        norm_w,
        norm_b,
        positions,
        cache[:, :half],
        cache[:, half:],
        EPS,
        QUANT_BLOCK,
        SCALE_FMT,
        WEIGHTS_SCALE,
        preshuffle=kv.shape[1] > 1,
        is_neox=is_neox,
    )


def _time_us(fn) -> float:
    ms = triton.testing.do_bench(fn, warmup=25, rep=100)
    return ms * 1000.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num-tokens", type=int, nargs="+", default=[1, 8, 32, 64, 256, 1024]
    )
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument(
        "--is-neox",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="RoPE layout. GLM-5.x sets indexer_rope_interleave, i.e. is_neox "
        "False; DeepSeek-V3.2 leaves it at the NeoX default.",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=3,
        help="Measure each point this many times and report the median. "
        "Process-level state (kernel tuning caches, clocks) moves these numbers "
        "more than do_bench's own variance does.",
    )
    args = parser.parse_args()

    if not current_platform.is_rocm():
        raise SystemExit("ROCm only")
    fp8 = current_platform.fp8_dtype()
    dev, dt = "cuda", torch.bfloat16

    def median(runs: list[float]) -> tuple[float, float, float]:
        ordered = sorted(runs)
        return ordered[len(ordered) // 2], ordered[0], ordered[-1]

    rows = []
    for num_tokens in args.num_tokens:
        num_blocks = (num_tokens + args.block_size - 1) // args.block_size + 2
        q = torch.randn(num_tokens, N_HEAD, HEAD_DIM, device=dev, dtype=dt)
        kw = torch.randn(num_tokens, HEAD_DIM + N_HEAD, device=dev, dtype=dt)
        positions = torch.randint(
            0, MAX_POS, (num_tokens,), device=dev, dtype=torch.int64
        )
        norm_w = torch.randn(HEAD_DIM, device=dev, dtype=dt)
        norm_b = torch.randn(HEAD_DIM, device=dev, dtype=dt)
        cache = torch.randn(MAX_POS, ROPE_DIM, device=dev, dtype=dt)
        kv = torch.zeros(
            num_blocks, args.block_size, HEAD_DIM + 4, dtype=fp8, device=dev
        )
        slots = torch.randperm(
            num_blocks * args.block_size, device=dev, dtype=torch.int64
        )[:num_tokens]
        q_out = torch.zeros((num_tokens, N_HEAD, HEAD_DIM), dtype=fp8, device=dev)
        w_out = torch.zeros((num_tokens, N_HEAD), dtype=torch.float32, device=dev)

        unfused_fn = functools.partial(
            _unfused,
            q,
            kw[:, :HEAD_DIM],
            kw[:, HEAD_DIM:],
            positions,
            cache,
            norm_w,
            norm_b,
            kv,
            slots,
            args.is_neox,
        )
        fused_fn = functools.partial(
            _fused,
            q,
            kw[:, :HEAD_DIM],
            kw[:, HEAD_DIM:],
            positions,
            cache,
            norm_w,
            norm_b,
            kv,
            slots,
            q_out,
            w_out,
            args.is_neox,
        )
        unfused_runs = [_time_us(unfused_fn) for _ in range(args.repeat)]
        fused_runs = [_time_us(fused_fn) for _ in range(args.repeat)]
        rows.append((num_tokens, median(unfused_runs), median(fused_runs)))

    print(
        f"{'tokens':>8} {'unfused us':>12} {'fused us':>10} {'speedup':>9}"
        f"   spread over {args.repeat} repeats"
    )
    for num_tokens, (u_med, u_lo, u_hi), (f_med, f_lo, f_hi) in rows:
        print(
            f"{num_tokens:>8} {u_med:>12.2f} {f_med:>10.2f} {u_med / f_med:>8.2f}x"
            f"   unfused {u_lo:.1f}-{u_hi:.1f}, fused {f_lo:.1f}-{f_hi:.1f}"
        )


if __name__ == "__main__":
    main()
