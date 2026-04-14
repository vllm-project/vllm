# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Correctness and benchmark tests for flash_attn_varlen_triton,
mirroring test_flash_attn.py so that perf can be compared
apple-to-apple against flash_attn_varlen_func.
"""

import pytest
import torch

from vllm.platforms import current_platform
from vllm.triton_utils import triton
from vllm.utils.math_utils import RCP_LN2
from vllm.utils.torch_utils import set_random_seed
from vllm.v1.attention.ops.triton_flash_attn_varlen import (
    TritonFlashAttnVarlenConfig,
    _fwd_kernel_varlen,
    _get_default_config,
    flash_attn_varlen_triton,
)

try:
    from vllm.vllm_flash_attn import (
        flash_attn_varlen_func,
        is_fa_version_supported,
    )

    HAS_FLASH_ATTN = True
except ImportError:
    HAS_FLASH_ATTN = False

# ── Parametrization constants (match test_flash_attn.py where possible) ──── #
NUM_HEADS = [(4, 4), (8, 2)]
HEAD_SIZES = [40, 72, 80, 128, 256]
DTYPES = [torch.bfloat16]

# Triton varlen causal requires q_len == k_len per sequence,
# so seq_lens entries are (seq_len,) meaning q_len = k_len = seq_len.
CAUSAL_SEQ_LENS = [
    [1, 18, 129],
    [523, 37, 2011],
]

# Non-causal allows different q_len and k_len.
NON_CAUSAL_SEQ_LENS = [
    [(1, 1328), (5, 18), (129, 463)],
    [(1, 523), (1, 37), (1, 2011)],
]

# ── Benchmark constants ──────────────────────────────────────────────────── #
BENCH_NUM_HEADS = [(32, 8)]
BENCH_HEAD_SIZES = [128]
BENCH_SEQ_LENS = [
    [512, 512, 512, 512],
    [1024, 1024],
    [2048],
    [128] * 16,
]
BENCH_WARMUP = 10
BENCH_ITERS = 50


# ── Reference implementation (contiguous KV, no paged cache) ────────────── #


def ref_varlen_attn(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    query_lens: list[int],
    kv_lens: list[int],
    scale: float,
    causal: bool,
) -> torch.Tensor:
    """Pure-PyTorch variable-length attention on contiguous packed Q/K/V."""
    num_seqs = len(query_lens)
    outputs: list[torch.Tensor] = []
    q_offset = 0
    k_offset = 0
    for i in range(num_seqs):
        q_len = query_lens[i]
        kv_len = kv_lens[i]
        q = query[q_offset : q_offset + q_len]
        q = q * scale
        k = key[k_offset : k_offset + kv_len]
        v = value[k_offset : k_offset + kv_len]

        # Expand KV heads for GQA
        if q.shape[1] != k.shape[1]:
            k = torch.repeat_interleave(k, q.shape[1] // k.shape[1], dim=1)
            v = torch.repeat_interleave(v, q.shape[1] // v.shape[1], dim=1)

        attn = torch.einsum("qhd,khd->hqk", q, k).float()
        if causal:
            mask = torch.ones(q_len, kv_len, device=q.device)
            mask = torch.triu(mask, diagonal=kv_len - q_len + 1).bool()
            attn.masked_fill_(mask, float("-inf"))
        attn = torch.softmax(attn, dim=-1).to(v.dtype)
        out = torch.einsum("hqk,khd->qhd", attn, v)

        outputs.append(out)
        q_offset += q_len
        k_offset += kv_len

    return torch.cat(outputs, dim=0)


# ── Helpers ──────────────────────────────────────────────────────────────── #


def _build_cu_seqlens(lens: list[int]) -> torch.Tensor:
    return torch.tensor([0] + lens, dtype=torch.int32).cumsum(dim=0, dtype=torch.int32)


# ═══════════════════════════════════════════════════════════════════════════ #
#  Correctness tests
# ═══════════════════════════════════════════════════════════════════════════ #


@pytest.mark.parametrize("seq_lens", CAUSAL_SEQ_LENS)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@torch.inference_mode()
def test_triton_varlen_causal(
    seq_lens: list[int],
    num_heads: tuple[int, int],
    head_size: int,
    dtype: torch.dtype,
) -> None:
    """Causal attention: q_len == k_len per sequence (triton constraint)."""
    torch.set_default_device(current_platform.device_type)
    set_random_seed(0)

    num_query_heads, num_kv_heads = num_heads
    assert num_query_heads % num_kv_heads == 0
    scale = head_size**-0.5

    query_lens = seq_lens
    kv_lens = seq_lens  # must be equal for causal
    total_q = sum(query_lens)
    total_k = sum(kv_lens)

    query = torch.randn(total_q, num_query_heads, head_size, dtype=dtype)
    key = torch.randn(total_k, num_kv_heads, head_size, dtype=dtype)
    value = torch.randn(total_k, num_kv_heads, head_size, dtype=dtype)

    cu_seqlens_q = _build_cu_seqlens(query_lens)
    cu_seqlens_k = _build_cu_seqlens(kv_lens)

    output = flash_attn_varlen_triton(
        q=query,
        k=key,
        v=value,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max(query_lens),
        max_seqlen_k=max(kv_lens),
        causal=True,
        softmax_scale=scale,
    )

    ref_output = ref_varlen_attn(
        query=query,
        key=key,
        value=value,
        query_lens=query_lens,
        kv_lens=kv_lens,
        scale=scale,
        causal=True,
    )

    torch.testing.assert_close(output, ref_output, atol=1.5e-2, rtol=1e-2)


@pytest.mark.parametrize("seq_lens", NON_CAUSAL_SEQ_LENS)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@torch.inference_mode()
def test_triton_varlen_non_causal(
    seq_lens: list[tuple[int, int]],
    num_heads: tuple[int, int],
    head_size: int,
    dtype: torch.dtype,
) -> None:
    """Non-causal attention: q_len and k_len can differ per sequence."""
    torch.set_default_device(current_platform.device_type)
    set_random_seed(0)

    num_query_heads, num_kv_heads = num_heads
    assert num_query_heads % num_kv_heads == 0
    scale = head_size**-0.5

    query_lens = [s[0] for s in seq_lens]
    kv_lens = [s[1] for s in seq_lens]
    total_q = sum(query_lens)
    total_k = sum(kv_lens)

    query = torch.randn(total_q, num_query_heads, head_size, dtype=dtype)
    key = torch.randn(total_k, num_kv_heads, head_size, dtype=dtype)
    value = torch.randn(total_k, num_kv_heads, head_size, dtype=dtype)

    cu_seqlens_q = _build_cu_seqlens(query_lens)
    cu_seqlens_k = _build_cu_seqlens(kv_lens)

    output = flash_attn_varlen_triton(
        q=query,
        k=key,
        v=value,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max(query_lens),
        max_seqlen_k=max(kv_lens),
        causal=False,
        softmax_scale=scale,
    )

    ref_output = ref_varlen_attn(
        query=query,
        key=key,
        value=value,
        query_lens=query_lens,
        kv_lens=kv_lens,
        scale=scale,
        causal=False,
    )

    torch.testing.assert_close(output, ref_output, atol=1.5e-2, rtol=1e-2)


@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@torch.inference_mode()
def test_triton_varlen_return_lse(
    num_heads: tuple[int, int],
    head_size: int,
    dtype: torch.dtype,
) -> None:
    """Verify that return_softmax_lse produces finite LSE values."""
    torch.set_default_device(current_platform.device_type)
    set_random_seed(0)

    num_query_heads, num_kv_heads = num_heads
    seq_lens = [64, 128]
    total_tokens = sum(seq_lens)
    scale = head_size**-0.5

    query = torch.randn(total_tokens, num_query_heads, head_size, dtype=dtype)
    key = torch.randn(total_tokens, num_kv_heads, head_size, dtype=dtype)
    value = torch.randn(total_tokens, num_kv_heads, head_size, dtype=dtype)

    cu_seqlens = _build_cu_seqlens(seq_lens)

    output, lse = flash_attn_varlen_triton(
        q=query,
        k=key,
        v=value,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=max(seq_lens),
        max_seqlen_k=max(seq_lens),
        causal=True,
        softmax_scale=scale,
        return_softmax_lse=True,
    )

    assert lse.shape == (num_query_heads, total_tokens)
    assert lse.dtype == torch.float32
    assert torch.all(torch.isfinite(lse))


# ═══════════════════════════════════════════════════════════════════════════ #
#  Benchmark: flash_attn_varlen_triton vs flash_attn_varlen_func
# ═══════════════════════════════════════════════════════════════════════════ #


def _bench_cuda_events(run_fn, warmup: int, iters: int) -> float:
    """Measure GPU kernel time using CUDA events (no host-sync bias)."""
    for _ in range(warmup):
        run_fn()
    torch.accelerator.synchronize()
    times: list[float] = []
    for _ in range(iters):
        start_ev = torch.Event(enable_timing=True)
        end_ev = torch.Event(enable_timing=True)
        start_ev.record()
        run_fn()
        end_ev.record()
        torch.accelerator.synchronize()
        times.append(start_ev.elapsed_time(end_ev))
    return sum(times) / len(times)


def _make_triton_runner(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    scale: float,
    causal: bool,
    head_size: int,
    config: TritonFlashAttnVarlenConfig | None = None,
):
    """Build a zero-overhead Triton kernel launcher (no Python wrapper sync)."""
    if config is None:
        config = _get_default_config(query.dtype)

    num_q_heads = query.shape[1]
    num_kv_heads = key.shape[1]
    kv_group_num = num_q_heads // num_kv_heads
    d_qk = query.shape[2]
    d_v = value.shape[2]
    num_seqs = cu_seqlens_q.numel() - 1

    sm_scale = float(scale) * RCP_LN2
    BLOCK_M = config.BLOCK_M
    BLOCK_DQK = triton.next_power_of_2(d_qk)
    BLOCK_DV = triton.next_power_of_2(d_v)
    EVEN_DQK = d_qk == BLOCK_DQK
    EVEN_DV = d_v == BLOCK_DV
    BLOCK_N = config.BLOCK_N

    q = query.contiguous()
    k = key.contiguous()
    v = value.contiguous()
    out = torch.empty_like(q)
    grid = (num_seqs, num_q_heads, triton.cdiv(max_seqlen_q, BLOCK_M))

    def run():
        _fwd_kernel_varlen[grid](
            q,
            k,
            v,
            sm_scale,
            cu_seqlens_q,
            cu_seqlens_k,
            out,
            out,
            q.stride(0),
            q.stride(1),
            k.stride(0),
            k.stride(1),
            v.stride(0),
            v.stride(1),
            out.stride(0),
            out.stride(1),
            0,
            0,
            kv_group_num=kv_group_num,
            BLOCK_M=BLOCK_M,
            BLOCK_DQK=BLOCK_DQK,
            BLOCK_DV=BLOCK_DV,
            BLOCK_N=BLOCK_N,
            IS_CAUSAL=causal,
            HAS_LSE=False,
            D_QK=d_qk,
            D_V=d_v,
            EVEN_DQK=EVEN_DQK,
            EVEN_DV=EVEN_DV,
            num_warps=config.num_warps,
            num_stages=config.num_stages,
        )

    return run


@pytest.mark.parametrize("seq_lens", BENCH_SEQ_LENS)
@pytest.mark.parametrize("num_heads", BENCH_NUM_HEADS)
@pytest.mark.parametrize("head_size", BENCH_HEAD_SIZES)
@pytest.mark.parametrize("causal", [True])
@torch.inference_mode()
def test_benchmark_triton_vs_flash_attn(
    seq_lens: list[int],
    num_heads: tuple[int, int],
    head_size: int,
    causal: bool,
) -> None:
    """
    Benchmark comparison using CUDA events for accurate GPU kernel timing.

    Run with::

        pytest tests/kernels/attention/test_triton_flash_attn.py \
            -k test_benchmark -v -s
    """
    torch.set_default_device(current_platform.device_type)
    dtype = torch.bfloat16
    num_query_heads, num_kv_heads = num_heads
    scale = head_size**-0.5

    query_lens = seq_lens
    kv_lens = seq_lens
    total_tokens = sum(seq_lens)

    query = torch.randn(total_tokens, num_query_heads, head_size, dtype=dtype)
    key = torch.randn(total_tokens, num_kv_heads, head_size, dtype=dtype)
    value = torch.randn(total_tokens, num_kv_heads, head_size, dtype=dtype)

    cu_seqlens_q = _build_cu_seqlens(query_lens)
    cu_seqlens_k = _build_cu_seqlens(kv_lens)
    max_seqlen_q = max(query_lens)
    max_seqlen_k = max(kv_lens)

    # Triton kernel (direct call, no wrapper overhead)
    triton_runner = _make_triton_runner(
        query,
        key,
        value,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        scale,
        causal,
        head_size,
    )
    triton_ms = _bench_cuda_events(
        triton_runner, warmup=BENCH_WARMUP, iters=BENCH_ITERS
    )

    # flash_attn_varlen_func (FA2 and FA3 where available)
    fa_results: dict[str, float] = {}
    if HAS_FLASH_ATTN:
        for ver in [2, 3]:
            if is_fa_version_supported(ver):

                def fa_runner(v=ver):
                    flash_attn_varlen_func(
                        q=query,
                        k=key,
                        v=value,
                        cu_seqlens_q=cu_seqlens_q,
                        cu_seqlens_k=cu_seqlens_k,
                        max_seqlen_q=max_seqlen_q,
                        max_seqlen_k=max_seqlen_k,
                        causal=causal,
                        softmax_scale=scale,
                        fa_version=v,
                    )

                ms = _bench_cuda_events(
                    fa_runner, warmup=BENCH_WARMUP, iters=BENCH_ITERS
                )
                fa_results[f"FA{ver}"] = ms

    # Report
    batch = len(seq_lens)
    label = (
        f"batch={batch} seqs={seq_lens} "
        f"heads=({num_query_heads},{num_kv_heads}) d={head_size}"
    )
    parts = [f"Triton: {triton_ms:.3f} ms"]
    for name, ms in fa_results.items():
        pct = ms / triton_ms * 100 if triton_ms > 0 else float("nan")
        parts.append(f"{name}: {ms:.3f} ms ({pct:.0f}% of Triton)")
    print(f"\n[BENCH] {label}")
    print(f"        {' | '.join(parts)}")
