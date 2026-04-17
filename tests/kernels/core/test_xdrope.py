# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Regression tests for XDRotaryEmbedding (xdrope).

Covers the dtype-mismatch bug reported in
https://github.com/vllm-project/vllm/issues/40165:
XDRotaryEmbedding.forward_{native,cuda} did not call
_match_cos_sin_cache_dtype(), so a float32 cos_sin_cache combined with
float16/bfloat16 query/key silently promoted outputs to float32,
crashing flash_attn downstream.
"""

import pytest
import torch

from vllm.model_executor.layers.rotary_embedding.xdrope import (
    XDRotaryEmbedding,
)
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed

# ── helpers ──────────────────────────────────────────────────────────

XDROPE_SECTION = [16, 16, 16, 16]  # P/W/H/T — 4 sections, 64 total
HEAD_SIZE = 128
ROTARY_DIM = 64  # sum(XDROPE_SECTION)
MAX_POSITION = 4096
BASE = 10000
SCALING_ALPHA = 1.0


def _make_xdrope(
    *,
    cache_dtype: torch.dtype,
    device: torch.device,
) -> XDRotaryEmbedding:
    """Create an XDRotaryEmbedding whose cos_sin_cache has *cache_dtype*."""
    rope = XDRotaryEmbedding(
        head_size=HEAD_SIZE,
        rotary_dim=ROTARY_DIM,
        max_position_embeddings=MAX_POSITION,
        base=BASE,
        is_neox_style=True,
        scaling_alpha=SCALING_ALPHA,
        dtype=cache_dtype,
        xdrope_section=XDROPE_SECTION,
    )
    return rope.to(device=device)


def _make_inputs(
    num_tokens: int,
    num_q_heads: int,
    num_kv_heads: int,
    dtype: torch.dtype,
    device: torch.device,
):
    """Return (positions, query, key) with the requested *dtype*."""
    set_random_seed(42)
    # 4-row positions for P/W/H/T
    positions = torch.randint(0, MAX_POSITION // 4, (4, num_tokens), device=device)
    query = torch.randn(num_tokens, num_q_heads * HEAD_SIZE, dtype=dtype, device=device)
    key = torch.randn(num_tokens, num_kv_heads * HEAD_SIZE, dtype=dtype, device=device)
    return positions, query, key


# ── dtype-mismatch regression tests ─────────────────────────────────
# The invariant under test:
#   output dtype == input query dtype,
#   regardless of cos_sin_cache dtype.


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(),
    reason="Requires CUDA or ROCm.",
)
@pytest.mark.parametrize(
    "query_dtype",
    [torch.float16, torch.bfloat16],
    ids=["fp16", "bf16"],
)
@pytest.mark.parametrize("num_tokens", [1, 11, 128])
def test_xdrope_dtype_mismatch_forward_native(
    default_vllm_config,
    query_dtype: torch.dtype,
    num_tokens: int,
):
    """forward_native must preserve query dtype even when cache is float32.

    Regression test for https://github.com/vllm-project/vllm/issues/40165
    """
    device = torch.device("cuda")
    rope = _make_xdrope(cache_dtype=torch.float32, device=device)

    # Sanity: cache is indeed float32 before the call
    assert rope.cos_sin_cache.dtype == torch.float32

    positions, query, key = _make_inputs(
        num_tokens,
        num_q_heads=8,
        num_kv_heads=2,
        dtype=query_dtype,
        device=device,
    )

    out_q, out_k = rope.forward_native(
        positions,
        query.clone(),
        key.clone(),
    )

    assert out_q.dtype == query_dtype, (
        f"forward_native output query dtype {out_q.dtype} != "
        f"input dtype {query_dtype} (cache was float32)"
    )
    assert out_k.dtype == query_dtype, (
        f"forward_native output key dtype {out_k.dtype} != "
        f"input dtype {query_dtype} (cache was float32)"
    )


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(),
    reason="Requires CUDA or ROCm.",
)
@pytest.mark.parametrize(
    "query_dtype",
    [torch.float16, torch.bfloat16],
    ids=["fp16", "bf16"],
)
@pytest.mark.parametrize("num_tokens", [1, 11, 128])
def test_xdrope_dtype_mismatch_forward_cuda(
    default_vllm_config,
    query_dtype: torch.dtype,
    num_tokens: int,
):
    """forward_cuda must preserve query dtype even when cache is float32.

    Regression test for https://github.com/vllm-project/vllm/issues/40165
    """
    device = torch.device("cuda")
    rope = _make_xdrope(cache_dtype=torch.float32, device=device)

    assert rope.cos_sin_cache.dtype == torch.float32

    positions, query, key = _make_inputs(
        num_tokens,
        num_q_heads=8,
        num_kv_heads=2,
        dtype=query_dtype,
        device=device,
    )

    out_q, out_k = rope.forward_cuda(
        positions,
        query.clone(),
        key.clone(),
    )

    assert out_q.dtype == query_dtype, (
        f"forward_cuda output query dtype {out_q.dtype} != "
        f"input dtype {query_dtype} (cache was float32)"
    )
    assert out_k.dtype == query_dtype, (
        f"forward_cuda output key dtype {out_k.dtype} != "
        f"input dtype {query_dtype} (cache was float32)"
    )


# ── same-dtype baseline (must always pass) ──────────────────────────


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(),
    reason="Requires CUDA or ROCm.",
)
@pytest.mark.parametrize(
    "dtype",
    [torch.float16, torch.bfloat16],
    ids=["fp16", "bf16"],
)
@pytest.mark.parametrize("num_tokens", [1, 11, 128])
def test_xdrope_native_vs_cuda_consistency(
    default_vllm_config,
    dtype: torch.dtype,
    num_tokens: int,
):
    """forward_native and forward_cuda must produce numerically close results.

    Mirrors the pattern used in test_mrope.py.
    """
    device = torch.device("cuda")
    rope = _make_xdrope(cache_dtype=dtype, device=device)

    positions, query, key = _make_inputs(
        num_tokens,
        num_q_heads=8,
        num_kv_heads=2,
        dtype=dtype,
        device=device,
    )

    q_native, k_native = rope.forward_native(
        positions,
        query.clone(),
        key.clone(),
    )
    q_cuda, k_cuda = rope.forward_cuda(
        positions,
        query.clone(),
        key.clone(),
    )

    torch.testing.assert_close(q_native, q_cuda, atol=1e-2, rtol=1.6e-2)
    torch.testing.assert_close(k_native, k_cuda, atol=1e-2, rtol=1.6e-2)
