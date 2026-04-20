# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Correctness tests for the Triton sparse MLA kernel.

Compares split-KV against the single-pass (`num_kv_splits=1`) path
produced by the same kernel — both paths must agree to within bf16 ULPs.
"""

import pytest
import torch

from vllm.platforms import current_platform
from vllm.v1.attention.ops.triton_sparse_mla_kernel import (
    _DIM_QK,
    triton_sparse_mla_attention,
)

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda_alike(),
    reason="Triton sparse MLA kernel requires CUDA/ROCm",
)


@pytest.fixture(scope="module")
def kv_cache():
    torch.manual_seed(0)
    return torch.randn(32768, 1, _DIM_QK, dtype=torch.bfloat16, device="cuda")


def _assert_split_matches_single_pass(
    num_tokens: int,
    num_heads: int,
    topk: int,
    num_kv_splits: int | None,
    kv_cache: torch.Tensor,
) -> None:
    torch.manual_seed(0)
    q = torch.randn(num_tokens, num_heads, _DIM_QK, dtype=torch.bfloat16, device="cuda")
    indices = torch.randint(
        0, kv_cache.shape[0], (num_tokens, 1, topk), dtype=torch.int32, device="cuda"
    )
    out_ref = triton_sparse_mla_attention(
        q,
        kv_cache,
        indices,
        sm_scale=0.1,
        num_kv_splits=1,
    )
    out = triton_sparse_mla_attention(
        q,
        kv_cache,
        indices,
        sm_scale=0.1,
        num_kv_splits=num_kv_splits,
    )
    torch.testing.assert_close(
        out.float(),
        out_ref.float(),
        atol=5e-2,
        rtol=5e-3,
    )


@pytest.mark.parametrize(
    "num_tokens,num_heads",
    [(1, 16), (1, 128), (8, 32), (32, 128), (128, 16)],
)
@pytest.mark.parametrize("topk", [1024, 2048, 4096])
@pytest.mark.parametrize("num_kv_splits", [2, 4, 8])
def test_split_kv_matches_single_pass(
    num_tokens, num_heads, topk, num_kv_splits, kv_cache
):
    _assert_split_matches_single_pass(
        num_tokens,
        num_heads,
        topk,
        num_kv_splits,
        kv_cache,
    )


@pytest.mark.parametrize("num_tokens", [1, 8, 32, 128])
def test_auto_split_matches_single_pass(num_tokens, kv_cache):
    _assert_split_matches_single_pass(
        num_tokens,
        num_heads=128,
        topk=2048,
        num_kv_splits=None,
        kv_cache=kv_cache,
    )


@pytest.mark.parametrize("num_kv_splits", [1, 2, 4, 8])
def test_short_prefill_no_nan(num_kv_splits, kv_cache):
    """Regression: short prefill where most topk slots are -1 sentinels.

    The indexer fills 2048 topk positions with only a handful of valid
    indices; the rest are -1. Before the NEG_LARGE sentinel fix, the online
    softmax produced NaN via `max(-inf, -inf) = -inf` and
    `exp2(-inf − -inf) = NaN`, poisoning every split.
    """
    torch.manual_seed(0)
    num_tokens, num_heads, topk = 5, 16, 2048
    q = torch.randn(num_tokens, num_heads, _DIM_QK, dtype=torch.bfloat16, device="cuda")
    indices = torch.full((num_tokens, 1, topk), -1, dtype=torch.int32, device="cuda")
    # Only the first `t+1` slots of each query hold valid indices; the
    # remaining ~2045 slots are -1, producing many all-invalid BLOCK_N tiles.
    for t in range(num_tokens):
        indices[t, 0, : t + 1] = torch.arange(
            64, 64 + t + 1, dtype=torch.int32, device="cuda"
        )
    out = triton_sparse_mla_attention(
        q, kv_cache, indices, sm_scale=0.0417, num_kv_splits=num_kv_splits
    )
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()
