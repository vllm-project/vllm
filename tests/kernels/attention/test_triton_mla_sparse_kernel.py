# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Correctness tests for the Triton sparse MLA kernel.

Compares split-KV against the single-pass (`num_kv_splits=1`) path
produced by the same kernel — both paths must agree to within bf16 ULPs.
"""

import pytest
import torch

from vllm.platforms import current_platform
from vllm.v1.attention.ops.triton_mla_sparse_kernel import (
    _DIM_QK,
    triton_mla_sparse_attention,
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
    ref = triton_mla_sparse_attention(
        q,
        kv_cache,
        indices,
        sm_scale=0.1,
        num_kv_splits=1,
        return_lse=True,
    )
    result = triton_mla_sparse_attention(
        q,
        kv_cache,
        indices,
        sm_scale=0.1,
        num_kv_splits=num_kv_splits,
        return_lse=True,
    )
    assert isinstance(ref, tuple)
    assert isinstance(result, tuple)
    out_ref, lse_ref = ref
    out, lse = result
    assert lse.shape == (num_tokens, num_heads)
    torch.testing.assert_close(
        out.float(),
        out_ref.float(),
        atol=5e-2,
        rtol=5e-3,
    )
    torch.testing.assert_close(lse, lse_ref, atol=5e-3, rtol=5e-3)


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
    out = triton_mla_sparse_attention(
        q, kv_cache, indices, sm_scale=0.0417, num_kv_splits=num_kv_splits
    )
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()


@pytest.mark.parametrize("num_kv_splits", [1, 4])
def test_return_lse_matches_dense_reference(num_kv_splits: int) -> None:
    torch.manual_seed(1)
    num_tokens, num_heads, topk = 2, 16, 128
    scale = _DIM_QK**-0.5
    q = torch.randn(num_tokens, num_heads, _DIM_QK, dtype=torch.bfloat16, device="cuda")
    kv = torch.randn(256, 1, _DIM_QK, dtype=torch.bfloat16, device="cuda")
    indices = torch.stack(
        [torch.randperm(kv.shape[0], device="cuda")[:topk] for _ in range(num_tokens)]
    ).to(torch.int32)[:, None, :]

    result = triton_mla_sparse_attention(
        q,
        kv,
        indices,
        sm_scale=scale,
        num_kv_splits=num_kv_splits,
        return_lse=True,
    )

    assert isinstance(result, tuple)
    _, lse = result
    for token in range(num_tokens):
        selected_kv = kv[indices[token, 0].long(), 0].float()
        logits = (q[token].float() @ selected_kv.T) * scale
        expected_lse = torch.logsumexp(logits, dim=-1)
        torch.testing.assert_close(lse[token], expected_lse, atol=5e-3, rtol=5e-3)


@pytest.mark.parametrize("num_kv_splits", [1, 4])
def test_empty_sparse_row_returns_merge_identity(num_kv_splits: int) -> None:
    q = torch.zeros(1, 16, _DIM_QK, dtype=torch.bfloat16, device="cuda")
    kv = torch.zeros(1, 1, _DIM_QK, dtype=torch.bfloat16, device="cuda")
    indices = torch.full((1, 1, 128), -1, dtype=torch.int32, device="cuda")

    result = triton_mla_sparse_attention(
        q,
        kv,
        indices,
        sm_scale=0.1,
        num_kv_splits=num_kv_splits,
        return_lse=True,
    )

    assert isinstance(result, tuple)
    output, lse = result
    assert torch.count_nonzero(output) == 0
    assert torch.isneginf(lse).all()
