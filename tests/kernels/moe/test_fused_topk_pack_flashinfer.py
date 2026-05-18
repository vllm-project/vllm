# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch.nn.functional as F

from vllm.model_executor.layers.fused_moe.utils import (
    fused_topk_softmax_pack_flashinfer,
)


def _torch_pack(topk_ids, topk_weights):
    return (topk_ids.to(torch.int32) << 16) | topk_weights.to(
        torch.bfloat16
    ).view(torch.int16)


def _torch_topk_softmax_pack(gating, topk, renormalize):
    _, topk_ids = torch.topk(gating, k=topk, dim=-1, sorted=False)
    topk_weights = gating.float().gather(1, topk_ids)
    if renormalize:
        topk_weights = F.softmax(topk_weights, dim=-1, dtype=torch.float32)
    return _torch_pack(topk_ids.to(torch.int32), topk_weights)


def _decode(packed):
    as_i64 = packed.to(torch.int64)
    ids = ((as_i64 >> 16) & 0xFFFF).to(torch.int32)
    low_i16 = (as_i64 & 0xFFFF).to(torch.int32).to(torch.int16)
    weights = low_i16.view(torch.bfloat16).to(torch.float32)
    return ids, weights


def _assert_packed_rows_match(got, expected):
    assert got.shape == expected.shape
    assert got.dtype == torch.int32 == expected.dtype
    if got.numel() == 0:
        return
    ids_g, w_g = _decode(got)
    ids_e, w_e = _decode(expected)
    sorted_ids_g, perm_g = ids_g.sort(dim=-1)
    sorted_ids_e, perm_e = ids_e.sort(dim=-1)
    assert torch.equal(sorted_ids_g, sorted_ids_e)
    w_g_sorted = torch.gather(w_g, 1, perm_g)
    w_e_sorted = torch.gather(w_e, 1, perm_e)
    torch.testing.assert_close(w_g_sorted, w_e_sorted, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize(
    "shape",
    [
        (1, 8, 2),
        (7, 32, 3),
        (16, 64, 4),
        (32, 32, 1),
        (128, 128, 8),
        (1024, 128, 8),
        (3, 256, 8),
        (4, 7, 5),
        (0, 128, 8),
    ],
)
@pytest.mark.parametrize("renormalize", [True, False])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_fused_topk_softmax_pack_flashinfer(shape, renormalize, dtype):
    num_tokens, num_experts, topk = shape
    torch.manual_seed(num_tokens * 131 + num_experts * 7 + topk)

    gating = torch.randn(num_tokens, num_experts, device="cuda", dtype=dtype)

    got = fused_topk_softmax_pack_flashinfer(gating, topk, renormalize)
    expected = _torch_topk_softmax_pack(gating, topk, renormalize)

    _assert_packed_rows_match(got, expected)


@pytest.mark.parametrize("num_tokens", [1, 16, 128, 1024, 4096])
@pytest.mark.parametrize("topk", [2, 4, 8])
def test_fused_topk_softmax_pack_perf_shapes(num_tokens, topk):
    num_experts = 256
    torch.manual_seed(42)
    gating = torch.randn(
        num_tokens, num_experts, device="cuda", dtype=torch.bfloat16
    )

    got = fused_topk_softmax_pack_flashinfer(gating, topk, renormalize=True)
    expected = _torch_topk_softmax_pack(gating, topk, renormalize=True)

    _assert_packed_rows_match(got, expected)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
