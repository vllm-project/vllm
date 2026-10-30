# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The fused GDN input projection matches one dense GEMM on both sides of the
dual split-K / two-GEMM token threshold."""

import pytest
import torch

from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
    GDN_AITER_IN_PROJ_DUAL_SPLITK_AVAILABLE,
    GDN_IN_PROJ_BA_SPLITS,
    MAX_FUSED_IN_PROJ_QKVZBA_TOKENS,
)

pytestmark = pytest.mark.skipif(
    not GDN_AITER_IN_PROJ_DUAL_SPLITK_AVAILABLE,
    reason="needs AITER's dual split-K GEMM and fused GDN decode",
)


@pytest.mark.parametrize(
    "num_tokens",
    [1, MAX_FUSED_IN_PROJ_QKVZBA_TOKENS, MAX_FUSED_IN_PROJ_QKVZBA_TOKENS + 1, 1000],
)
@torch.inference_mode()
def test_in_proj_qkvzba_matches_dense_gemm(num_tokens: int) -> None:
    hidden, qkvz_size, ba_size = 5120, 1024, 96
    torch.manual_seed(0)
    x = torch.randn(num_tokens, hidden, dtype=torch.bfloat16, device="cuda")
    weight = (
        torch.randn(qkvz_size + ba_size, hidden, dtype=torch.bfloat16, device="cuda")
        * 0.05
    )
    qkvzba, qkvz, ba, ba_partials = torch.ops.vllm.qwen_gdn_in_proj_qkvzba(
        x, weight, ba_size
    )
    assert ba_partials.shape == (GDN_IN_PROJ_BA_SPLITS, num_tokens, ba_size)
    if num_tokens <= MAX_FUSED_IN_PROJ_QKVZBA_TOKENS:
        qkvz, ba = qkvzba[:, :qkvz_size], ba_partials.sum(0)
    ref = (x.float() @ weight.float().t()).to(x.dtype)
    torch.testing.assert_close(
        torch.cat([qkvz, ba.to(x.dtype)], dim=-1), ref, atol=0.5, rtol=1e-2
    )
