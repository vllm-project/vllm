# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MuseSpark1x ROCm RMSNorm that hands AITER experts a per-route copy."""

import pytest
import torch

from vllm.models.muse_spark.amd.ops.fused_norm_expand import (
    fused_rmsnorm_expand,
    take_expanded,
)
from vllm.models.muse_spark.common.ops.fused_rmsnorm import fused_rmsnorm_fwd
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm(), reason="requires a ROCm GPU"
)


@pytest.mark.parametrize("input_scale", [False, True])
@torch.inference_mode()
def test_rmsnorm_expand_hands_aiter_the_norm_once_per_route(input_scale: bool) -> None:
    """The experts get the norm repeated top_k times, only for its own output."""
    num_tokens, top_k, hidden = 7, 8, 4096
    x = torch.randn(num_tokens, hidden, device="cuda", dtype=torch.bfloat16) * 3
    w = torch.randn(hidden, device="cuda", dtype=torch.bfloat16) * 0.1
    ref = fused_rmsnorm_fwd(x, w, 1e-6, gain_center=1.0, input_scale=input_scale)

    first = fused_rmsnorm_expand(x, w, 1e-6, 1.0, input_scale, top_k)
    torch.testing.assert_close(first, ref)
    assert take_expanded(ref, top_k) is None
    assert take_expanded(first.clone(), top_k) is None
    assert take_expanded(first, top_k - 1) is None
    expanded = take_expanded(first, top_k)
    assert expanded is not None and expanded.shape == (num_tokens * top_k, hidden)
    torch.testing.assert_close(
        expanded,
        first[:, None, :].expand(num_tokens, top_k, hidden).reshape(-1, hidden),
        atol=0,
        rtol=0,
    )
