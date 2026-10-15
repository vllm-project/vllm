# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.platforms import current_platform
from vllm.third_party.flash_linear_attention.ops.kda import fused_recurrent_kda

DEVICE = current_platform.device_type

pytestmark = pytest.mark.skipif(
    not (current_platform.is_cuda_alike() or current_platform.is_xpu()),
    reason="KDA Triton kernels require a CUDA-alike or XPU device.",
)


@pytest.mark.parametrize("head_dim", [40, 64, 100])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_fused_recurrent_kda_non_pow2_head_dim(head_dim, dtype):
    """NaN gate padding must not contaminate an otherwise zero output."""
    q = torch.zeros(1, 1, 1, head_dim, device=DEVICE, dtype=dtype)
    storage = torch.full((2 * head_dim,), float("nan"), device=DEVICE)
    g = storage[:head_dim].view_as(q).zero_()
    state = torch.zeros(1, 1, head_dim, head_dim, device=DEVICE)
    out, _ = fused_recurrent_kda(
        q, q, q, g, q[..., 0], initial_state=state, inplace_final_state=False
    )
    torch.testing.assert_close(out, q)
