# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.platforms import current_platform
from vllm.third_party.flash_linear_attention.ops.kda import fused_recurrent_kda

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda_alike(), reason="Requires CUDA or ROCm"
)


@pytest.mark.parametrize(
    "head_k_dim, head_v_dim, num_v_heads, v_dtype",
    [
        (64, 32, 1, torch.float16),
        (64, 64, 1, torch.float16),
        (32, 64, 1, torch.float16),
        (64, 32, 2, torch.float16),
        (64, 64, 1, torch.bfloat16),
    ],
    ids=["k64-v32", "k64-v64", "k32-v64", "k64-v32-hv2", "v-bf16"],
)
def test_fused_recurrent_kda_output_has_value_shape(
    head_k_dim, head_v_dim, num_v_heads, v_dtype
):
    """The output must have v's shape and k's dtype (#68415).
    The kernel writes HV heads of width V.
    Zero queries make every output value zero."""
    device = "cuda"
    num_tokens = 4
    q = torch.zeros(1, num_tokens, 1, head_k_dim, dtype=torch.float16, device=device)
    k = torch.ones_like(q)
    v = torch.ones(1, num_tokens, num_v_heads, head_v_dim, dtype=v_dtype, device=device)
    g = -torch.ones(
        1, num_tokens, num_v_heads, head_k_dim, dtype=torch.float32, device=device
    )
    beta = torch.ones(1, num_tokens, num_v_heads, dtype=torch.float16, device=device)
    state = torch.zeros(
        num_tokens + 1,
        num_v_heads,
        head_v_dim,
        head_k_dim,
        dtype=torch.float32,
        device=device,
    )

    out, _ = fused_recurrent_kda(
        q,
        k,
        v,
        g,
        beta,
        initial_state=state,
        cu_seqlens=torch.arange(num_tokens + 1, dtype=torch.int32, device=device),
        ssm_state_indices=torch.arange(
            1, num_tokens + 1, dtype=torch.int32, device=device
        ),
    )

    assert out.shape == v.shape
    assert out.dtype == k.dtype
    assert torch.count_nonzero(out) == 0
