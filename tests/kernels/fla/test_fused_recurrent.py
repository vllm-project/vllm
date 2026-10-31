# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""With ``inplace_final_state=False`` the recurrent delta-rule kernels return
the state after every token, one row per token. A dense ``[B, T]`` batch must
return the same rows as the same tokens packed into a varlen batch.
"""

import pytest
import torch

from vllm.platforms import current_platform
from vllm.third_party.flash_linear_attention.ops import (
    fused_recurrent_gated_delta_rule,
    fused_sigmoid_gating_delta_rule_update,
)
from vllm.third_party.flash_linear_attention.ops.kda import fused_recurrent_kda

DEVICE = current_platform.device_type

pytestmark = pytest.mark.skipif(
    not (current_platform.is_cuda_alike() or current_platform.is_xpu()),
    reason="Gated delta rule Triton kernels require a CUDA-alike or XPU device.",
)

B, T, H, D = 3, 4, 2, 16


def run_sigmoid_gating(q, k, v, g, beta, **kwargs):
    zeros = torch.zeros(H, device=q.device)
    return fused_sigmoid_gating_delta_rule_update(
        A_log=zeros,
        a=g.reshape(-1, H),
        b=beta.reshape(-1, H),
        dt_bias=zeros,
        q=q,
        k=k,
        v=v,
        **kwargs,
    )


@pytest.mark.parametrize(
    "run,g_shape",
    [
        (fused_recurrent_gated_delta_rule, (B, T, H)),
        (fused_recurrent_kda, (B, T, H, D)),
        (run_sigmoid_gating, (B, T, H)),
    ],
    ids=["gdn", "kda", "sigmoid_gating"],
)
def test_non_inplace_dense_batch_matches_varlen(run, g_shape):
    device = torch.device(DEVICE)
    generator = torch.Generator(device=device).manual_seed(0)

    def randn(*shape):
        return torch.randn(shape, device=device, generator=generator)

    q, k, v = randn(B, T, H, D), randn(B, T, H, D), randn(B, T, H, D)
    g = -randn(*g_shape).sigmoid()
    beta = randn(B, T, H).sigmoid()
    kwargs = dict(
        scale=D**-0.5,
        initial_state=randn(B + 1, H, D, D),
        inplace_final_state=False,
        ssm_state_indices=torch.arange(1, B + 1, device=device, dtype=torch.int32),
        use_qk_l2norm_in_kernel=True,
    )

    o, states = run(q=q, k=k, v=v, g=g, beta=beta, **kwargs)

    def flat(x):
        return x.reshape(1, B * T, *x.shape[2:])

    cu_seqlens = torch.arange(0, B * T + 1, T, device=device, dtype=torch.int32)
    o_ref, states_ref = run(
        *map(flat, (q, k, v, g, beta)), cu_seqlens=cu_seqlens, **kwargs
    )

    torch.testing.assert_close(states, states_ref)
    torch.testing.assert_close(flat(o), o_ref)
