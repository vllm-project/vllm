# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-5.3-Flash KDA decode kernel: token-strided q/k/v/beta inputs.

The decode path hands the recurrent kernel column slices of the merged
``q|k|v`` conv output and of the fused ``qkvbfg_a`` projection (beta). The
kernel must read those in place, bit-identically to contiguous copies.
"""

import pytest
import torch

from vllm.models.glm5next.nvidia.ops.third_party.kda import fused_recurrent_kda
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda(), reason="CUDA-only Triton kernel"
)


@pytest.mark.parametrize("num_seqs", [1, 7])
@pytest.mark.parametrize("num_heads", [16])
@pytest.mark.parametrize("head_dim", [128])
def test_fused_recurrent_kda_strided_inputs_match_contiguous(
    num_seqs: int, num_heads: int, head_dim: int
):
    torch.manual_seed(0)
    device = torch.device("cuda")
    proj = num_heads * head_dim
    # Merged conv output [T, q|k|v] and fused projection [T, qkv|b|f_a|g_a].
    qkv = torch.randn(num_seqs, 3 * proj, dtype=torch.bfloat16, device=device)
    projected = torch.randn(
        num_seqs,
        3 * proj + num_heads + 2 * head_dim,
        dtype=torch.bfloat16,
        device=device,
    )
    g = torch.randn(
        1, num_seqs, num_heads, head_dim, dtype=torch.bfloat16, device=device
    )
    a_log = torch.randn(1, 1, num_heads, 1, dtype=torch.float32, device=device)
    dt_bias = torch.randn(proj, dtype=torch.float32, device=device)
    cu_seqlens = torch.arange(num_seqs + 1, dtype=torch.int32, device=device)
    state_indices = torch.randperm(num_seqs, device=device).to(torch.int32) + 1
    state = torch.randn(
        num_seqs + 1, num_heads, head_dim, head_dim, dtype=torch.float32, device=device
    )

    def rearr(x: torch.Tensor) -> torch.Tensor:
        return x.reshape(1, -1, num_heads, head_dim)

    q, k, v = qkv.split(proj, dim=-1)
    beta = projected[:, 3 * proj : 3 * proj + num_heads].unsqueeze(0)
    if num_seqs > 1:
        assert not rearr(q).is_contiguous() and not beta.is_contiguous()

    def run(q, k, v, beta, state):
        out, _ = fused_recurrent_kda(
            q=rearr(q),
            k=rearr(k),
            v=rearr(v),
            g=g,
            beta=beta,
            initial_state=state,
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=cu_seqlens,
            ssm_state_indices=state_indices,
            sigmoid_beta=True,
            a_log=a_log,
            g_bias=dt_bias,
            compute_gate=True,
            lower_bound=-5.0,
        )
        return out, state

    state_ref = state.clone()
    out_ref, state_ref = run(
        q.contiguous(), k.contiguous(), v.contiguous(), beta.contiguous(), state_ref
    )
    out, state = run(q, k, v, beta, state)
    torch.testing.assert_close(out, out_ref, rtol=0, atol=0)
    torch.testing.assert_close(state, state_ref, rtol=0, atol=0)
