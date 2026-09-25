# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ROCm Kimi-K3 sigmoid-gating kernel vs a float32 recurrence.

The CUDA Qwen kernel is covered by tests/kernels/test_fused_sigmoid_gating_delta_rule.py.
This file covers vllm.models.kimi_k3.amd.ops.fused_sigmoid_gating, including the
Kimi gate_lower_bound path that kernel does not implement.
"""

import pytest
import torch
import torch.nn.functional as F

from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda_alike(),
    reason="Kimi-K3 sigmoid-gating kernel requires a CUDA/ROCm GPU",
)

DEVICE = current_platform.device_type


def _recurrent_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    initial_state: torch.Tensor,
    indices: torch.Tensor,
    cu_seqlens: torch.Tensor,
    lower_bound: float | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Match fused_sigmoid_gating_delta_rule_update, one token at a time."""
    scale = k.shape[-1] ** -0.5
    state = initial_state.float().clone()
    out = torch.empty_like(v, dtype=torch.float32)
    num_seqs = cu_seqlens.numel() - 1
    for seq in range(num_seqs):
        bos = int(cu_seqlens[seq])
        eos = int(cu_seqlens[seq + 1])
        hidden = state[int(indices[seq, 0])].clone()
        for step, token in enumerate(range(bos, eos)):
            query = q[0, token].float()
            key = k[0, token].float()
            query = query * torch.rsqrt((query * query).sum(-1, keepdim=True) + 1e-6)
            key = key * torch.rsqrt((key * key).sum(-1, keepdim=True) + 1e-6)
            query = query * scale
            gate_input = a[0, token].float() + dt_bias.float()
            if lower_bound is None:
                gate = -A_log.float().exp()[:, None] * F.softplus(gate_input)
            else:
                gate = lower_bound * torch.sigmoid(
                    A_log.float().exp()[:, None] * gate_input
                )
            beta = torch.sigmoid(b[0, token].float())
            hidden = hidden * gate.exp()[:, None, :]
            value = v[0, token].float() - (hidden * key[:, None, :]).sum(-1)
            value = value * beta[:, None]
            hidden = hidden + value[:, :, None] * key[:, None, :]
            out[0, token] = (hidden * query[:, None, :]).sum(-1)
            state[int(indices[seq, step])] = hidden
    return out, state


@pytest.mark.parametrize("lower_bound", [-5.0, None])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("strided_beta", [False, True])
@torch.inference_mode()
def test_amd_fused_sigmoid_gating_matches_recurrence(
    lower_bound: float | None,
    dtype: torch.dtype,
    strided_beta: bool,
) -> None:
    from vllm.models.kimi_k3.amd.ops.fused_sigmoid_gating import (
        fused_sigmoid_gating_delta_rule_update,
    )

    torch.manual_seed(0)
    num_seqs, seq_len, num_heads, head_dim = 2, 3, 4, 128
    total = num_seqs * seq_len
    q = torch.randn(1, total, num_heads, head_dim, device=DEVICE, dtype=dtype)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    a = torch.randn(1, total, num_heads, head_dim, device=DEVICE, dtype=dtype)
    if strided_beta:
        beta_storage = torch.randn(
            1, total, num_heads + 3, device=DEVICE, dtype=dtype
        )
        b = beta_storage[..., :num_heads]
    else:
        b = torch.randn(1, total, num_heads, device=DEVICE, dtype=dtype)
    A_log = torch.randn(num_heads, device=DEVICE, dtype=torch.float32)
    dt_bias = torch.randn(num_heads, head_dim, device=DEVICE, dtype=torch.float32)
    cu_seqlens = torch.arange(0, total + 1, seq_len, device=DEVICE, dtype=torch.int32)
    # Slot 0 is the reserved null block. Each sequence owns seq_len contiguous slots.
    indices = (
        torch.arange(1, total + 1, device=DEVICE, dtype=torch.int32).view(
            num_seqs, seq_len
        )
    )
    state = torch.randn(
        total + 1,
        num_heads,
        head_dim,
        head_dim,
        device=DEVICE,
        dtype=dtype,
    )

    ref_out, ref_state = _recurrent_reference(
        q,
        k,
        v,
        a,
        b,
        A_log,
        dt_bias,
        state,
        indices,
        cu_seqlens,
        lower_bound,
    )
    out, final_state = fused_sigmoid_gating_delta_rule_update(
        A_log=A_log,
        a=a,
        b=b,
        dt_bias=dt_bias,
        q=q,
        k=k,
        v=v,
        initial_state=state.clone(),
        inplace_final_state=True,
        cu_seqlens=cu_seqlens,
        ssm_state_indices=indices,
        use_qk_l2norm_in_kernel=True,
        lower_bound=lower_bound,
    )

    atol = 2e-2 if dtype is torch.bfloat16 else 1e-4
    rtol = 2e-2 if dtype is torch.bfloat16 else 1e-4
    torch.testing.assert_close(out.float(), ref_out, atol=atol, rtol=rtol)
    torch.testing.assert_close(final_state.float(), ref_state, atol=atol, rtol=rtol)
