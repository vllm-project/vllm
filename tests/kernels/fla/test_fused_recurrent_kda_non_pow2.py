# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch.nn.functional as F

from vllm.platforms import current_platform
from vllm.third_party.flash_linear_attention.ops.kda import fused_recurrent_kda

DEVICE = current_platform.device_type

pytestmark = pytest.mark.skipif(
    not (current_platform.is_cuda_alike() or current_platform.is_xpu()),
    reason="KDA Triton kernels require a CUDA-alike or XPU device.",
)


def _ref_fused_recurrent_kda(q, k, v, g, beta, h0, cu_seqlens, scale):
    """Per-token fp32 reference of the KDA recurrence with L2-normalised q/k.

    q, k: [1, T, H, K]; v: [1, T, HV, V]; g: [1, T, HV, K] (log decay);
    beta: [1, T, HV]; h0: [N, HV, V, K] (initial state of each sequence).
    """
    _, _, H, K = k.shape
    HV = v.shape[2]
    q, k, v, g, beta = (x.float() for x in (q[0], k[0], v[0], g[0], beta[0]))
    q = F.normalize(q, dim=-1) * scale
    k = F.normalize(k, dim=-1)
    out = torch.empty_like(v)
    for n in range(len(cu_seqlens) - 1):
        state = h0[n].float().clone()  # [HV, V, K]
        for t in range(int(cu_seqlens[n]), int(cu_seqlens[n + 1])):
            for hv in range(HV):
                kt = k[t, hv // (HV // H)]
                qt = q[t, hv // (HV // H)]
                s = state[hv] * torch.exp(g[t, hv])[None, :]
                u = (v[t, hv] - (s * kt[None, :]).sum(-1)) * beta[t, hv]
                s = s + u[:, None] * kt[None, :]
                state[hv] = s
                out[t, hv] = (s * qt[None, :]).sum(-1)
    return out[None]


@pytest.mark.parametrize("head_dim", [40, 100])
@pytest.mark.parametrize("seq_lens", [[1], [3, 2]])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_fused_recurrent_kda_non_pow2_head_dim(
    head_dim: int, seq_lens: list[int], dtype: torch.dtype
) -> None:
    """The kernel rounds K up to a power of two. The per-key gate ``g`` must be
    masked like every other operand, otherwise the padding lanes read past the
    end of ``g`` and feed whatever is there (NaN here) into the state."""
    torch.manual_seed(0)
    H = HV = 2
    K = V = head_dim
    num_tokens = sum(seq_lens)
    num_seqs = len(seq_lens)
    max_len = max(seq_lens)
    cu_seqlens = torch.tensor(
        [0, *torch.tensor(seq_lens).cumsum(0).tolist()],
        dtype=torch.int32,
        device=DEVICE,
    )
    # One state slot per (sequence, token); slot 0 is the null block.
    ssm_state_indices = (
        torch.arange(num_seqs * max_len, dtype=torch.int32, device=DEVICE).view(
            num_seqs, max_len
        )
        + 1
    )
    q = torch.randn(1, num_tokens, H, K, device=DEVICE, dtype=dtype)
    k = torch.randn(1, num_tokens, H, K, device=DEVICE, dtype=dtype)
    v = torch.randn(1, num_tokens, HV, V, device=DEVICE, dtype=dtype)
    beta = torch.rand(1, num_tokens, HV, device=DEVICE, dtype=dtype)
    state_pool = torch.randn(
        num_seqs * max_len + 1, HV, V, K, device=DEVICE, dtype=torch.float32
    )
    h0 = state_pool[ssm_state_indices[:, 0].long()]

    # `g` is the last tensor of its allocation and is followed by NaNs, so an
    # unmasked read of the padding lanes of the last (token, head) row is both
    # visible to the output and, under compute-sanitizer, out of bounds.
    g_numel = num_tokens * HV * K
    g_storage = torch.full((g_numel + 2 * K,), float("nan"), device=DEVICE)
    g = g_storage[:g_numel].view(1, num_tokens, HV, K)
    g.copy_(-torch.rand(1, num_tokens, HV, K, device=DEVICE))

    scale = K**-0.5
    ref = _ref_fused_recurrent_kda(q, k, v, g, beta, h0, cu_seqlens.tolist(), scale)
    out, _ = fused_recurrent_kda(
        q,
        k,
        v,
        g,
        beta,
        scale=scale,
        initial_state=state_pool.clone(),
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=cu_seqlens,
        ssm_state_indices=ssm_state_indices,
    )

    assert torch.isfinite(out.float()).all()
    tol = 1e-4 if dtype == torch.float32 else 3e-2
    torch.testing.assert_close(out.float(), ref, atol=tol, rtol=tol)
