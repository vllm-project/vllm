# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.v1.attention.ops.triton_mla_sparse import triton_bf16_mla_sparse_interface


# https://github.com/deepseek-ai/FlashMLA/blob/main/tests/ref.py#L7
def _merge_two_lse(
    lse0: torch.Tensor, lse1: torch.Tensor | None, s_q: int, h_q: int
) -> torch.Tensor:
    if lse1 is None:
        return lse0
    else:
        return torch.logsumexp(
            torch.stack([lse0.view(s_q, h_q), lse1.broadcast_to(s_q, h_q)], dim=0),
            dim=0,
        )


# Adapted from https://github.com/deepseek-ai/FlashMLA/blob/main/tests/ref.py#L19
def reference_mla_sparse_prefill(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    d_v: int,
    topk_length: torch.Tensor | None = None,
    attn_sink: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns:
    - o: [s_q, h_q, dv]
    - o_fp32: [s_q, h_q, dv]
    - max_logits: [s_q, h_q]
    - lse: [s_q, h_q]
    """
    s_q, h_q, d_qk = q.shape
    s_kv, _, _ = kv.shape
    _, _, topk = indices.shape

    indices = indices.clone().squeeze(1)
    if topk_length is not None:
        mask = torch.arange(topk, device=topk_length.device).unsqueeze(0).broadcast_to(
            s_q, topk
        ) >= topk_length.unsqueeze(1)  # [s_q, topk]
        indices[mask] = -1
    invalid_mask = (indices < 0) | (indices >= s_kv)  # [s_q, topk]
    indices[invalid_mask] = 0

    q = q.float()
    gathered_kv = (
        kv.index_select(dim=0, index=indices.flatten()).reshape(s_q, topk, d_qk).float()
    )  # [s_q, topk, d_qk]
    P = q @ gathered_kv.transpose(1, 2)  # [s_q, h_q, topk]
    P *= sm_scale
    P[invalid_mask.unsqueeze(1).broadcast_to(P.shape)] = float("-inf")

    orig_lse = torch.logsumexp(P, dim=-1)  # [s_q, h_q]
    max_logits = P.max(dim=-1).values  # [s_q, h_q]

    lse_for_o = _merge_two_lse(orig_lse, attn_sink, s_q, h_q)
    if not torch.is_inference_mode_enabled():
        lse_for_o = lse_for_o.clone()
    lse_for_o[lse_for_o == float("-inf")] = float(
        "+inf"
    )  # So that corresponding O will be 0
    s_for_o = torch.exp(P - lse_for_o.unsqueeze(-1))
    out = s_for_o @ gathered_kv[..., :d_v]  # [s_q, h_q, dv]

    lonely_q_mask = orig_lse == float("-inf")  # [s_q, h_q]
    orig_lse[lonely_q_mask] = float("+inf")
    return (out.to(torch.bfloat16), out, max_logits, orig_lse)


@pytest.mark.parametrize("device_str", ["cuda", "xpu"])
@pytest.mark.skipif(
    not torch.cuda.is_available() and not torch.xpu.is_available(),
    reason="CUDA or XPU is required",
)
def test_bf16_triton_sparse_mla(device_str):
    if device_str == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    if device_str == "xpu" and not torch.xpu.is_available():
        pytest.skip("XPU is not available")

    device = torch.device(device_str)
    s_q = 1
    s_kv = 256
    h_q = 64  # kernel expects multiple of 64
    h_kv = 1
    d_qk = 576
    d_v = 512
    topk = 128
    dtype = torch.bfloat16

    torch.random.manual_seed(1234)

    q = torch.randn((s_q, h_q, d_qk), dtype=dtype, device=device)
    kv = torch.randn((s_kv, h_kv, d_qk), dtype=dtype, device=device)
    indices = torch.full((s_q, h_kv, topk), -1, dtype=torch.int32, device=device)
    for t in range(s_q):
        for h in range(h_kv):
            i_i = torch.randperm(max(1, t))[:topk]
            indices[t, h, : len(i_i)] = i_i

    sm_scale = d_qk**-0.5

    out, max_logits, lse = triton_bf16_mla_sparse_interface(
        q, kv, indices, sm_scale, d_v
    )
    assert out.shape == (s_q, h_q, d_v)
    assert max_logits.shape == (s_q, h_q)
    assert lse.shape == (s_q, h_q)

    ref_out, ref_out_fp32, ref_max_logits, ref_lse = reference_mla_sparse_prefill(
        q, kv, indices, sm_scale, d_v
    )
    assert torch.allclose(out, ref_out, atol=1e-2, rtol=1e-2)
    assert torch.allclose(max_logits, ref_max_logits, atol=1e-3, rtol=1e-3)
    assert torch.allclose(lse, ref_lse, atol=1e-3, rtol=1e-3)
