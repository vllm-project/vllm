# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The fused q/k norm + RoPE + cache-write kernel must reproduce the unfused
vLLM ops (rms_norm, rotary_embedding, reshape_and_cache_flash) it replaces."""

import pytest
import torch

from vllm import _custom_ops as ops
from vllm.models.flint.ops.qk_norm_rope import qk_norm_rope_cache
from vllm.platforms import current_platform

HQ, HKV, D = 32, 4, 128
BLOCK_SIZE, NUM_BLOCKS = 16, 64


@pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA only")
@pytest.mark.parametrize("num_tokens", [1, 7, 256, 1000])
@pytest.mark.parametrize("apply_rope", [True, False])
@torch.inference_mode()
def test_matches_unfused_ops(num_tokens: int, apply_rope: bool):
    torch.manual_seed(0)
    dev = "cuda"
    qkv = torch.randn(num_tokens, (HQ + 2 * HKV) * D, device=dev).bfloat16() * 3
    positions = torch.randint(0, 100_000, (num_tokens,), device=dev)
    q_w = torch.empty(D, device=dev).normal_(1.0, 0.2).bfloat16()
    k_w = torch.empty(D, device=dev).normal_(1.0, 0.2).bfloat16()
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, D, 2, device=dev).float() / D))
    freqs = torch.outer(torch.arange(262144, device=dev).float(), inv_freq)
    cos_sin = torch.cat([freqs.cos(), freqs.sin()], dim=-1)
    # Padded tokens (beyond the slot mapping) and slot -1 must not be written.
    slots = torch.randperm(NUM_BLOCKS * BLOCK_SIZE, device=dev)[:num_tokens]
    slot_mapping = slots[: max(1, num_tokens - 1)].clone()
    slot_mapping[0] = -1

    q, k, v = qkv.split([HQ * D, HKV * D, HKV * D], dim=-1)
    ref_q = torch.empty(num_tokens, HQ, D, device=dev, dtype=torch.bfloat16)
    ref_k = torch.empty(num_tokens, HKV, D, device=dev, dtype=torch.bfloat16)
    ops.rms_norm(ref_q, q.reshape(-1, HQ, D), q_w, 1e-6)
    ops.rms_norm(ref_k, k.reshape(-1, HKV, D), k_w, 1e-6)
    ref_q = ref_q.reshape(q.shape)
    ref_k = ref_k.reshape(k.shape)
    if apply_rope:
        # fp32 cos/sin with bf16 q/k, as in the reference implementation.
        ops.rotary_embedding(positions, ref_q, ref_k, D, cos_sin, True)
    ref_cache = torch.zeros(NUM_BLOCKS, 2, BLOCK_SIZE, HKV, D, device=dev).bfloat16()
    torch.ops._C_cache_ops.reshape_and_cache_flash(
        ref_k.reshape(-1, HKV, D),
        v.reshape(-1, HKV, D),
        ref_cache[:, 0],
        ref_cache[:, 1],
        slot_mapping,
        "auto",
        torch.ones((), device=dev),
        torch.ones((), device=dev),
    )

    cache = torch.zeros_like(ref_cache)
    out_qkv = qkv.clone()
    out_q = qk_norm_rope_cache(
        out_qkv,
        positions,
        q_w,
        k_w,
        1e-6,
        cos_sin if apply_rope else None,
        cache[:, 0],
        cache[:, 1],
        slot_mapping,
        HQ,
        HKV,
        D,
    )
    out_k = out_qkv[:, HQ * D : (HQ + HKV) * D]
    # Same ops and roundings; only the RMS reduction order differs.
    torch.testing.assert_close(out_q, ref_q, atol=4e-3, rtol=1.6e-2)
    torch.testing.assert_close(out_k, ref_k, atol=4e-3, rtol=1.6e-2)
    assert (out_q != ref_q).float().mean() < 0.01
    assert (out_k != ref_k).float().mean() < 0.01
    torch.testing.assert_close(cache, ref_cache, atol=4e-3, rtol=1.6e-2)
    assert torch.equal(out_qkv[:, (HQ + HKV) * D :], v)
