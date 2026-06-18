# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import pytest
import torch

from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm(), reason="ROCm-specific tests"
)

# DeepSeek-V3/R1 MLA dimensions
KV_LORA_RANK = 512
QK_NOPE_HEAD_DIM = 512
QK_ROPE_HEAD_DIM = 64
NUM_TOKENS = 4
NUM_Q_HEADS = 128
BLOCK_SIZE = 16


def _setup_fused_inputs(device: str = "cuda"):
    torch.manual_seed(42)
    q = torch.randn(
        NUM_TOKENS, NUM_Q_HEADS, QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM,
        device=device,
    )
    positions = torch.randint(0, 8192, (NUM_TOKENS,), device=device)
    cos_sin_cache = torch.randn(8192, QK_ROPE_HEAD_DIM, device=device)
    slot_mapping = torch.arange(NUM_TOKENS, dtype=torch.long, device=device)
    kv_cache = torch.zeros(
        16, BLOCK_SIZE, KV_LORA_RANK + QK_ROPE_HEAD_DIM, device=device
    )
    q_nope = q[..., :QK_NOPE_HEAD_DIM]
    q_pe = q[..., QK_NOPE_HEAD_DIM:]
    kv_c = torch.randn(NUM_TOKENS, KV_LORA_RANK, device=device)
    k_pe = torch.randn(NUM_TOKENS, QK_ROPE_HEAD_DIM, device=device)
    q_out = torch.zeros_like(q)
    k_scale = torch.tensor([1.0], device=device)
    q_scale = torch.tensor([1.0], device=device)
    half = QK_ROPE_HEAD_DIM // 2
    cos_cache = cos_sin_cache[:, :half]
    sin_cache = cos_sin_cache[:, half:]
    return (
        q_nope, q_pe, kv_c, k_pe, kv_cache, q_out,
        slot_mapping, k_scale, q_scale, positions,
        cos_cache, sin_cache,
    )


def _skip_if_no_kernel():
    from vllm._aiter_ops import rocm_aiter_ops
    if not rocm_aiter_ops.has_fused_rope_mla_kv_cache():
        pytest.skip("fused_qk_rope_concat_and_cache_mla kernel absent on this build")


def test_rope_applied_to_pe():
    """q_out[..., QK_NOPE_HEAD_DIM:] must differ from q_pe — RoPE was applied."""
    from vllm._aiter_ops import rocm_aiter_ops
    _skip_if_no_kernel()

    device = "cuda"
    (q_nope, q_pe, kv_c, k_pe, kv_cache, q_out,
     slot_mapping, k_scale, q_scale, positions,
     cos_cache, sin_cache) = _setup_fused_inputs(device)

    rocm_aiter_ops.fused_rope_and_mla_kv_cache_write(
        q_nope=q_nope, q_pe=q_pe, kv_c=kv_c, k_pe=k_pe,
        kv_cache=kv_cache, q_out=q_out, slot_mapping=slot_mapping,
        k_scale=k_scale, q_scale=q_scale, positions=positions,
        cos_cache=cos_cache, sin_cache=sin_cache, is_neox=True,
    )

    assert not torch.allclose(q_out[..., QK_NOPE_HEAD_DIM:], q_pe), (
        "pe part of q_out is identical to input q_pe — RoPE was not applied"
    )


def test_slot_mapping_respected():
    """Only slots in slot_mapping should be written — others stay zero."""
    from vllm._aiter_ops import rocm_aiter_ops
    _skip_if_no_kernel()

    device = "cuda"
    (q_nope, q_pe, kv_c, k_pe, kv_cache, q_out,
     _slot_mapping, k_scale, q_scale, positions,
     cos_cache, sin_cache) = _setup_fused_inputs(device)

    # Write only to slots 0 and 1; all other cache entries must stay zero.
    slot_mapping = torch.tensor([0, 1, 0, 1], dtype=torch.long, device=device)

    rocm_aiter_ops.fused_rope_and_mla_kv_cache_write(
        q_nope=q_nope, q_pe=q_pe, kv_c=kv_c, k_pe=k_pe,
        kv_cache=kv_cache, q_out=q_out, slot_mapping=slot_mapping,
        k_scale=k_scale, q_scale=q_scale, positions=positions,
        cos_cache=cos_cache, sin_cache=sin_cache, is_neox=True,
    )

    assert kv_cache[0, 0].any(), "Slot 0 should be written"
    assert kv_cache[0, 1].any(), "Slot 1 should be written"
    assert not kv_cache[0, 2:].any(), "Slots 2+ must remain zero"


