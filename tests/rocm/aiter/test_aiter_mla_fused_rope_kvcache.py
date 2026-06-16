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


def _setup_fused_inputs(device: str = "cuda"):
    """Build all inputs needed by fused RoPE + KV-cache write tests.

    Note: production MLARoPEKVCacheCatFusionPass passes a single
    cos_sin_cache tensor to concat_and_cache_mla_rope_fused. This
    helper exercises fused_rope_and_mla_kv_cache_write directly,
    which takes pre-split cos/sin tensors.
    """
    q = torch.randn(
        NUM_TOKENS, NUM_Q_HEADS,
        QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM,
        device=device,
    )
    positions = torch.randint(0, 8192, (NUM_TOKENS,), device=device)
    cos_sin_cache = torch.randn(8192, QK_ROPE_HEAD_DIM, device=device)
    slot_mapping = torch.arange(NUM_TOKENS, dtype=torch.long, device=device)
    kv_cache = torch.zeros(
        16, 16, KV_LORA_RANK + QK_ROPE_HEAD_DIM, device=device
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
    """Skip test if the AITER fused kernel is not available."""
    from vllm._aiter_ops import rocm_aiter_ops
    if not rocm_aiter_ops.has_fused_rope_mla_kv_cache():
        pytest.skip(
            "fused_qk_rope_concat_and_cache_mla kernel absent "
            "on this build"
        )


def test_kv_cache_written_after_fused_call():
    """KV-cache slots in slot_mapping must be non-zero after fused call.

    Verifies the kernel actually wrote kv_c + k_pe to the cache.
    If this fails, decode attention reads stale data — wrong outputs.
    """
    from vllm._aiter_ops import rocm_aiter_ops
    _skip_if_no_kernel()

    device = "cuda"
    (q_nope, q_pe, kv_c, k_pe, kv_cache, q_out,
     slot_mapping, k_scale, q_scale, positions,
     cos_cache, sin_cache) = _setup_fused_inputs(device)

    kv_cache_before = kv_cache.clone()

    rocm_aiter_ops.fused_rope_and_mla_kv_cache_write(
        q_nope=q_nope, q_pe=q_pe, kv_c=kv_c, k_pe=k_pe,
        kv_cache=kv_cache, q_out=q_out, slot_mapping=slot_mapping,
        k_scale=k_scale, q_scale=q_scale, positions=positions,
        cos_cache=cos_cache, sin_cache=sin_cache, is_neox=True,
    )

    assert not torch.equal(kv_cache, kv_cache_before), (
        "KV-cache was not updated by fused_rope_and_mla_kv_cache_write"
    )


def test_q_out_updated_after_fused_call():
    """q_out must be non-zero after fused call — RoPE was applied.

    Verifies the kernel applied RoPE to q_pe and wrote it into q_out.
    If this fails, the query vector is un-rotated and attention scores
    are incorrect.
    """
    from vllm._aiter_ops import rocm_aiter_ops
    _skip_if_no_kernel()

    device = "cuda"
    (q_nope, q_pe, kv_c, k_pe, kv_cache, q_out,
     slot_mapping, k_scale, q_scale, positions,
     cos_cache, sin_cache) = _setup_fused_inputs(device)

    assert not q_out.any(), "q_out should be all-zero before the call"

    rocm_aiter_ops.fused_rope_and_mla_kv_cache_write(
        q_nope=q_nope, q_pe=q_pe, kv_c=kv_c, k_pe=k_pe,
        kv_cache=kv_cache, q_out=q_out, slot_mapping=slot_mapping,
        k_scale=k_scale, q_scale=q_scale, positions=positions,
        cos_cache=cos_cache, sin_cache=sin_cache, is_neox=True,
    )

    assert q_out.any(), (
        "q_out was not updated — RoPE was not applied to q_pe"
    )


def test_slot_mapping_respected():
    """Only slots in slot_mapping should be written — others stay zero.

    Verifies the kernel writes to exactly the requested cache slots.
    If slot_mapping is ignored, other requests' KV-cache entries
    are corrupted or writes are silently dropped.
    """
    from vllm._aiter_ops import rocm_aiter_ops
    _skip_if_no_kernel()

    device = "cuda"
    (q_nope, q_pe, kv_c, k_pe, kv_cache, q_out,
     _slot_mapping, k_scale, q_scale, positions,
     cos_cache, sin_cache) = _setup_fused_inputs(device)

    # Write only to slots 0 and 1
    slot_mapping = torch.tensor(
        [0, 1, 0, 1], dtype=torch.long, device=device
    )

    rocm_aiter_ops.fused_rope_and_mla_kv_cache_write(
        q_nope=q_nope, q_pe=q_pe, kv_c=kv_c, k_pe=k_pe,
        kv_cache=kv_cache, q_out=q_out, slot_mapping=slot_mapping,
        k_scale=k_scale, q_scale=q_scale, positions=positions,
        cos_cache=cos_cache, sin_cache=sin_cache, is_neox=True,
    )

    assert kv_cache[0].any(), "Slot 0 should be written"
    assert kv_cache[1].any(), "Slot 1 should be written"
    assert not kv_cache[2:].any(), (
        "Slots not in slot_mapping must remain zero"
    )
