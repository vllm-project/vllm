# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Correctness tests for OSCAR INT2 KV-cache quantization.

Validates:
  * config slot geometry,
  * the store + dequant round-trip (INT2 pack/unpack consistency vs a pure
    PyTorch reference quantizer),
  * the fused INT2 decode-attention kernel vs a reference attention computed
    on the dequantized cache.

Run: ``pytest tests/quantization/test_oscar.py``.
"""

import pytest
import torch

from vllm.model_executor.layers.quantization.oscar.config import OscarConfig
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda(), reason="OSCAR kernels require CUDA/Triton."
)


def _ref_int2_quant_dequant(x: torch.Tensor, levels: int) -> torch.Tensor:
    """Per-vector asymmetric uniform quant-dequant, matching the kernel."""
    vmin = x.amin(dim=-1, keepdim=True)
    vmax = x.amax(dim=-1, keepdim=True)
    scale = ((vmax - vmin) / (levels - 1)).clamp_min(1e-8)
    # fp16 scale/zero storage, as in the kernel.
    scale = scale.half().float()
    vmin = vmin.half().float()
    q = torch.clamp(((x - vmin) / scale + 0.5).to(torch.int32), 0, levels - 1)
    return q.float() * scale + vmin


def test_config_geometry():
    c = OscarConfig.from_cache_dtype("oscar_int2", 128)
    assert c.key_levels == 4 and c.value_levels == 4
    assert c.key_data_bytes == 32 and c.value_data_bytes == 32
    assert c.key_packed_size == 36 and c.value_packed_size == 36
    assert c.slot_size_aligned == 72


def _make_cache(num_blocks, block_size, num_kv_heads, slot_size, device):
    return torch.zeros(
        num_blocks,
        block_size,
        num_kv_heads,
        slot_size,
        dtype=torch.uint8,
        device=device,
    )


@pytest.mark.parametrize("head_dim", [64, 128])
def test_store_dequant_roundtrip(head_dim):
    from vllm.v1.attention.ops.triton_oscar_decode import oscar_full_dequant_kv
    from vllm.v1.attention.ops.triton_oscar_store import oscar_store

    torch.manual_seed(0)
    device = "cuda"
    cfg = OscarConfig.from_cache_dtype("oscar_int2", head_dim)
    N, Hk = 40, 4
    block_size, num_blocks = 16, 8

    key = torch.randn(N, Hk, head_dim, device=device)
    value = torch.randn(N, Hk, head_dim, device=device)
    cache = _make_cache(num_blocks, block_size, Hk, cfg.slot_size_aligned, device)
    # Contiguous slots 0..N-1.
    slot_mapping = torch.arange(N, device=device, dtype=torch.int32)

    oscar_store(
        key,
        value,
        cache,
        slot_mapping,
        key_levels=cfg.key_levels,
        value_levels=cfg.value_levels,
        key_packed_size=cfg.key_packed_size,
        data_bytes=cfg.key_data_bytes,
    )

    # Single contiguous block table covering all tokens.
    block_table = torch.arange(num_blocks, device=device, dtype=torch.int32).view(1, -1)
    k_deq, v_deq = oscar_full_dequant_kv(
        cache,
        block_table,
        N,
        Hk,
        head_dim,
        cfg.key_levels,
        cfg.value_levels,
        cfg.key_data_bytes,
        cfg.key_packed_size,
        cfg.value_data_bytes,
    )

    k_ref = _ref_int2_quant_dequant(key, cfg.key_levels)
    v_ref = _ref_int2_quant_dequant(value, cfg.value_levels)

    # fp16 storage of the dequantized result -> compare in fp16 tolerance.
    torch.testing.assert_close(k_deq.float(), k_ref, atol=2e-3, rtol=2e-3)
    torch.testing.assert_close(v_deq.float(), v_ref, atol=2e-3, rtol=2e-3)


@pytest.mark.parametrize("head_dim,Hq,Hk", [(128, 8, 2), (64, 4, 4)])
def test_decode_attention_matches_reference(head_dim, Hq, Hk):
    from vllm.v1.attention.ops.triton_oscar_decode import (
        oscar_decode_attention,
        oscar_full_dequant_kv,
    )
    from vllm.v1.attention.ops.triton_oscar_store import oscar_store

    torch.manual_seed(1)
    device = "cuda"
    cfg = OscarConfig.from_cache_dtype("oscar_int2", head_dim)
    B = 2
    seq_len = 48
    block_size, num_blocks = 16, B * 8
    scale = head_dim**-0.5

    # Per-batch contiguous storage.
    key = torch.randn(B, seq_len, Hk, head_dim, device=device)
    value = torch.randn(B, seq_len, Hk, head_dim, device=device)
    cache = _make_cache(num_blocks, block_size, Hk, cfg.slot_size_aligned, device)

    blocks_per_req = num_blocks // B
    block_table = torch.arange(num_blocks, device=device, dtype=torch.int32).view(
        B, blocks_per_req
    )

    # Store all tokens for both requests.
    for b in range(B):
        base = b * blocks_per_req * block_size
        slot = torch.arange(base, base + seq_len, device=device, dtype=torch.int32)
        oscar_store(
            key[b],
            value[b],
            cache,
            slot,
            key_levels=cfg.key_levels,
            value_levels=cfg.value_levels,
            key_packed_size=cfg.key_packed_size,
            data_bytes=cfg.key_data_bytes,
        )

    q = torch.randn(B, Hq, head_dim, device=device)
    seq_lens = torch.full((B,), seq_len, device=device, dtype=torch.int32)

    out = oscar_decode_attention(
        q,
        cache,
        block_table,
        seq_lens,
        scale,
        key_levels=cfg.key_levels,
        value_levels=cfg.value_levels,
        key_data_bytes=cfg.key_data_bytes,
        key_packed_size=cfg.key_packed_size,
        value_data_bytes=cfg.value_data_bytes,
        max_num_kv_splits=4,
    )

    # Reference: attention over the *dequantized* cache (what the kernel reads).
    ref = torch.empty(B, Hq, head_dim, device=device)
    g = Hq // Hk
    for b in range(B):
        k_deq, v_deq = oscar_full_dequant_kv(
            cache,
            block_table[b : b + 1],
            seq_len,
            Hk,
            head_dim,
            cfg.key_levels,
            cfg.value_levels,
            cfg.key_data_bytes,
            cfg.key_packed_size,
            cfg.value_data_bytes,
        )
        for h in range(Hq):
            kh = k_deq[:, h // g, :].float()
            vh = v_deq[:, h // g, :].float()
            s = (q[b, h].float() @ kh.t()) * scale
            p = torch.softmax(s, dim=-1)
            ref[b, h] = p @ vh

    torch.testing.assert_close(out.float(), ref, atol=5e-3, rtol=5e-3)
