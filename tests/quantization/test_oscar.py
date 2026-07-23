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

import types

import pytest
import torch

from vllm.model_executor.layers.quantization.oscar.config import OscarConfig
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda(), reason="OSCAR kernels require CUDA/Triton."
)


def _make_impl(Hq, Hk, head_dim, monkeypatch, sink=0, recent=0, staging=0):
    """Build an OscarAttentionImpl outside an engine context."""
    from vllm.v1.attention.backends.fa_utils import get_flash_attn_version
    from vllm.v1.attention.backends.oscar_attn import OscarAttentionImpl

    monkeypatch.setenv("VLLM_OSCAR_SINK_TOKENS", str(sink))
    monkeypatch.setenv("VLLM_OSCAR_RECENT_TOKENS", str(recent))
    monkeypatch.setenv("VLLM_OSCAR_STAGING_TOKENS", str(staging))
    impl = object.__new__(OscarAttentionImpl)
    impl.num_heads, impl.head_size, impl.scale = Hq, head_dim, head_dim**-0.5
    impl.num_kv_heads = Hk
    impl.kv_cache_dtype = "oscar_int2"
    impl.cfg = OscarConfig.from_cache_dtype("oscar_int2", head_dim)
    impl.fa_version = get_flash_attn_version(head_size=head_dim)
    impl.max_num_kv_splits = 8
    impl.window_enabled = (
        impl.cfg.sink_tokens > 0 or impl.cfg.recent_tokens > 0
    ) and impl.cfg.staging_tokens > 0
    return impl


def _identity_layer():
    return types.SimpleNamespace(layer_name="model.layers.0.self_attn.attn")


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


def _rand_rotation_ckpt(tmp_path, name, head_dim, num_layers=1, seed=0):
    g = torch.Generator().manual_seed(seed)
    layers = {}
    for lid in range(num_layers):
        q, _ = torch.linalg.qr(torch.randn(head_dim, head_dim, generator=g))
        layers[lid] = {"layer_id": lid, "rotation": q.float()}
    path = tmp_path / f"{name}.pt"
    torch.save({"format_version": 1, "layers": layers}, path)
    return str(path)


def _store_all(impl, layer, key, value, cache, slots):
    from vllm.v1.attention.ops.triton_oscar_store import oscar_store

    k_rot = impl._rotate_clip(key, layer._oscar_Rk, impl.cfg.k_clip_ratio)
    v_rot = impl._rotate_clip(value, layer._oscar_Rv, impl.cfg.v_clip_ratio)
    oscar_store(
        k_rot,
        v_rot,
        cache,
        slots,
        key_levels=impl.cfg.key_levels,
        value_levels=impl.cfg.value_levels,
        key_packed_size=impl.cfg.key_packed_size,
        data_bytes=impl.cfg.key_data_bytes,
    )


@pytest.mark.parametrize("window", [False, True])
def test_prefill_fastpath_mixed_batch(monkeypatch, tmp_path, window):
    """A fresh prefill batched with a shorter continuation must not silently
    drop the continuation's cached context (regression test for the
    max_query_len == max_seq_len fast-path)."""
    from vllm.v1.attention.backends.oscar_attn import OscarMetadata
    from vllm.v1.attention.ops.triton_oscar_decode import oscar_full_dequant_kv

    torch.manual_seed(0)
    device = "cuda"
    H, D = 4, 64
    Q_A, S_A = 32, 32
    Q_B, S_B, CACHED_B = 16, 32, 16
    monkeypatch.setenv(
        "VLLM_OSCAR_K_ROTATION_PATH", _rand_rotation_ckpt(tmp_path, "k", D, seed=1)
    )
    monkeypatch.setenv(
        "VLLM_OSCAR_V_ROTATION_PATH", _rand_rotation_ckpt(tmp_path, "v", D, seed=2)
    )
    impl = _make_impl(
        H,
        H,
        D,
        monkeypatch,
        sink=16 if window else 0,
        recent=32 if window else 0,
        staging=4096 if window else 0,
    )
    layer = _identity_layer()
    impl._ensure_rotations(layer, torch.device(device))

    cfg = impl.cfg
    block_size, num_blocks = 16, 4
    cache = torch.zeros(
        num_blocks,
        block_size,
        H,
        cfg.slot_size_aligned,
        dtype=torch.uint8,
        device=device,
    )
    kb_cached = torch.randn(CACHED_B, H, D, device=device, dtype=torch.bfloat16)
    vb_cached = torch.randn(CACHED_B, H, D, device=device, dtype=torch.bfloat16)
    slots_b = torch.arange(32, 32 + CACHED_B, device=device, dtype=torch.int32)
    _store_all(impl, layer, kb_cached.float(), vb_cached.float(), cache, slots_b)

    N = Q_A + Q_B
    q = torch.randn(N, H, D, device=device, dtype=torch.bfloat16)
    k = torch.randn(N, H, D, device=device, dtype=torch.bfloat16)
    v = torch.randn(N, H, D, device=device, dtype=torch.bfloat16)

    meta = OscarMetadata(
        seq_lens=torch.tensor([S_A, S_B], device=device, dtype=torch.int32),
        slot_mapping=torch.arange(N, device=device, dtype=torch.int64),
        block_table=torch.tensor([[0, 1], [2, 3]], device=device, dtype=torch.int32),
        query_start_loc=torch.tensor([0, Q_A, N], device=device, dtype=torch.int32),
        num_actual_tokens=N,
        max_query_len=32,
        max_seq_len=32,
        is_prefill=True,
        has_context=True,  # request B attends 16 cached tokens
        query_start_loc_cpu=torch.tensor([0, Q_A, N], dtype=torch.int32),
        seq_lens_cpu=torch.tensor([S_A, S_B], dtype=torch.int32),
    )
    if window:
        impl._ensure_staging(layer, cache)
        # stage request B's cached prefix as if written when it was prefilled
        stage_meta = OscarMetadata(
            seq_lens=torch.tensor([CACHED_B], device=device, dtype=torch.int32),
            slot_mapping=slots_b.to(torch.int64),
            block_table=meta.block_table[1:],
            query_start_loc=torch.tensor(
                [0, CACHED_B], device=device, dtype=torch.int32
            ),
            num_actual_tokens=CACHED_B,
        )
        impl._staging_write(layer, kb_cached, vb_cached, stage_meta)

    out = impl._prefill_attention(q, k, v, cache, meta, layer)
    out_b = out[Q_A:]

    # reference for B: cached prefix (exact if staged, else dequant+unrotate)
    if window:
        k_pre, v_pre = kb_cached.float(), vb_cached.float()
    else:
        k_deq, v_deq = oscar_full_dequant_kv(
            cache,
            meta.block_table[1:2],
            CACHED_B,
            H,
            D,
            cfg.key_levels,
            cfg.value_levels,
            cfg.key_data_bytes,
            cfg.key_packed_size,
            cfg.value_data_bytes,
        )
        k_pre = k_deq.float() @ layer._oscar_RkT
        v_pre = v_deq.float() @ layer._oscar_RvT
    k_full = torch.cat([k_pre, k[Q_A:].float()], 0)
    v_full = torch.cat([v_pre, v[Q_A:].float()], 0)
    qb = q[Q_A:].float()
    mask = (
        torch.arange(S_B, device=device)[None, :]
        <= torch.arange(Q_B, device=device)[:, None] + CACHED_B
    )
    ref = torch.nn.functional.scaled_dot_product_attention(
        qb.transpose(0, 1),
        k_full.transpose(0, 1),
        v_full.transpose(0, 1),
        attn_mask=mask,
        scale=impl.scale,
    ).transpose(0, 1)
    rel = (out_b.float() - ref).norm() / ref.norm()
    assert rel < 2e-2, f"continuation attended wrong context: relL2={rel:.4f}"


def test_decode_window_matches_reference(monkeypatch, tmp_path):
    """Windowed decode (BF16 sink + recent, INT2 middle, LSE merge) must match
    a straightforward reference over the mixed-precision K/V."""
    from vllm.v1.attention.backends.oscar_attn import OscarMetadata
    from vllm.v1.attention.ops.triton_oscar_decode import oscar_full_dequant_kv

    torch.manual_seed(1)
    device = "cuda"
    Hq, Hk, D = 8, 2, 128
    B, S = 2, 100
    SINK, RECENT = 16, 32
    monkeypatch.setenv(
        "VLLM_OSCAR_K_ROTATION_PATH", _rand_rotation_ckpt(tmp_path, "k", D, seed=3)
    )
    monkeypatch.setenv(
        "VLLM_OSCAR_V_ROTATION_PATH", _rand_rotation_ckpt(tmp_path, "v", D, seed=4)
    )
    monkeypatch.setenv("VLLM_OSCAR_K_CLIP_RATIO", "0.96")
    monkeypatch.setenv("VLLM_OSCAR_V_CLIP_RATIO", "0.92")
    impl = _make_impl(Hq, Hk, D, monkeypatch, sink=SINK, recent=RECENT, staging=4096)
    layer = _identity_layer()
    impl._ensure_rotations(layer, torch.device(device))

    cfg = impl.cfg
    block_size, blocks_per_req = 16, 8
    cache = torch.zeros(
        B * blocks_per_req,
        block_size,
        Hk,
        cfg.slot_size_aligned,
        dtype=torch.uint8,
        device=device,
    )
    block_table = torch.arange(
        B * blocks_per_req, device=device, dtype=torch.int32
    ).view(B, blocks_per_req)
    impl._ensure_staging(layer, cache)

    key = torch.randn(B, S, Hk, D, device=device, dtype=torch.bfloat16)
    val = torch.randn(B, S, Hk, D, device=device, dtype=torch.bfloat16)
    slots = []
    for b in range(B):
        base = b * blocks_per_req * block_size
        slots.append(torch.arange(base, base + S, device=device, dtype=torch.int32))
    # INT2 copy for every token
    for b in range(B):
        _store_all(impl, layer, key[b].float(), val[b].float(), cache, slots[b])
    # staging writes: one prefill step for [0, S-1), then a decode step for S-1
    pre_meta = OscarMetadata(
        seq_lens=torch.full((B,), S - 1, device=device, dtype=torch.int32),
        slot_mapping=torch.cat([s[: S - 1] for s in slots]).to(torch.int64),
        block_table=block_table,
        query_start_loc=torch.arange(0, (S - 1) * (B + 1), S - 1, device=device)[
            : B + 1
        ].to(torch.int32),
        num_actual_tokens=B * (S - 1),
    )
    impl._staging_write(
        layer,
        key[:, : S - 1].reshape(-1, Hk, D),
        val[:, : S - 1].reshape(-1, Hk, D),
        pre_meta,
    )
    dec_meta = OscarMetadata(
        seq_lens=torch.full((B,), S, device=device, dtype=torch.int32),
        slot_mapping=torch.stack([s[S - 1] for s in slots]).to(torch.int64),
        block_table=block_table,
        query_start_loc=torch.arange(B + 1, device=device, dtype=torch.int32),
        num_actual_tokens=B,
    )
    impl._staging_write(layer, key[:, S - 1], val[:, S - 1], dec_meta)

    q = torch.randn(B, Hq, D, device=device, dtype=torch.bfloat16)
    out = impl._decode_attention(q, cache, dec_meta, layer)

    # reference: sink/recent from exact bf16 values, middle from the INT2 cache
    g = Hq // Hk
    cut = S - RECENT  # staging coverage is contiguous here
    ref = torch.empty(B, Hq, D, device=device)
    for b in range(B):
        k_deq, v_deq = oscar_full_dequant_kv(
            cache,
            block_table[b : b + 1],
            S,
            Hk,
            D,
            cfg.key_levels,
            cfg.value_levels,
            cfg.key_data_bytes,
            cfg.key_packed_size,
            cfg.value_data_bytes,
        )
        k_mix = k_deq.float() @ layer._oscar_RkT
        v_mix = v_deq.float() @ layer._oscar_RvT
        k_mix[:SINK] = key[b, :SINK].float()
        v_mix[:SINK] = val[b, :SINK].float()
        k_mix[cut:] = key[b, cut:].float()
        v_mix[cut:] = val[b, cut:].float()
        for h in range(Hq):
            kh, vh = k_mix[:, h // g], v_mix[:, h // g]
            p = torch.softmax((q[b, h].float() @ kh.T) * impl.scale, -1)
            ref[b, h] = p @ vh
    rel = (out.float() - ref).norm() / ref.norm()
    assert rel < 2e-2, f"windowed decode mismatch: relL2={rel:.4f}"


def test_staging_eviction_falls_back_to_int2(monkeypatch, tmp_path):
    """With every staging tag invalidated, windowed decode must equal the
    plain all-INT2 decode exactly (graceful degradation)."""
    torch.manual_seed(2)
    device = "cuda"
    Hq, Hk, D = 4, 4, 64
    B, S = 2, 50
    monkeypatch.setenv(
        "VLLM_OSCAR_K_ROTATION_PATH", _rand_rotation_ckpt(tmp_path, "k", D, seed=5)
    )
    monkeypatch.setenv(
        "VLLM_OSCAR_V_ROTATION_PATH", _rand_rotation_ckpt(tmp_path, "v", D, seed=6)
    )
    from vllm.v1.attention.backends.oscar_attn import OscarMetadata

    impl = _make_impl(Hq, Hk, D, monkeypatch, sink=16, recent=16, staging=64)
    layer = _identity_layer()
    impl._ensure_rotations(layer, torch.device(device))
    cfg = impl.cfg
    block_size, blocks_per_req = 16, 4
    cache = torch.zeros(
        B * blocks_per_req,
        block_size,
        Hk,
        cfg.slot_size_aligned,
        dtype=torch.uint8,
        device=device,
    )
    block_table = torch.arange(
        B * blocks_per_req, device=device, dtype=torch.int32
    ).view(B, blocks_per_req)
    impl._ensure_staging(layer, cache)
    key = torch.randn(B, S, Hk, D, device=device)
    val = torch.randn(B, S, Hk, D, device=device)
    for b in range(B):
        base = b * blocks_per_req * block_size
        s = torch.arange(base, base + S, device=device, dtype=torch.int32)
        _store_all(impl, layer, key[b], val[b], cache, s)
    dec_meta = OscarMetadata(
        seq_lens=torch.full((B,), S, device=device, dtype=torch.int32),
        slot_mapping=torch.tensor(
            [S - 1, blocks_per_req * block_size + S - 1], device=device
        ),
        block_table=block_table,
        query_start_loc=torch.arange(B + 1, device=device, dtype=torch.int32),
        num_actual_tokens=B,
    )
    # simulate total eviction: no staging tag is valid
    layer._oscar_slot_owner.fill_(-1)
    q = torch.randn(B, Hq, D, device=device, dtype=torch.bfloat16)
    out_windowed = impl._decode_attention(q, cache, dec_meta, layer)
    impl.window_enabled = False
    out_plain = impl._decode_attention(q, cache, dec_meta, layer)
    torch.testing.assert_close(
        out_windowed.float(), out_plain.float(), atol=1e-4, rtol=1e-4
    )
