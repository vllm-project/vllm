# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bit-exact kernel equivalence for MLA decode/write kernels on the
cross-layer (block-major) KV cache layout.

The cross-layer layout carves each layer's per-block page out of a single
unified slot, so the per-layer view has an inflated ``stride(0)`` (the full
unified slot) and a non-zero storage offset. These tests confirm the MLA
kernels behind the backends that opt in to the layout (FlashMLA dense,
FlashInfer MLA dense, FlashMLA fp8 sparse, plus the ``concat_and_cache_mla``
write) honor that strided view bit-identically to a contiguous per-layer
cache, and that writes do not bleed into neighbouring layers' segments.
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="MLA cache kernels require CUDA"
)


def test_concat_and_cache_mla_into_unified_slot_view():
    """concat_and_cache_mla must write correctly into a per-layer view whose
    block stride is the full unified slot (block-major), with zero bleed into
    the other layers' segments of the same slot."""
    from vllm import _custom_ops as ops

    torch.manual_seed(0)
    dev = "cuda"
    kv_lora_rank = 512
    pe = 64
    entry = kv_lora_rank + pe
    page = 64
    num_blocks = 32
    ntok = 200

    kv_c = torch.randn(ntok, kv_lora_rank, device=dev, dtype=torch.bfloat16)
    k_pe = torch.randn(ntok, pe, device=dev, dtype=torch.bfloat16)
    slot = torch.randperm(num_blocks * page, device=dev, dtype=torch.int64)[:ntok]
    scale = torch.tensor(1.0, device=dev)

    def write(cache):
        ops.concat_and_cache_mla(kv_c, k_pe, cache, slot, "auto", scale)

    # Contiguous per-layer reference: (num_blocks, page, entry).
    ref = torch.zeros(num_blocks, page, entry, device=dev, dtype=torch.bfloat16)
    write(ref)

    # Unified slot holding three layer pages per block. Carve the middle
    # layer's view (non-zero offset, block stride == full unified slot).
    layer_page_elems = page * entry
    n_layers = 3
    unified_slot_elems = n_layers * layer_page_elems
    big = torch.zeros(num_blocks, unified_slot_elems, device=dev, dtype=torch.bfloat16)
    flat = big.view(-1)
    offset = layer_page_elems  # middle layer
    view = torch.as_strided(
        flat,
        size=(num_blocks, page, entry),
        stride=(unified_slot_elems, entry, 1),
        storage_offset=offset,
    )
    assert not view.is_contiguous()
    assert view.stride(0) == unified_slot_elems
    write(view)

    # Bit-exact equivalence and zero bleed into the neighbour segments.
    max_diff = (ref.float() - view.float()).abs().max().item()
    assert max_diff == 0.0, f"max|Δ| = {max_diff}"

    neighbour_lo = torch.as_strided(
        flat, (num_blocks, layer_page_elems), (unified_slot_elems, 1), 0
    )
    neighbour_hi = torch.as_strided(
        flat,
        (num_blocks, layer_page_elems),
        (unified_slot_elems, 1),
        2 * layer_page_elems,
    )
    assert neighbour_lo.abs().max().item() == 0.0
    assert neighbour_hi.abs().max().item() == 0.0


def test_flashmla_dense_decode_unified_slot_view():
    """FlashMLA dense decode (FLASHMLA backend, e.g. Kimi-K2-style dense MLA
    on Hopper) must read a unified-slot block-major view bit-identically to a
    contiguous per-layer cache."""
    import vllm.v1.attention.ops.flashmla as fm

    ok, reason = fm.is_flashmla_dense_supported()
    if not ok:
        pytest.skip(reason)

    torch.manual_seed(0)
    dev = "cuda"
    dt = torch.bfloat16
    head_dim = 576
    hdv = 512
    h_q = 128
    page = 64
    num_blocks = 64
    bs = 4
    n_layers = 3
    layer = 1

    q = torch.randn(bs, 1, h_q, head_dim, device=dev, dtype=dt) * 0.1
    kv_data = torch.randn(num_blocks, page, 1, head_dim, device=dev, dtype=dt) * 0.1

    # (A) contiguous per-layer reference.
    cache_contiguous = kv_data.clone().contiguous()

    # (B) unified slot: view one layer -> inflated stride(0), non-zero offset.
    unified = (
        torch.randn(num_blocks, n_layers, page, 1, head_dim, device=dev, dtype=dt) * 0.1
    )
    unified[:, layer].copy_(kv_data)
    cache_view = unified[:, layer]
    assert not cache_view.is_contiguous()
    assert cache_view.stride(0) == n_layers * page * 1 * head_dim

    max_blk = num_blocks // bs
    block_table = torch.arange(num_blocks, device=dev, dtype=torch.int32).view(
        bs, max_blk
    )
    cache_seqlens = torch.full((bs,), max_blk * page, device=dev, dtype=torch.int32)

    def run(kc):
        meta, num_splits = fm.get_mla_metadata()
        out, _ = fm.flash_mla_with_kvcache(
            q=q,
            k_cache=kc,
            block_table=block_table,
            cache_seqlens=cache_seqlens,
            head_dim_v=hdv,
            tile_scheduler_metadata=meta,
            num_splits=num_splits,
            softmax_scale=head_dim**-0.5,
            causal=True,
        )
        return out.clone().float()

    out_ref = run(cache_contiguous)
    out_view = run(cache_view)
    assert torch.isfinite(out_ref).all()
    assert out_ref.abs().max().item() > 0.0
    assert (out_ref - out_view).abs().max().item() == 0.0


def test_flashinfer_mla_dense_decode_unified_slot_view():
    """FlashInfer MLA dense decode must read a unified-slot block-major view
    (inflated stride(0), non-zero storage offset) bit-identically to a
    contiguous per-layer cache."""
    try:
        from flashinfer.decode import trtllm_batch_decode_with_kv_cache_mla
    except ImportError:
        pytest.skip("flashinfer is not available")
    from vllm.platforms import current_platform

    if not current_platform.is_device_capability_family(100):
        pytest.skip("FlashInfer trtllm-gen MLA requires sm100")

    torch.manual_seed(0)
    dev = "cuda"
    dt = torch.bfloat16
    kv_lora_rank = 512
    qk_rope_head_dim = 64
    qk_nope_head_dim = 128
    head_dim = kv_lora_rank + qk_rope_head_dim  # 576
    num_qo_heads = 128
    page = 64
    num_blocks = 64
    bs = 4
    n_layers = 3  # >1 so the per-layer view's block stride is inflated.
    layer = 1

    q = torch.randn(bs, 1, num_qo_heads, head_dim, device=dev, dtype=dt)
    kv_data = torch.randn(num_blocks, 1, page, head_dim, device=dev, dtype=dt)

    # (A) contiguous per-layer reference.
    kv_contiguous = kv_data.clone().contiguous()

    # (B) unified slot: block b of every layer packed together; view one layer
    # -> stride(0) is n_layers x larger and storage offset is non-zero.
    unified = torch.randn(num_blocks, n_layers, 1, page, head_dim, device=dev, dtype=dt)
    unified[:, layer].copy_(kv_data)
    kv_view = unified[:, layer]
    assert not kv_view.is_contiguous()
    assert kv_view.stride(0) == n_layers * 1 * page * head_dim

    max_blk = num_blocks // bs
    block_tables = torch.arange(num_blocks, device=dev, dtype=torch.int32).view(
        bs, max_blk
    )
    seq_lens = torch.full((bs,), max_blk * page, device=dev, dtype=torch.int32)
    ws = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device=dev)
    scale = head_dim**-0.5

    def run(kv):
        return trtllm_batch_decode_with_kv_cache_mla(
            query=q,
            kv_cache=kv,
            workspace_buffer=ws,
            qk_nope_head_dim=qk_nope_head_dim,
            kv_lora_rank=kv_lora_rank,
            qk_rope_head_dim=qk_rope_head_dim,
            block_tables=block_tables,
            seq_lens=seq_lens,
            max_seq_len=int(seq_lens.max().item()),
            bmm1_scale=scale,
            bmm2_scale=1.0,
        ).clone()

    out_ref = run(kv_contiguous).float()
    out_view = run(kv_view).float()
    assert torch.isfinite(out_ref).all()
    assert (out_ref - out_view).abs().max().item() == 0.0


def test_flashmla_fp8_sparse_decode_unified_slot_view():
    """FlashMLA fp8 sparse decode (DeepSeek V3.2/V4 DSA path) must read a
    unified-slot block-major view bit-identically to a contiguous fp8_ds_mla
    cache, with finite nonzero output."""
    import vllm.v1.attention.ops.flashmla as fm

    ok, reason = fm.is_flashmla_sparse_supported()
    if not ok:
        pytest.skip(reason)

    torch.manual_seed(0)
    dev = "cuda"
    entry = 656  # fp8_ds_mla bytes per token
    page = 64
    num_blocks = 32
    h_q = 128
    head_dim = 576
    hdv = 512
    batch = 2
    topk = 128
    n_layers = 3
    layer = 1

    q = torch.randn(batch, 1, h_q, head_dim, device=dev, dtype=torch.bfloat16) * 0.1

    # Structurally valid fp8 ds_mla payload: 512B fp8 + 16B f32 scales + 128B
    # bf16 rope (random bytes corrupt the scale region and yield NaNs).
    nope = (torch.randn(num_blocks, page, 1, 512, device=dev) * 0.1).to(
        torch.float8_e4m3fn
    )
    scales = torch.ones(num_blocks, page, 1, 4, device=dev, dtype=torch.float32)
    rope = (torch.randn(num_blocks, page, 1, 64, device=dev) * 0.1).to(torch.bfloat16)
    payload = torch.cat(
        [
            nope.view(torch.uint8).view(num_blocks, page, 1, 512),
            scales.view(torch.uint8).view(num_blocks, page, 1, 16),
            rope.view(torch.uint8).view(num_blocks, page, 1, 128),
        ],
        dim=-1,
    ).contiguous()
    assert payload.shape[-1] == entry and payload.dtype == torch.uint8

    # (A) contiguous reference.
    cache_contiguous = payload.clone().contiguous()

    # (B) unified slot: view one layer -> inflated stride(0), non-zero offset.
    unified = torch.randint(
        0, 256, (num_blocks, n_layers, page, 1, entry), device=dev, dtype=torch.uint8
    )
    unified[:, layer].copy_(payload)
    cache_view = unified[:, layer]
    assert not cache_view.is_contiguous()
    assert cache_view.stride(0) == n_layers * page * 1 * entry

    # Sparse indices: each batch uses its own disjoint blocks.
    blocks_per_batch = num_blocks // batch
    idx = torch.full((batch, 1, topk), -1, device=dev, dtype=torch.int32)
    for b in range(batch):
        slots: list[int] = []
        for blk in range(b * blocks_per_batch, (b + 1) * blocks_per_batch):
            slots.extend(blk * page + off for off in range(page))
        slots_t = torch.tensor(slots[:topk], device=dev, dtype=torch.int32)
        idx[b, 0, : slots_t.numel()] = slots_t

    def run(kc):
        meta, num_splits = fm.get_mla_metadata()
        out, _ = fm.flash_mla_with_kvcache(
            q=q,
            k_cache=kc,
            block_table=None,
            cache_seqlens=None,
            head_dim_v=hdv,
            tile_scheduler_metadata=meta,
            is_fp8_kvcache=True,
            indices=idx,
            softmax_scale=head_dim**-0.5,
        )
        return out.clone().float()

    out_ref = run(cache_contiguous)
    out_view = run(cache_view)
    assert torch.isfinite(out_ref).all()
    assert out_ref.abs().max().item() > 0.0
    assert (out_ref - out_view).abs().max().item() == 0.0


def test_indexer_k_quant_and_cache_into_unified_slot_view():
    """indexer_k_quant_and_cache (DeepSeek V3.2/V4 DSA indexer K write) must
    write correctly into a per-layer view whose block stride is the full
    unified slot, with zero bleed into the other layers' segments."""
    from vllm import _custom_ops as ops

    torch.manual_seed(0)
    dev = "cuda"
    head_dim = 128
    quant_block_size = 128
    block_size = 64
    num_blocks = 16
    ntok = 100
    # Indexer cache layout per token: head_dim fp8 bytes followed by
    # head_dim * 4 / quant_block_size scale bytes.
    cache_stride = head_dim + head_dim * 4 // quant_block_size

    k = torch.randn(ntok, head_dim, device=dev, dtype=torch.bfloat16)
    slot = torch.randperm(num_blocks * block_size, device=dev, dtype=torch.int64)[:ntok]

    def write(cache):
        ops.indexer_k_quant_and_cache(k, cache, slot, quant_block_size, "ue8m0")

    # Contiguous per-layer reference.
    ref = torch.zeros(
        num_blocks, block_size, cache_stride, device=dev, dtype=torch.uint8
    )
    write(ref)

    # Unified slot holding three layer pages per block; carve the middle one.
    n_layers = 3
    layer = 1
    unified = torch.zeros(
        num_blocks, n_layers, block_size, cache_stride, device=dev, dtype=torch.uint8
    )
    view = unified[:, layer]
    assert not view.is_contiguous()
    assert view.stride(0) == n_layers * block_size * cache_stride
    write(view)

    assert torch.equal(ref, view.contiguous())
    # Zero bleed into the neighbour layers' segments.
    assert unified[:, 0].abs().max().item() == 0
    assert unified[:, 2].abs().max().item() == 0


def test_flashattn_mla_dense_decode_unified_slot_view():
    """FA3 decode (FLASH_ATTN_MLA backend) must read a unified-slot
    block-major view bit-identically to a contiguous per-layer cache."""
    try:
        from vllm.vllm_flash_attn import flash_attn_varlen_func
    except ImportError:
        pytest.skip("vllm_flash_attn is not available")
    from vllm.v1.attention.backends.fa_utils import flash_attn_supports_mla

    if not flash_attn_supports_mla():
        pytest.skip("FA3 MLA requires a Hopper device")

    torch.manual_seed(0)
    dev = "cuda"
    dt = torch.bfloat16
    kv_lora_rank = 512
    rope_dim = 64
    entry = kv_lora_rank + rope_dim  # 576
    h_q = 16
    page = 64
    num_blocks = 64
    bs = 4
    n_layers = 3
    layer = 1

    q_pe = torch.randn(bs, h_q, rope_dim, device=dev, dtype=dt) * 0.1
    q_nope = torch.randn(bs, h_q, kv_lora_rank, device=dev, dtype=dt) * 0.1
    kv_data = torch.randn(num_blocks, page, entry, device=dev, dtype=dt) * 0.1

    # (A) contiguous per-layer reference.
    cache_contiguous = kv_data.clone().contiguous()

    # (B) unified slot: view one layer -> inflated stride(0), non-zero offset.
    unified = torch.randn(num_blocks, n_layers, page, entry, device=dev, dtype=dt) * 0.1
    unified[:, layer].copy_(kv_data)
    cache_view = unified[:, layer]
    assert not cache_view.is_contiguous()
    assert cache_view.stride(0) == n_layers * page * entry

    max_blk = num_blocks // bs
    block_table = torch.arange(num_blocks, device=dev, dtype=torch.int32).view(
        bs, max_blk
    )
    seq_lens = torch.full((bs,), max_blk * page, device=dev, dtype=torch.int32)
    cu_seqlens_q = torch.arange(bs + 1, device=dev, dtype=torch.int32)

    def run(cache):
        kv_c_cache = cache[..., :kv_lora_rank]
        k_pe_cache = cache[..., kv_lora_rank:]
        out = flash_attn_varlen_func(
            q=q_pe,
            k=k_pe_cache.unsqueeze(-2),  # Add head dim of 1
            v=kv_c_cache.unsqueeze(-2),  # Add head dim of 1
            q_v=q_nope,
            max_seqlen_q=1,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_k=int(seq_lens.max().item()),
            seqused_k=seq_lens,
            block_table=block_table,
            softmax_scale=entry**-0.5,
            causal=True,
            fa_version=3,
        )
        return out.clone().float()

    out_ref = run(cache_contiguous)
    out_view = run(cache_view)
    assert torch.isfinite(out_ref).all()
    assert out_ref.abs().max().item() > 0.0
    assert (out_ref - out_view).abs().max().item() == 0.0


def test_flashmla_dense_fp8_decode_unified_slot_view():
    """FlashMLA dense fp8 decode (FLASHMLA backend with quantized KV cache)
    must read a unified-slot block-major view bit-identically to a contiguous
    per-layer fp8 cache."""
    import vllm.v1.attention.ops.flashmla as fm

    ok, reason = fm.is_flashmla_dense_supported()
    if not ok:
        pytest.skip(reason)

    torch.manual_seed(0)
    dev = "cuda"
    head_dim = 576
    hdv = 512
    h_q = 128
    page = 64
    num_blocks = 64
    bs = 4
    n_layers = 3
    layer = 1

    q = torch.randn(bs, 1, h_q, head_dim, device=dev, dtype=torch.bfloat16) * 0.1
    kv_data = (torch.randn(num_blocks, page, head_dim, device=dev) * 0.1).to(
        torch.float8_e4m3fn
    )

    # (A) contiguous per-layer reference.
    cache_contiguous = kv_data.clone().contiguous()

    # (B) unified slot: view one layer -> inflated stride(0), non-zero offset.
    unified = (torch.randn(num_blocks, n_layers, page, head_dim, device=dev) * 0.1).to(
        torch.float8_e4m3fn
    )
    unified[:, layer].copy_(kv_data)
    cache_view = unified[:, layer]
    assert not cache_view.is_contiguous()
    assert cache_view.stride(0) == n_layers * page * head_dim

    max_blk = num_blocks // bs
    block_table = torch.arange(num_blocks, device=dev, dtype=torch.int32).view(
        bs, max_blk
    )
    cache_seqlens = torch.full((bs,), max_blk * page, device=dev, dtype=torch.int32)
    descale = torch.ones(1, device=dev, dtype=torch.float32)

    def run(kc):
        tile_md, num_splits = fm.get_mla_metadata_dense_fp8(cache_seqlens, h_q, 1)
        out, _ = fm.flash_mla_with_kvcache_fp8(
            q=q,
            k_cache=kc.unsqueeze(-2),  # Add head dim of 1
            block_table=block_table,
            cache_seqlens=cache_seqlens,
            head_dim_v=hdv,
            tile_scheduler_metadata=tile_md,
            num_splits=num_splits,
            softmax_scale=head_dim**-0.5,
            causal=True,
            descale_q=descale,
            descale_k=descale,
        )
        return out.clone().float()

    out_ref = run(cache_contiguous)
    out_view = run(cache_view)
    assert torch.isfinite(out_ref).all()
    assert out_ref.abs().max().item() > 0.0
    assert (out_ref - out_view).abs().max().item() == 0.0


def test_flashinfer_mla_dense_fp8_decode_unified_slot_view():
    """FlashInfer MLA dense decode with an fp8 KV cache must read a
    unified-slot block-major view bit-identically to a contiguous per-layer
    cache."""
    try:
        from flashinfer.decode import trtllm_batch_decode_with_kv_cache_mla
    except ImportError:
        pytest.skip("flashinfer is not available")
    from vllm.platforms import current_platform

    if not current_platform.is_device_capability_family(100):
        pytest.skip("FlashInfer trtllm-gen MLA requires sm100")

    torch.manual_seed(0)
    dev = "cuda"
    kv_lora_rank = 512
    qk_rope_head_dim = 64
    qk_nope_head_dim = 128
    head_dim = kv_lora_rank + qk_rope_head_dim  # 576
    num_qo_heads = 128
    page = 64
    num_blocks = 64
    bs = 4
    n_layers = 3
    layer = 1

    # With a quantized KV cache the decode query is quantized to fp8 as well
    # (trtllm-gen has no bf16-query x fp8-cache decode kernel).
    q = (torch.randn(bs, 1, num_qo_heads, head_dim, device=dev) * 0.1).to(
        torch.float8_e4m3fn
    )
    kv_data = (torch.randn(num_blocks, 1, page, head_dim, device=dev) * 0.1).to(
        torch.float8_e4m3fn
    )

    # (A) contiguous per-layer reference.
    kv_contiguous = kv_data.clone().contiguous()

    # (B) unified slot: view one layer -> inflated stride(0), non-zero offset.
    unified = (
        torch.randn(num_blocks, n_layers, 1, page, head_dim, device=dev) * 0.1
    ).to(torch.float8_e4m3fn)
    unified[:, layer].copy_(kv_data)
    kv_view = unified[:, layer]
    assert not kv_view.is_contiguous()
    assert kv_view.stride(0) == n_layers * 1 * page * head_dim

    max_blk = num_blocks // bs
    block_tables = torch.arange(num_blocks, device=dev, dtype=torch.int32).view(
        bs, max_blk
    )
    seq_lens = torch.full((bs,), max_blk * page, device=dev, dtype=torch.int32)
    ws = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device=dev)
    scale = head_dim**-0.5

    def run(kv):
        return trtllm_batch_decode_with_kv_cache_mla(
            query=q,
            kv_cache=kv,
            workspace_buffer=ws,
            qk_nope_head_dim=qk_nope_head_dim,
            kv_lora_rank=kv_lora_rank,
            qk_rope_head_dim=qk_rope_head_dim,
            block_tables=block_tables,
            seq_lens=seq_lens,
            max_seq_len=int(seq_lens.max().item()),
            bmm1_scale=scale,
            bmm2_scale=1.0,
        ).clone()

    out_ref = run(kv_contiguous).float()
    out_view = run(kv_view).float()
    assert torch.isfinite(out_ref).all()
    assert (out_ref - out_view).abs().max().item() == 0.0
