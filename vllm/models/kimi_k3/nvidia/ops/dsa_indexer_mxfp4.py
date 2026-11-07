# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MXFP4 writers of the Kimi K3.1 DSA indexer: the key LayerNorm + MXFP4 quant +
cache insert (standalone, and fused into the DSA prologue) and the MXFP4 query quant.

Format (DeepSeek V4's MXFP4 indexer cache, what DeepGEMM's ``fp8_fp4_*mqa_logits``
read): one UE8M0 scale ``2^ceil(log2(amax / 6))`` per 32 values, E2M1 values (round to
nearest even, saturating), two per byte with the even element in the low nibble. A
cache block of ``B`` tokens is ``[B x D/2 E2M1 | B x D/32 UE8M0]``, ``D/2 + D/32`` =
68 B/token for ``D = 128``. The query is ``[T, P, D/2]`` E2M1 with its four UE8M0
bytes per head packed into one int32 (``q_scale [T, P]``); the head weights take only
``weight_scale`` (the MX scales stay in the logits MMA).

Used only with an explicit ``--attention-config '{"indexer_kv_dtype": "mxfp4"}'``
(``kimi_k3_indexer_uses_fp4``); the default FP8 cache keeps the writers of
``dsa_indexer.py``, whose LayerNorm this key insert repeats op for op. Triton only (no
CuTe import), so every engine of a P/D pair writes the same bytes.
"""

import torch

from vllm.model_executor.layers.sparse_attn_indexer import MXFP4_BLOCK_SIZE
from vllm.models.kimi_k3.common.dsa_cache import indexer_cache_entry_bytes
from vllm.models.kimi_k3.nvidia.ops.dsa_indexer import (
    _FP8_DTYPE,
    _FP8_MAX,
    _USE_FNUZ,
    _dsa_fused_prologue_kernel,
    _fill_topk,
)
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton

_MXFP4_BLOCK = tl.constexpr(MXFP4_BLOCK_SIZE)


@triton.jit
def _fp32x2_to_fp4x2(x_lo, x_hi):
    # $1 is the high nibble, $2 the low nibble.
    return tl.inline_asm_elementwise(
        """
        {
            .reg .b8 tmp;
            cvt.rn.satfinite.e2m1x2.f32 tmp, $1, $2;
            cvt.u32.u8 $0, tmp;
        }
        """,
        constraints="=r,f,f",
        args=[x_hi, x_lo],
        dtype=tl.uint32,
        is_pure=True,
        pack=1,
    ).to(tl.uint8)


@triton.jit
def _mxfp4_quant(x, HEAD_DIM: tl.constexpr):
    """MXFP4 of one ``HEAD_DIM`` fp32 row, same math as DeepSeek V4's
    ``_quantize_mxfp4_pair``. Returns ``(packed [HEAD_DIM // 2] uint8 with the
    even element in the low nibble, ue8m0 [HEAD_DIM // 32] uint8)``.

    cutedsl/dsa/index.py (mxfp4_scale) replays the PTX Triton emits for this
    scale: keep the two in step."""
    NUM_BLOCKS: tl.constexpr = HEAD_DIM // _MXFP4_BLOCK
    x_lo, x_hi = tl.split(tl.reshape(x, (NUM_BLOCKS, _MXFP4_BLOCK // 2, 2)))
    amax = tl.maximum(tl.max(tl.abs(x_lo), axis=1), tl.max(tl.abs(x_hi), axis=1))
    # ue8m0 block scale 2^ceil(log2(amax / 6)), floored at 6 * 2^-126.
    log2_scale = tl.math.ceil(tl.math.log2(tl.maximum(amax, 6.0 * (2**-126)) / 6.0))
    log2_scale = tl.minimum(tl.maximum(log2_scale, -127.0), 127.0)
    inv_scale = (1.0 / tl.math.exp2(log2_scale))[:, None]
    packed = _fp32x2_to_fp4x2(x_lo * inv_scale, x_hi * inv_scale)
    return tl.reshape(packed, (HEAD_DIM // 2,)), (log2_scale + 127.0).to(tl.uint8)


@triton.jit
def _mxfp4_k_insert(
    tok,
    k_ptr,
    norm_w_ptr,
    norm_b_ptr,
    norm_eps,
    slot_mapping_ptr,
    num_slots,
    cache_ptr,
    cache_block_size,
    cache_block_stride,
    HEAD_DIM: tl.constexpr,
):
    # Padded rows and (under DCP) tokens owned by another rank have no slot.
    slot = tl.load(slot_mapping_ptr + tok, mask=tok < num_slots, other=-1)
    if slot >= 0:
        # The stock FP8 insert's LayerNorm (dsa_indexer._index_k_insert), op for op.
        offs = tl.arange(0, HEAD_DIM)
        x = tl.load(k_ptr + offs).to(tl.float32)
        mean = tl.sum(x, axis=0) / HEAD_DIM
        diff = x - mean
        rstd = tl.rsqrt(tl.sum(diff * diff, axis=0) / HEAD_DIM + norm_eps)
        w = tl.load(norm_w_ptr + offs).to(tl.float32)
        b = tl.load(norm_b_ptr + offs).to(tl.float32)
        # Round to bf16 like the unfused path so the cached bytes match.
        y = (diff * rstd * w + b).to(k_ptr.dtype.element_ty).to(tl.float32)

        # Block layout: [block_size x HEAD_DIM / 2 packed e2m1 |
        # block_size x HEAD_DIM / 32 ue8m0], as DeepSeek V4's MXFP4 cache.
        NUM_SCALES: tl.constexpr = HEAD_DIM // _MXFP4_BLOCK
        packed, ue8m0 = _mxfp4_quant(y, HEAD_DIM)
        block_start = (slot // cache_block_size) * cache_block_stride
        block_off = slot % cache_block_size
        data = cache_ptr + block_start + block_off * (HEAD_DIM // 2)
        tl.store(data + tl.arange(0, HEAD_DIM // 2), packed)
        scale_byte = (
            block_start + cache_block_size * (HEAD_DIM // 2) + block_off * NUM_SCALES
        )
        tl.store(cache_ptr + scale_byte + tl.arange(0, NUM_SCALES), ue8m0)


@triton.jit
def _mxfp4_k_norm_quant_insert_kernel(
    k_ptr,
    k_stride,
    norm_w_ptr,
    norm_b_ptr,
    norm_eps,
    slot_mapping_ptr,
    num_slots,
    cache_ptr,
    cache_block_size,
    cache_block_stride,
    topk_ptr,
    topk_stride,
    HEAD_DIM: tl.constexpr,
    TOPK: tl.constexpr,
    TOPK_BLOCK: tl.constexpr,
):
    tok = tl.program_id(0).to(tl.int64)
    if tl.program_id(1) == 1:
        # Padding rows included.
        _fill_topk(topk_ptr + tok * topk_stride, TOPK, TOPK_BLOCK)
    else:
        _mxfp4_k_insert(
            tok,
            k_ptr + tok * k_stride,
            norm_w_ptr,
            norm_b_ptr,
            norm_eps,
            slot_mapping_ptr,
            num_slots,
            cache_ptr,
            cache_block_size,
            cache_block_stride,
            HEAD_DIM,
        )


def _cache_bytes(kv_cache: torch.Tensor, head_dim: int) -> torch.Tensor:
    u8_cache = kv_cache.view(torch.uint8)
    assert u8_cache.dim() == 3
    assert u8_cache.shape[-1] == indexer_cache_entry_bytes(head_dim, True), (
        u8_cache.shape
    )
    return u8_cache


def mxfp4_k_norm_quant_insert(
    k: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_bias: torch.Tensor,
    norm_eps: float,
    slot_mapping: torch.Tensor,
    kv_cache: torch.Tensor,
    topk_indices_buffer: torch.Tensor,
    num_tokens: int,
) -> None:
    """``indexer_k_norm_quant_insert`` for the MXFP4 cache: LayerNorm + MXFP4
    quant + cache insert of the indexer key, and the -1 fill of the top-k rows
    ``[:num_tokens]``.

    ``kv_cache`` is ``[num_blocks, block_size, indexer_cache_entry_bytes]`` uint8,
    possibly a strided page view of a packed cache. Rows past ``slot_mapping`` or
    with a negative slot are not inserted.
    """
    if num_tokens == 0:
        return
    head_dim = k.shape[-1]
    assert k.stride(-1) == 1
    assert head_dim % MXFP4_BLOCK_SIZE == 0
    assert triton.next_power_of_2(head_dim) == head_dim
    cache = _cache_bytes(kv_cache, head_dim)
    topk = topk_indices_buffer.shape[-1]
    _mxfp4_k_norm_quant_insert_kernel[(num_tokens, 2)](
        k,
        k.stride(0),
        norm_weight,
        norm_bias,
        norm_eps,
        slot_mapping,
        slot_mapping.shape[0],
        cache,
        cache.shape[1],
        cache.stride(0),
        topk_indices_buffer,
        topk_indices_buffer.stride(0),
        HEAD_DIM=head_dim,
        TOPK=topk,
        TOPK_BLOCK=min(1024, triton.next_power_of_2(topk)),
        num_warps=1,
    )


@triton.jit
def _mxfp4_q_quant_kernel(
    q_ptr,
    q_stride0,
    q_stride1,
    w_ptr,
    w_stride,
    q_out_ptr,
    q_out_stride0,
    q_scale_ptr,
    q_scale_stride0,
    w_out_ptr,
    w_out_stride,
    weight_scale,
    num_heads,
    HEAD_DIM: tl.constexpr,
    HEADS_PER_PROGRAM: tl.constexpr,
    EVEN_HEADS: tl.constexpr,
):
    """HEADS_PER_PROGRAM heads of one token. The MXFP4 blocks never straddle two
    heads, so the heads quantize as one row, each block exactly as alone."""
    HB: tl.constexpr = HEADS_PER_PROGRAM
    NUM_SCALES: tl.constexpr = HEAD_DIM // _MXFP4_BLOCK
    tok = tl.program_id(0).to(tl.int64)
    head0 = tl.program_id(1) * HB
    offs = tl.arange(0, HB * HEAD_DIM)
    head = head0 + offs // HEAD_DIM
    q_in = q_ptr + tok * q_stride0 + head * q_stride1 + offs % HEAD_DIM
    q = tl.load(q_in) if EVEN_HEADS else tl.load(q_in, mask=head < num_heads, other=0.0)
    # The ue8m0 block scales go to q_scale; the logits kernel applies them.
    packed, ue8m0 = _mxfp4_quant(q.to(tl.float32), HB * HEAD_DIM)
    # Packed rows of the output heads are contiguous (q_out [T, P, D / 2]).
    out = head0 * (HEAD_DIM // 2) + tl.arange(0, HB * HEAD_DIM // 2)
    scale = head0 * NUM_SCALES + tl.arange(0, HB * NUM_SCALES)
    heads = head0 + tl.arange(0, HB)
    q_out = q_out_ptr + tok * q_out_stride0 + out
    q_scale = q_scale_ptr + tok * q_scale_stride0 + scale
    w_in = w_ptr + tok * w_stride + heads
    w_out = w_out_ptr + tok * w_out_stride + heads
    if EVEN_HEADS:
        tl.store(q_out, packed)
        tl.store(q_scale, ue8m0)
        tl.store(w_out, tl.load(w_in).to(tl.float32) * weight_scale)
    else:
        tl.store(q_out, packed, mask=out < num_heads * (HEAD_DIM // 2))
        tl.store(q_scale, ue8m0, mask=scale < num_heads * NUM_SCALES)
        w = tl.load(w_in, mask=heads < num_heads)
        tl.store(w_out, w.to(tl.float32) * weight_scale, mask=heads < num_heads)


def mxfp4_q_quant(
    index_q: torch.Tensor,
    index_weights: torch.Tensor,
    weight_scale: float,
    padded_num_heads: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize ``index_q`` ``[T, H, D]`` to MXFP4 (one ue8m0 scale per 32
    values), the query format of the MXFP4 indexer key cache, and scale
    ``index_weights`` ``[T, H]`` by ``weight_scale``.

    Returns ``(q_packed [T, P, D // 2] uint8, q_scale [T, P] int32 packing the
    four ue8m0 bytes of each head, weights [T, P] fp32)`` where ``P =
    padded_num_heads``; heads ``[H, P)`` are zero.
    """
    num_tokens, num_heads, head_dim = index_q.shape
    assert index_q.stride(-1) == 1
    assert index_weights.shape == (num_tokens, num_heads)
    assert head_dim == 4 * MXFP4_BLOCK_SIZE
    alloc = torch.empty if padded_num_heads == num_heads else torch.zeros
    device = index_q.device
    q_packed = alloc(
        (num_tokens, padded_num_heads, head_dim // 2), dtype=torch.uint8, device=device
    )
    q_scale = alloc((num_tokens, padded_num_heads), dtype=torch.int32, device=device)
    weights = alloc((num_tokens, padded_num_heads), dtype=torch.float32, device=device)
    if num_tokens == 0:
        return q_packed, q_scale, weights
    q_scale_bytes = q_scale.view(torch.uint8)
    # One warp per program; heads per program by token count (GB300, 32 heads:
    # 2 at decode rows, 8 from 128 tokens, 16 from 1K; 16K tokens: 35 us
    # against 266 us at one head per program).
    hb = 2 if num_tokens < 128 else 8 if num_tokens < 1024 else 16
    heads_per_program = min(hb, triton.next_power_of_2(num_heads))
    grid = (num_tokens, triton.cdiv(num_heads, heads_per_program))
    _mxfp4_q_quant_kernel[grid](
        index_q,
        index_q.stride(0),
        index_q.stride(1),
        index_weights,
        index_weights.stride(0),
        q_packed,
        q_packed.stride(0),
        q_scale_bytes,
        q_scale_bytes.stride(0),
        weights,
        weights.stride(0),
        weight_scale,
        num_heads,
        HEAD_DIM=head_dim,
        HEADS_PER_PROGRAM=heads_per_program,
        EVEN_HEADS=num_heads % heads_per_program == 0,
        num_warps=1,
    )
    return q_packed, q_scale, weights


@triton.jit
def _mxfp4_fused_prologue_kernel(
    q_c_ptr,
    q_c_stride,
    q_norm_w_ptr,
    q_out_ptr,
    kv_c_ptr,
    kv_c_stride,
    kv_norm_w_ptr,
    kv_out_ptr,
    k_pe_ptr,
    k_pe_stride,
    index_k_ptr,
    index_k_stride,
    k_norm_w_ptr,
    k_norm_b_ptr,
    eps,
    mla_slot_ptr,
    num_mla_slots,
    mla_cache_ptr,
    mla_scale_ptr,
    mla_rope_ptr,
    mla_block_size,
    mla_block_stride,
    mla_entry_stride,
    index_slot_ptr,
    num_index_slots,
    index_cache_ptr,
    index_block_size,
    index_block_stride,
    topk_ptr,
    topk_stride,
    Q_DIM: tl.constexpr,
    Q_BLOCK: tl.constexpr,
    KV_DIM: tl.constexpr,
    ROPE_DIM: tl.constexpr,
    INDEX_DIM: tl.constexpr,
    TOPK: tl.constexpr,
    TOPK_BLOCK: tl.constexpr,
    USE_PDL: tl.constexpr,
    FP8_MAX: tl.constexpr,
    USE_FNUZ: tl.constexpr,
):
    """The stock FP8 prologue (dsa_indexer._dsa_fused_prologue_kernel) with its
    program 0, the indexer key insert, writing MXFP4: programs 1-3 (fp8_ds_mla
    insert, q_a RMSNorm, top-k fill) run the stock kernel's own code."""
    if tl.program_id(1) == 0:
        if USE_PDL:
            tl.extra.cuda.gdc_wait()
            tl.extra.cuda.gdc_launch_dependents()
        tok = tl.program_id(0).to(tl.int64)
        _mxfp4_k_insert(
            tok,
            index_k_ptr + tok * index_k_stride,
            k_norm_w_ptr,
            k_norm_b_ptr,
            eps,
            index_slot_ptr,
            num_index_slots,
            index_cache_ptr,
            index_block_size,
            index_block_stride,
            INDEX_DIM,
        )
    else:
        # The stock kernel's FP8 key insert (its program 0) is dead code here:
        # no index slots, and its cache pointers are never dereferenced.
        _dsa_fused_prologue_kernel(
            q_c_ptr,
            q_c_stride,
            q_norm_w_ptr,
            q_out_ptr,
            kv_c_ptr,
            kv_c_stride,
            kv_norm_w_ptr,
            kv_out_ptr,
            k_pe_ptr,
            k_pe_stride,
            index_k_ptr,
            index_k_stride,
            k_norm_w_ptr,
            k_norm_b_ptr,
            eps,
            mla_slot_ptr,
            num_mla_slots,
            mla_cache_ptr,
            mla_scale_ptr,
            mla_rope_ptr,
            mla_block_size,
            mla_block_stride,
            mla_entry_stride,
            index_slot_ptr,
            0,
            mla_cache_ptr,
            mla_scale_ptr,
            index_block_size,
            index_block_stride,
            topk_ptr,
            topk_stride,
            Q_DIM,
            Q_BLOCK,
            KV_DIM,
            ROPE_DIM,
            INDEX_DIM,
            TOPK,
            TOPK_BLOCK,
            False,
            USE_PDL,
            FP8_MAX,
            USE_FNUZ,
        )


def mxfp4_fused_prologue(
    q_c: torch.Tensor,
    kv_c: torch.Tensor,
    k_pe: torch.Tensor,
    index_k: torch.Tensor,
    q_norm_weight: torch.Tensor,
    kv_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    k_norm_bias: torch.Tensor,
    eps: float,
    topk_indices_buffer: torch.Tensor,
    mla_cache: torch.Tensor | None,
    mla_slot_mapping: torch.Tensor | None,
    index_cache: torch.Tensor | None,
    index_slot_mapping: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``dsa_fused_prologue`` for the MXFP4 indexer cache, in one launch: the
    q_a / kv_a RMSNorms, the fp8_ds_mla insert of ``[kv_c_normed | k_pe]`` and
    the -1 fill of the top-k rows (the stock kernel's code), and the indexer key
    LayerNorm + MXFP4 insert.

    ``mla_cache`` (``[num_blocks, block_size, 656]``) and ``index_cache``
    (``[num_blocks, block_size, indexer_cache_entry_bytes]``) are uint8, possibly
    strided page views of a packed cache; ``None`` (profile runs) skips the
    insert. Rows past a slot mapping or with a negative slot are not inserted.

    Returns ``(q_c_normed, kv_c_normed)``.
    """
    num_tokens, q_dim = q_c.shape
    kv_dim, rope_dim, index_dim = kv_c.shape[1], k_pe.shape[1], index_k.shape[1]
    assert all(t.stride(-1) == 1 for t in (q_c, kv_c, k_pe, index_k))
    assert index_dim % MXFP4_BLOCK_SIZE == 0
    q_out = torch.empty((num_tokens, q_dim), dtype=q_c.dtype, device=q_c.device)
    kv_out = torch.empty((num_tokens, kv_dim), dtype=kv_c.dtype, device=kv_c.device)
    if num_tokens == 0:
        return q_out, kv_out

    # No cache (profile runs): an empty slot mapping disables the insert.
    def placeholder(entry_bytes: int) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.empty(1, 1, entry_bytes, dtype=torch.uint8, device=q_c.device),
            torch.empty(0, dtype=torch.int64, device=q_c.device),
        )

    if mla_cache is None or mla_slot_mapping is None:
        mla_cache, mla_slot_mapping = placeholder(4)
    if index_cache is None or index_slot_mapping is None:
        index_cache, index_slot_mapping = placeholder(
            indexer_cache_entry_bytes(index_dim, True)
        )
    mla = mla_cache.view(torch.uint8)
    index = _cache_bytes(index_cache, index_dim)
    topk = topk_indices_buffer.shape[-1]
    use_pdl = current_platform.is_arch_support_pdl()
    _mxfp4_fused_prologue_kernel[(num_tokens, 4)](
        q_c,
        q_c.stride(0),
        q_norm_weight,
        q_out,
        kv_c,
        kv_c.stride(0),
        kv_norm_weight,
        kv_out,
        k_pe,
        k_pe.stride(0),
        index_k,
        index_k.stride(0),
        k_norm_weight,
        k_norm_bias,
        eps,
        mla_slot_mapping,
        mla_slot_mapping.shape[0],
        mla.view(_FP8_DTYPE),
        mla.view(torch.float32),
        mla.view(torch.bfloat16),
        mla.shape[1],
        mla.stride(0),
        mla.stride(1),
        index_slot_mapping,
        index_slot_mapping.shape[0],
        index,
        index.shape[1],
        index.stride(0),
        topk_indices_buffer,
        topk_indices_buffer.stride(0),
        Q_DIM=q_dim,
        Q_BLOCK=triton.next_power_of_2(q_dim),
        KV_DIM=kv_dim,
        ROPE_DIM=rope_dim,
        INDEX_DIM=index_dim,
        TOPK=topk,
        TOPK_BLOCK=min(1024, triton.next_power_of_2(topk)),
        USE_PDL=use_pdl,
        FP8_MAX=_FP8_MAX,
        USE_FNUZ=_USE_FNUZ,
        launch_pdl=use_pdl,
    )
    return q_out, kv_out
