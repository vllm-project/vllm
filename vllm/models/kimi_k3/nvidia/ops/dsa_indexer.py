# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused Triton prologues for the Kimi K3.1 DSA layer, replacing the small
launch-bound kernels between the projection GEMMs and `sparse_attn_indexer`.

The quant helpers and `dsa_fused_prologue` follow `fused_norm_rope` in
`deepseek_v32/common/kernels.py` (without RoPE); they are copied so this
model does not depend on another model's private kernels.
"""

import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.deep_gemm import mqa_logits_use_mx_scales
from vllm.v1.attention.ops.common import pack_ue8m0_x4

_FP8_DTYPE = current_platform.fp8_dtype()
_USE_FNUZ = torch.float8_e4m3fnuz == _FP8_DTYPE
_FP8_MAX = 224.0 if _USE_FNUZ else 448.0


@triton.jit
def _ue8m0_scale(amax, AMAX_FLOOR: tl.constexpr, FP8_MAX: tl.constexpr):
    scale = tl.div_rn(tl.maximum(amax, AMAX_FLOOR), FP8_MAX)
    return tl.math.exp2(tl.math.ceil(tl.math.log2(scale)))


@triton.jit
def _fill_topk(row_ptr, TOPK: tl.constexpr, TOPK_BLOCK: tl.constexpr):
    for i in range(0, TOPK, TOPK_BLOCK):
        offs = i + tl.arange(0, TOPK_BLOCK)
        tl.store(row_ptr + offs, -1, mask=offs < TOPK)


@triton.jit
def _index_k_insert(
    tok,
    k_ptr,
    norm_w_ptr,
    norm_b_ptr,
    norm_eps,
    slot_mapping_ptr,
    num_slots,
    cache_ptr,
    cache_scale_ptr,
    cache_block_size,
    cache_block_stride,
    HEAD_DIM: tl.constexpr,
    MX_SCALE: tl.constexpr,
    FP8_MAX: tl.constexpr,
    USE_FNUZ: tl.constexpr,
):
    # Padded rows and (under DCP) tokens owned by another rank have no slot.
    slot = tl.load(slot_mapping_ptr + tok, mask=tok < num_slots, other=-1)
    if slot >= 0:
        offs = tl.arange(0, HEAD_DIM)
        x = tl.load(k_ptr + offs).to(tl.float32)
        mean = tl.sum(x, axis=0) / HEAD_DIM
        diff = x - mean
        rstd = tl.rsqrt(tl.sum(diff * diff, axis=0) / HEAD_DIM + norm_eps)
        w = tl.load(norm_w_ptr + offs).to(tl.float32)
        b = tl.load(norm_b_ptr + offs).to(tl.float32)
        # Round to bf16 like the unfused path so the cached bytes match.
        y = (diff * rstd * w + b).to(k_ptr.dtype.element_ty).to(tl.float32)

        # Same math as the CUDA indexer_k_quant_and_cache kernel.
        scale = _ue8m0_scale(tl.max(tl.abs(y), axis=0), 1e-4, FP8_MAX)
        fp8_dtype = tl.float8e4b8 if USE_FNUZ else tl.float8e4nv
        y_fp8 = tl.div_rn(y, scale).to(fp8_dtype)

        # Block layout: [block_size x HEAD_DIM fp8 values | block_size x 4B scale].
        block_start = (slot // cache_block_size) * cache_block_stride
        block_off = slot % cache_block_size
        tl.store(cache_ptr + block_start + block_off * HEAD_DIM + offs, y_fp8)
        scale_byte = block_start + cache_block_size * HEAD_DIM + block_off * 4
        if MX_SCALE:
            scale = pack_ue8m0_x4(scale)
        tl.store(cache_scale_ptr + scale_byte // 4, scale)


@triton.jit
def _indexer_k_norm_quant_insert_kernel(
    k_ptr,
    k_stride,
    norm_w_ptr,
    norm_b_ptr,
    norm_eps,
    slot_mapping_ptr,
    num_slots,
    cache_ptr,
    cache_scale_ptr,
    cache_block_size,
    cache_block_stride,
    topk_ptr,
    topk_stride,
    HEAD_DIM: tl.constexpr,
    TOPK: tl.constexpr,
    TOPK_BLOCK: tl.constexpr,
    MX_SCALE: tl.constexpr,
    FP8_MAX: tl.constexpr,
    USE_FNUZ: tl.constexpr,
):
    tok = tl.program_id(0).to(tl.int64)
    if tl.program_id(1) == 1:
        # Padding rows included.
        _fill_topk(topk_ptr + tok * topk_stride, TOPK, TOPK_BLOCK)
    else:
        _index_k_insert(
            tok,
            k_ptr + tok * k_stride,
            norm_w_ptr,
            norm_b_ptr,
            norm_eps,
            slot_mapping_ptr,
            num_slots,
            cache_ptr,
            cache_scale_ptr,
            cache_block_size,
            cache_block_stride,
            HEAD_DIM,
            MX_SCALE,
            FP8_MAX,
            USE_FNUZ,
        )


def indexer_k_norm_quant_insert(
    k: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_bias: torch.Tensor,
    norm_eps: float,
    slot_mapping: torch.Tensor,
    kv_cache: torch.Tensor,
    topk_indices_buffer: torch.Tensor,
    num_tokens: int,
) -> None:
    """LayerNorm + FP8 quant + cache insert of the indexer key, and the -1
    fill of the top-k rows `[:num_tokens]`.

    `kv_cache` is `[num_blocks, block_size, head_dim + 4]` uint8, possibly
    a strided page view of a packed cache. Rows past `slot_mapping` or with
    a negative slot are not inserted.
    """
    if num_tokens == 0:
        return
    head_dim = k.shape[-1]
    assert k.stride(-1) == 1
    assert kv_cache.dim() == 3 and kv_cache.shape[-1] == head_dim + 4
    assert triton.next_power_of_2(head_dim) == head_dim
    mx_scale = mqa_logits_use_mx_scales()
    u8_cache = kv_cache.view(torch.uint8)
    topk = topk_indices_buffer.shape[-1]
    _indexer_k_norm_quant_insert_kernel[(num_tokens, 2)](
        k,
        k.stride(0),
        norm_weight,
        norm_bias,
        norm_eps,
        slot_mapping,
        slot_mapping.shape[0],
        u8_cache.view(_FP8_DTYPE),
        u8_cache.view(torch.int32 if mx_scale else torch.float32),
        u8_cache.shape[1],
        u8_cache.stride(0),
        topk_indices_buffer,
        topk_indices_buffer.stride(0),
        HEAD_DIM=head_dim,
        TOPK=topk,
        TOPK_BLOCK=min(1024, triton.next_power_of_2(topk)),
        MX_SCALE=mx_scale,
        FP8_MAX=_FP8_MAX,
        USE_FNUZ=_USE_FNUZ,
        num_warps=1,
    )


@triton.jit
def _indexer_q_quant_fold_weights_kernel(
    q_ptr,
    q_stride0,
    q_stride1,
    w_ptr,
    w_stride,
    q_fp8_ptr,
    q_fp8_stride0,
    q_fp8_stride1,
    w_out_ptr,
    w_out_stride,
    weight_scale,
    HEAD_DIM: tl.constexpr,
    FP8_MAX: tl.constexpr,
    USE_FNUZ: tl.constexpr,
):
    tok = tl.program_id(0).to(tl.int64)
    head = tl.program_id(1)
    offs = tl.arange(0, HEAD_DIM)
    q = tl.load(q_ptr + tok * q_stride0 + head * q_stride1 + offs).to(tl.float32)
    # Same math as per_token_group_quant_fp8(use_ue8m0=True) (eps=1e-10).
    scale = _ue8m0_scale(tl.max(tl.abs(q), axis=0), 1e-10, FP8_MAX)
    fp8_dtype = tl.float8e4b8 if USE_FNUZ else tl.float8e4nv
    q_fp8 = tl.clamp(tl.div_rn(q, scale), -FP8_MAX, FP8_MAX).to(fp8_dtype)
    tl.store(q_fp8_ptr + tok * q_fp8_stride0 + head * q_fp8_stride1 + offs, q_fp8)
    w = tl.load(w_ptr + tok * w_stride + head).to(tl.float32)
    tl.store(w_out_ptr + tok * w_out_stride + head, w * scale * weight_scale)


def indexer_q_quant_fold_weights(
    index_q: torch.Tensor,
    index_weights: torch.Tensor,
    weight_scale: float,
    padded_num_heads: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize `index_q` `[T, H, D]` to FP8 per head (ue8m0 scales) and fold each
    head's scale into `index_weights` `[T, H]` together with `weight_scale`.

    Returns `(q_fp8 [T, P, D], weights [T, P] fp32)` where `P =
    padded_num_heads`; heads `[H, P)` are zero.
    """
    num_tokens, num_heads, head_dim = index_q.shape
    assert index_q.stride(-1) == 1
    assert index_weights.shape == (num_tokens, num_heads)
    assert triton.next_power_of_2(head_dim) == head_dim
    alloc = torch.empty if padded_num_heads == num_heads else torch.zeros
    q_fp8 = alloc(
        (num_tokens, padded_num_heads, head_dim),
        dtype=_FP8_DTYPE,
        device=index_q.device,
    )
    weights = alloc(
        (num_tokens, padded_num_heads), dtype=torch.float32, device=index_q.device
    )
    if num_tokens == 0:
        return q_fp8, weights
    _indexer_q_quant_fold_weights_kernel[(num_tokens, num_heads)](
        index_q,
        index_q.stride(0),
        index_q.stride(1),
        index_weights,
        index_weights.stride(0),
        q_fp8,
        q_fp8.stride(0),
        q_fp8.stride(1),
        weights,
        weights.stride(0),
        weight_scale,
        HEAD_DIM=head_dim,
        FP8_MAX=_FP8_MAX,
        USE_FNUZ=_USE_FNUZ,
        num_warps=1,
    )
    return q_fp8, weights


@triton.jit
def _rms_norm_row(x_ptr, w_ptr, eps, SIZE: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs, mask=offs < SIZE, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + offs, mask=offs < SIZE, other=0.0).to(tl.float32)
    return x * tl.rsqrt(tl.sum(x * x, axis=0) / SIZE + eps) * w


@triton.jit
def _dsa_fused_prologue_kernel(
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
    index_scale_ptr,
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
    MX_SCALE: tl.constexpr,
    USE_PDL: tl.constexpr,
    FP8_MAX: tl.constexpr,
    USE_FNUZ: tl.constexpr,
):
    tok = tl.program_id(0).to(tl.int64)
    pid = tl.program_id(1)
    if USE_PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()

    if pid == 0:
        _index_k_insert(
            tok,
            index_k_ptr + tok * index_k_stride,
            k_norm_w_ptr,
            k_norm_b_ptr,
            eps,
            index_slot_ptr,
            num_index_slots,
            index_cache_ptr,
            index_scale_ptr,
            index_block_size,
            index_block_stride,
            INDEX_DIM,
            MX_SCALE,
            FP8_MAX,
            USE_FNUZ,
        )
    elif pid == 1:
        # The normed row is written for every token (the prefill path reads it).
        offs = tl.arange(0, KV_DIM)
        y = _rms_norm_row(
            kv_c_ptr + tok * kv_c_stride, kv_norm_w_ptr, eps, KV_DIM, KV_DIM
        )
        y = y.to(kv_out_ptr.dtype.element_ty)
        tl.store(kv_out_ptr + tok * KV_DIM + offs, y)
        slot = tl.load(mla_slot_ptr + tok, mask=tok < num_mla_slots, other=-1)
        if slot >= 0:
            # fp8_ds_mla, same math as concat_and_cache_ds_mla: KV_DIM fp8 values
            # with power-of-two per-128 tile scales (fp32), then k_pe in bf16.
            base = (slot // mla_block_size) * mla_block_stride + (
                slot % mla_block_size
            ) * mla_entry_stride
            NUM_TILES: tl.constexpr = KV_DIM // 128
            tiles = tl.reshape(y.to(tl.float32), (NUM_TILES, 128))
            amax = tl.max(tl.abs(tiles), axis=1, keep_dims=True)
            scale = tl.math.exp2(
                tl.math.ceil(tl.math.log2(tl.maximum(amax / FP8_MAX, 1e-4)))
            )
            fp8_dtype = tl.float8e4b8 if USE_FNUZ else tl.float8e4nv
            tl.store(
                mla_cache_ptr + base + offs,
                tl.reshape((tiles / scale).to(fp8_dtype), (KV_DIM,)),
            )
            tl.store(
                mla_scale_ptr + base // 4 + KV_DIM // 4 + tl.arange(0, NUM_TILES),
                tl.reshape(scale, (NUM_TILES,)),
            )
            rope = tl.arange(0, ROPE_DIM)
            k_pe = tl.load(k_pe_ptr + tok * k_pe_stride + rope)
            tl.store(
                mla_rope_ptr + base // 2 + KV_DIM // 2 + 2 * NUM_TILES + rope, k_pe
            )
    elif pid == 2:
        offs = tl.arange(0, Q_BLOCK)
        y = _rms_norm_row(q_c_ptr + tok * q_c_stride, q_norm_w_ptr, eps, Q_DIM, Q_BLOCK)
        tl.store(
            q_out_ptr + tok * Q_DIM + offs,
            y.to(q_out_ptr.dtype.element_ty),
            mask=offs < Q_DIM,
        )
    else:
        # Padding rows included.
        _fill_topk(topk_ptr + tok * topk_stride, TOPK, TOPK_BLOCK)


def dsa_fused_prologue(
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
    """`fused_norm_rope` for the NoPE Kimi K3.1 DSA layer: the q_a / kv_a
    RMSNorms, the fp8_ds_mla insert of `[kv_c_normed | k_pe]`, the indexer key
    LayerNorm + FP8 insert and the -1 fill of the top-k rows, in one launch.

    `mla_cache` (`[num_blocks, block_size, 656]`) and `index_cache`
    (`[num_blocks, block_size, head_dim + 4]`) are uint8, possibly strided
    page views of a packed cache; `None` (profile runs) skips the insert. Rows
    past a slot mapping or with a negative slot are not inserted.

    Returns `(q_c_normed, kv_c_normed)`.
    """
    num_tokens, q_dim = q_c.shape
    kv_dim, rope_dim, index_dim = kv_c.shape[1], k_pe.shape[1], index_k.shape[1]
    assert all(t.stride(-1) == 1 for t in (q_c, kv_c, k_pe, index_k))
    q_out = torch.empty((num_tokens, q_dim), dtype=q_c.dtype, device=q_c.device)
    kv_out = torch.empty((num_tokens, kv_dim), dtype=kv_c.dtype, device=kv_c.device)
    if num_tokens == 0:
        return q_out, kv_out

    # No cache (profile runs): an empty slot mapping disables the insert.
    def placeholder() -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.empty(1, 1, 4, dtype=torch.uint8, device=q_c.device),
            torch.empty(0, dtype=torch.int64, device=q_c.device),
        )

    if mla_cache is None or mla_slot_mapping is None:
        mla_cache, mla_slot_mapping = placeholder()
    if index_cache is None or index_slot_mapping is None:
        index_cache, index_slot_mapping = placeholder()
    mla = mla_cache.view(torch.uint8)
    index = index_cache.view(torch.uint8)
    mx_scale = mqa_logits_use_mx_scales()
    topk = topk_indices_buffer.shape[-1]
    use_pdl = current_platform.is_arch_support_pdl()
    _dsa_fused_prologue_kernel[(num_tokens, 4)](
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
        index.view(_FP8_DTYPE),
        index.view(torch.int32 if mx_scale else torch.float32),
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
        MX_SCALE=mx_scale,
        USE_PDL=use_pdl,
        FP8_MAX=_FP8_MAX,
        USE_FNUZ=_USE_FNUZ,
        launch_pdl=use_pdl,
    )
    return q_out, kv_out
