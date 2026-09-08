# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
from vllm.model_executor.warmup.jit_warmup import kernel_launcher
from vllm.model_executor.warmup.jit_warmup_triton_helper import (
    LaunchSpec,
    TritonWarmupTensor,
    VllmTritonJitKernel,
)
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import is_quantized_kv_cache

# Cache of tiny 1-element dummy tensors (per device, dtype) reused by the
# has_indexer=False path so the indexer args don't allocate every call.
_DUMMY_CACHE: dict[tuple, torch.Tensor] = {}


def _dummy(shape: tuple, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    key = (shape, dtype, device)
    t = _DUMMY_CACHE.get(key)
    if t is None:
        t = torch.empty(shape, dtype=dtype, device=device)
        _DUMMY_CACHE[key] = t
    return t


@triton.jit
def _rms_norm(x, w, eps, HIDDEN_SIZE: tl.constexpr):
    x = x.to(tl.float32)
    mean_sq = tl.sum(x * x, axis=0) / HIDDEN_SIZE
    rrms = tl.rsqrt(mean_sq + eps)
    w = w.to(tl.float32)
    return (x * rrms) * w


@triton.jit
def _get_cos_sin(
    cos_sin_cache_ptr,
    cos_sin_cache_stride,
    pos,
    HALF_ROT_DIM: tl.constexpr,
):
    block = tl.arange(0, HALF_ROT_DIM)
    cos = tl.load(cos_sin_cache_ptr + pos * cos_sin_cache_stride + block)
    cos = cos.to(tl.float32)
    sin = tl.load(cos_sin_cache_ptr + pos * cos_sin_cache_stride + block + HALF_ROT_DIM)
    sin = sin.to(tl.float32)
    return cos, sin


@triton.jit
def _fp8_ue8m0_quantize(vals):
    """Quantize float32 values to FP8 E4M3 with a ue8m0 (power-of-2) scale.

    Returns (fp8_vals, scale) so the caller can store them or reuse the scale.
    """
    vals = vals.to(tl.float32)
    amax = tl.max(tl.abs(vals))
    scale = tl.div_rn(tl.maximum(amax, 1e-4), 448.0)
    scale = tl.math.exp2(tl.math.ceil(tl.math.log2(scale)))
    fp8_vals = tl.div_rn(vals, scale).to(tl.float8e4nv)
    return fp8_vals, scale


@triton.jit
def _fp8_quant_and_cache_write(
    vals,
    mask,
    slot_idx,
    kv_cache_ptr,
    kv_cache_scale_ptr,
    cache_block_size,
    cache_stride,
    offsets,
    HEAD_DIM: tl.constexpr,
):
    k_fp8, scale = _fp8_ue8m0_quantize(vals)

    block_idx = slot_idx // cache_block_size
    block_offset = slot_idx % cache_block_size
    block_start = block_idx * cache_block_size * cache_stride

    tl.store(
        kv_cache_ptr + block_start + block_offset * HEAD_DIM + offsets,
        k_fp8,
        mask=mask,
    )
    scale_byte_off = block_start + cache_block_size * HEAD_DIM + block_offset * 4
    tl.store(kv_cache_scale_ptr + scale_byte_off // 4, scale)


@triton.jit
def _fused_norm_rope_kernel(
    pos_ptr,
    # Q RMS norm
    q_c_ptr,
    q_c_stride,
    q_rms_norm_w_ptr,
    q_rms_eps,
    q_c_out_ptr,
    q_c_out_stride,
    Q_DIM: tl.constexpr,
    Q_BLOCK_SIZE: tl.constexpr,
    # KV RMS norm
    kv_ptr,
    kv_stride,
    kv_rms_norm_w_ptr,
    kv_rms_eps,
    kv_out_ptr,
    kv_out_stride,
    KV_DIM: tl.constexpr,
    # KV RoPE
    kpe_ptr,
    kpe_stride,
    kpe_rope_cos_sin_cache_ptr,
    kpe_rope_cos_sin_cache_stride,
    kpe_out_ptr,
    kpe_out_stride,
    KPE_HALF_ROT_DIM: tl.constexpr,
    # Index K layer norm
    index_k_ptr,
    index_k_stride,
    index_k_layer_norm_w_ptr,
    index_k_layer_norm_bias_ptr,
    index_k_layer_norm_eps,
    INDEX_K_DIM: tl.constexpr,
    INDEX_K_BLOCK_SIZE: tl.constexpr,
    # Index K RoPE
    index_k_rope_cos_sin_cache_ptr,
    index_k_rope_cos_sin_cache_stride,
    index_k_out_ptr,
    index_k_out_stride,
    INDEX_K_HALF_ROT_DIM: tl.constexpr,
    # Cache params (shared by indexer K and MLA)
    slot_mapping_ptr,
    # Index K FP8 cache
    indexer_cache_ptr,
    indexer_cache_scale_ptr,
    indexer_cache_block_size,
    indexer_cache_stride,
    # MLA KV cache (concat kv_c_normed + k_pe_roped, uses slot_mapping_ptr)
    mla_cache_ptr,
    mla_cache_block_stride,
    mla_cache_entry_stride,
    MLA_CACHE_BLOCK_SIZE: tl.constexpr,
    MLA_CACHE_FP8: tl.constexpr,
    mla_cache_scale_ptr,
    # fp8_ds_mla cache views (block-scaled fp8 NoPE + unquantized bf16 RoPE).
    # mla_cache_ptr is the fp8 (1-byte) view, so the block/entry strides above
    # are byte offsets; these two share the same buffer as fp32 / bf16 views.
    mla_cache_ds_scale_ptr,
    mla_cache_ds_rope_ptr,
    MLA_CACHE_DS_MLA: tl.constexpr,
    MLA_NUM_TILES: tl.constexpr,
    MLA_TILE_DIM: tl.constexpr,
    # Top k indices
    topk_indices_ptr,
    topk_indices_stride,
    TOPK: tl.constexpr,
    TOPK_BLOCK_SIZE: tl.constexpr,
    HAS_INDEXER: tl.constexpr,
    INDEX_ROPE_INTERLEAVE: tl.constexpr,
    USE_PDL: tl.constexpr,
):
    tok_idx = tl.program_id(0).to(tl.int64)
    pid = tl.program_id(1)
    if USE_PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    if pid == 3:
        if not HAS_INDEXER:
            # Shared layer: reuse the previous indexer layer's top-k; do not
            # clear the buffer.
            return
        # Fill top k indices buffer with -1
        for i in range(0, TOPK, TOPK_BLOCK_SIZE):
            offset = i + tl.arange(0, TOPK_BLOCK_SIZE)
            mask = offset < TOPK
            tl.store(
                topk_indices_ptr + tok_idx * topk_indices_stride + offset,
                -1,
                mask=mask,
            )
        return

    if pid == 2:
        # Q RMS norm. Runs for every row: under DCP a negative slot only
        # means another rank owns this token's KV slot, but the query is
        # still needed on every rank (queries are not sharded), so the
        # slot-based skip below must not gate it. Padding rows do harmless
        # row-local extra work (no position load, no cache write).
        q_block = tl.arange(0, Q_BLOCK_SIZE)
        q_mask = q_block < Q_DIM
        q_c = tl.load(q_c_ptr + tok_idx * q_c_stride + q_block, mask=q_mask, other=0.0)
        q_c_rms_w = tl.load(q_rms_norm_w_ptr + q_block, mask=q_mask)
        q_c = _rms_norm(q_c, q_c_rms_w, q_rms_eps, Q_DIM)
        tl.store(q_c_out_ptr + tok_idx * q_c_out_stride + q_block, q_c, mask=q_mask)
        return

    if slot_mapping_ptr is None:
        if kv_out_ptr is None and kpe_out_ptr is None and index_k_out_ptr is None:
            return
    elif tl.load(slot_mapping_ptr + tok_idx) < 0 and (
        pid != 1 or (kv_out_ptr is None and kpe_out_ptr is None)
    ):
        # Padding, or (under DCP) a token whose KV slot another rank owns.
        # Dense prefill still consumes the materialized normalized/rotated K
        # rows on every rank, so pid 1 must produce them even when this rank
        # must not write the owner-local cache slot.
        return

    if pid == 1:
        # KV RMS Norm + KV RoPE + MLA concat_and_cache.
        # Merged so the normed kv_c and RoPE'd k_pe can be written
        # to the MLA KV cache directly without a separate kernel.

        # KV RMS Norm (result stays in registers for MLA cache write)
        kv_block = tl.arange(0, KV_DIM)
        kv_c = tl.load(kv_ptr + tok_idx * kv_stride + kv_block)
        kv_c_rms_w = tl.load(kv_rms_norm_w_ptr + kv_block)
        kv_c = _rms_norm(kv_c, kv_c_rms_w, kv_rms_eps, KV_DIM)

        # KV RoPE (interleaved) on k_pe — in registers only.
        # k_pe is not needed after the cache write (MLA decode reads
        # from kv_cache), so we skip writing back to kpe_ptr.
        pos = tl.load(pos_ptr + tok_idx)
        cos, sin = _get_cos_sin(
            kpe_rope_cos_sin_cache_ptr,
            kpe_rope_cos_sin_cache_stride,
            pos,
            KPE_HALF_ROT_DIM,
        )
        dim_off = tl.arange(0, KPE_HALF_ROT_DIM)
        kpe_base = kpe_ptr + tok_idx * kpe_stride
        x1 = tl.load(kpe_base + dim_off * 2).to(tl.float32)
        x2 = tl.load(kpe_base + dim_off * 2 + 1).to(tl.float32)
        r1 = x1 * cos - x2 * sin
        r2 = x2 * cos + x1 * sin

        # PCP materializes K for the cross-rank gather, then inserts it into cache.
        if kv_out_ptr is not None:
            tl.store(kv_out_ptr + tok_idx * kv_out_stride + kv_block, kv_c)
        if kpe_out_ptr is not None:
            tl.store(
                kpe_out_ptr + tok_idx * kpe_out_stride + dim_off * 2,
                r1.to(kpe_out_ptr.dtype.element_ty),
            )
            tl.store(
                kpe_out_ptr + tok_idx * kpe_out_stride + dim_off * 2 + 1,
                r2.to(kpe_out_ptr.dtype.element_ty),
            )

        if slot_mapping_ptr is not None:
            # MLA concat_and_cache: write [kv_c_normed, k_pe_roped] to cache.
            if mla_cache_entry_stride == 0:
                return

            slot_idx = tl.load(slot_mapping_ptr + tok_idx)
            if slot_idx < 0:
                return
            mla_block_size = MLA_CACHE_BLOCK_SIZE
            mla_block_idx = slot_idx // mla_block_size
            mla_block_off = slot_idx % mla_block_size

            if MLA_CACHE_DS_MLA:
                # fp8_ds_mla layout (DeepSeek-V3.2, KV_DIM == 512): per-128-element
                # tile of the NoPE is dynamically quantized to fp8 with its own
                # float32 scale; the RoPE tail is stored unquantized in bf16.
                #   bytes [0, KV_DIM)            : KV_DIM fp8 NoPE values
                #   bytes [KV_DIM, KV_DIM + 16)  : MLA_NUM_TILES float32 scales
                #   bytes [KV_DIM + 16, ...)     : 2 * KPE_HALF_ROT_DIM bf16 RoPE
                # mla_cache_block_stride / mla_cache_entry_stride are byte strides
                # (mla_cache_ptr is the 1-byte fp8 view of the uint8 cache).
                byte_base = (
                    mla_block_idx * mla_cache_block_stride
                    + mla_block_off * mla_cache_entry_stride
                )
                kv_2d = tl.reshape(kv_c, (MLA_NUM_TILES, MLA_TILE_DIM))
                tile_amax = tl.max(tl.abs(kv_2d), axis=1, keep_dims=True)
                # scale = amax / 448 (fp8 e4m3 max), matching the reference
                # concat_and_cache_ds_mla kernel; floored to FLT_MIN.
                tile_scale = tl.maximum(tile_amax * (1.0 / 448.0), 1.1754944e-38)
                kv_c_fp8 = tl.reshape((kv_2d / tile_scale).to(tl.float8e4nv), (KV_DIM,))
                tl.store(mla_cache_ptr + byte_base + kv_block, kv_c_fp8)
                tile_off = tl.arange(0, MLA_NUM_TILES)
                tl.store(
                    mla_cache_ds_scale_ptr + byte_base // 4 + KV_DIM // 4 + tile_off,
                    tl.reshape(tile_scale, (MLA_NUM_TILES,)),
                )
                rope_dst = mla_cache_ds_rope_ptr + byte_base // 2 + (KV_DIM // 2 + 8)
                tl.store(rope_dst + dim_off * 2, r1.to(tl.bfloat16))
                tl.store(rope_dst + dim_off * 2 + 1, r2.to(tl.bfloat16))
                return

            dst = (
                mla_cache_ptr
                + mla_block_idx * mla_cache_block_stride
                + mla_block_off * mla_cache_entry_stride
            )
            # kv_c_normed (KV_DIM elements)
            if MLA_CACHE_FP8:
                scale = tl.load(mla_cache_scale_ptr)
                kv_c_fp8 = (kv_c.to(tl.float32) / scale).to(tl.float8e4nv)
                tl.store(dst + kv_block, kv_c_fp8)
            else:
                tl.store(dst + kv_block, kv_c)
            # k_pe_roped (from registers, interleaved layout)
            if MLA_CACHE_FP8:
                tl.store(dst + KV_DIM + dim_off * 2, (r1 / scale).to(tl.float8e4nv))
                tl.store(
                    dst + KV_DIM + dim_off * 2 + 1,
                    (r2 / scale).to(tl.float8e4nv),
                )
            else:
                tl.store(dst + KV_DIM + dim_off * 2, r1)
                tl.store(dst + KV_DIM + dim_off * 2 + 1, r2)
    elif pid == 0:
        if not HAS_INDEXER:
            # Shared layer: no indexer K to process.
            return
        # Fused: Index K LayerNorm + RoPE + FP8 quant + cache write.
        # Eliminates the separate indexer_k_quant_and_cache kernel launch.

        index_k_block = tl.arange(0, INDEX_K_BLOCK_SIZE)
        index_k_mask = index_k_block < INDEX_K_DIM
        index_k = tl.load(
            index_k_ptr + tok_idx * index_k_stride + index_k_block,
            mask=index_k_mask,
            other=0.0,
        ).to(tl.float32)
        index_k_w = tl.load(
            index_k_layer_norm_w_ptr + index_k_block, mask=index_k_mask
        ).to(tl.float32)
        index_k_b = tl.load(
            index_k_layer_norm_bias_ptr + index_k_block, mask=index_k_mask
        ).to(tl.float32)

        # 1. LayerNorm. Keep (mean, rstd) so the RoPE rotation partner can be
        #    re-normalized in registers below, avoiding a global scratch buffer.
        mean = tl.sum(index_k, axis=0) / INDEX_K_DIM
        diff = tl.where(index_k_mask, index_k - mean, 0.0)
        var = tl.sum(diff * diff, axis=0) / INDEX_K_DIM
        rstd = tl.rsqrt(var + index_k_layer_norm_eps)
        normed = (index_k - mean) * rstd * index_k_w + index_k_b

        # 2. RoPE on the rotation region. Supports both interleaved (adjacent
        #    pairs, e.g. GLM-5.2) and NeoX (split-half, e.g. DeepSeek-V3.2). The
        #    rotation partner is gathered from the read-only inputs and
        #    re-normalized with the same (mean, rstd) — no scratch, no atomics.
        pos = tl.load(pos_ptr + tok_idx)
        in_rope = index_k_block < 2 * INDEX_K_HALF_ROT_DIM
        if INDEX_ROPE_INTERLEAVE:
            # pair i = block // 2; partner = block ^ 1; even -> -sin, odd -> +sin.
            cos_idx = index_k_block // 2
            partner_offs = tl.where(in_rope, index_k_block ^ 1, index_k_block)
            sign = tl.where(index_k_block % 2 == 0, -1.0, 1.0)
        else:
            # NeoX: pair across halves; partner = block ^ HALF.
            cos_idx = index_k_block % INDEX_K_HALF_ROT_DIM
            partner_offs = tl.where(
                in_rope, index_k_block ^ INDEX_K_HALF_ROT_DIM, index_k_block
            )
            sign = tl.where(index_k_block < INDEX_K_HALF_ROT_DIM, -1.0, 1.0)
        cos_full = tl.load(
            index_k_rope_cos_sin_cache_ptr
            + pos * index_k_rope_cos_sin_cache_stride
            + cos_idx,
            mask=in_rope,
            other=1.0,
        ).to(tl.float32)
        sin_full = tl.load(
            index_k_rope_cos_sin_cache_ptr
            + pos * index_k_rope_cos_sin_cache_stride
            + INDEX_K_HALF_ROT_DIM
            + cos_idx,
            mask=in_rope,
            other=0.0,
        ).to(tl.float32)
        # normed[partner_offs] == (raw_partner - mean) * rstd * w_partner +
        # b_partner: gather the raw partner and its norm affine (read-only
        # loads), then apply the same per-token mean/rstd.
        raw_partner = tl.load(
            index_k_ptr + tok_idx * index_k_stride + partner_offs,
            mask=index_k_mask,
            other=0.0,
        ).to(tl.float32)
        w_partner = tl.load(
            index_k_layer_norm_w_ptr + partner_offs, mask=index_k_mask
        ).to(tl.float32)
        b_partner = tl.load(
            index_k_layer_norm_bias_ptr + partner_offs, mask=index_k_mask
        ).to(tl.float32)
        normed_partner = (raw_partner - mean) * rstd * w_partner + b_partner
        roped = normed * cos_full + sign * normed_partner * sin_full
        result = tl.where(in_rope, roped, normed)

        if index_k_out_ptr is not None:
            tl.store(
                index_k_out_ptr + tok_idx * index_k_out_stride + index_k_block,
                result.to(index_k_out_ptr.dtype.element_ty),
                mask=index_k_mask,
            )

        # PCP inserts index K after gathering; other paths write it directly.
        if indexer_cache_ptr is not None and slot_mapping_ptr is not None:
            slot_idx = tl.load(slot_mapping_ptr + tok_idx)
            _fp8_quant_and_cache_write(
                result,
                index_k_mask,
                slot_idx,
                indexer_cache_ptr,
                indexer_cache_scale_ptr,
                indexer_cache_block_size,
                indexer_cache_stride,
                index_k_block,
                INDEX_K_DIM,
            )


class FusedNormRopeKernel(VllmTritonJitKernel["FusedNormRopeKernel.CompileKey"]):
    """Warmup owner for fused Q/KV norm, RoPE, and cache insertion."""

    kernel = staticmethod(_fused_norm_rope_kernel)
    topk_block_size = 1024

    @dataclass(frozen=True)
    class CompileKey:
        q_dim: int
        q_block_size: int
        kv_dim: int
        kpe_half_rot_dim: int
        index_k_dim: int
        index_k_block_size: int
        index_k_half_rot_dim: int
        mla_cache_fp8: bool
        mla_cache_ds_mla: bool
        mla_num_tiles: int
        mla_tile_dim: int
        topk: int
        has_indexer: bool
        index_rope_interleave: bool
        use_pdl: bool
        slot_mapping_present: bool
        kv_out_present: bool
        kpe_out_present: bool
        index_k_out_present: bool
        indexer_cache_present: bool
        block_size: int
        act_dtype: torch.dtype
        cos_sin_dtype: torch.dtype
        topk_dtype: torch.dtype

    def dispatch(  # type: ignore[override]
        self,
        *,
        q_lora_rank: int,
        kv_lora_rank: int,
        qk_rope_head_dim: int,
        index_head_dim: int,
        topk: int,
        use_pcp: bool,
        has_indexer: bool,
        index_rope_interleave: bool,
        use_pdl: bool,
        mla_kv_cache_dtype: str,
        block_size: int,
        act_dtype: torch.dtype,
        cos_sin_dtype: torch.dtype,
        topk_dtype: torch.dtype,
        has_cache: bool,
    ) -> "FusedNormRopeKernel.CompileKey":
        half_rot = qk_rope_head_dim // 2
        mla_present = has_cache and not use_pcp
        ds_mla = mla_kv_cache_dtype == "fp8_ds_mla"
        mla_fp8 = is_quantized_kv_cache(mla_kv_cache_dtype) and not ds_mla
        mla_num_tiles = kv_lora_rank // 128 if (mla_present and ds_mla) else 1
        mla_tile_dim = kv_lora_rank // mla_num_tiles if ds_mla else 1
        return self.CompileKey(
            q_dim=q_lora_rank,
            q_block_size=triton.next_power_of_2(q_lora_rank),
            kv_dim=kv_lora_rank,
            kpe_half_rot_dim=half_rot,
            index_k_dim=index_head_dim if has_indexer else 1,
            index_k_block_size=triton.next_power_of_2(index_head_dim)
            if has_indexer
            else 1,
            index_k_half_rot_dim=half_rot,
            mla_cache_fp8=mla_fp8,
            mla_cache_ds_mla=ds_mla,
            mla_num_tiles=mla_num_tiles,
            mla_tile_dim=mla_tile_dim,
            topk=topk,
            has_indexer=has_indexer,
            index_rope_interleave=index_rope_interleave,
            use_pdl=use_pdl,
            slot_mapping_present=mla_present,
            kv_out_present=use_pcp,
            kpe_out_present=use_pcp,
            index_k_out_present=has_indexer and use_pcp,
            indexer_cache_present=has_indexer and mla_present,
            block_size=block_size,
            act_dtype=act_dtype,
            cos_sin_dtype=cos_sin_dtype,
            topk_dtype=topk_dtype,
        )

    def get_warmup_keys(  # type: ignore[override]
        self,
        *,
        q_lora_rank: int,
        kv_lora_rank: int,
        qk_rope_head_dim: int,
        index_head_dim: int,
        topk: int,
        use_pcp: bool,
        has_indexer: bool,
        index_rope_interleave: bool,
        use_pdl: bool,
        mla_kv_cache_dtype: str,
        block_size: int,
        act_dtype: torch.dtype,
        cos_sin_dtype: torch.dtype,
        topk_dtype: torch.dtype,
    ) -> "list[FusedNormRopeKernel.CompileKey]":
        return self._trace_dispatch(self.dispatch)(
            q_lora_rank=q_lora_rank,
            kv_lora_rank=kv_lora_rank,
            qk_rope_head_dim=qk_rope_head_dim,
            index_head_dim=index_head_dim,
            topk=topk,
            use_pcp=use_pcp,
            has_indexer=has_indexer,
            index_rope_interleave=index_rope_interleave,
            use_pdl=use_pdl,
            mla_kv_cache_dtype=mla_kv_cache_dtype,
            block_size=block_size,
            act_dtype=act_dtype,
            cos_sin_dtype=cos_sin_dtype,
            topk_dtype=topk_dtype,
            has_cache=(True, False),
        )

    def warmup_inputs(
        self, compile_key: "FusedNormRopeKernel.CompileKey"
    ) -> dict[str, Any]:
        k = compile_key
        qk_rope = 2 * k.kpe_half_rot_dim
        index_shape = (1, k.index_k_dim)
        activation = TritonWarmupTensor(k.act_dtype)
        q = TritonWarmupTensor(k.act_dtype, shape=(1, k.q_dim))
        kv = TritonWarmupTensor(k.act_dtype, shape=(1, k.kv_dim))
        kpe = TritonWarmupTensor(k.act_dtype, shape=(1, qk_rope))
        index_k = TritonWarmupTensor(k.act_dtype, shape=index_shape)
        cos_sin = TritonWarmupTensor(k.cos_sin_dtype, shape=(1, qk_rope))
        fp32 = TritonWarmupTensor(torch.float32)
        mla_fp8_view = k.slot_mapping_present and (
            k.mla_cache_ds_mla or k.mla_cache_fp8
        )
        idx_cache_entry = k.index_k_dim + k.index_k_dim // 128 * 4
        return dict(
            positions=TritonWarmupTensor(torch.int64),
            q_c=q,
            q_rms_norm_w=activation,
            q_rms_eps=0.0,
            q_c_out=q,
            kv_c=kv,
            kv_rms_norm_w=activation,
            kv_rms_eps=0.0,
            kv_c_out=kv if k.kv_out_present else None,
            k_pe=kpe,
            k_rope_cos_sin_cache=cos_sin,
            k_pe_out=kpe if k.kpe_out_present else None,
            index_k=index_k,
            index_k_layer_norm_w=fp32,
            index_k_layer_norm_bias=fp32,
            index_k_layer_norm_eps=0.0,
            index_k_rope_cos_sin_cache=cos_sin,
            index_k_out=index_k if k.index_k_out_present else None,
            slot_mapping=(
                TritonWarmupTensor(torch.int64) if k.slot_mapping_present else None
            ),
            indexer_k_cache=(
                TritonWarmupTensor(torch.float8_e4m3fn)
                if k.indexer_cache_present
                else None
            ),
            idx_cache_scale_view=fp32 if k.indexer_cache_present else None,
            idx_cache_block_size=k.block_size if k.indexer_cache_present else 1,
            idx_cache_stride=idx_cache_entry if k.indexer_cache_present else 0,
            mla_kv_cache=TritonWarmupTensor(
                torch.float8_e4m3fn if mla_fp8_view else torch.bfloat16
            ),
            mla_block_stride=k.kv_dim if k.slot_mapping_present else 0,
            mla_entry_stride=k.kv_dim if k.slot_mapping_present else 0,
            mla_cache_fp8=k.mla_cache_fp8,
            mla_k_scale=fp32,
            mla_ds_scale_view=fp32,
            mla_ds_rope_view=TritonWarmupTensor(torch.bfloat16),
            mla_cache_ds_mla=k.mla_cache_ds_mla,
            mla_num_tiles=k.mla_num_tiles,
            mla_tile_dim=k.mla_tile_dim,
            topk_indices_buffer=TritonWarmupTensor(
                k.topk_dtype,
                shape=(1, k.topk),
            ),
            has_indexer=k.has_indexer,
            index_rope_interleave=k.index_rope_interleave,
            use_pdl=k.use_pdl,
        )

    @kernel_launcher
    def __call__(
        self,
        positions: torch.Tensor,
        q_c: torch.Tensor,
        q_rms_norm_w: torch.Tensor,
        q_rms_eps: float,
        q_c_out: torch.Tensor,
        kv_c: torch.Tensor,
        kv_rms_norm_w: torch.Tensor,
        kv_rms_eps: float,
        kv_c_out: torch.Tensor | None,
        k_pe: torch.Tensor,
        k_rope_cos_sin_cache: torch.Tensor,
        k_pe_out: torch.Tensor | None,
        index_k: torch.Tensor,
        index_k_layer_norm_w: torch.Tensor,
        index_k_layer_norm_bias: torch.Tensor,
        index_k_layer_norm_eps: float,
        index_k_rope_cos_sin_cache: torch.Tensor,
        index_k_out: torch.Tensor | None,
        slot_mapping: torch.Tensor | None,
        indexer_k_cache: torch.Tensor | None,
        idx_cache_scale_view: torch.Tensor | None,
        idx_cache_block_size: int,
        idx_cache_stride: int,
        mla_kv_cache: torch.Tensor,
        mla_block_stride: int,
        mla_entry_stride: int,
        mla_cache_fp8: bool,
        mla_k_scale: torch.Tensor,
        mla_ds_scale_view: torch.Tensor,
        mla_ds_rope_view: torch.Tensor,
        mla_cache_ds_mla: bool,
        mla_num_tiles: int,
        mla_tile_dim: int,
        topk_indices_buffer: torch.Tensor,
        has_indexer: bool,
        index_rope_interleave: bool,
        use_pdl: bool,
    ) -> LaunchSpec:
        num_tokens = positions.shape[0]
        q_dim = q_c.shape[-1]
        kv_dim = kv_c.shape[-1]
        index_k_dim = index_k.shape[-1]
        topk = topk_indices_buffer.shape[-1]
        kv_c_out_stride = kv_c_out.stride(0) if kv_c_out is not None else 0
        k_pe_out_stride = k_pe_out.stride(0) if k_pe_out is not None else 0
        index_k_out_stride = index_k_out.stride(0) if index_k_out is not None else 0
        return (num_tokens, 4), dict(
            pos_ptr=positions,
            q_c_ptr=q_c,
            q_c_stride=q_c.stride(0),
            q_rms_norm_w_ptr=q_rms_norm_w,
            q_rms_eps=q_rms_eps,
            q_c_out_ptr=q_c_out,
            q_c_out_stride=q_c_out.stride(0),
            Q_DIM=q_dim,
            Q_BLOCK_SIZE=triton.next_power_of_2(q_dim),
            kv_ptr=kv_c,
            kv_stride=kv_c.stride(0),
            kv_rms_norm_w_ptr=kv_rms_norm_w,
            kv_rms_eps=kv_rms_eps,
            kv_out_ptr=kv_c_out,
            kv_out_stride=kv_c_out_stride,
            KV_DIM=kv_dim,
            kpe_ptr=k_pe,
            kpe_stride=k_pe.stride(0),
            kpe_rope_cos_sin_cache_ptr=k_rope_cos_sin_cache,
            kpe_rope_cos_sin_cache_stride=k_rope_cos_sin_cache.stride(0),
            kpe_out_ptr=k_pe_out,
            kpe_out_stride=k_pe_out_stride,
            KPE_HALF_ROT_DIM=k_rope_cos_sin_cache.shape[-1] // 2,
            index_k_ptr=index_k,
            index_k_stride=index_k.stride(0),
            index_k_layer_norm_w_ptr=index_k_layer_norm_w,
            index_k_layer_norm_bias_ptr=index_k_layer_norm_bias,
            index_k_layer_norm_eps=index_k_layer_norm_eps,
            INDEX_K_DIM=index_k_dim,
            INDEX_K_BLOCK_SIZE=triton.next_power_of_2(index_k_dim),
            index_k_rope_cos_sin_cache_ptr=index_k_rope_cos_sin_cache,
            index_k_rope_cos_sin_cache_stride=index_k_rope_cos_sin_cache.stride(0),
            index_k_out_ptr=index_k_out,
            index_k_out_stride=index_k_out_stride,
            INDEX_K_HALF_ROT_DIM=index_k_rope_cos_sin_cache.shape[-1] // 2,
            slot_mapping_ptr=slot_mapping,
            indexer_cache_ptr=indexer_k_cache,
            indexer_cache_scale_ptr=idx_cache_scale_view,
            indexer_cache_block_size=idx_cache_block_size,
            indexer_cache_stride=idx_cache_stride,
            mla_cache_ptr=mla_kv_cache,
            mla_cache_block_stride=mla_block_stride,
            mla_cache_entry_stride=mla_entry_stride,
            MLA_CACHE_FP8=mla_cache_fp8,
            mla_cache_scale_ptr=mla_k_scale,
            mla_cache_ds_scale_ptr=mla_ds_scale_view,
            mla_cache_ds_rope_ptr=mla_ds_rope_view,
            MLA_CACHE_DS_MLA=mla_cache_ds_mla,
            MLA_NUM_TILES=mla_num_tiles,
            MLA_TILE_DIM=mla_tile_dim,
            topk_indices_ptr=topk_indices_buffer,
            topk_indices_stride=topk_indices_buffer.stride(0),
            TOPK=topk,
            TOPK_BLOCK_SIZE=self.topk_block_size,
            HAS_INDEXER=has_indexer,
            INDEX_ROPE_INTERLEAVE=index_rope_interleave,
            USE_PDL=use_pdl,
            launch_pdl=use_pdl,
        )


def fused_norm_rope(
    positions: torch.Tensor,
    q_c: torch.Tensor,
    q_rms_norm_w: torch.Tensor,
    q_rms_eps: float,
    kv_c: torch.Tensor,
    kv_rms_norm_w: torch.Tensor,
    kv_rms_eps: float,
    k_pe: torch.Tensor,
    k_rope_cos_sin_cache: torch.Tensor,
    index_k: torch.Tensor | None,
    index_k_layer_norm_w: torch.Tensor | None,
    index_k_layer_norm_bias: torch.Tensor | None,
    index_k_layer_norm_eps: float,
    index_k_rope_cos_sin_cache: torch.Tensor | None,
    topk_indices_buffer: torch.Tensor,
    # Cache params for fused writes (single slot_mapping for both caches)
    slot_mapping: torch.Tensor | None = None,
    indexer_k_cache: torch.Tensor | None = None,
    mla_kv_cache: torch.Tensor | None = None,
    mla_kv_cache_dtype: str = "auto",
    mla_k_scale: torch.Tensor | None = None,
    has_indexer: bool = True,
    index_rope_interleave: bool = False,
    q_c_out: torch.Tensor | None = None,
    kv_c_out: torch.Tensor | None = None,
    k_pe_out: torch.Tensor | None = None,
    index_k_out: torch.Tensor | None = None,
) -> torch.Tensor:
    assert positions.ndim == 1
    assert q_c.ndim == 2
    assert kv_c.ndim == 2
    assert k_pe.ndim == 2
    assert topk_indices_buffer.ndim == 2

    kv_dim = kv_c.shape[-1]
    device = positions.device

    # Shared (no-indexer) layers: substitute cached 1-element dummies so the
    # kernel launches cleanly; pid 0/3 (indexer + topk fill) skipped by
    # HAS_INDEXER and never dereference them.
    if not has_indexer:
        indexer_k_cache = None
        index_k = _dummy((1, 1), q_c.dtype, device)
        index_k_layer_norm_w = _dummy((1,), torch.float32, device)
        index_k_layer_norm_bias = _dummy((1,), torch.float32, device)
        index_k_rope_cos_sin_cache = k_rope_cos_sin_cache
    assert index_k is not None
    assert index_k_rope_cos_sin_cache is not None
    if indexer_k_cache is not None or mla_kv_cache is not None:
        assert slot_mapping is not None
    else:
        slot_mapping = None

    # --- Indexer K cache setup ---
    if indexer_k_cache is not None:
        assert slot_mapping is not None
        idx_cache_scale_view = indexer_k_cache.view(torch.uint8).view(torch.float32)
        idx_cache_block_size = indexer_k_cache.shape[1]
        idx_cache_stride = indexer_k_cache.shape[2]
        if indexer_k_cache.dtype == torch.uint8:
            indexer_k_cache = indexer_k_cache.view(torch.float8_e4m3fn)
    else:
        idx_cache_scale_view = None
        idx_cache_block_size = 1
        idx_cache_stride = 0

    # --- MLA KV cache setup ---
    mla_cache_ds_mla = mla_kv_cache_dtype == "fp8_ds_mla"
    mla_cache_fp8 = is_quantized_kv_cache(mla_kv_cache_dtype) and not mla_cache_ds_mla
    mla_num_tiles = 1
    mla_ds_scale_view = torch.empty(0, dtype=torch.float32, device=device)
    mla_ds_rope_view = torch.empty(0, dtype=torch.bfloat16, device=device)
    if mla_kv_cache is not None:
        mla_block_size = mla_kv_cache.shape[1]
        if mla_cache_ds_mla:
            # 656-byte custom layout addressed in bytes; mla_cache_ptr is the
            # 1-byte fp8 view, so block/entry strides are byte offsets and the
            # fp32/bf16 views share the same buffer.
            assert kv_dim == 512, "fp8_ds_mla requires kv_lora_rank == 512"
            mla_num_tiles = kv_dim // 128
            u8_cache = mla_kv_cache.view(torch.uint8)
            mla_block_stride = u8_cache.stride(0)
            mla_entry_stride = u8_cache.stride(1)
            mla_ds_scale_view = u8_cache.view(torch.float32)
            mla_ds_rope_view = u8_cache.view(torch.bfloat16)
            mla_kv_cache = u8_cache.view(torch.float8_e4m3fn)
        else:
            mla_block_stride = mla_kv_cache.stride(0)
            mla_entry_stride = mla_kv_cache.stride(1)
            if mla_cache_fp8 and mla_kv_cache.dtype == torch.uint8:
                mla_kv_cache = mla_kv_cache.view(torch.float8_e4m3fn)
        if mla_k_scale is None:
            mla_k_scale = torch.ones(1, dtype=torch.float32, device=device)
    else:
        # Dummy cache values; a zero entry stride disables the cache write.
        mla_kv_cache = torch.empty(0, dtype=torch.bfloat16, device=device)
        mla_block_stride = 0
        mla_entry_stride = 0
        mla_block_size = 1
        mla_k_scale = _dummy((1,), torch.float32, device)

    if q_c_out is None:
        q_c_out = torch.empty_like(q_c)
    if kv_c_out is not None:
        assert kv_c_out.shape == kv_c.shape
    if k_pe_out is not None:
        assert k_pe_out.shape == k_pe.shape
    if index_k_out is not None:
        assert index_k_out.shape == index_k.shape
    use_pdl = current_platform.is_arch_support_pdl()
    mla_tile_dim = kv_dim // mla_num_tiles if mla_cache_ds_mla else 1
    # The kernel-owner reproduces this exact launch (strides read from the
    # tensors inside __call__); prep above stays in the wrapper.
    _FUSED_NORM_ROPE_KERNEL(
        positions,
        q_c,
        q_rms_norm_w,
        q_rms_eps,
        q_c_out,
        kv_c,
        kv_rms_norm_w,
        kv_rms_eps,
        kv_c_out,
        k_pe,
        k_rope_cos_sin_cache,
        k_pe_out,
        index_k,
        index_k_layer_norm_w,
        index_k_layer_norm_bias,
        index_k_layer_norm_eps,
        index_k_rope_cos_sin_cache,
        index_k_out,
        slot_mapping,
        indexer_k_cache,
        idx_cache_scale_view,
        idx_cache_block_size,
        idx_cache_stride,
        mla_kv_cache,
        mla_block_stride,
        mla_entry_stride,
        mla_block_size,
        mla_cache_fp8,
        mla_k_scale,
        mla_ds_scale_view,
        mla_ds_rope_view,
        mla_cache_ds_mla,
        mla_num_tiles,
        mla_tile_dim,
        topk_indices_buffer,
        has_indexer,
        index_rope_interleave,
        use_pdl,
    )
    return q_c_out


@triton.jit
def _fused_q_kernel(
    pos_ptr,
    # MQA query PE: RoPE + FP8 pack into output tail
    q_pe_ptr,
    q_pe_stride0,
    q_pe_stride1,
    NUM_Q_HEADS: tl.constexpr,
    q_pe_cos_sin_ptr,
    q_pe_cos_sin_stride,
    Q_PE_HALF_ROT_DIM: tl.constexpr,
    # Index Q RoPE
    index_q_ptr,
    index_q_stride0,
    index_q_stride1,
    NUM_INDEX_Q_HEADS: tl.constexpr,
    index_q_cos_sin_ptr,
    index_q_cos_sin_stride,
    INDEX_Q_HALF_ROT_DIM: tl.constexpr,
    # Index Q Quantize
    index_q_fp8_ptr,
    index_q_fp8_stride0,
    index_q_fp8_stride1,
    INDEX_Q_HEAD_DIM: tl.constexpr,
    # MQA query pack: quantize ql_nope and RoPE+quantize q_pe into mqa_q_fp8
    ql_nope_ptr,
    ql_nope_stride0,
    ql_nope_stride1,
    mqa_q_fp8_ptr,
    mqa_q_fp8_stride0,
    mqa_q_fp8_stride1,
    q_scale_ptr,
    QL_NOPE_DIM: tl.constexpr,
    QL_NOPE_BLOCK: tl.constexpr,
    # bf16 MQA query RoPE output (when QUANTIZE_MQA is False); the NoPE part is
    # consumed directly from ql_nope, so only the RoPE'd q_pe is written here.
    q_pe_out_ptr,
    q_pe_out_stride0,
    q_pe_out_stride1,
    # Index weights
    index_weights_ptr,
    index_weights_stride,
    index_weights_softmax_scale,
    index_weights_head_scale,
    index_weights_out_ptr,
    index_weights_out_stride,
    HAS_INDEXER: tl.constexpr,
    INDEX_ROPE_INTERLEAVE: tl.constexpr,
    QUANTIZE_MQA: tl.constexpr,
    USE_PDL: tl.constexpr,
):
    tok_idx = tl.program_id(0).to(tl.int64)
    pid = tl.program_id(1)
    head_idx = tl.program_id(2)
    if USE_PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()

    if pid == 2:
        # ql_nope quantize + pack into the front of mqa_q_fp8. On the bf16
        # query path ql_nope is consumed as-is (no pack), so skip entirely.
        if not QUANTIZE_MQA:
            return
        if 2 * head_idx >= NUM_Q_HEADS:
            return

        scale = tl.load(q_scale_ptr)
        for local_head in range(2):
            q_head_idx = head_idx * 2 + local_head
            if q_head_idx < NUM_Q_HEADS:
                ql_nope_off = tl.arange(0, QL_NOPE_BLOCK)
                ql_nope_mask = ql_nope_off < QL_NOPE_DIM
                ql_nope = tl.load(
                    ql_nope_ptr
                    + tok_idx * ql_nope_stride0
                    + q_head_idx * ql_nope_stride1
                    + ql_nope_off,
                    mask=ql_nope_mask,
                ).to(tl.float32)
                ql_nope_fp8 = (ql_nope / scale).to(tl.float8e4nv)
                tl.store(
                    mqa_q_fp8_ptr
                    + tok_idx * mqa_q_fp8_stride0
                    + q_head_idx * mqa_q_fp8_stride1
                    + ql_nope_off,
                    ql_nope_fp8,
                    mask=ql_nope_mask,
                )
        return
    elif pid == 0:
        # q_pe RoPE + quantize + pack into the tail of mqa_q_fp8.
        if 2 * head_idx >= NUM_Q_HEADS:
            return

        pos = tl.load(pos_ptr + tok_idx)
        cos, sin = _get_cos_sin(
            q_pe_cos_sin_ptr,
            q_pe_cos_sin_stride,
            pos,
            Q_PE_HALF_ROT_DIM,
        )

        scale = tl.load(q_scale_ptr)
        for local_head in range(2):
            q_head_idx = head_idx * 2 + local_head
            if q_head_idx < NUM_Q_HEADS:
                rot_off = tl.arange(0, Q_PE_HALF_ROT_DIM)
                x1 = tl.load(
                    q_pe_ptr
                    + tok_idx * q_pe_stride0
                    + q_head_idx * q_pe_stride1
                    + rot_off * 2,
                ).to(tl.float32)
                x2 = tl.load(
                    q_pe_ptr
                    + tok_idx * q_pe_stride0
                    + q_head_idx * q_pe_stride1
                    + rot_off * 2
                    + 1
                ).to(tl.float32)
                r1 = x1 * cos - x2 * sin
                r2 = x2 * cos + x1 * sin
                if QUANTIZE_MQA:
                    tl.store(
                        mqa_q_fp8_ptr
                        + tok_idx * mqa_q_fp8_stride0
                        + q_head_idx * mqa_q_fp8_stride1
                        + QL_NOPE_DIM
                        + rot_off * 2,
                        (r1 / scale).to(tl.float8e4nv),
                    )
                    tl.store(
                        mqa_q_fp8_ptr
                        + tok_idx * mqa_q_fp8_stride0
                        + q_head_idx * mqa_q_fp8_stride1
                        + QL_NOPE_DIM
                        + rot_off * 2
                        + 1,
                        (r2 / scale).to(tl.float8e4nv),
                    )
                else:
                    # bf16 query: write the RoPE'd q_pe unquantized.
                    out_ty = q_pe_out_ptr.dtype.element_ty
                    q_pe_dst = (
                        q_pe_out_ptr
                        + tok_idx * q_pe_out_stride0
                        + q_head_idx * q_pe_out_stride1
                    )
                    tl.store(q_pe_dst + rot_off * 2, r1.to(out_ty))
                    tl.store(q_pe_dst + rot_off * 2 + 1, r2.to(out_ty))
        return
    elif pid == 1:
        # Index Q RoPE + fp8 quant, all in registers. The roped bf16 index_q is
        # never consumed (only the fp8 below is), so we avoid an in-place
        # store-then-reload round-trip.
        if not HAS_INDEXER:
            return
        if head_idx >= NUM_INDEX_Q_HEADS:
            return

        pos = tl.load(pos_ptr + tok_idx)
        index_q_block = tl.arange(0, INDEX_Q_HEAD_DIM)
        iq_base = index_q_ptr + tok_idx * index_q_stride0 + head_idx * index_q_stride1
        index_q = tl.load(iq_base + index_q_block).to(tl.float32)

        # RoPE in registers (interleaved for GLM-5.2, NeoX for DeepSeek-V3.2),
        # gathering the rotation partner from the read-only input.
        in_rope = index_q_block < 2 * INDEX_Q_HALF_ROT_DIM
        if INDEX_ROPE_INTERLEAVE:
            cos_idx = index_q_block // 2
            partner_offs = tl.where(in_rope, index_q_block ^ 1, index_q_block)
            sign = tl.where(index_q_block % 2 == 0, -1.0, 1.0)
        else:
            cos_idx = index_q_block % INDEX_Q_HALF_ROT_DIM
            partner_offs = tl.where(
                in_rope, index_q_block ^ INDEX_Q_HALF_ROT_DIM, index_q_block
            )
            sign = tl.where(index_q_block < INDEX_Q_HALF_ROT_DIM, -1.0, 1.0)
        cos_full = tl.load(
            index_q_cos_sin_ptr + pos * index_q_cos_sin_stride + cos_idx,
            mask=in_rope,
            other=1.0,
        ).to(tl.float32)
        sin_full = tl.load(
            index_q_cos_sin_ptr
            + pos * index_q_cos_sin_stride
            + INDEX_Q_HALF_ROT_DIM
            + cos_idx,
            mask=in_rope,
            other=0.0,
        ).to(tl.float32)
        partner = tl.load(iq_base + partner_offs).to(tl.float32)
        roped = index_q * cos_full + sign * partner * sin_full
        index_q = tl.where(in_rope, roped, index_q)

        # Index Q Quantize (from registers)
        index_q_fp8, index_q_scale = _fp8_ue8m0_quantize(index_q)
        tl.store(
            index_q_fp8_ptr
            + tok_idx * index_q_fp8_stride0
            + head_idx * index_q_fp8_stride1
            + index_q_block,
            index_q_fp8,
        )

        # Index weights update
        index_weights = tl.load(
            index_weights_ptr + tok_idx * index_weights_stride + head_idx
        )
        index_weights = index_weights.to(tl.float32)
        index_weights *= index_q_scale
        index_weights *= index_weights_softmax_scale
        index_weights *= index_weights_head_scale
        tl.store(
            index_weights_out_ptr + tok_idx * index_weights_out_stride + head_idx,
            index_weights,
        )


class FusedQTritonKernel(VllmTritonJitKernel["FusedQTritonKernel.CompileKey"]):
    """Warmup owner for the Triton fused MQA/indexer query path."""

    kernel = staticmethod(_fused_q_kernel)
    num_warps = 1

    @dataclass(frozen=True)
    class CompileKey:
        num_q_heads: int
        q_pe_half_rot_dim: int
        num_index_q_heads: int
        index_q_half_rot_dim: int
        index_q_head_dim: int
        ql_nope_dim: int
        ql_nope_block: int
        has_indexer: bool
        index_rope_interleave: bool
        quantize_mqa: bool
        use_pdl: bool
        act_dtype: torch.dtype
        cos_sin_dtype: torch.dtype

    def dispatch(  # type: ignore[override]
        self,
        *,
        num_q_heads: int,
        qk_rope_head_dim: int,
        kv_lora_rank: int,
        index_n_head: int,
        index_head_dim: int,
        has_indexer: bool,
        index_rope_interleave: bool,
        quantize_mqa: bool,
        use_pdl: bool,
        act_dtype: torch.dtype,
        cos_sin_dtype: torch.dtype,
    ) -> "FusedQTritonKernel.CompileKey":
        half_rot = qk_rope_head_dim // 2
        return self.CompileKey(
            num_q_heads=num_q_heads,
            q_pe_half_rot_dim=half_rot,
            num_index_q_heads=index_n_head if has_indexer else 1,
            index_q_half_rot_dim=half_rot,
            index_q_head_dim=index_head_dim if has_indexer else 1,
            ql_nope_dim=kv_lora_rank,
            ql_nope_block=triton.next_power_of_2(kv_lora_rank),
            has_indexer=has_indexer,
            index_rope_interleave=index_rope_interleave,
            quantize_mqa=quantize_mqa,
            use_pdl=use_pdl,
            act_dtype=act_dtype,
            cos_sin_dtype=cos_sin_dtype,
        )

    def get_warmup_keys(  # type: ignore[override]
        self,
        *,
        num_q_heads: int,
        qk_rope_head_dim: int,
        kv_lora_rank: int,
        index_n_head: int,
        index_head_dim: int,
        has_indexer: bool,
        index_rope_interleave: bool,
        quantize_mqa: bool,
        use_pdl: bool,
        act_dtype: torch.dtype,
        cos_sin_dtype: torch.dtype,
    ) -> "list[FusedQTritonKernel.CompileKey]":
        return self._trace_dispatch(self.dispatch)(
            num_q_heads=num_q_heads,
            qk_rope_head_dim=qk_rope_head_dim,
            kv_lora_rank=kv_lora_rank,
            index_n_head=index_n_head,
            index_head_dim=index_head_dim,
            has_indexer=has_indexer,
            index_rope_interleave=index_rope_interleave,
            quantize_mqa=quantize_mqa,
            use_pdl=use_pdl,
            act_dtype=act_dtype,
            cos_sin_dtype=cos_sin_dtype,
        )

    def warmup_inputs(
        self, compile_key: "FusedQTritonKernel.CompileKey"
    ) -> dict[str, Any]:
        act = compile_key.act_dtype
        cos_sin = compile_key.cos_sin_dtype
        mqa_out_dtype = torch.float8_e4m3fn if compile_key.quantize_mqa else act
        iw_dtype = act if compile_key.has_indexer else torch.float32
        rope_dim = 2 * compile_key.q_pe_half_rot_dim
        index_shape = (
            (1, compile_key.num_index_q_heads, compile_key.index_q_head_dim)
            if compile_key.has_indexer
            else (1, 1, 1)
        )
        return dict(
            positions=TritonWarmupTensor(torch.int64),
            q_pe=TritonWarmupTensor(
                act,
                shape=(1, compile_key.num_q_heads, rope_dim),
            ),
            q_pe_cos_sin_cache=TritonWarmupTensor(
                cos_sin,
                shape=(1, rope_dim),
            ),
            index_q=TritonWarmupTensor(act, shape=index_shape),
            index_q_cos_sin_cache=TritonWarmupTensor(
                cos_sin,
                shape=(1, rope_dim),
            ),
            ql_nope=TritonWarmupTensor(
                act,
                shape=(1, compile_key.num_q_heads, compile_key.ql_nope_dim),
            ),
            q_scale=TritonWarmupTensor(torch.float32),
            index_q_fp8=TritonWarmupTensor(
                torch.float8_e4m3fn,
                shape=index_shape,
            ),
            mqa_q_fp8=TritonWarmupTensor(
                mqa_out_dtype,
                shape=(
                    1,
                    compile_key.num_q_heads,
                    compile_key.ql_nope_dim + rope_dim,
                ),
            ),
            q_pe_out=TritonWarmupTensor(
                mqa_out_dtype,
                shape=(1, compile_key.num_q_heads, rope_dim),
            ),
            index_weights=TritonWarmupTensor(
                iw_dtype,
                shape=(1, compile_key.num_index_q_heads),
            ),
            index_weights_out=TritonWarmupTensor(
                torch.float32,
                shape=(1, compile_key.num_index_q_heads),
            ),
            index_weights_softmax_scale=0.0,
            index_weights_head_scale=0.0,
            has_indexer=compile_key.has_indexer,
            index_rope_interleave=compile_key.index_rope_interleave,
            quantize_mqa=compile_key.quantize_mqa,
        )

    @kernel_launcher
    def __call__(
        self,
        positions: torch.Tensor,
        q_pe: torch.Tensor,
        q_pe_cos_sin_cache: torch.Tensor,
        index_q: torch.Tensor,
        index_q_cos_sin_cache: torch.Tensor,
        ql_nope: torch.Tensor,
        q_scale: torch.Tensor,
        index_q_fp8: torch.Tensor,
        mqa_q_fp8: torch.Tensor,
        q_pe_out: torch.Tensor,
        index_weights: torch.Tensor,
        index_weights_out: torch.Tensor,
        index_weights_softmax_scale: float,
        index_weights_head_scale: float,
        has_indexer: bool,
        index_rope_interleave: bool,
        quantize_mqa: bool,
    ) -> LaunchSpec:
        num_tokens = positions.shape[0]
        num_q_heads = q_pe.shape[1]
        num_index_q_heads = index_q.shape[1]
        index_q_head_dim = index_q.shape[2]
        # Grid's 3rd dim must cover the MQA-pack heads (pid 0/2 iterate 2 heads
        # each) and, when present, the indexer heads (pid 1).
        mqa_grid_heads = (num_q_heads + 1) // 2
        grid_heads = max(mqa_grid_heads, num_index_q_heads)
        use_pdl = current_platform.is_arch_support_pdl()
        return (num_tokens, 3, grid_heads), dict(
            pos_ptr=positions,
            q_pe_ptr=q_pe,
            q_pe_stride0=q_pe.stride(0),
            q_pe_stride1=q_pe.stride(1),
            NUM_Q_HEADS=num_q_heads,
            q_pe_cos_sin_ptr=q_pe_cos_sin_cache,
            q_pe_cos_sin_stride=q_pe_cos_sin_cache.stride(0),
            Q_PE_HALF_ROT_DIM=q_pe_cos_sin_cache.shape[-1] // 2,
            index_q_ptr=index_q,
            index_q_stride0=index_q.stride(0),
            index_q_stride1=index_q.stride(1),
            NUM_INDEX_Q_HEADS=num_index_q_heads,
            index_q_cos_sin_ptr=index_q_cos_sin_cache,
            index_q_cos_sin_stride=index_q_cos_sin_cache.stride(0),
            INDEX_Q_HALF_ROT_DIM=index_q_cos_sin_cache.shape[-1] // 2,
            index_q_fp8_ptr=index_q_fp8,
            index_q_fp8_stride0=index_q_fp8.stride(0),
            index_q_fp8_stride1=index_q_fp8.stride(1),
            INDEX_Q_HEAD_DIM=index_q_head_dim,
            ql_nope_ptr=ql_nope,
            ql_nope_stride0=ql_nope.stride(0),
            ql_nope_stride1=ql_nope.stride(1),
            mqa_q_fp8_ptr=mqa_q_fp8,
            mqa_q_fp8_stride0=mqa_q_fp8.stride(0),
            mqa_q_fp8_stride1=mqa_q_fp8.stride(1),
            q_scale_ptr=q_scale,
            QL_NOPE_DIM=ql_nope.shape[2],
            QL_NOPE_BLOCK=triton.next_power_of_2(ql_nope.shape[2]),
            q_pe_out_ptr=q_pe_out,
            q_pe_out_stride0=q_pe_out.stride(0),
            q_pe_out_stride1=q_pe_out.stride(1),
            index_weights_ptr=index_weights,
            index_weights_stride=index_weights.stride(0),
            index_weights_softmax_scale=index_weights_softmax_scale,
            index_weights_head_scale=index_weights_head_scale,
            index_weights_out_ptr=index_weights_out,
            index_weights_out_stride=index_weights_out.stride(0),
            HAS_INDEXER=has_indexer,
            INDEX_ROPE_INTERLEAVE=index_rope_interleave,
            QUANTIZE_MQA=quantize_mqa,
            USE_PDL=use_pdl,
            launch_pdl=use_pdl,
            # num_warps=1 is optimal here: each program is a single 128-element
            # rope+quant, so the kernel is program-count/occupancy bound, not
            # per-program compute bound (swept 1/2/4/8 — 1 wins or ties).
            num_warps=self.num_warps,
        )


def fused_q(
    positions: torch.Tensor,
    q_pe: torch.Tensor,
    q_pe_cos_sin_cache: torch.Tensor,
    index_q: torch.Tensor | None,
    index_q_cos_sin_cache: torch.Tensor | None,
    ql_nope: torch.Tensor,
    q_scale: torch.Tensor,
    # Index weights
    index_weights: torch.Tensor | None,
    index_weights_softmax_scale: float,
    index_weights_head_scale: float,
    has_indexer: bool = True,
    index_rope_interleave: bool = False,
    quantize_mqa: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fuse the MQA-query and indexer-query RoPE/quantization.

    Returns ``(index_q_fp8, index_weights_out, mqa_q)``. When ``quantize_mqa``
    is True (FlashInfer sparse, fp8 query) ``mqa_q`` is a single fp8 tensor
    packing ``[ql_nope; q_pe]``. When False (FlashMLA sparse, bf16 query) it is
    the RoPE'd ``q_pe`` in bf16; the caller pairs it with ``ql_nope`` as the
    ``(ql_nope, q_pe)`` tuple the backend expects.
    """
    assert positions.ndim == 1
    assert positions.dtype == torch.int64
    assert q_pe.ndim == 3
    assert q_pe_cos_sin_cache.ndim == 2
    assert ql_nope.ndim == 3
    assert ql_nope.shape[:2] == q_pe.shape[:2]
    assert q_scale.dtype == torch.float32 and q_scale.numel() == 1
    if not has_indexer:
        # Shared layer: cached 1-element dummies; pid 1 skipped by HAS_INDEXER
        # and never dereferences them.
        index_q = _dummy((1, 1, 1), q_pe.dtype, q_pe.device)
        index_q_cos_sin_cache = q_pe_cos_sin_cache
        index_weights = _dummy((1, 1), torch.float32, q_pe.device)
    assert index_q is not None and index_q.ndim == 3
    assert index_q_cos_sin_cache is not None
    assert index_weights is not None
    # fused_q is shared with the ROCm path, and the CuTeDSL module imports
    # cutlass at module scope, so only reach for it on CUDA.
    cutedsl_kernel: Callable[..., None] | None = None
    if current_platform.is_cuda():
        from vllm.models.deepseek_v32.nvidia.ops.fused_q_cutedsl import (
            _FUSED_Q_CUTEDSL_KERNEL,
            is_fused_q_cutedsl_supported,
        )

        if is_fused_q_cutedsl_supported(
            q_pe,
            index_q,
            ql_nope,
            has_indexer=has_indexer,
            quantize_mqa=quantize_mqa,
        ):
            # The JIT-warmup owner singleton is directly callable with the same
            # signature the inline launch used, so runtime reuses the executor
            # compiled at startup warmup.
            cutedsl_kernel = _FUSED_Q_CUTEDSL_KERNEL
    if quantize_mqa:
        # fp8 path: pack [ql_nope; q_pe] into a single fp8 tensor.
        mqa_q_fp8 = torch.empty(
            q_pe.shape[0],
            q_pe.shape[1],
            ql_nope.shape[2] + q_pe.shape[2],
            dtype=torch.float8_e4m3fn,
            device=q_pe.device,
        )
        # Placeholder; pid 0 packs q_pe into mqa_q_fp8 instead.
        q_pe_out = mqa_q_fp8
        mqa_q = mqa_q_fp8
    else:
        # bf16 path: only the RoPE'd q_pe is produced; ql_nope used directly.
        q_pe_out = torch.empty_like(q_pe)
        mqa_q_fp8 = q_pe_out  # unused placeholder for the fp8 pack pointer
        mqa_q = q_pe_out

    index_q_fp8 = torch.empty_like(index_q, dtype=torch.float8_e4m3fn)
    index_weights_out = torch.empty_like(index_weights, dtype=torch.float32)
    if cutedsl_kernel is not None:
        cutedsl_kernel(
            positions,
            q_pe,
            q_pe_cos_sin_cache,
            ql_nope,
            q_scale,
            mqa_q,
            index_q,
            index_q_cos_sin_cache,
            index_weights,
            index_weights_softmax_scale,
            index_weights_head_scale,
            index_q_fp8,
            index_weights_out,
            has_indexer=has_indexer,
            index_rope_interleave=index_rope_interleave,
        )
        return index_q_fp8, index_weights_out, mqa_q

    _FUSED_Q_TRITON_KERNEL(
        positions,
        q_pe,
        q_pe_cos_sin_cache,
        index_q,
        index_q_cos_sin_cache,
        ql_nope,
        q_scale,
        index_q_fp8,
        mqa_q_fp8,
        q_pe_out,
        index_weights,
        index_weights_out,
        index_weights_softmax_scale,
        index_weights_head_scale,
        has_indexer,
        index_rope_interleave,
        quantize_mqa,
    )
    return index_q_fp8, index_weights_out, mqa_q


@triton.jit
def _fused_eh_norm_kernel(
    pos_ptr,
    embeds_ptr,
    embeds_stride,
    prev_ptr,
    prev_stride,
    enorm_w_ptr,
    hnorm_w_ptr,
    eps,
    out_ptr,
    out_stride,
    H: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """MTP input fusion: zero embeds at position 0, RMSNorm(embeds) with enorm
    and RMSNorm(prev_hidden) with hnorm, written side-by-side into ``out``
    ([N, 2H]) ready for the eh_proj GEMM. Replaces where + 2x RMSNorm + cat."""
    tok = tl.program_id(0)
    off = tl.arange(0, BLOCK)
    mask = off < H

    pos = tl.load(pos_ptr + tok)
    e = tl.load(embeds_ptr + tok * embeds_stride + off, mask=mask, other=0.0)
    e = tl.where(pos == 0, 0.0, e.to(tl.float32))
    ew = tl.load(enorm_w_ptr + off, mask=mask)
    e_normed = _rms_norm(e, ew, eps, H)
    tl.store(out_ptr + tok * out_stride + off, e_normed, mask=mask)

    p = tl.load(prev_ptr + tok * prev_stride + off, mask=mask, other=0.0)
    hw = tl.load(hnorm_w_ptr + off, mask=mask)
    p_normed = _rms_norm(p, hw, eps, H)
    tl.store(out_ptr + tok * out_stride + H + off, p_normed, mask=mask)


class FusedEhNormKernel(VllmTritonJitKernel["FusedEhNormKernel.CompileKey"]):
    """Warmup owner for the MTP embed/hidden RMSNorm fusion."""
    kernel = staticmethod(_fused_eh_norm_kernel)

    @dataclass(frozen=True)
    class CompileKey:
        hidden_size: int
        block_size: int
        dtype: torch.dtype

    def dispatch(
        self,
        *,
        h: int,
        dtype: torch.dtype,
    ) -> CompileKey:  # type: ignore[override]
        return self.CompileKey(
            hidden_size=h,
            block_size=triton.next_power_of_2(h),
            dtype=dtype,
        )

    def get_warmup_keys(self, vllm_config: Any) -> list[CompileKey]:
        speculative_config = vllm_config.speculative_config
        assert speculative_config is not None
        draft_model_config = speculative_config.draft_model_config
        return self._trace_dispatch(self.dispatch)(
            h=draft_model_config.hf_config.hidden_size,
            dtype=draft_model_config.dtype,
        )

    def warmup_inputs(self, compile_key: CompileKey) -> dict[str, Any]:
        hidden = TritonWarmupTensor(
            compile_key.dtype,
            shape=(1, compile_key.hidden_size),
        )
        return dict(
            positions=TritonWarmupTensor(torch.int64),
            inputs_embeds=hidden,
            previous_hidden=hidden,
            enorm_w=hidden,
            hnorm_w=hidden,
            eps=0.0,
            out=TritonWarmupTensor(
                compile_key.dtype,
                shape=(1, 2 * compile_key.hidden_size),
            ),
        )

    @kernel_launcher
    def __call__(
        self,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor,
        previous_hidden: torch.Tensor,
        enorm_w: torch.Tensor,
        hnorm_w: torch.Tensor,
        eps: float,
        out: torch.Tensor,
    ) -> LaunchSpec:
        n, h = inputs_embeds.shape
        return (n,), dict(
            pos_ptr=positions,
            embeds_ptr=inputs_embeds,
            embeds_stride=inputs_embeds.stride(0),
            prev_ptr=previous_hidden,
            prev_stride=previous_hidden.stride(0),
            enorm_w_ptr=enorm_w,
            hnorm_w_ptr=hnorm_w,
            eps=eps,
            out_ptr=out,
            out_stride=out.stride(0),
            H=h,
            BLOCK=triton.next_power_of_2(h),
        )


def fused_eh_norm(
    positions: torch.Tensor,
    inputs_embeds: torch.Tensor,
    previous_hidden: torch.Tensor,
    enorm_w: torch.Tensor,
    hnorm_w: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Returns cat([enorm(masked embeds), hnorm(prev_hidden)]) -> [N, 2H]."""
    n, h = inputs_embeds.shape
    out = torch.empty(n, 2 * h, dtype=inputs_embeds.dtype, device=inputs_embeds.device)
    _FUSED_EH_NORM_KERNEL(
        positions,
        inputs_embeds,
        previous_hidden,
        enorm_w,
        hnorm_w,
        eps,
        out,
    )
    return out


_FUSED_NORM_ROPE_KERNEL = FusedNormRopeKernel()
_FUSED_Q_TRITON_KERNEL = FusedQTritonKernel()
_FUSED_EH_NORM_KERNEL = FusedEhNormKernel()
