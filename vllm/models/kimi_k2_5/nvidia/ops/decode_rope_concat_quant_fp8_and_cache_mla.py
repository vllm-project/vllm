# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused decode-query RoPE + concat + FP8 quant + paged KV-cache write kernel.

This Blackwell (SM10x) CuTe DSL kernel specializes the Kimi-K2.5 NVFP4 MLA
decode path: it applies RoPE to the decode query, concatenates it with the
absorbed (no-RoPE) query, and per-tensor FP8-quantizes the result, fused with
writing the (already-RoPE'd) latent KV into the paged FP8 KV cache.

The public entry point is
:func:`_run_kimik25_decode_rope_concat_quant_fp8_and_cache_mla` (torch tensors
in, FP8 query out). Compilation is cached per constexpr configuration by
:func:`_compile_decode_rope_concat_quant_fp8_and_cache_mla`, which builds fake
tensors with symbolic shapes/strides and calls ``cute.compile``, following the
convention in ``vllm/v1/attention/ops/deepseek_v4_ops``. The compiled executor
is then called directly with torch tensors and sources its launch stream from
the TVM-FFI environment.
"""

from functools import cache

import cutlass
import cutlass.cute as cute
import torch
from cuda.bindings.driver import CUstream
from cutlass import BFloat16, Float32, Int64, Uint8
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

from vllm.platforms import current_platform

from .cutedsl_utils import _fake, _fake_stream


@dsl_user_op
def _cvt_f32_to_e4m3(a: cutlass.Float32, *, loc=None, ip=None) -> cutlass.Uint32:
    return cutlass.Uint32(
        llvm.inline_asm(
            T.i32(),
            [cutlass.Float32(a).ir_value(loc=loc, ip=ip)],
            """
            {
                .reg .b16 fp8_pair;
                .reg .f32 zero;
                mov.f32 zero, 0f00000000;
                cvt.rn.satfinite.e4m3x2.f32 fp8_pair, zero, $1;
                cvt.u32.u16 $0, fp8_pair;
            }
            """,
            "=r,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@cute.kernel
def kimik25_decode_rope_concat_quant_fp8_and_cache_mla_kernel(
    positions: cute.Tensor,  # (B,)
    ql_nope: cute.Tensor,  # (B, N, q_lora_dim)
    q_pe: cute.Tensor,  # (B, N, (2, pe_dim // 2))
    q_out: cute.Tensor,  # uint8 bytes, (B, N, q_lora_dim + pe_dim)
    cos_sin_cache: cute.Tensor,  # (max_position_embeddings, pe_dim)
    q_scale: cute.Tensor,  # (1,)
    kv_c: cute.Tensor,  # (Sp, kv_lora_rank)
    k_pe: cute.Tensor,  # (Sp, pe_dim)
    kv_cache: cute.Tensor,  # (num_blocks, block_size, kv_lora_rank + pe_dim)
    slot_mapping: cute.Tensor,  # (Sp,)
    kv_scale: cute.Tensor,  # (1,)
    q_lora_dim: cutlass.Constexpr,
    kv_lora_rank: cutlass.Constexpr,
    pe_dim: cutlass.Constexpr,
    kv_cache_block_factor: cutlass.Constexpr,
):
    """
    This kernel writes the (already-RoPE'd) latent KV into the paged FP8
    cache, and builds the FP8 decode query by applying RoPE to the query,
    concatenating it with the absorbed query, and quantizing.

    All quantization is per-tensor FP8 (e4m3): KV by kv_scale, query by q_scale.
    This assumes q_lora_dim == kv_lora_rank == 512, pe_dim == 64,
    q_lora_dim % 256 == 0, and kv_lora_rank % kv_cache_block_factor == 0.

    The grid is linearized and divided into two regions of blocks:
     - [0, Sp * (kv_cache_block_factor + 1)): KV cache write, with
       (kv_cache_block_factor + 1) blocks per KV token. Tokens whose slot is
       negative (padding) are skipped. Each thread writes one element:
         - split 0: the pe_dim already-RoPE'd K values (k_pe); first pe_dim
           threads active.
         - splits 1 .. kv_cache_block_factor: one kv_lora_rank /
           kv_cache_block_factor chunk of the latent KV (kv_c).
     - [Sp * (kv_cache_block_factor + 1), end): decode query, with
       (q_lora_dim // 256 + 1) blocks per (token, head). For each:
         - block_kind 0 .. q_lora_dim // 256 - 1: quantize one 256-value tile of
           the absorbed no-RoPE query (ql_nope); each thread handles two values.
         - the last block_kind: RoPE the pe_dim query (q_pe) and quantize it into
           the tail of q_out; only the first pe_dim // 2 threads active, each
           handling one rotary pair.
    """
    tid, _, _ = cute.arch.thread_idx()
    linear_block, _, _ = cute.arch.block_idx()

    kv_cache_splits: cutlass.Constexpr = kv_cache_block_factor + 1
    num_cache_blocks = slot_mapping.shape[0] * kv_cache_splits

    if linear_block < num_cache_blocks:
        token_idx = linear_block // kv_cache_splits
        split_idx = linear_block % kv_cache_splits
        kv_c_elems_per_split: cutlass.Constexpr = kv_lora_rank // kv_cache_block_factor

        slot_idx = slot_mapping[token_idx]
        if slot_idx >= 0:
            block_size = kv_cache.shape[1]
            cache_block_idx = slot_idx // block_size
            cache_block_offset = slot_idx % block_size
            scale_value = kv_scale[0].to(cutlass.Float32)

            if split_idx > 0:
                kv_c_idx = (split_idx - 1) * kv_c_elems_per_split + tid
                kv_c_val = kv_c[token_idx, kv_c_idx].to(cutlass.Float32)
                kv_cache[cache_block_idx, cache_block_offset, kv_c_idx] = cutlass.Uint8(
                    _cvt_f32_to_e4m3(kv_c_val / scale_value) & cutlass.Uint32(0xFF)
                )
            else:
                if tid < pe_dim:
                    k_pe_val = k_pe[token_idx, tid].to(cutlass.Float32)
                    kv_cache[
                        cache_block_idx,
                        cache_block_offset,
                        kv_lora_rank + tid,
                    ] = cutlass.Uint8(
                        _cvt_f32_to_e4m3(k_pe_val / scale_value) & cutlass.Uint32(0xFF)
                    )
    else:
        decode_block = linear_block - num_cache_blocks
        q_lora_tiles: cutlass.Constexpr = q_lora_dim // 256
        decode_tile_count: cutlass.Constexpr = q_lora_tiles + 1
        block_kind = decode_block % decode_tile_count
        token_head_block = decode_block // decode_tile_count
        token_idx = token_head_block // ql_nope.shape[1]
        head_idx = token_head_block % ql_nope.shape[1]

        scale_value = q_scale[0]
        ql_nope_paired = cute.logical_divide(ql_nope, (1, 1, 2))
        q_out_paired = cute.logical_divide(q_out, (1, 1, 2))
        half_pe_dim: cutlass.Constexpr = pe_dim // 2

        if block_kind < q_lora_tiles:
            q_pair_idx = block_kind * 128 + tid
            in_scratch = cute.make_rmem_tensor(2, dtype=cutlass.BFloat16)
            out_scratch = cute.make_rmem_tensor(2, dtype=cutlass.Uint8)
            cute.autovec_copy(
                ql_nope_paired[token_idx, head_idx, (None, q_pair_idx)],
                in_scratch,
            )
            for i in cutlass.range_constexpr(2):
                q_val = in_scratch[i].to(cutlass.Float32)
                out_scratch[i] = cutlass.Uint8(
                    _cvt_f32_to_e4m3(q_val / scale_value) & cutlass.Uint32(0xFF)
                )
            cute.autovec_copy(
                out_scratch,
                q_out_paired[token_idx, head_idx, (None, q_pair_idx)],
            )
        elif tid < half_pe_dim:
            pos = positions[token_idx]
            cos = cos_sin_cache[pos, tid]
            sin = cos_sin_cache[pos, tid + half_pe_dim]
            in_scratch = cute.make_rmem_tensor(2, dtype=cutlass.BFloat16)
            out_scratch = cute.make_rmem_tensor(2, dtype=cutlass.Uint8)
            cute.autovec_copy(q_pe[token_idx, head_idx, (None, tid)], in_scratch)
            a = in_scratch[0]
            b = in_scratch[1]
            qx = (a * cos - b * sin).to(cutlass.BFloat16)
            qy = (a * sin + b * cos).to(cutlass.BFloat16)
            out_scratch[0] = cutlass.Uint8(
                _cvt_f32_to_e4m3(qx.to(cutlass.Float32) / scale_value)
                & cutlass.Uint32(0xFF)
            )
            out_scratch[1] = cutlass.Uint8(
                _cvt_f32_to_e4m3(qy.to(cutlass.Float32) / scale_value)
                & cutlass.Uint32(0xFF)
            )
            cute.autovec_copy(
                out_scratch,
                q_out_paired[token_idx, head_idx, (None, q_lora_dim // 2 + tid)],
            )


@cute.jit
def kimik25_decode_rope_concat_quant_fp8_and_cache_mla(
    positions: cute.Tensor,
    ql_nope: cute.Tensor,
    q_pe: cute.Tensor,
    q_out: cute.Tensor,
    cos_sin_cache: cute.Tensor,
    q_scale: cute.Tensor,
    kv_c: cute.Tensor,
    k_pe: cute.Tensor,
    kv_cache: cute.Tensor,
    slot_mapping: cute.Tensor,
    kv_scale: cute.Tensor,
    q_lora_dim: cutlass.Constexpr,
    kv_lora_rank: cutlass.Constexpr,
    pe_dim: cutlass.Constexpr,
    kv_cache_block_factor: cutlass.Constexpr,
    stream: CUstream,
):
    assert q_lora_dim == 512
    assert kv_lora_rank == 512
    assert pe_dim == 64
    assert pe_dim % 2 == 0
    assert kv_cache_block_factor > 0
    assert kv_lora_rank % kv_cache_block_factor == 0
    assert kv_lora_rank // kv_cache_block_factor == 128
    assert q_lora_dim % 256 == 0
    assert ql_nope.stride[2] == 1
    assert q_pe.stride[2] == 1
    assert kv_c.stride[1] == 1
    assert k_pe.stride[1] == 1
    assert kv_cache.stride[2] == 1

    sp = positions.shape[0]
    ql_nope = cute.make_tensor(
        ql_nope.iterator,
        cute.make_layout(
            (sp, ql_nope.shape[1], q_lora_dim),
            stride=(
                cute.assume(ql_nope.stride[0], divby=2),
                cute.assume(ql_nope.stride[1], divby=2),
                1,
            ),
        ),
    )
    q_pe = cute.make_tensor(
        q_pe.iterator,
        cute.make_layout(
            (sp, q_pe.shape[1], (2, pe_dim // 2)),
            stride=(
                cute.assume(q_pe.stride[0], divby=2),
                cute.assume(q_pe.stride[1], divby=2),
                (1, 2),
            ),
        ),
    )
    q_out = cute.make_tensor(
        q_out.iterator,
        cute.make_layout(
            (sp, q_out.shape[1], q_lora_dim + pe_dim),
            stride=(
                cute.assume(q_out.stride[0], divby=2),
                cute.assume(q_out.stride[1], divby=2),
                1,
            ),
        ),
    )
    kv_c = cute.make_tensor(
        kv_c.iterator,
        cute.make_layout(
            (kv_c.shape[0], kv_lora_rank),
            stride=(kv_c.stride[0], 1),
        ),
    )
    k_pe = cute.make_tensor(
        k_pe.iterator,
        cute.make_layout(
            (k_pe.shape[0], pe_dim),
            stride=(k_pe.stride[0], 1),
        ),
    )
    kv_cache = cute.make_tensor(
        kv_cache.iterator,
        cute.make_layout(
            (kv_cache.shape[0], kv_cache.shape[1], kv_lora_rank + pe_dim),
            stride=(kv_cache.stride[0], kv_cache.stride[1], 1),
        ),
    )

    q_lora_tiles: cutlass.Constexpr = q_lora_dim // 256
    decode_blocks = sp * ql_nope.shape[1] * (q_lora_tiles + 1)
    cache_blocks = slot_mapping.shape[0] * (kv_cache_block_factor + 1)
    kimik25_decode_rope_concat_quant_fp8_and_cache_mla_kernel(
        positions,
        ql_nope,
        q_pe,
        q_out,
        cos_sin_cache,
        q_scale,
        kv_c,
        k_pe,
        kv_cache,
        slot_mapping,
        kv_scale,
        q_lora_dim,
        kv_lora_rank,
        pe_dim,
        kv_cache_block_factor,
    ).launch(
        grid=(cache_blocks + decode_blocks, 1, 1),
        block=(128, 1, 1),
        stream=stream,
    )


@cache
def _compile_decode_rope_concat_quant_fp8_and_cache_mla(
    q_lora_dim: int,
    kv_lora_rank: int,
    pe_dim: int,
    kv_cache_block_factor: int,
    num_local_heads: int,
):
    positions = _fake(Int64, (cute.sym_int(),), (cute.sym_int64(),), align=16)
    ql_nope = _fake(
        BFloat16,
        (cute.sym_int(), num_local_heads, q_lora_dim),
        (cute.sym_int64(divisibility=2), cute.sym_int64(divisibility=2), 1),
        align=16,
    )
    q_pe = _fake(
        BFloat16,
        (cute.sym_int(), num_local_heads, pe_dim),
        (cute.sym_int64(divisibility=2), cute.sym_int64(divisibility=2), 1),
        align=16,
    )
    q_out = _fake(
        Uint8,
        (cute.sym_int(), num_local_heads, q_lora_dim + pe_dim),
        (cute.sym_int64(divisibility=2), cute.sym_int64(divisibility=2), 1),
        align=16,
    )
    cos_sin_cache = _fake(
        BFloat16, (cute.sym_int(), pe_dim), (cute.sym_int64(), 1), align=16
    )
    q_scale = _fake(Float32, (1,), (1,), align=4)
    kv_c = _fake(
        BFloat16, (cute.sym_int(), kv_lora_rank), (cute.sym_int64(), 1), align=16
    )
    k_pe = _fake(BFloat16, (cute.sym_int(), pe_dim), (cute.sym_int64(), 1), align=16)
    kv_cache = _fake(
        Uint8,
        (cute.sym_int(), cute.sym_int(), kv_lora_rank + pe_dim),
        (cute.sym_int64(), cute.sym_int64(), 1),
        align=16,
    )
    slot_mapping = _fake(Int64, (cute.sym_int(),), (cute.sym_int64(),), align=16)
    kv_scale = _fake(Float32, (1,), (1,), align=4)
    return cute.compile(
        kimik25_decode_rope_concat_quant_fp8_and_cache_mla,
        positions,
        ql_nope,
        q_pe,
        q_out,
        cos_sin_cache,
        q_scale,
        kv_c,
        k_pe,
        kv_cache,
        slot_mapping,
        kv_scale,
        q_lora_dim,
        kv_lora_rank,
        pe_dim,
        kv_cache_block_factor,
        _fake_stream(),
        options="--enable-tvm-ffi",
    )


def _run_kimik25_decode_rope_concat_quant_fp8_and_cache_mla(
    *,
    positions: torch.Tensor,
    ql_nope: torch.Tensor,
    q_pe: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    q_scale: torch.Tensor,
    kv_c: torch.Tensor,
    k_pe: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    kv_cache_dtype: str,
    kv_scale: torch.Tensor,
) -> torch.Tensor:
    assert kv_cache_dtype in {"fp8", "fp8_e4m3"}

    q_lora_dim = ql_nope.shape[2]
    kv_lora_rank = kv_c.shape[1]
    pe_dim = q_pe.shape[2]
    assert q_lora_dim == 512, "Kimi-K2.5 NVFP4 expects kv_lora_rank=512"
    assert kv_lora_rank == 512, "Kimi-K2.5 NVFP4 expects kv_lora_rank=512"
    assert pe_dim == 64, "Kimi-K2.5 NVFP4 expects qk_rope_head_dim=64"
    assert k_pe.shape[1] == pe_dim
    assert ql_nope.shape[:2] == q_pe.shape[:2]

    kv_cache_block_factor = 4
    q_out = torch.empty(
        (ql_nope.shape[0], ql_nope.shape[1], q_lora_dim + pe_dim),
        device=ql_nope.device,
        dtype=torch.uint8,
    )
    _compile_decode_rope_concat_quant_fp8_and_cache_mla(
        q_lora_dim, kv_lora_rank, pe_dim, kv_cache_block_factor, ql_nope.shape[1]
    )(
        positions,
        ql_nope,
        q_pe,
        q_out,
        cos_sin_cache,
        q_scale.view(1),
        kv_c,
        k_pe,
        kv_cache,
        slot_mapping,
        kv_scale.view(1),
    )
    return q_out.view(current_platform.fp8_dtype())
