# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa
"""Raw TMA ops and descriptor builders.

`tma_utils.py` is the canonical owner for raw TMA inline-asm helpers and TMA
descriptor construction. Non-TMA store/layout helpers are re-exported from
`copy_utils.py` for backward compatibility.
"""

import ctypes

import cutlass._mlir.dialects.cute as cute_ir
import cutlass._mlir.dialects.cute_nvgpu as cute_nvgpu_ir
from cutlass import Int32, Int64
from cutlass._mlir.dialects import _cute_nvgpu_ops_gen as cute_nvgpu_gen
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

# Raw TMA Ops

TMA_CACHE_EVICT_FIRST = 0x12F0000000000000
TMA_CACHE_EVICT_LAST = 0x14F0000000000000


@dsl_user_op
def tma_tile_load(
    smem_ptr,
    smem_byte_offset,
    tma_desc_ptr,
    col_idx,
    row_idx,
    mbar_ptr,
    *,
    loc=None,
    ip=None,
):
    """cp.async.bulk.tensor.2d.shared::cta.global.tile with mbar completion."""
    llvm.inline_asm(
        T.i32(),
        [
            smem_ptr.toint().ir_value(loc=loc, ip=ip),
            Int32(smem_byte_offset).ir_value(loc=loc, ip=ip),
            tma_desc_ptr.toint().ir_value(loc=loc, ip=ip),
            Int32(col_idx).ir_value(loc=loc, ip=ip),
            Int32(row_idx).ir_value(loc=loc, ip=ip),
            mbar_ptr.toint().ir_value(loc=loc, ip=ip),
        ],
        "{\n"
        ".reg .u32 sa, ma;\n"
        "cvt.u32.u64 sa, $1;\n"
        "add.u32 sa, sa, $2;\n"
        "cvt.u32.u64 ma, $6;\n"
        "cp.async.bulk.tensor.2d.shared::cta.global.tile"
        ".mbarrier::complete_tx::bytes [sa], [$3, {$4, $5}], [ma];\n"
        "mov.u32 $0, 0;\n"
        "}\n",
        "=r,l,r,l,r,r,l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def tma_gather4(
    smem_ptr,
    smem_byte_offset,
    tma_desc_ptr,
    col_idx,
    row0,
    row1,
    row2,
    row3,
    mbar_ptr,
    *,
    loc=None,
    ip=None,
):
    """cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4 with mbar."""
    llvm.inline_asm(
        T.i32(),
        [
            smem_ptr.toint().ir_value(loc=loc, ip=ip),
            Int32(smem_byte_offset).ir_value(loc=loc, ip=ip),
            tma_desc_ptr.toint().ir_value(loc=loc, ip=ip),
            Int32(col_idx).ir_value(loc=loc, ip=ip),
            Int32(row0).ir_value(loc=loc, ip=ip),
            Int32(row1).ir_value(loc=loc, ip=ip),
            Int32(row2).ir_value(loc=loc, ip=ip),
            Int32(row3).ir_value(loc=loc, ip=ip),
            mbar_ptr.toint().ir_value(loc=loc, ip=ip),
        ],
        "{\n"
        ".reg .u32 sa, ma;\n"
        "cvt.u32.u64 sa, $1;\n"
        "add.u32 sa, sa, $2;\n"
        "cvt.u32.u64 ma, $9;\n"
        "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
        ".mbarrier::complete_tx::bytes [sa], [$3, {$4, $5, $6, $7, $8}], [ma];\n"
        "mov.u32 $0, 0;\n"
        "}\n",
        "=r,l,r,l,r,r,r,r,r,l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def prefetch_tma_desc_raw(tma_desc_ptr, *, loc=None, ip=None):
    """Prefetch a raw TMA descriptor pointer into the descriptor cache."""
    ptr_i64 = tma_desc_ptr.toint().ir_value(loc=loc, ip=ip)
    ptr_i64_align_ty = cute_ir.ConstrainedIntType.get(128, ptr_i64.type.width)
    ptr_i64_align = cute_ir.assume(ptr_i64_align_ty, ptr_i64, loc=loc, ip=ip)
    ptr_ty = cute_ir.PtrType.get(
        cute_nvgpu_ir.TmaDescriptorTiledType.get(),
        cute_ir.AddressSpace.gmem,
        128,
    )
    desc_ptr = cute_ir.inttoptr(ptr_ty, ptr_i64_align, loc=loc, ip=ip)
    cute_nvgpu_gen.arch_prefetch_tma_desc(desc_ptr.value, loc=loc, ip=ip)


@dsl_user_op
def tma_tile_prefetch(
    tma_desc_ptr,
    col_idx,
    row_idx,
    cache_hint=TMA_CACHE_EVICT_FIRST,
    *,
    loc=None,
    ip=None,
):
    """cp.async.bulk.prefetch.tensor.2d.L2.global.tile with cache hint."""
    llvm.inline_asm(
        None,
        [
            tma_desc_ptr.toint().ir_value(loc=loc, ip=ip),
            Int32(col_idx).ir_value(loc=loc, ip=ip),
            Int32(row_idx).ir_value(loc=loc, ip=ip),
            Int64(cache_hint).ir_value(loc=loc, ip=ip),
        ],
        "cp.async.bulk.prefetch.tensor.2d.L2.global.tile.L2::cache_hint "
        "[$0, {$1, $2}], $3;\n",
        "l,r,r,l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def tma_gather4_prefetch(
    tma_desc_ptr,
    col_idx,
    row0,
    row1,
    row2,
    row3,
    cache_hint=TMA_CACHE_EVICT_LAST,
    *,
    loc=None,
    ip=None,
):
    """cp.async.bulk.prefetch.tensor.2d.L2.global.tile::gather4 with cache hint."""
    llvm.inline_asm(
        None,
        [
            tma_desc_ptr.toint().ir_value(loc=loc, ip=ip),
            Int32(col_idx).ir_value(loc=loc, ip=ip),
            Int32(row0).ir_value(loc=loc, ip=ip),
            Int32(row1).ir_value(loc=loc, ip=ip),
            Int32(row2).ir_value(loc=loc, ip=ip),
            Int32(row3).ir_value(loc=loc, ip=ip),
            Int64(cache_hint).ir_value(loc=loc, ip=ip),
        ],
        "cp.async.bulk.prefetch.tensor.2d.L2.global.tile::gather4.L2::cache_hint "
        "[$0, {$1, $2, $3, $4, $5}], $6;\n",
        "l,r,r,r,r,r,l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def tma_tile_load_cached(
    smem_ptr,
    smem_byte_offset,
    tma_desc_ptr,
    col_idx,
    row_idx,
    mbar_ptr,
    cache_hint=TMA_CACHE_EVICT_FIRST,
    *,
    loc=None,
    ip=None,
):
    """cp.async.bulk.tensor.2d.shared::cta.global.tile with cache hint and mbar."""
    llvm.inline_asm(
        T.i32(),
        [
            smem_ptr.toint().ir_value(loc=loc, ip=ip),
            Int32(smem_byte_offset).ir_value(loc=loc, ip=ip),
            tma_desc_ptr.toint().ir_value(loc=loc, ip=ip),
            Int32(col_idx).ir_value(loc=loc, ip=ip),
            Int32(row_idx).ir_value(loc=loc, ip=ip),
            mbar_ptr.toint().ir_value(loc=loc, ip=ip),
            Int64(cache_hint).ir_value(loc=loc, ip=ip),
        ],
        "{\n"
        ".reg .u32 sa, ma;\n"
        "cvt.u32.u64 sa, $1;\n"
        "add.u32 sa, sa, $2;\n"
        "cvt.u32.u64 ma, $6;\n"
        "cp.async.bulk.tensor.2d.shared::cta.global.tile"
        ".mbarrier::complete_tx::bytes.L2::cache_hint "
        "[sa], [$3, {$4, $5}], [ma], $7;\n"
        "mov.u32 $0, 0;\n"
        "}\n",
        "=r,l,r,l,r,r,l,l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def tma_gather4_cached(
    smem_ptr,
    smem_byte_offset,
    tma_desc_ptr,
    col_idx,
    row0,
    row1,
    row2,
    row3,
    mbar_ptr,
    cache_hint=TMA_CACHE_EVICT_LAST,
    *,
    loc=None,
    ip=None,
):
    """cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4 with cache hint."""
    llvm.inline_asm(
        None,
        [
            smem_ptr.toint().ir_value(loc=loc, ip=ip),
            Int32(smem_byte_offset).ir_value(loc=loc, ip=ip),
            tma_desc_ptr.toint().ir_value(loc=loc, ip=ip),
            Int32(col_idx).ir_value(loc=loc, ip=ip),
            Int32(row0).ir_value(loc=loc, ip=ip),
            Int32(row1).ir_value(loc=loc, ip=ip),
            Int32(row2).ir_value(loc=loc, ip=ip),
            Int32(row3).ir_value(loc=loc, ip=ip),
            mbar_ptr.toint().ir_value(loc=loc, ip=ip),
            Int64(cache_hint).ir_value(loc=loc, ip=ip),
        ],
        "{\n"
        ".reg .u32 sa, ma;\n"
        "cvt.u32.u64 sa, $0;\n"
        "add.u32 sa, sa, $1;\n"
        "cvt.u32.u64 ma, $8;\n"
        "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
        ".mbarrier::complete_tx::bytes.cta_group::1.L2::cache_hint "
        "[sa], [$2, {$3, $4, $5, $6, $7}], [ma], $9;\n"
        "}\n",
        "l,r,l,r,r,r,r,r,l,l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def tma_tile_store(
    tma_desc_ptr,
    col_idx,
    row_idx,
    smem_ptr,
    smem_byte_offset,
    *,
    loc=None,
    ip=None,
):
    """cp.async.bulk.tensor.2d.global.shared::cta.bulk_group store."""
    llvm.inline_asm(
        T.i32(),
        [
            tma_desc_ptr.toint().ir_value(loc=loc, ip=ip),
            Int32(col_idx).ir_value(loc=loc, ip=ip),
            Int32(row_idx).ir_value(loc=loc, ip=ip),
            smem_ptr.toint().ir_value(loc=loc, ip=ip),
            Int32(smem_byte_offset).ir_value(loc=loc, ip=ip),
        ],
        "{\n"
        ".reg .u32 sa;\n"
        "cvt.u32.u64 sa, $4;\n"
        "add.u32 sa, sa, $5;\n"
        "cp.async.bulk.tensor.2d.global.shared::cta.bulk_group"
        " [$1, {$2, $3}], [sa];\n"
        "mov.u32 $0, 0;\n"
        "}\n",
        "=r,l,r,r,l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


# Descriptor Builders

_TMA_DESC_BYTES = 128


def _encode_tma_desc_2d_bytes(tensor_2d, *, box_x, box_y, context: str) -> bytes:
    import cuda.bindings.driver as cuda
    import torch

    if tensor_2d.ndim != 2:
        raise ValueError(
            f"{context} tensor must be rank-2, got {tuple(tensor_2d.shape)}"
        )
    rows, cols = tensor_2d.shape
    if tensor_2d.stride(-1) != 1:
        raise ValueError(f"{context} tensor must be contiguous in the last dimension")
    dtype_map = {
        torch.float16: cuda.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
        torch.bfloat16: cuda.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
    }
    if tensor_2d.dtype not in dtype_map:
        raise TypeError(
            f"Unsupported dtype for {context} TMA descriptor: {tensor_2d.dtype}"
        )

    sizes = [cuda.cuuint64_t(cols), cuda.cuuint64_t(rows)]
    strides = [cuda.cuuint64_t(tensor_2d.stride(0) * tensor_2d.element_size())]
    box = [cuda.cuuint32_t(box_x), cuda.cuuint32_t(box_y)]
    elem_stride = [cuda.cuuint32_t(1), cuda.cuuint32_t(1)]
    err, tm = cuda.cuTensorMapEncodeTiled(
        dtype_map[tensor_2d.dtype],
        2,
        tensor_2d.data_ptr(),
        sizes,
        strides,
        box,
        elem_stride,
        cuda.CUtensorMapInterleave.CU_TENSOR_MAP_INTERLEAVE_NONE,
        cuda.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_128B,
        cuda.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
        cuda.CUtensorMapFloatOOBfill.CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE,
    )
    assert err == cuda.CUresult.CUDA_SUCCESS, f"TMA encode failed: {err}"
    buf = (ctypes.c_uint8 * _TMA_DESC_BYTES).from_address(tm.getPtr())
    return bytes(buf)


def _desc_bytes_to_device_tensor(desc_bytes: bytes | bytearray, *, device):
    import torch

    desc_bytes = bytes(desc_bytes)
    device = torch.device(device)
    if device.type != "cuda":
        raise ValueError(f"TMA descriptors require a CUDA device, got {device}")

    host_desc = torch.empty((len(desc_bytes),), dtype=torch.uint8, pin_memory=True)
    host_desc.copy_(torch.frombuffer(bytearray(desc_bytes), dtype=torch.uint8))
    device_desc = torch.empty((len(desc_bytes),), dtype=torch.uint8, device=device)
    stream = torch.cuda.current_stream(device)
    with torch.cuda.stream(stream):
        device_desc.copy_(host_desc, non_blocking=True)
    device_desc.record_stream(stream)
    # Keep the staging buffer alive for the async copy without caching descriptors.
    device_desc._tma_host_desc = host_desc
    return device_desc


def create_flat_gather4_tma_desc(tensor_2d, box_x=64):
    """Create a gather4 CUtensorMap descriptor for a flat 2D row-major tensor."""
    if tensor_2d.ndim != 2:
        raise ValueError(
            f"tensor_2d must be rank-2 [rows, dim], got {tuple(tensor_2d.shape)}"
        )
    desc = _encode_tma_desc_2d_bytes(
        tensor_2d,
        box_x=box_x,
        box_y=1,
        context="gather4",
    )
    return _desc_bytes_to_device_tensor(desc, device=tensor_2d.device)


def create_q_gather4_tma_desc(q_flat, box_x=64):
    return create_flat_gather4_tma_desc(q_flat, box_x=box_x)


def create_strided_2d_tma_desc(tensor_2d, *, box_x, box_y):
    """Create a CUtensorMap descriptor for a rank-2 tensor with arbitrary row stride."""
    desc = _encode_tma_desc_2d_bytes(
        tensor_2d,
        box_x=box_x,
        box_y=box_y,
        context="strided 2D",
    )
    return _desc_bytes_to_device_tensor(desc, device=tensor_2d.device)


def create_paged_kv_tma_descs(kv_paged, *, box_x=64, box_y=128):
    """Create per-KV-head token-major TMA descriptors for paged [P, S, H, D] storage."""
    import torch

    if kv_paged.ndim != 4:
        raise ValueError(
            f"kv_paged must be rank-4 [num_pages, page_size, H, D], got {tuple(kv_paged.shape)}"
        )
    num_pages, page_size, head_kv, dim = kv_paged.shape
    total_rows = num_pages * page_size
    row_stride = head_kv * dim
    desc_table = bytearray()
    for h in range(head_kv):
        head_view = torch.as_strided(
            kv_paged,
            size=(total_rows, dim),
            stride=(row_stride, 1),
            storage_offset=h * dim,
        )
        desc_table.extend(
            _encode_tma_desc_2d_bytes(
                head_view,
                box_x=box_x,
                box_y=box_y,
                context="paged KV",
            )
        )
    return _desc_bytes_to_device_tensor(desc_table, device=kv_paged.device).reshape(
        head_kv, _TMA_DESC_BYTES
    )


def view_paged_kv_as_blocks(kv_paged, *, blk_kv):
    """Return a no-copy [num_physical_blocks, blk_kv, H, D] view for page_size >= blk_kv."""
    if kv_paged.ndim != 4:
        raise ValueError(
            f"kv_paged must be rank-4 [num_pages, page_size, H, D], got {tuple(kv_paged.shape)}"
        )
    num_pages, page_size, head_kv, dim = kv_paged.shape
    if page_size % blk_kv != 0:
        raise ValueError(
            f"page_size ({page_size}) must be divisible by blk_kv ({blk_kv})"
        )
    blocks_per_page = page_size // blk_kv
    return kv_paged.view(num_pages, blocks_per_page, blk_kv, head_kv, dim).view(
        num_pages * blocks_per_page, blk_kv, head_kv, dim
    )


def create_flat_kv_tma_descs(kv_flat, *, box_x=64, box_y=128):
    """Create per-KV-head token-major TMA descriptors for flat [total_k, H, D] storage."""
    import torch

    if kv_flat.ndim != 3:
        raise ValueError(
            f"kv_flat must be rank-3 [total_k, H, D], got {tuple(kv_flat.shape)}"
        )
    total_k, head_kv, dim = kv_flat.shape
    row_stride = head_kv * dim
    desc_table = bytearray()
    for h in range(head_kv):
        head_view = torch.as_strided(
            kv_flat,
            size=(total_k, dim),
            stride=(row_stride, 1),
            storage_offset=h * dim,
        )
        desc_table.extend(
            _encode_tma_desc_2d_bytes(
                head_view,
                box_x=box_x,
                box_y=box_y,
                context="flat KV",
            )
        )
    return _desc_bytes_to_device_tensor(desc_table, device=kv_flat.device).reshape(
        head_kv, _TMA_DESC_BYTES
    )


# Compatibility Re-exports

from .copy_utils import (
    atomic_add_broadcast_i32,
    atomic_add_i32,
    convert_layout_acc_mn,
    convert_layout_from_tmem16x256b_to_acc_sm90,
    make_16x256b_tensor_mn_view,
    real_col_to_stg128_fake_col,
    real_col_to_stg128_half_fake_col,
    stg128_fake_col_to_real_col,
    stg128_half_fake_col_to_real_col,
    stg_64_bf16,
    stg_64_f16,
    stg_128,
    stg_128_bf16,
    stg_128_bf16_cs,
    stg_128_cs,
    stg_128_f16,
    stg_128_f16_cs,
)

__all__ = [
    "TMA_CACHE_EVICT_FIRST",
    "TMA_CACHE_EVICT_LAST",
    "atomic_add_broadcast_i32",
    "atomic_add_i32",
    "convert_layout_acc_mn",
    "convert_layout_from_tmem16x256b_to_acc_sm90",
    "create_flat_gather4_tma_desc",
    "create_flat_kv_tma_descs",
    "create_paged_kv_tma_descs",
    "create_q_gather4_tma_desc",
    "create_strided_2d_tma_desc",
    "make_16x256b_tensor_mn_view",
    "prefetch_tma_desc_raw",
    "real_col_to_stg128_fake_col",
    "real_col_to_stg128_half_fake_col",
    "stg128_fake_col_to_real_col",
    "stg128_half_fake_col_to_real_col",
    "stg_128",
    "stg_128_cs",
    "stg_128_bf16",
    "stg_128_bf16_cs",
    "stg_128_f16",
    "stg_128_f16_cs",
    "stg_64_bf16",
    "stg_64_f16",
    "tma_gather4",
    "tma_gather4_cached",
    "tma_gather4_prefetch",
    "tma_tile_load",
    "tma_tile_load_cached",
    "tma_tile_prefetch",
    "tma_tile_store",
    "view_paged_kv_as_blocks",
]
