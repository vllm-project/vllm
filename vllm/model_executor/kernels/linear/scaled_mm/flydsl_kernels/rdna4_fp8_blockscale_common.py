# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Shared host validation and gfx120x helpers for block-scaled FP8 GEMM."""

import flydsl.expr as fx
from flydsl.expr.typing import Vector as Vec

WMMA_M = 16
WMMA_N = 16
WMMA_K = 16
SCALE_K = 128
WAVE_SIZE = 32


def _make_buffer(tensor, elem_ty, width: int, num_records_bytes):
    """Build a flat raw-buffer view addressable by an element offset."""
    alignment = max(1, elem_ty.width * width // 8)
    ptr_ty = fx.PointerType.get(elem_ty.ir_type, fx.AddressSpace.Global, alignment)
    base = fx.inttoptr(ptr_ty, fx.Int64(fx.ptrtoint(fx.get_iter(tensor))))
    view = fx.Tensor(fx.make_view(base, fx.make_layout((width, 1), (1, 1))))
    return fx.rocdl.make_buffer_tensor(
        view,
        max_size=False,
        num_records_bytes=num_records_bytes,
    )


def _load_f32(buffer, index):
    atom = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32)
    fragment = fx.make_rmem_tensor(1, fx.Float32)
    fx.copy(atom, fx.slice(buffer, (None, index)), fragment)
    return Vec(fragment.load())[0]


def _load_fp8_fragment_buffer(buffer, index, fragment):
    atom = fx.make_copy_atom(fx.rocdl.BufferCopy64b(), fx.Float8E4M3FN)
    fx.copy(atom, fx.slice(buffer, (None, index)), fragment)


def _load_fp8_fragment_ptr(ptr, index, fragment):
    packed = fx.ptr_load(ptr + index, result_type=fx.Vector.make_type(8, fx.Uint8))
    fragment.store(Vec(packed.bitcast(fx.Float8E4M3FN)))


def _store_bf16(buffer, index, value):
    atom = fx.make_copy_atom(fx.rocdl.BufferCopy16b(), fx.BFloat16)
    fragment = fx.make_rmem_tensor(1, fx.BFloat16)
    fragment.store(Vec.from_elements([value], fx.BFloat16))
    fx.copy(atom, fragment, fx.slice(buffer, (None, index)))


def _f32_to_bf16_rne(value):
    """Round FP32 to BF16 without gfx120x NaN-classification lowering."""
    bits = fx.Float32(value).bitcast(fx.Uint32)
    rounded = bits + fx.Uint32(0x7FFF) + ((bits >> fx.Uint32(16)) & fx.Uint32(1))
    return fx.Uint16(rounded >> fx.Uint32(16)).bitcast(fx.BFloat16)
