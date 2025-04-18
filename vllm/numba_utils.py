# SPDX-License-Identifier: Apache-2.0
"""
Utilities for numba-based operations
"""

from ctypes import c_void_p
from typing import overload

from llvmlite import ir
from numba import jit, types
from numba.core.base import BaseContext
from numba.core.cgutils import raw_memcpy
from numba.core.typing.context import BaseContext as TypingContext
from numba.core.typing.templates import Signature
from numba.extending import intrinsic


@overload  # type: ignore[misc]
def numba_array_memcpy(dst_arr: c_void_p, dst_offset: int, src_arr: c_void_p,
                       src_offset: int, elem_count: int) -> None:
    """
    Copy a block of data within a numpy array for numba nopython mode.

    Parameters:
        dst_arr: c_void_p           # pointer to destination array buffer
        dst_offset: int             # start offset (in elements) for destination
        src_arr: c_void_p           # pointer to source array buffer
        src_offset: int             # start offset (in elements) for source
        elem_count: int             # number of elements to copy
    """
    ...


@intrinsic(inline=True)
def numba_array_memcpy(
    typingctx: TypingContext,
    dst_arr: types.ArrayCTypes,
    dst_offset: types.Integer,
    src_arr: types.ArrayCTypes,
    src_offset: types.Integer,
    elem_count: types.Integer,
):
    """
    LLVM intrinsic for `numba_array_memcpy`

    This is the LLVM IR code generator, for actual func signature 
    please refer to the overload above.
    """

    assert dst_arr.dtype == src_arr.dtype, \
        "dst_arr and src_arr must have the same dtype"

    def codegen(
        context: BaseContext,
        builder: ir.IRBuilder,
        signature: Signature,
        args: tuple[
            ir.Value,
            ir.Value,
            ir.Value,
            ir.Value,
            ir.Value,
        ],
    ):
        dst, dst_offset, src, src_offset, elem_count = args

        dst_ptr = builder.gep(dst, [dst_offset])
        src_ptr = builder.gep(src, [src_offset])
        item_size = context.get_abi_sizeof(dst_ptr.type)

        raw_memcpy(builder, dst_ptr, src_ptr, elem_count, item_size)

        return context.get_dummy_value()

    sig = types.void(
        types.CPointer(dst_arr.dtype),
        dst_offset,
        types.CPointer(src_arr.dtype),
        src_offset,
        elem_count,
    )

    return sig, codegen


@jit(nopython=True, inline="always")
def numba_cdiv(a: int, b: int) -> int:
    """
    Compute integer ceiling division in numba.
    """
    return -(a // -b)
