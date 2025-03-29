# SPDX-License-Identifier: Apache-2.0

from numba import types
from numba.extending import intrinsic
from numba.core.cgutils import raw_memcpy

@intrinsic
def numba_array_memcpy(typingctx, dst_arr, dst_offset, src_arr, src_offset, elem_count):
    """calling C memcpy for numpy array in numba no-python code"""

    assert dst_arr.dtype == src_arr.dtype

    def codegen(context, builder, signature, args):
        dst, dst_offset, src, src_offset, elem_count = args

        dst_ptr = builder.gep(dst, [dst_offset])
        src_ptr = builder.gep(src, [src_offset])
        item_size = context.get_abi_sizeof(dst_ptr.type)

        raw_memcpy(builder, dst_ptr, src_ptr, elem_count, item_size)

        return context.get_dummy_value()
    
    sig = types.void(
        types.CPointer(dst_arr.dtype),
        dst_offset,
        types.CPointer(dst_arr.dtype),
        src_offset,
        elem_count,
    )
    
    return sig, codegen
