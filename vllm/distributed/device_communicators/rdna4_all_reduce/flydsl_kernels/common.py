# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import flydsl.expr as fx

MAX_BLOCKS = 80

_PACK_I32 = 4


def global_pointer(address, dtype, alignment):
    return fx.inttoptr(
        fx.PointerType.get(
            elem_ty=dtype.ir_type,
            address_space=fx.AddressSpace.Global,
            alignment=alignment,
        ),
        address,
    )


def _pack_view(address, pack_index):
    pointer_type = fx.PointerType.get(
        elem_ty=fx.Int32.ir_type,
        address_space=fx.AddressSpace.Global,
        alignment=16,
    )
    pointer = fx.inttoptr(pointer_type, address)
    pointer = pointer + fx.Int64(pack_index) * fx.Int64(_PACK_I32)
    return fx.make_view(pointer, fx.make_layout(_PACK_I32, 1))


def load_pack_128b(address, pack_index, *, nontemporal: bool = False):
    if nontemporal:
        byte_address = address + fx.Int64(pack_index) * fx.Int64(16)
        return fx.generic_load(
            global_pointer(byte_address, fx.Int32, 16),
            count=_PACK_I32,
            nontemporal=True,
        )
    copy_atom = fx.make_copy_atom(fx.UniversalCopy128b(), fx.Int32)
    register = fx.make_rmem_tensor(_PACK_I32, fx.Int32)
    fx.copy(
        copy_atom,
        _pack_view(address, pack_index),
        register,
    )
    return fx.memref_load_vec(register)


def store_pack_128b(
    address,
    pack_index,
    value,
    *,
    nontemporal: bool = False,
):
    if nontemporal:
        byte_address = address + fx.Int64(pack_index) * fx.Int64(16)
        fx.generic_store(
            global_pointer(byte_address, fx.Int32, 16),
            value,
            nontemporal=True,
        )
    else:
        copy_atom = fx.make_copy_atom(fx.UniversalCopy128b(), fx.Int32)
        register = fx.make_rmem_tensor(_PACK_I32, fx.Int32)
        fx.memref_store_vec(value, register)
        fx.copy(
            copy_atom,
            register,
            _pack_view(address, pack_index),
        )
