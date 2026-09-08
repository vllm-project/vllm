# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import flydsl.expr as fx

MAX_BLOCKS = 80

_PACK_I32 = 4


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
        return fx.rocdl.global_load(
            byte_address,
            fx.Int32,
            vector_width=_PACK_I32,
            alignment=16,
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
        fx.rocdl.global_store(
            byte_address,
            value,
            alignment=16,
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
