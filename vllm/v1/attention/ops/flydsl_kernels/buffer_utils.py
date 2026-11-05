# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Buffer-resource (V#) views shared by the FlyDSL decode kernels.

The descriptor sits in SGPRs and each access adds a 32-bit voffset, so offsets
into one view must stay below 4 GiB; move the base with ``base_byte_offset``
for anything further away.
"""

import flydsl.expr as fx

# Nominal view extent. The real bound is the descriptor's num_records, which
# make_buffer_tensor() leaves at 0xFFFFFFFF bytes (no OOB checking).
_VIEW_ELEMS = 0xFFFFFFFF


def buffer_view(tensor, elem, unit_elems=1, base_byte_offset=None):
    """Buffer view of a kernel tensor argument, indexed in ``elem`` elements.

    ``view[i]`` loads or stores one element. With ``unit_elems > 1``,
    :func:`load_vec` / :func:`store_vec` access ``unit_elems`` consecutive
    elements starting at any element index ``i``.
    """
    it = fx.get_iter(tensor)
    if base_byte_offset is not None:
        it = fx.add_offset(fx.recast_iter(fx.Uint8, it), base_byte_offset)
    elem_ptr = fx.PointerType.get(elem.ir_type, fx.AddressSpace.Global, elem.width // 8)
    it = fx.recast_iter(elem_ptr, it)
    if unit_elems == 1:
        layout = fx.make_layout(_VIEW_ELEMS, 1)
    else:
        layout = fx.make_layout((_VIEW_ELEMS, unit_elems), (1, 1))
    return fx.rocdl.make_buffer_tensor(fx.make_view(it, layout))


def _copy_atom(elem, n, cache_modifier=0):
    return fx.make_copy_atom(fx.rocdl.BufferCopy(n * elem.width, cache_modifier), elem)


def load_vec(view, index, n, cache_modifier=0):
    """Load ``n`` elements at element ``index`` of a ``unit_elems=n`` view."""
    elem = view.element_type
    frag = fx.make_rmem_tensor(n, elem)
    fx.copy(_copy_atom(elem, n, cache_modifier), fx.slice(view, (index, None)), frag)
    return fx.Vector(frag.load())


def store_vec(view, index, value, n):
    """Store the ``n``-element vector ``value`` at element ``index``."""
    elem = view.element_type
    frag = fx.make_rmem_tensor(n, elem)
    frag.store(value)
    fx.copy(_copy_atom(elem, n), frag, fx.slice(view, (index, None)))
