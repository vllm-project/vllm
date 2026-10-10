# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""gemm1 tile of the mono MoE launch: AITER's a4w4 stage-1 body
(``aiter.ops.flydsl.kernels.mxfp4_gemm1._gemm1_body``: MXFP4 x MXFP4, inline
quant of x, SiTUv2, the MXFP4 intermediate and its scales out), unchanged
except for how it stores its outputs.

With ``wt_out`` the body's global stores (the intermediate and its scales) are
device scope, so they write through the XCD's L2 and the m-block counter needs
no L2 writeback before a gemm2 tile on another XCD reads them. The body builds
its global outputs with ``_global_scalar_tiles`` and writes them only through
``_scalar_store``; both are swapped in AITER's module for the duration of the
trace and restored after it.
"""

import contextlib

import flydsl.expr as fx
from aiter.ops.flydsl.kernels import communication_ops_utils as comm_ops
from aiter.ops.flydsl.kernels import mxfp4_gemm1 as _aiter_g1
from flydsl._mlir.dialects import llvm as _llvm


class _GlobalOut:
    """Stand-in for a global output view: its base address and element type."""

    def __init__(self, addr_i64, numeric_cls):
        self.addr_i64 = addr_i64
        self.numeric_cls = numeric_cls
        self.stores = 0


def _wt_store(out, idx, value, numeric_cls):
    assert numeric_cls is out.numeric_cls
    nbytes = numeric_cls.width // 8
    _llvm.StoreOp(
        numeric_cls(value).ir_value(),
        comm_ops._ptr_plus(out.addr_i64, idx, nbytes),
        alignment=nbytes,
        ordering=_llvm.AtomicOrdering.monotonic,
        syncscope=fx.rocdl.SyncScope.AgentOneAs,
    )
    out.stores += 1


@contextlib.contextmanager
def _write_through_outputs():
    tiles_fn = _aiter_g1._global_scalar_tiles
    store_fn = _aiter_g1._scalar_store
    outs = []

    def global_tiles(addr_i64, numeric_cls, num_elems):
        outs.append(_GlobalOut(addr_i64, numeric_cls))
        return outs[-1]

    def scalar_store(tiles, idx, value, numeric_cls):
        if isinstance(tiles, _GlobalOut):
            _wt_store(tiles, idx, value, numeric_cls)
        else:
            store_fn(tiles, idx, value, numeric_cls)

    _aiter_g1._global_scalar_tiles = global_tiles
    _aiter_g1._scalar_store = scalar_store
    try:
        yield outs
    finally:
        _aiter_g1._global_scalar_tiles = tiles_fn
        _aiter_g1._scalar_store = store_fn
    # The intermediate, then its scales (one view per element width, the
    # tile shape picks which one is written). Anything else means AITER
    # changed how the body writes them, and stores that miss the swap would
    # race with gemm2 on another XCD.
    if len(outs) < 2 or outs[0].stores == 0 or not any(o.stores for o in outs[1:]):
        raise RuntimeError(
            "mono MoE: AITER's _gemm1_body no longer writes its outputs through "
            "_global_scalar_tiles/_scalar_store; update stages/gemm1.py"
        )


def _gemm1_body(*args, wt_out=False, **kwargs):
    if not wt_out:
        return _aiter_g1._gemm1_body(*args, **kwargs)
    with _write_through_outputs():
        return _aiter_g1._gemm1_body(*args, **kwargs)
