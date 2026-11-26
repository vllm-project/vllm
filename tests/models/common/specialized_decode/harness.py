# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run one of a model's real device helpers on the GPU, one row per lane.

The FlyDSL half of the primitive tier (T1) of #68016. T1 is a debugging aid,
not the numerical contract; the contract is `DecodeNumerics` (T2), which
compares a decode at its public entry point and holds however the kernel is
written.

Nothing here computes anything. `run` builds a map kernel whose thread i calls
the helper under test on row i of the inputs and stores what it returns, so the
arithmetic under test is the model's own and never a re-implementation of it.
What it is compared against is `reference.py`; the inputs that make a rounding
rule observable, and the comparison itself, are in `inputs.py`.

The split is by what the code does, not by what it is written in: this file is
the only one that needs FlyDSL, and `reference.py` and `inputs.py` are plain
torch. A model whose helpers are not FlyDSL device functions a test kernel can
call -- one whose kernels emit their instructions inline, say -- cannot be
checked at this tier at all, and relies on T2 plus a stage-level check instead.
It can still use the other two files.

See `tests/models/deepseek_v41/amd/mono/test_primitives.py` for a caller.
"""

import hashlib
import importlib
from collections.abc import Callable
from functools import cache
from pathlib import Path
from typing import Any

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import range_constexpr
from flydsl.expr.typing import Constexpr, Int64, T

from tests.models.common.specialized_decode.inputs import DEV, F32, I32

# A FlyDSL device function. It takes and returns FlyDSL expressions rather than
# tensors, so there is nothing more precise to say about it than "callable".
DeviceFn = Callable[..., Any]

_BLOCK = 256


def _rsrc(pointer: Any) -> Any:
    return bo.create_buffer_resource_from_addr(pointer)


def _load_words(src: Any, at: Any, n: int) -> list[Any]:
    """`n` Int32 at word `at` of `src`, in loads of up to 4 words."""
    out: list = []
    while len(out) < n:
        w = min(4, n - len(out))
        v = bo.buffer_load(_rsrc(src), at + len(out), vec_width=w, dtype=T.i32)
        out += [fx.Int32(v)] if w == 1 else [fx.Vector(v)[j] for j in range(w)]
    return [fx.Int32(x) for x in out]


@cache
def _sources_digest(package: str) -> str:
    """Of every file of `package`: FlyDSL's compile cache keys a kernel on
    the sources of its own directory only, not on the helpers it calls, so a
    test kernel would otherwise keep running against a stale helper."""
    h = hashlib.sha256()
    root = Path(importlib.import_module(package).__file__).parent
    for f in sorted(root.rglob("*.py")):
        h.update(f.read_bytes())
    return h.hexdigest()


def _map_kernel(key: str, fn: DeviceFn, n_in: int, n_out: int) -> Any:
    """A kernel whose thread i calls `fn` on words i n_in .. of `src`
    (Int32) and stores the n_out Int32 it returns at i n_out .. of `dst`.
    A wave is lanes 64 j .. 64 j + 63 of the rows.

    Deliberately not memoised. `fn` is a fresh closure at every call site, so
    a cache on it could never hit; FlyDSL's own cache, keyed on the kernel
    name and `sources`, is what stops a recompile.
    """

    @flyc.kernel(name=f"mono_prim_{key}", known_block_size=[_BLOCK, 1, 1])
    def kern(src: Int64, dst: Int64):
        i = fx.block_idx.x * _BLOCK + fx.thread_idx.x
        outs = fn(*_load_words(src, i * n_in, n_in))
        for j in range_constexpr(n_out):
            bo.buffer_store(fx.Int32(outs[j]), _rsrc(dst), i * n_out + j)

    @flyc.jit
    def launch(src: Int64, dst: Int64, grid: fx.Int32, sources: Constexpr[str]):
        kern(src, dst).launch(grid=(grid,), block=(_BLOCK,))

    return launch


def run(
    key: str,
    fn: DeviceFn,
    n_out: int,
    *cols: torch.Tensor,
    package: str,
) -> torch.Tensor:
    """`fn` -- a device helper -- over the rows of the int32 / float32
    columns `cols`, on the GPU, one row per lane -> int32 [rows, n_out].

    `key` has to be distinct per call site: it names the kernel, and FlyDSL
    caches on that name and `package`'s digest, so two call sites sharing a
    key would both get whichever kernel compiled first and the second would
    quietly assert against the first one's helper. `package` is the model
    package the helper comes from (see `_sources_digest`).
    """
    rows = len(cols[0])
    assert all(len(c) == rows for c in cols), (
        f"every column needs the same number of rows, got {[len(c) for c in cols]}"
    )
    src = torch.cat(
        [(c.view(I32) if c.dtype == F32 else c).view(rows, -1) for c in cols], 1
    )
    pad = -rows % _BLOCK
    src = torch.cat([src, src[:1].expand(pad, -1)]).contiguous().to(DEV)
    dst = torch.zeros(src.shape[0], n_out, dtype=I32, device=DEV)
    _map_kernel(key, fn, src.shape[1], n_out)(
        src.data_ptr(), dst.data_ptr(), src.shape[0] // _BLOCK, _sources_digest(package)
    )
    torch.accelerator.synchronize()
    return dst[:rows]


def f32(word: Any) -> Any:
    """An Int32 expression as the Float32 of the same bits."""
    return fx.Int32(word).bitcast(fx.Float32)


def i32(value: Any) -> Any:
    """A Float32 expression as the Int32 of the same bits."""
    return fx.Float32(value).bitcast(fx.Int32)
