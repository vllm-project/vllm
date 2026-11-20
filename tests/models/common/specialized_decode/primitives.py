# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The machinery a specialized decode's primitive tests need, no model in it.

This is the primitive tier (T1) of #68016, and it is a debugging aid rather
than the numerical contract. It reaches into a model's internals on purpose,
so it cannot survive a kernel rewrite. The contract is `DecodeNumerics` (T2),
which compares a decode's output at the public entry point and holds however
the kernel is written.

A primitive test calls a model's *real* device helper -- the one its kernels
call -- on every lane of a thin map kernel, and compares what comes back
against the closed-form float64 definition in `reference.py`. Nothing here
re-implements a rounding: `run` only moves words onto the GPU, hands them to
the helper under test and brings the result back, and the generators below
only build the inputs that make a rounding rule observable (the ties, the
subnormals, the overflow edges).

The per-model file supplies the helpers and the assertions; see
`tests/models/deepseek_v41/amd/mono/test_primitives.py` for the shape of one.
This file assumes only that the helpers are FlyDSL device functions a test
kernel can call. A model whose kernels emit their instructions inline, with no
callable helper to import, cannot be checked at this tier at all, and relies
on T2 plus a stage-level check instead.
"""

import hashlib
import importlib
import math
from functools import cache, cached_property
from pathlib import Path

import flydsl.compiler as flyc
import flydsl.expr as fx
import pytest
import torch
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import range_constexpr
from flydsl.expr.typing import Constexpr, Int64, T

from tests.models.common.specialized_decode import reference as R

DEV = "cuda"
F32, I32 = torch.float32, torch.int32
_BLOCK = 256


# ---------------------------------------------------------------- libraries
class HelperSet:
    """One copy of a model's device helpers: the `modules` of `package`.

    A model usually carries several. DeepSeek-V4.1 has one on the MoE and
    seam side and another on the attention side, and each is its own copy of
    the same rounding and packing rules, so each has to be checked on its
    own: a bug fixed in one is not fixed in the other.
    """

    def __init__(self, package: str, modules: tuple[str, ...]):
        self.package = package
        self.modules = modules

    @cached_property
    def _loaded(self) -> list:
        return [importlib.import_module(f"{self.package}.{m}") for m in self.modules]

    def find(self, name: str):
        """`name` as this set defines or imports it, else None."""
        for module in self._loaded:
            if hasattr(module, name):
                return getattr(module, name)
        return None


def own_helper(libs: dict[str, HelperSet], lib: str, name: str):
    """`libs[lib]`'s `name`, or None where that is an earlier set's object
    rather than a copy of its own, so each distinct helper is checked once
    against the set that carries it."""
    fn = libs[lib].find(name)
    for other in libs:
        if other == lib:
            return fn
        if fn is not None and libs[other].find(name) is fn:
            return None
    return None


def params(libs: dict[str, HelperSet], *names: str) -> list:
    """The `libs` keys carrying a copy of every helper of `names`, as pytest
    params. Empty `names` means every key."""
    return [
        pytest.param(lib, id=lib)
        for lib in libs
        if all(own_helper(libs, lib, n) is not None for n in names)
    ]


# ---------------------------------------------------------------- harness
def _rsrc(p):
    return bo.create_buffer_resource_from_addr(p)


def _load_words(src, at, n):
    """`n` Int32 at word `at` of `src`, in loads of up to 4 words."""
    out: list = []
    while len(out) < n:
        w = min(4, n - len(out))
        v = bo.buffer_load(_rsrc(src), at + len(out), vec_width=w, dtype=T.i32)
        out += [fx.Int32(v)] if w == 1 else [fx.Vector(v)[j] for j in range(w)]
    return [fx.Int32(x) for x in out]


@cache
def sources_digest(package: str) -> str:
    """Of every file of `package`: FlyDSL's compile cache keys a kernel on
    the sources of its own directory only, not on the helpers it calls."""
    h = hashlib.sha256()
    root = Path(importlib.import_module(package).__file__).parent
    for f in sorted(root.rglob("*.py")):
        h.update(f.read_bytes())
    return h.hexdigest()


@cache
def _map_kernel(key, fn, n_in, n_out):
    """A kernel whose thread i calls `fn` on words i n_in .. of `src`
    (Int32) and stores the n_out Int32 it returns at i n_out .. of `dst`.
    A wave is lanes 64 j .. 64 j + 63 of the rows."""

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


def run(key, fn, n_out, *cols, sources: str):
    """`fn` -- a device helper -- over the rows of the int32 / float32
    columns `cols`, on the GPU, one row per lane -> int32 [rows, n_out].

    `sources` keys the compile cache; pass `sources_digest(package)` of the
    model whose helpers are under test.
    """
    src = torch.cat(
        [(c.view(I32) if c.dtype == F32 else c).view(len(c), -1) for c in cols], 1
    )
    n = src.shape[0]
    pad = -n % _BLOCK
    src = torch.cat([src, src[:1].expand(pad, -1)]).contiguous().to(DEV)
    dst = torch.zeros(src.shape[0], n_out, dtype=I32, device=DEV)
    _map_kernel(key, fn, src.shape[1], n_out)(
        src.data_ptr(), dst.data_ptr(), src.shape[0] // _BLOCK, sources
    )
    torch.accelerator.synchronize()
    return dst[:n]


def f32(w):
    return fx.Int32(w).bitcast(fx.Float32)


def i32(v):
    return fx.Float32(v).bitcast(fx.Int32)


def bits(x):
    return x.float().view(I32)


# ---------------------------------------------------------------- inputs
def full(v):
    return torch.full((1,), v, device=DEV)


def gen(seed):
    return torch.Generator(device=DEV).manual_seed(seed)


def log_uniform(n, lo, hi, g, signed=True):
    x = torch.exp2(torch.empty(n, dtype=torch.float64, device=DEV).uniform_(
        lo, hi, generator=g))  # fmt: skip
    if signed:
        x = x * (torch.randint(0, 2, (n,), device=DEV, generator=g) * 2 - 1)
    return x.float()


def edge_floats():
    """Every f32 exponent field with the mantissas next to the code / rounding
    boundaries, positive."""
    mant = torch.tensor(
        [0, 1, 2, 0x200000, 0x3FFFFF, 0x400000, 0x400001, 0x7FFFFE, 0x7FFFFF]
    )
    bit_patterns = (torch.arange(255)[:, None] << 23) | mant
    return bit_patterns.flatten().to(I32).view(F32).to(DEV)


def neighbours(x):
    """`x` (float32) and its f32 neighbours."""
    inf = torch.full_like(x, math.inf)
    return torch.cat([torch.nextafter(x, -inf), x, torch.nextafter(x, inf)])


def fp8_bytes(fnuz):
    """Every E4M3 byte that is not a NaN."""
    b = torch.arange(256, device=DEV)
    nan = b == 0x80 if fnuz else (b & 0x7F) == 0x7F
    return b[~nan].to(torch.uint8)


def fp8_domain(fnuz):
    """Every E4M3 magnitude and every midpoint between neighbours, each with
    its f32 neighbours, both signs; past the range up to inf; 2^20 random."""
    fp8_dtype = torch.float8_e4m3fnuz if fnuz else torch.float8_e4m3fn
    codes = torch.arange(0, 127, dtype=torch.uint8).view(fp8_dtype)
    magnitudes = codes.double().to(DEV)
    midpoints = (magnitudes[1:] + magnitudes[:-1]) / 2
    past_max = torch.tensor([464.0, 480, 1e6, 3e38], device=DEV)
    values = neighbours(torch.cat([magnitudes, midpoints, past_max]).float())
    random_values = log_uniform(1 << 20, -14, 10, gen(0))
    infinities = torch.tensor([math.inf, -math.inf], device=DEV)
    values = torch.cat([values, -values, random_values, infinities])
    return torch.cat([values, values.new_tensor([1e-40, -1e-40])])


def all_byte_words(exclude=()):
    """Words covering every allowed byte value in every byte position, with
    random other bytes."""
    b = torch.tensor([v for v in range(256) if v not in exclude], device=DEV)
    allowed = b.to(torch.uint8)
    g = gen(10)
    words = []
    for pos in range(4):
        w = allowed[torch.randint(0, len(b), (len(b), 4), device=DEV, generator=g)]
        w[:, pos] = allowed
        words.append(w)
    return torch.cat(words).contiguous().view(I32).flatten()


def e2m1_bytes(codes):
    """E2M1 values of the 8 nibbles of int32 words, low nibble first."""
    nib = (codes.long()[:, None] >> (4 * torch.arange(8, device=DEV))) & 0xF
    return R.E2M1_VALUES.to(DEV)[nib]


# ---------------------------------------------------------------- assertions
def assert_equal(got, ref, what, x):
    """`got == ref` row for row; a failure shows the first rows and their x."""
    bad = (got != ref).view(len(ref), -1).any(1).nonzero().flatten()[:3]
    assert bad.numel() == 0, (
        f"{what} differs at x={x[bad].tolist()}: "
        f"got {got[bad].tolist()}, want {ref[bad].tolist()}"
    )
