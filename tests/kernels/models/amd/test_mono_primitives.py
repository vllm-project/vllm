# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The device helpers of the AMD monokernels against float64 / exact
references (tests/models/mono_reference.py), one helper at a time, from a
test kernel that calls it on every lane.

Roundings, packings, scale codes and data movement are bit-exact: there is a
closed form for each, so a tolerance would only hide a wrong one.

Each model carries its own copy of these helpers, so each has to be checked
separately. This starts with DeepSeek-V4.1; a model is added by naming its
package in `_PKG`, its modules in `_LIBS`, and pointing one more `-k` command
at it from the step that already builds it."""

import hashlib
import importlib
import math
from functools import cache
from pathlib import Path

import pytest
import torch

from tests.models import mono_reference as R
from vllm.platforms import current_platform

if not current_platform.is_rocm():
    pytest.skip("the mono kernels need ROCm", allow_module_level=True)
pytest.importorskip("flydsl")
pytest.importorskip("aiter")

import flydsl.compiler as flyc  # noqa: E402
import flydsl.expr as fx  # noqa: E402
from aiter.ops.flydsl.kernels import buffer_ops as bo  # noqa: E402
from flydsl.expr import range_constexpr  # noqa: E402
from flydsl.expr.typing import Constexpr, Int64, T  # noqa: E402

from vllm.platforms.rocm import on_gfx942, on_mi3xx  # noqa: E402

if not on_mi3xx():
    pytest.skip("the mono kernels are MI300/MI350 only", allow_module_level=True)

_PKG = {
    "dsv41": "vllm.models.deepseek_v41.amd.mono",
}
_LIBS = {
    "dsv41": (
        "dsv41",
        ("common.ops", "common.mx", "common.gfx942", "stages.moe", "stages.seam"),
    ),
    "dsv41_attn": ("dsv41", ("attention.device",)),
}
# gfx942 builds only DeepSeek-V4.1's kernels.
_ON_942 = ("dsv41", "dsv41_attn")
DEV = "cuda"
F32, I32 = torch.float32, torch.int32


@cache
def _modules(lib):
    pkg, mods = _LIBS[lib]
    return [importlib.import_module(f"{_PKG[pkg]}.{m}") for m in mods]


def _find(lib, name):
    for m in _modules(lib):
        if hasattr(m, name):
            return getattr(m, name)
    return None


def _fn(lib, name):
    """Helper `name` of `lib`; None where it is an earlier library's
    (imported, not a copy)."""
    fn = _find(lib, name)
    for other in _LIBS:
        if other == lib:
            return fn
        if fn is not None and _find(other, name) is fn:
            return None


def _gfx942(lib):
    return lib in _ON_942 and _find("dsv41", "GFX942")


def _libs(*names, gfx950_only=False):
    """The libraries defining every helper of `names`."""
    out = []
    for lib in _LIBS:
        skip = on_gfx942() and (gfx950_only or lib not in _ON_942)
        marks = pytest.mark.skipif(skip, reason="gfx950 only")
        out.append(pytest.param(lib, marks=marks, id=lib))
    return [p for p in out if all(_has(p.values[0], n) for n in names)]


def _has(lib, name):
    try:
        return _fn(lib, name) is not None
    except ImportError:
        return False


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
def _sources_digest():
    """Of every file of the mono packages: FlyDSL's compile cache keys a
    kernel on the sources of its own directory only, not on these helpers."""
    h = hashlib.sha256()
    for pkg in _PKG.values():
        root = Path(importlib.import_module(pkg).__file__).parent
        for f in sorted(root.rglob("*.py")):
            h.update(f.read_bytes())
    return h.hexdigest()


@cache
def _map_kernel(key, fn, n_in, n_out):
    """A kernel whose thread i calls `fn` on words i n_in .. of `src`
    (Int32) and stores the n_out Int32 it returns at i n_out .. of `dst`.
    A wave is lanes 64 j .. 64 j + 63 of the rows."""

    @flyc.kernel(name=f"mono_prim_{key}", known_block_size=[256, 1, 1])
    def kern(src: Int64, dst: Int64):
        i = fx.block_idx.x * 256 + fx.thread_idx.x
        outs = fn(*_load_words(src, i * n_in, n_in))
        for j in range_constexpr(n_out):
            bo.buffer_store(fx.Int32(outs[j]), _rsrc(dst), i * n_out + j)

    @flyc.jit
    def launch(src: Int64, dst: Int64, grid: fx.Int32, sources: Constexpr[str]):
        kern(src, dst).launch(grid=(grid,), block=(256,))

    return launch


def _run(key, fn, n_out, *cols):
    """`fn` over the rows of the int32 / float32 columns `cols` -> int32
    [rows, n_out]."""
    src = torch.cat(
        [(c.view(I32) if c.dtype == F32 else c).view(len(c), -1) for c in cols], 1
    )
    n = src.shape[0]
    pad = -n % 256
    src = torch.cat([src, src[:1].expand(pad, -1)]).contiguous().to(DEV)
    dst = torch.zeros(src.shape[0], n_out, dtype=I32, device=DEV)
    _map_kernel(key, fn, src.shape[1], n_out)(
        src.data_ptr(), dst.data_ptr(), src.shape[0] // 256, _sources_digest()
    )
    torch.accelerator.synchronize()
    return dst[:n]


def f32(w):
    return fx.Int32(w).bitcast(fx.Float32)


def i32(v):
    return fx.Float32(v).bitcast(fx.Int32)


def _bits(x):
    return x.float().view(I32)


def _full(v):
    return torch.full((1,), v, device=DEV)


def _gen(seed):
    return torch.Generator(device=DEV).manual_seed(seed)


def _log_uniform(n, lo, hi, g, signed=True):
    x = torch.exp2(torch.empty(n, dtype=torch.float64, device=DEV).uniform_(
        lo, hi, generator=g))  # fmt: skip
    if signed:
        x = x * (torch.randint(0, 2, (n,), device=DEV, generator=g) * 2 - 1)
    return x.float()


def _edge_floats():
    """Every f32 exponent field with the mantissas next to the code / rounding
    boundaries, positive."""
    mant = torch.tensor(
        [0, 1, 2, 0x200000, 0x3FFFFF, 0x400000, 0x400001, 0x7FFFFE, 0x7FFFFF]
    )
    bits = (torch.arange(255)[:, None] << 23) | mant
    return bits.flatten().to(I32).view(F32).to(DEV)


def _neighbours(x):
    """`x` (float32) and its f32 neighbours."""
    inf = torch.full_like(x, math.inf)
    return torch.cat([torch.nextafter(x, -inf), x, torch.nextafter(x, inf)])


def _assert_equal(got, ref, what, x):
    """`got == ref` row for row; a failure shows the first rows and their x."""
    bad = (got != ref).view(len(ref), -1).any(1).nonzero().flatten()[:3]
    assert bad.numel() == 0, (
        f"{what} differs at x={x[bad].tolist()}: "
        f"got {got[bad].tolist()}, want {ref[bad].tolist()}"
    )


# ---------------------------------------------------------------- bf16
@pytest.mark.parametrize("lib", _libs("bf16_round", "bf16_pair", "bf_lo", "bf_hi"))
def test_bf16_round_and_pack(lib):
    """bf16_round, bf16_pair, bf_lo, bf_hi on every bf16 pattern next to its
    rounding boundaries (ties, carries into the exponent, overflow to inf,
    subnormals) and 2^20 random words: RNE, bit for bit."""
    ops = {n: _fn(lib, n) for n in ("bf16_round", "bf16_pair", "bf_lo", "bf_hi")}

    def body(a, b):
        p = fx.Float32(ops["bf16_pair"](f32(a), f32(b))).bitcast(fx.Int32)
        return [
            i32(ops["bf16_round"](f32(a))),
            p,
            i32(ops["bf_lo"](p)),
            i32(ops["bf_hi"](p)),
        ]

    hi = torch.arange(1 << 16, dtype=torch.int64)[:, None] << 16
    lo = torch.tensor([0, 1, 0x7FFF, 0x8000, 0x8001, 0xFFFF])
    g = torch.Generator().manual_seed(0)
    rnd = torch.randint(-(1 << 31), 1 << 31, (1 << 20,), generator=g)
    a = torch.cat([(hi | lo).flatten(), rnd]).to(I32).view(F32).to(DEV)
    a = torch.where(torch.isnan(a), torch.zeros_like(a), a)
    b = a.roll(1)
    got = _run(f"bf16_{lib}", body, 4, a, b)
    ra, rb = _bits(R.bf16_rne(a)), _bits(R.bf16_rne(b))
    assert torch.equal(got[:, 0], ra), "bf16_round"
    assert torch.equal(got[:, 1], (ra >> 16 & 0xFFFF) | (rb & -65536)), "bf16_pair"
    assert torch.equal(got[:, 2], ra), "bf_lo"
    assert torch.equal(got[:, 3], rb), "bf_hi"
    low = a.view(I32) & 0xFFFF
    tie_up = (low == 0x8000) & (a.view(I32) >> 16 & 1 == 1)
    assert tie_up.any() and ((low == 0x8000) & ~tie_up).any()
    assert (torch.isinf(R.bf16_rne(a)) & torch.isfinite(a)).any()
    assert ((a.abs() < R.FLT_MIN) & (a != 0)).any()


# ---------------------------------------------------------------- fp8
def _fp8_domain(fnuz):
    """Every E4M3 magnitude and every midpoint between neighbours, each with
    its f32 neighbours, both signs; past the range up to inf; 2^20 random."""
    dt = torch.float8_e4m3fnuz if fnuz else torch.float8_e4m3fn
    vals = torch.arange(0, 127, dtype=torch.uint8).view(dt).double().to(DEV)
    mids = (vals[1:] + vals[:-1]) / 2
    x = torch.cat([vals, mids, torch.tensor([464.0, 480, 1e6, 3e38], device=DEV)])
    x = _neighbours(x.float())
    rnd = _log_uniform(1 << 20, -14, 10, _gen(0))
    x = torch.cat([x, -x, rnd, torch.tensor([math.inf, -math.inf], device=DEV)])
    return torch.cat([x, x.new_tensor([1e-40, -1e-40])])


@pytest.mark.parametrize("lib", _libs("fp8_pack4", "clamp_fp8"))
def test_fp8_clamp_and_pack(lib):
    """fp8_pack4(clamp_fp8(x)): the saturating E4M3 RNE of x, byte for byte
    (FNUZ with a maximum of 224 where the library targets gfx942)."""
    clamp, pack = _fn(lib, "clamp_fp8"), _fn(lib, "fp8_pack4")
    fnuz = _gfx942(lib)
    x = _fp8_domain(fnuz)
    x = torch.cat([x, x.new_zeros(-x.numel() % 4)])
    got = _run(
        f"fp8_{lib}",
        lambda *w: [pack(*[clamp(f32(v)) for v in w])],
        1,
        *x.view(-1, 4).T,
    )
    dt = torch.float8_e4m3fnuz if fnuz else torch.float8_e4m3fn
    ref = R.to_e4m3_rne(x, fnuz).to(dt).view(torch.uint8)
    _assert_equal(got.view(torch.uint8).flatten(), ref, "fp8", x)
    fmax = 224.0 if fnuz else 448.0
    q = x.double().abs() / (2.0**-10 if fnuz else 2.0**-9)
    assert ((q == q.floor() + 0.5) & (q < 8)).any()
    assert (x.abs() > fmax).any() and (ref.view(dt).double().abs() == fmax).any()


# ---------------------------------------------------------------- scale codes
_CODE_RULES = {
    "code_ceil": lambda v, fmax: R.ceil_code(v).clamp(1, 254),
    "vllm_mx_code": R.mx_code_vllm,
    "mx_code": R.mx_code_vllm,
    "e8m0_roundup": lambda v, fmax: R.e8m0_roundup6(v),
}


def _code_libs():
    return [p for p in _libs() if any(_has(p.values[0], n) for n in _CODE_RULES)]


@pytest.mark.parametrize("lib", _code_libs())
def test_scale_codes(lib):
    """code_ceil, vllm_mx_code / mx_code and e8m0_roundup on every f32
    exponent with the mantissas around a power of two, the values whose
    quotient by 448 or 6 is a power of two and their neighbours, 0 and the
    subnormals; pow2 / rcp_pow2 on every code they take. Bit-exact."""
    names = [n for n in _CODE_RULES if _has(lib, n)]
    fns = [_fn(lib, n) for n in names]
    p2, rp2 = _fn(lib, "pow2"), _fn(lib, "rcp_pow2")

    def body(v, code):
        out = [fx.Int32(f(f32(v))) for f in fns]
        out.append(i32(p2(code)) if p2 else code)
        out.append(i32(rp2(code)) if rp2 else code)
        return out

    k = torch.arange(-149, 128, dtype=torch.float64, device=DEV)
    pow2s = torch.exp2(k)
    edges = torch.cat([448 * pow2s, 6 * pow2s, pow2s * R.RCP6_F32**-1])
    v = torch.cat([_edge_floats(), _neighbours(edges[edges < 3e38].float())])
    v = v[torch.isfinite(v) & (v >= 0)]
    code = torch.arange(v.numel(), device=DEV, dtype=I32) % 253 + 1
    got = _run(f"codes_{lib}", body, len(fns) + 2, v, code)
    fmax = _find(lib, "FP8_MAX")
    for j, n in enumerate(names):
        _assert_equal(got[:, j], _CODE_RULES[n](v, fmax), n, v)
    if p2:
        assert torch.equal(got[:, -2].view(F32).double(), R.e8m0(code))
    if rp2:
        assert torch.equal(got[:, -1].view(F32).double(), 1 / R.e8m0(code))
    assert (v == 0).any() and ((v > 0) & (v < R.FLT_MIN)).any()


# ---------------------------------------------------------------- conversions
def _fp8_bytes(fnuz):
    b = torch.arange(256, device=DEV)
    nan = b == 0x80 if fnuz else (b & 0x7F) == 0x7F
    return b[~nan].to(torch.uint8)


@pytest.mark.parametrize("lib", _libs("fp8x8_bf16"))
def test_fp8x8_bf16(lib):
    """fp8x8_bf16(w0, w1, scale): every E4M3 byte (FNUZ on gfx942) in every
    position times a power-of-two scale -> bf16, exact."""
    cvt = _fn(lib, "fp8x8_bf16")
    fnuz = _gfx942(lib)
    b = _fp8_bytes(fnuz)
    words = torch.stack([b.roll(j) for j in range(8)], 1).contiguous().view(I32)
    codes = torch.arange(110, 145, device=DEV)
    w = words.repeat(codes.numel(), 1)
    c = codes.repeat_interleave(words.shape[0])
    got = _run(
        f"f8bf_{lib}",
        lambda w0, w1, sc: [fx.Vector(cvt(w0, w1, f32(sc << 23)))[j] for j in range(4)],
        4,
        w,
        c.to(I32),
    )
    ref = R.dequant_fp8(w.contiguous().view(torch.uint8), c[:, None], (1, 8), fnuz)
    _assert_equal(got.view(torch.bfloat16).double(), ref, "fp8x8_bf16", c)


def _all_byte_words(exclude=()):
    """Words covering every allowed byte value in every byte position, with
    random other bytes."""
    b = torch.tensor([v for v in range(256) if v not in exclude], device=DEV)
    allowed = b.to(torch.uint8)
    g = _gen(10)
    words = []
    for pos in range(4):
        w = allowed[torch.randint(0, len(b), (len(b), 4), device=DEV, generator=g)]
        w[:, pos] = allowed
        words.append(w)
    return torch.cat(words).contiguous().view(I32).flatten()


@pytest.mark.parametrize("lib", _libs("ocp_to_fnuz"))
def test_ocp_to_fnuz(lib):
    """Every OCP byte in every position -> the FNUZ byte of half its value:
    the same byte, but OCP's -0 (0x80, the FNUZ NaN) to 0x00."""
    cvt = _fn(lib, "ocp_to_fnuz")
    w = _all_byte_words()
    got = _run(f"o2f_{lib}", lambda w: [cvt(w)], 1, w).view(torch.uint8).flatten()
    ocp = w.view(torch.uint8)
    _assert_equal(got, torch.where(ocp == 0x80, 0, ocp), "ocp_to_fnuz", ocp)
    ok = (ocp & 0x7F) != 0x7F
    halved = got[ok].view(torch.float8_e4m3fnuz).double() * 2
    assert torch.equal(halved, ocp[ok].view(torch.float8_e4m3fn).double())


def _e2m1_bytes(codes):
    """E2M1 values of the 8 nibbles of int32 words, low nibble first."""
    nib = (codes.long()[:, None] >> (4 * torch.arange(8, device=DEV))) & 0xF
    return R.E2M1.to(DEV)[nib]


@pytest.mark.parametrize("lib", _libs("fp4x8_fnuz"))
def test_fp4x8_fnuz(lib):
    """Eight E2M1 nibbles (no -0) -> FNUZ bytes of the values times 2^-7,
    the even K in the first word, the odd in the second: exact."""
    cvt = _fn(lib, "fp4x8_fnuz")
    neg0 = [v for v in range(256) if (v & 0xF) == 8 or (v >> 4) == 8]
    w = _all_byte_words(exclude=neg0)
    got = _run(f"f4f_{lib}", lambda w: list(cvt(w)), 2, w)
    vals = got.contiguous().view(torch.uint8).view(torch.float8_e4m3fnuz).double()
    ref = _e2m1_bytes(w)[:, [0, 2, 4, 6, 1, 3, 5, 7]] * 2.0**-7
    _assert_equal(vals.view(-1, 8), ref, "fp4x8_fnuz", w[:, None])


# ---------------------------------------------------------------- layouts
@pytest.mark.parametrize("lib", _libs("scale_index"))
def test_scale_index(lib):
    """scale_index(row, col, cols) on every scale of a 64 x 256 tile is where
    AITER's shuffle_scale `view(rows / 32, 2, 16, cols / 8, 2, 4)
    .permute(0, 3, 5, 2, 4, 1)` puts that scale (host integers)."""
    rows, cols = 64, 256
    count = torch.arange(rows * cols)
    shuffled = count.view(rows // 32, 2, 16, cols // 8, 2, 4)
    shuffled = shuffled.permute(0, 3, 5, 2, 4, 1).flatten()
    where = torch.empty_like(count)
    where[shuffled] = torch.arange(rows * cols)
    r, c = count // cols, count % cols
    assert torch.equal(_fn(lib, "scale_index")(r, c, cols), where)
