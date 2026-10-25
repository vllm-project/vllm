# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The device helpers of the AMD monokernels against float64 / exact
references (tests/models/mono_reference.py), one helper at a time, from a
test kernel that calls it on every lane.

The helper libraries are separate copies, so each runs here:
  dsv41, dsv41_attn    vllm/models/deepseek_v41/amd/mono (common, stages,
                       attention/device.py); MiniMax-M3 uses its common
  dsv4pro, dsv4pro_attn vllm/models/deepseek_v4/amd/mono
  kimi                 vllm/models/kimi_k3/amd/specialized_decode/mono
  minimax              vllm/models/minimax_m3/amd/specialized_decode/mono

Roundings, packings, scale codes and data movement are bit-exact against the
references. The hardware transcendentals are held to AMD's documented 1 ulp
and the helpers built on them to twice the error measured over 20 seeds."""

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

from vllm.platforms.rocm import on_gfx942, on_gfx950  # noqa: E402

if not (on_gfx942() or on_gfx950()):
    pytest.skip("the mono kernels target gfx942 / gfx950", allow_module_level=True)

_PKG = {
    "dsv41": "vllm.models.deepseek_v41.amd.mono",
    "dsv4pro": "vllm.models.deepseek_v4.amd.mono",
    "kimi": "vllm.models.kimi_k3.amd.specialized_decode.mono",
    "minimax": "vllm.models.minimax_m3.amd.specialized_decode.mono",
}
_LIBS = {
    "dsv41": (
        "dsv41",
        ("common.ops", "common.mx", "common.gfx942", "stages.moe", "stages.seam"),
    ),
    "dsv41_attn": ("dsv41", ("attention.device",)),
    "dsv4pro": ("dsv4pro", ("common.ops", "common.mx", "stages.moe", "stages.seam")),
    "dsv4pro_attn": ("dsv4pro", ("attention.device",)),
    "kimi": ("kimi", ("common.ops", "common.mx", "common.ranks", "stages.moe")),
    "kimi_kda": ("kimi", ("attention.kda",)),
    "minimax": ("minimax", ("a16w4", "ffn")),
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
    """Helper ``name`` of ``lib``; None where it is an earlier library's
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
    """The libraries defining every helper of ``names``."""
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
    """``n`` Int32 at word ``at`` of ``src``, in loads of up to 4 words."""
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
    """A kernel whose thread i calls ``fn`` on words i n_in .. of ``src``
    (Int32) and stores the n_out Int32 it returns at i n_out .. of ``dst``.
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
    """``fn`` over the rows of the int32 / float32 columns ``cols`` -> int32
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
    """``x`` (float32) and its f32 neighbours."""
    inf = torch.full_like(x, math.inf)
    return torch.cat([torch.nextafter(x, -inf), x, torch.nextafter(x, inf)])


def _assert_equal(got, ref, what, x):
    """``got == ref`` row for row; a failure shows the first rows and their x."""
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


# ---------------------------------------------------------------- fp4
def _e2m1_bytes(codes):
    """E2M1 values of the 8 nibbles of int32 words, low nibble first."""
    nib = (codes.long()[:, None] >> (4 * torch.arange(8, device=DEV))) & 0xF
    return R.E2M1.to(DEV)[nib]


@pytest.mark.parametrize("lib", _libs("dequant", gfx950_only=True))
def test_fp4_dequant(lib):
    """a16w4.dequant (cvt_scalef32_pk_bf16_fp4): every byte in every
    position of a word, times every scale code the weights take: exact."""
    deq = _fn(lib, "dequant")
    b = torch.arange(256, device=DEV, dtype=torch.int64)
    rnd = torch.randint(0, 1 << 32, (4096,), device=DEV, generator=_gen(1))
    words = torch.cat([b * 0x01010101, rnd])
    codes = torch.arange(100, 141, device=DEV)
    w = words.repeat(codes.numel()).to(I32)
    c = codes.repeat_interleave(words.numel())
    got = _run(
        f"deq_{lib}",
        lambda w, c: [fx.Vector(deq(w, f32(c << 23)))[j] for j in range(4)],
        4,
        w,
        c.to(I32),
    )
    ref = _e2m1_bytes(w) * R.e8m0(c)[:, None]
    _assert_equal(got.view(torch.bfloat16).double(), ref, "dequant", w[:, None])


def _fp4_domain(dtype):
    """Every E2M1 value and midpoint, past 6 up to 100, with their ``dtype``
    neighbours and both signs, times scale codes 100 .. 140; zeros with a
    scale of 0 (an all-zero group's code 0)."""
    m = R.E2M1[:8].tolist() + [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 7.0, 100.0]
    codes = torch.arange(100, 141, dtype=torch.float64)
    x = (torch.tensor(m, dtype=torch.float64)[:, None] * R.e8m0(codes)).flatten()
    x = _neighbours(x.float().to(DEV)).to(dtype)
    sc = R.e8m0(codes.to(DEV)).repeat(len(m)).repeat(3)
    x, sc = torch.cat([x, -x]), torch.cat([sc, sc])
    perm = torch.randperm(x.numel(), device=DEV, generator=_gen(2))
    x = torch.cat([x[perm], x.new_zeros(64)])
    sc = torch.cat([sc[perm], sc.new_zeros(64)])
    n = x.numel() // 8 * 8
    return x[:n].view(-1, 8), sc[:n].view(-1, 8)[:, 0]


@pytest.mark.parametrize("bf16", [False, True], ids=["f32", "bf16"])
@pytest.mark.parametrize("lib", _libs("pk_fp4_f32", "pk_fp4_bf16", gfx950_only=True))
def test_fp4_quant(lib, bf16):
    """pk_fp4_f32 / pk_fp4_bf16 (cvt_scalef32_pk_fp4_*): x / scale to E2M1,
    RNE, saturating at 6, into byte j of the word: exact at every midpoint."""
    pk = _fn(lib, "pk_fp4_bf16" if bf16 else "pk_fp4_f32")
    x, sc = _fp4_domain(torch.bfloat16 if bf16 else F32)

    def body(*w):
        out, scale = fx.Int32(0), f32(w[-1])
        for j in range(4):
            if bf16:
                out = pk(out, w[j], scale, j)
            else:
                out = pk(out, f32(w[2 * j]), f32(w[2 * j + 1]), scale, j)
        return [out]

    words = x.contiguous().view(I32) if bf16 else x
    got = _run(f"fp4q_{lib}_{bf16}", body, 1, words, sc.float())
    xs = x.double() / sc.double()[:, None]
    ref = R.to_e2m1_rne(torch.where(sc[:, None] == 0, 0.0, xs))
    _assert_equal(_e2m1_bytes(got[:, 0]), ref, "fp4", x)
    assert (xs.abs() > 6).any() and (ref.abs() == 6).any()
    assert (xs.abs() * 4 % 2 == 1).any()


# ---------------------------------------------------------------- transcendentals
def _mixed(n, g, lo, hi, small=(-30, 6)):
    """Half uniform in [lo, hi], half log-uniform in +-2^small."""
    u = torch.empty(n // 2, device=DEV).uniform_(lo, hi, generator=g)
    return torch.cat([u, _log_uniform(n - n // 2, small[0], small[1], g)])


# name -> (inputs(n, g), reference, floor, ulp bound). The bound of the bare
# instructions is AMD's documented 1 ulp; the others are twice the maximum
# over 20 seeds (calibrated with mono_reference, as recorded in the PR).
_TRANS = {
    "hw_exp2": (lambda n, g: [_mixed(n, g, -126, 127)], torch.exp2, 0.0, 1.0),
    "hw_rcp": (
        lambda n, g: [_log_uniform(n, -126, 126, g)],
        torch.reciprocal,
        0.0,
        1.0,
    ),
    "hw_rsq": (
        lambda n, g: [_log_uniform(n, -126, 127, g, signed=False)],
        torch.rsqrt,
        0.0,
        1.0,
    ),
    "sigmoid": (lambda n, g: [_mixed(n, g, -80, 80)], torch.sigmoid, 0.0, 4.2),
    "_sigmoid": (lambda n, g: [_mixed(n, g, -80, 80)], torch.sigmoid, 0.0, 4.5),
    "_score": (
        lambda n, g: [torch.cat([_mixed(n - 3, g, -40, 40), _neighbours(_full(20.0))])],
        lambda x: R.router_scores(x, "sqrt_softplus").square(),
        1.0,
        4.7,
    ),
    "tanh_libm": (lambda n, g: [_log_uniform(n, -24, 6, g)], torch.tanh, 1.0, 1.5),
    "tanh_aiter": (lambda n, g: [_log_uniform(n, -24, 6, g)], torch.tanh, 1.0, 2.0),
    "situ_vllm": (
        lambda n, g: [_mixed(n, g, -40, 40), _mixed(n, g, -80, 80)],
        R.situ,
        1.0,
        18.0,
    ),
    "situ_aiter": (
        lambda n, g: [_mixed(n, g, -40, 40), _mixed(n, g, -80, 80)],
        R.situ,
        1.0,
        18.0,
    ),
    "_swiglu": (
        lambda n, g: [_mixed(n, g, -12, 12), _mixed(n, g, -12, 12)],
        R.swiglu_oai,
        1.0,
        4.7,
    ),
}


def _trans_cases():
    return [
        pytest.param(p.values[0], name, marks=p.marks, id=f"{p.id}-{name}")
        for name in _TRANS
        for p in _libs(name)
    ]


def _condition(ref, xs, h=2.0**-30):
    """Sum over the inputs of |x df/dx / f| (central differences in float64):
    the relative error a 1-ulp change of the inputs alone causes, in ulps."""
    f = ref(*xs)
    c = torch.zeros_like(f)
    for i, x in enumerate(xs):
        up = [y * (1 + h) if j == i else y for j, y in enumerate(xs)]
        dn = [y * (1 - h) if j == i else y for j, y in enumerate(xs)]
        c += ((ref(*up) - ref(*dn)) / (2 * h * f)).abs().nan_to_num(0.0)
    return f, c


def trans_error(lib, name, seed, n=1 << 22):
    """Max error of helper ``name`` of ``lib`` over n inputs of the seed, in
    f32 ulps of max(|ref|, floor) over 1 + the condition number (a
    correctly rounded result of correctly rounded inputs scores <= 1)."""
    inputs, ref, floor, _ = _TRANS[name]
    fn = _fn(lib, name)
    xs = inputs(n, _gen(seed))
    got = _run(f"t_{lib}_{name}", lambda *w: [i32(fn(*map(f32, w)))], 1, *xs)
    got = got.view(F32)[:, 0].double()
    if name == "_score":
        got = got.square()
    f, cond = _condition(ref, [x.double() for x in xs])
    if name.startswith("hw_"):
        cond = 0
    return (R.ulps(got, f, F32, floor) / (1 + cond)).max().item()


@pytest.mark.parametrize(("lib", "name"), _trans_cases())
def test_transcendentals(lib, name):
    bound = _TRANS[name][3]
    assert bound is not None, f"{name}: not calibrated"
    err = trans_error(lib, name, seed=0)
    assert err <= bound, f"{name}: {err:.3g} ulp > {bound}"


@pytest.mark.parametrize("lib", _libs("div_rn"))
def test_div_rn(lib):
    """div_rn(x, d, RN(1 / d)): the IEEE quotient, bit for bit."""
    g = _gen(3)
    x, d = (_log_uniform(1 << 22, -60, 60, g) for _ in range(2))
    r = (1 / d.double()).float()
    div = _fn(lib, "div_rn")
    got = _run(f"div_{lib}", lambda *w: [i32(div(*map(f32, w)))], 1, x, d, r)
    _assert_equal(got[:, 0], _bits(R.f32_rne(x.double() / d.double())), "div_rn", x)


# ---------------------------------------------------------------- reductions
def _emulate_butterfly(v, offsets, op):
    """The lane-xor butterfly over 64-lane waves, in float32, in order."""
    v = v.view(-1, 64).clone()
    lane = torch.arange(64, device=v.device)
    for off in offsets:
        v = op(v, v[:, lane ^ off])
    return v.flatten()


@pytest.mark.parametrize("op", ["add", "max"])
@pytest.mark.parametrize("offsets", [(32, 16, 8, 4, 2, 1), (16, 8, 4, 2, 1)])
@pytest.mark.parametrize("lib", _libs("butterfly"))
def test_butterfly(lib, offsets, op):
    """butterfly(v, offsets, op): every lane holds the float32 reduction in
    exactly the documented order (the order is the numerical contract)."""
    bf = _fn(lib, "butterfly")
    dop = fx.max if op == "max" else (lambda a, b: a + b)
    top = torch.maximum if op == "max" else torch.add
    v = _log_uniform(64 * 4096, -20, 20, _gen(4))
    got = _run(
        f"bfly_{lib}_{''.join(map(str, offsets))}_{op}",
        lambda w: [i32(bf(f32(w), offsets, dop))],
        1,
        v,
    )
    ref = _emulate_butterfly(v, offsets, top)
    _assert_equal(got[:, 0], ref.view(I32), "butterfly", v)
    if op == "add":
        other = _emulate_butterfly(v, offsets[::-1], top)
        assert not torch.equal(ref, other), "inputs blind to the order"


@pytest.mark.parametrize("tp", [2, 4, 8])
@pytest.mark.parametrize("lib", _libs("sum_partials"))
def test_sum_partials(lib, tp):
    """sum_partials: the ranks' bf16 pair partials summed in float32 in rank
    order 0 .. tp - 1, bit for bit."""
    sp = _fn(lib, "sum_partials")
    w = torch.randn(1 << 16, tp, 2, device=DEV, generator=_gen(5))
    w = w * torch.exp2(torch.randint(-8, 8, w.shape, device=DEV, generator=_gen(6)))
    w = w.bfloat16()

    def body(*ws):
        polled = [[ws[s]] for s in range(tp)]
        return [i32(x) for x in sp(lambda reqs: polled, None, lambda s: s, tp)]

    got = _run(f"sp_{lib}_{tp}", body, 2, w.view(-1, tp * 2).view(I32))
    acc = w[:, 0].float()
    for s in range(1, tp):
        acc = acc + w[:, s].float()
    _assert_equal(got.view(F32), acc, "sum_partials", w.float().sum(1))
    alt = w.float().flip(1)[:, 0]
    for s in range(1, tp):
        alt = alt + w.float().flip(1)[:, s]
    assert tp == 2 or not torch.equal(acc, alt), "inputs blind to the rank order"


# ---------------------------------------------------------------- MFMA
def _lanes(x, k):
    """[P, 16, K] -> the [P 64, k] lane operands: lane l holds row l % 16,
    K k (l / 16) .. + k."""
    p = x.shape[0]
    return x.view(p, 16, 4, k).permute(0, 2, 1, 3).reshape(p * 64, k)


def _acc_lanes(c):
    """[P, 16, 16] C <-> lanes: lane l holds rows 4 (l / 16) .. + 3 of
    column l % 16."""
    return c.view(-1, 4, 4, 16).transpose(2, 3).reshape(-1, 4)


def _from_acc_lanes(d):
    return d.view(-1, 4, 16, 4).transpose(2, 3).reshape(-1, 16, 16)


def _mfma_error(got, a, b, c, k, block=None):
    """D = C + A B^T: the error over the fp32 accumulation scale (k + 1)
    2^-24 (|C| + |A| |B|^T), the a-priori bound being 1; or, ``block`` given,
    over |C| + R.blockmax_cond (an FP8 MFMA's scale)."""
    d = _from_acc_lanes(got.view(F32).double())
    ref = c + a @ b.transpose(1, 2)
    if block is None:
        scale = (k + 1) * 2.0**-24 * (c.abs() + a.abs() @ b.abs().transpose(1, 2))
    else:
        scale = c.abs() + R.blockmax_cond(a, b, block)
    return ((d - ref).abs() / scale).max().item()


def _bf16_operand(p, g):
    x = torch.randn(p, 16, 32, device=DEV, generator=g)
    return (
        x * torch.exp2(torch.randint(-20, 20, x.shape, device=DEV, generator=g))
    ).bfloat16()


@pytest.mark.parametrize(
    ("lib", "name"),
    [pytest.param(*p.values, "mfma_bf16", id=p.id) for p in _libs("mfma_bf16")]
    + [pytest.param("dsv41", "mfma_bf16_k32", id="dsv41-k32")],
)
def test_mfma_bf16(lib, name):
    """The 16x16x32 bf16 MFMA (gfx942: two 16x16x16) on its documented lane
    layout, against the float64 product."""
    mfma = _fn(lib, name)
    g = _gen(7)
    a, b = _bf16_operand(256, g), _bf16_operand(256, g)
    c = torch.randn(256, 16, 16, device=DEV, generator=g)

    def body(*w):
        x = [fx.Vector.from_elements(list(w[i : i + 4]), fx.Int32) for i in (0, 4)]
        acc = fx.Vector.from_elements([f32(v) for v in w[8:]], fx.Float32)
        d = mfma(x[0].bitcast(fx.BFloat16), x[1].bitcast(fx.BFloat16), acc)
        return [i32(d[i]) for i in range(4)]

    ops = [_lanes(t, 8).contiguous().view(I32) for t in (a, b)]
    got = _run(f"mfma_{lib}_{name}", body, 4, *ops, _acc_lanes(c))
    err = _mfma_error(got, a.double(), b.double(), c.double(), 32)
    assert err <= 1, f"{err:.3g} of the fp32 accumulation bound"


def _fp8_lanes(q):
    """[P, 16, 128] E4M3 -> the f8f6f4 MFMA's lane operands: lane l holds
    row l % 16, K 16 j .. + 16 and 64 + 16 j .. + 16 for j = l / 16."""
    p = q.shape[0]
    halves = q.view(p, 16, 2, 4, 16).permute(0, 3, 1, 2, 4)
    return halves.reshape(p * 64, 32).contiguous()


def _mx_operand(p, fmt, g):
    """Random MXFP8 (no NaN) or MXFP4 [P, 16, 128] -> (lane words, float64
    values without scale)."""
    if fmt == 0:
        q = torch.randint(0, 256, (p, 16, 128), device=DEV, generator=g)
        q = torch.where((q & 0x7F) == 0x7F, 0, q).to(torch.uint8)
        return _fp8_lanes(q).view(I32), q.view(torch.float8_e4m3fn).double()
    q = torch.randint(0, 256, (p, 16, 64), device=DEV, generator=g, dtype=torch.uint8)
    v = R.dequant_mxfp4(q, q.new_full((p, 16, 4), 127))
    return _lanes(q, 16).contiguous().view(I32), v


def _scaled_cases():
    fmts = [(0, 0), (4, 0), (4, 4)]
    out = []
    for p in _libs("mfma_scaled", gfx950_only=True):
        lib = p.values[0]
        sels = [(0, 0)] if lib.endswith("_attn") else [(0, 0), (1, 2), (3, 1)]
        for f in fmts if not lib.endswith("_attn") else fmts[:1]:
            for sel in sels:
                ident = f"{lib}-{'fp4' if f[0] else 'fp8'}x{'fp4' if f[1] else 'fp8'}"
                out.append(
                    pytest.param(lib, f, sel, marks=p.marks, id=f"{ident}-{sel}")
                )
    return out


def mfma_scaled_error(lib, fmts, sel, seed, p=256):
    """The scaled MFMA of ``lib`` on random operands of the formats ``fmts``
    whose scale words hold 4 codes, byte ``sel`` the lane's -> (error over
    the fp32 bound, error over the FP8 block scale, the operands' largest
    product range within a block)."""
    mfma = _fn(lib, "mfma_scaled")
    g = _gen(seed)
    (wa, va), (wb, vb) = (_mx_operand(p, f, g) for f in fmts)
    codes = torch.randint(100, 155, (2, p * 64, 4), device=DEV, generator=g)
    sw = (codes << (8 * torch.arange(4, device=DEV))).sum(-1).to(I32)
    c = torch.randn(p, 16, 16, device=DEV, generator=g)
    na, nb = wa.shape[1], wb.shape[1]
    extra = () if lib.endswith("_attn") else sel

    def body(*w):
        a = fx.Vector.from_elements(list(w[:na]), fx.Int32)
        b = fx.Vector.from_elements(list(w[na : na + nb]), fx.Int32)
        acc = fx.Vector.from_elements([f32(v) for v in w[na + nb : -2]], fx.Float32)
        d = mfma(a, b, acc, w[-2], w[-1], *fmts, *extra)
        return [i32(d[i]) for i in range(4)]

    key = f"mfs_{lib}_{fmts[0]}{fmts[1]}_{sel[0]}{sel[1]}"
    got = _run(key, body, 4, wa, wb, _acc_lanes(c), sw[0], sw[1])

    def scaled(v, which):
        code = codes[which, :, sel[which]].view(p, 4, 16).transpose(1, 2)
        return v * R.e8m0(code).repeat_interleave(32, -1)

    a, b, c = scaled(va, 0), scaled(vb, 1), c.double()
    span = max((v.abs().max() / v[v != 0].abs().min()).item() for v in (va, vb))
    return _mfma_error(got, a, b, c, 128), _mfma_error(got, a, b, c, 128, 32), span


# The FP8 operand cases' error over |C| + R.blockmax_cond: twice the maximum
# over 20 seeds. They exceed the fp32 bound: the MFMA drops a product 2^-14
# below the largest of its 32-wide K block.
_FP8_MFMA_BOUND = 1.5e-3


@pytest.mark.parametrize(("lib", "fmts", "sel"), _scaled_cases())
def test_mfma_scaled(lib, fmts, sel):
    """The 16x16x128 scaled MFMA (f8f6f4) with E8M0 scale words whose byte
    ``sel`` is the lane's (row or column, K block l / 16) code: against
    the float64 product of the exactly dequantized operands."""
    e32, eblk, span = mfma_scaled_error(lib, fmts, sel, seed=8)
    if fmts == (4, 4):
        assert e32 <= 1, f"{e32:.3g} of the fp32 accumulation bound"
    else:
        assert span > 2.0**14, "operands too narrow to show the FP8 MFMA's precision"
        assert eblk <= _FP8_MFMA_BOUND, f"{eblk:.3g} > {_FP8_MFMA_BOUND}"


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


@pytest.mark.parametrize("lib", _libs("fp8x8_bf16_pk", gfx950_only=True))
def test_fp8x8_bf16_pk(lib):
    cvt = _fn(lib, "fp8x8_bf16_pk")
    b = _fp8_bytes(False)
    w = torch.stack([b.roll(j) for j in range(8)], 1).contiguous().view(I32)
    got = _run(
        f"f8pk_{lib}",
        lambda w0, w1: [
            fx.Vector(cvt([w0, w1])).bitcast(fx.Int32)[j] for j in range(4)
        ],
        4,
        w,
    )
    ref = w.view(torch.uint8).view(torch.float8_e4m3fn).double()
    _assert_equal(got.view(torch.bfloat16).double(), ref, "fp8x8_bf16_pk", w)


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
    AITER's shuffle_scale ``view(rows / 32, 2, 16, cols / 8, 2, 4)
    .permute(0, 3, 5, 2, 4, 1)`` puts that scale (host integers)."""
    rows, cols = 64, 256
    count = torch.arange(rows * cols)
    shuffled = count.view(rows // 32, 2, 16, cols // 8, 2, 4)
    shuffled = shuffled.permute(0, 3, 5, 2, 4, 1).flatten()
    where = torch.empty_like(count)
    where[shuffled] = torch.arange(rows * cols)
    r, c = count // cols, count % cols
    assert torch.equal(_fn(lib, "scale_index")(r, c, cols), where)
