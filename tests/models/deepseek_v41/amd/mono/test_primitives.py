# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4.1's AMD monokernel device helpers against float64 / exact
references (`tests/models/common/specialized_decode/reference.py`), one helper
at a time, from a test kernel that calls the helper itself on every lane.

Roundings, packings, scale codes and data movement are bit-exact: there is a
closed form for each, so a tolerance would only hide a wrong one.

This is the primitive tier (T1) of #68016, a debugging aid that reaches into
this model's internals on purpose. It is not the numerical contract; that is
`DecodeNumerics` (T2), at the decoder layer's public entry point.

The model carries two copies of these helpers -- one on the MoE and seam side,
one on the attention side -- and `_LIBS` checks each, since a bug fixed in one
is not fixed in the other. The model-agnostic machinery lives in
`tests/models/common/specialized_decode/primitives.py`; another model's helpers
belong in a file under its own `tests/models/<model>/` directory, over its own
`_LIBS`."""

import pytest
import torch

from tests.models.common.specialized_decode import reference as R
from vllm.platforms import current_platform

if not current_platform.is_rocm():
    pytest.skip("the mono kernels need ROCm", allow_module_level=True)
pytest.importorskip("flydsl")
pytest.importorskip("aiter")

import flydsl.expr as fx  # noqa: E402

from tests.models.common.specialized_decode import primitives as H  # noqa: E402
from vllm.models.deepseek_v41.amd.mono.common.arch import GFX942  # noqa: E402
from vllm.platforms.rocm import on_mi3xx  # noqa: E402

if not on_mi3xx():
    pytest.skip("the mono kernels are MI300/MI350 only", allow_module_level=True)

_PKG = "vllm.models.deepseek_v41.amd.mono"
_LIBS = {
    "dsv41": H.HelperSet(
        _PKG,
        ("common.ops", "common.mx", "common.gfx942", "stages.moe", "stages.seam"),
    ),
    "dsv41_attn": H.HelperSet(_PKG, ("attention.device",)),
}
DEV, F32, I32 = H.DEV, H.F32, H.I32


def _fn(lib, name):
    """Helper `name` of `lib`; None where it is an earlier library's
    (imported, not a copy)."""
    return H.own_helper(_LIBS, lib, name)


def _has(lib, name):
    return _fn(lib, name) is not None


def _libs(*names):
    """The libraries defining every helper of `names`."""
    return H.params(_LIBS, *names)


def _run(key, fn, n_out, *cols):
    """`fn` over the rows of the int32 / float32 columns `cols` -> int32
    [rows, n_out]."""
    return H.run(key, fn, n_out, *cols, sources=H.sources_digest(_PKG))


# ---------------------------------------------------------------- bf16
@pytest.mark.parametrize("lib", _libs("bf16_round", "bf16_pair", "bf_lo", "bf_hi"))
def test_bf16_round_and_pack(lib):
    """bf16_round, bf16_pair, bf_lo, bf_hi on every bf16 pattern next to its
    rounding boundaries (ties, carries into the exponent, overflow to inf,
    subnormals) and 2^20 random words: RNE, bit for bit."""
    ops = {n: _fn(lib, n) for n in ("bf16_round", "bf16_pair", "bf_lo", "bf_hi")}

    def body(a, b):
        p = fx.Float32(ops["bf16_pair"](H.f32(a), H.f32(b))).bitcast(fx.Int32)
        return [
            H.i32(ops["bf16_round"](H.f32(a))),
            p,
            H.i32(ops["bf_lo"](p)),
            H.i32(ops["bf_hi"](p)),
        ]

    hi = torch.arange(1 << 16, dtype=torch.int64)[:, None] << 16
    lo = torch.tensor([0, 1, 0x7FFF, 0x8000, 0x8001, 0xFFFF])
    g = torch.Generator().manual_seed(0)
    rnd = torch.randint(-(1 << 31), 1 << 31, (1 << 20,), generator=g)
    a = torch.cat([(hi | lo).flatten(), rnd]).to(I32).view(F32).to(DEV)
    a = torch.where(torch.isnan(a), torch.zeros_like(a), a)
    b = a.roll(1)
    got = _run(f"bf16_{lib}", body, 4, a, b)
    ra, rb = H.bits(R.bf16_rne(a)), H.bits(R.bf16_rne(b))
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
@pytest.mark.parametrize("lib", _libs("fp8_pack4", "clamp_fp8"))
def test_fp8_clamp_and_pack(lib):
    """fp8_pack4(clamp_fp8(x)): the saturating E4M3 RNE of x, byte for byte
    (FNUZ with a maximum of 224 where the kernels target gfx942)."""
    clamp, pack = _fn(lib, "clamp_fp8"), _fn(lib, "fp8_pack4")
    values = H.fp8_domain(GFX942)
    values = torch.cat([values, values.new_zeros(-values.numel() % 4)])
    got = _run(
        f"fp8_{lib}",
        lambda *words: [pack(*[clamp(H.f32(w)) for w in words])],
        1,
        *values.view(-1, 4).T,
    )
    fp8_dtype = torch.float8_e4m3fnuz if GFX942 else torch.float8_e4m3fn
    ref = R.to_e4m3_rne(values, GFX942).to(fp8_dtype).view(torch.uint8)
    H.assert_equal(got.view(torch.uint8).flatten(), ref, "fp8", values)
    fmax = 224.0 if GFX942 else 448.0
    # In units of the smallest subnormal, so `.5` is a tie the RNE has to
    # break and `< 8` keeps it inside the subnormal range.
    subnormal_ulps = values.double().abs() / (2.0**-10 if GFX942 else 2.0**-9)
    ties = subnormal_ulps == subnormal_ulps.floor() + 0.5
    assert (ties & (subnormal_ulps < 8)).any()
    assert (values.abs() > fmax).any()
    assert (ref.view(fp8_dtype).double().abs() == fmax).any()


# ---------------------------------------------------------------- scale codes
_CODE_RULES = {
    "code_ceil": lambda v, fmax: R.ceil_code(v).clamp(1, 254),
    "vllm_mx_code": R.mx_code_vllm,
    "mx_code": R.mx_code_vllm,
}


def _code_libs():
    return [p for p in _libs() if any(_has(p.values[0], n) for n in _CODE_RULES)]


@pytest.mark.parametrize("lib", _code_libs())
def test_scale_codes(lib):
    """code_ceil and vllm_mx_code / mx_code on every f32 exponent with the
    mantissas around a power of two, the values whose quotient by the fp8
    maximum is a power of two and their neighbours, 0 and the subnormals;
    pow2 / rcp_pow2 on every code they take. Bit-exact."""
    names = [n for n in _CODE_RULES if _has(lib, n)]
    fns = [_fn(lib, n) for n in names]
    p2, rp2 = _fn(lib, "pow2"), _fn(lib, "rcp_pow2")

    def body(v, code):
        out = [fx.Int32(f(H.f32(v))) for f in fns]
        out.append(H.i32(p2(code)) if p2 else code)
        out.append(H.i32(rp2(code)) if rp2 else code)
        return out

    k = torch.arange(-149, 128, dtype=torch.float64, device=DEV)
    pow2s = torch.exp2(k)
    # 224 * 2^k is 448 * 2^(k - 1), so these are the edges of both maxima.
    edges = 448 * pow2s
    v = torch.cat([H.edge_floats(), H.neighbours(edges[edges < 3e38].float())])
    v = v[torch.isfinite(v) & (v >= 0)]
    code = torch.arange(v.numel(), device=DEV, dtype=I32) % 253 + 1
    got = _run(f"codes_{lib}", body, len(fns) + 2, v, code)
    fmax = _LIBS[lib].find("FP8_MAX")
    for j, n in enumerate(names):
        H.assert_equal(got[:, j], _CODE_RULES[n](v, fmax), n, v)
    if p2:
        assert torch.equal(got[:, -2].view(F32).double(), R.e8m0(code))
    if rp2:
        assert torch.equal(got[:, -1].view(F32).double(), 1 / R.e8m0(code))
    assert (v == 0).any() and ((v > 0) & (v < R.FLT_MIN)).any()


# ---------------------------------------------------------------- conversions
@pytest.mark.parametrize("lib", _libs("fp8x8_bf16"))
def test_fp8x8_bf16(lib):
    """fp8x8_bf16(w0, w1, scale): every E4M3 byte (FNUZ on gfx942) in every
    position times a power-of-two scale -> bf16, exact."""
    cvt = _fn(lib, "fp8x8_bf16")
    b = H.fp8_bytes(GFX942)
    words = torch.stack([b.roll(j) for j in range(8)], 1).contiguous().view(I32)
    codes = torch.arange(110, 145, device=DEV)
    w = words.repeat(codes.numel(), 1)
    c = codes.repeat_interleave(words.shape[0])
    got = _run(
        f"f8bf_{lib}",
        lambda w0, w1, sc: [
            fx.Vector(cvt(w0, w1, H.f32(sc << 23)))[j] for j in range(4)
        ],
        4,
        w,
        c.to(I32),
    )
    ref = R.dequant_fp8(w.contiguous().view(torch.uint8), c[:, None], (1, 8), GFX942)
    H.assert_equal(got.view(torch.bfloat16).double(), ref, "fp8x8_bf16", c)


@pytest.mark.parametrize("lib", _libs("ocp_to_fnuz"))
def test_ocp_to_fnuz(lib):
    """Every OCP byte in every position -> the FNUZ byte of half its value:
    the same byte, but OCP's -0 (0x80, the FNUZ NaN) to 0x00."""
    cvt = _fn(lib, "ocp_to_fnuz")
    w = H.all_byte_words()
    got = _run(f"o2f_{lib}", lambda w: [cvt(w)], 1, w).view(torch.uint8).flatten()
    ocp = w.view(torch.uint8)
    H.assert_equal(got, torch.where(ocp == 0x80, 0, ocp), "ocp_to_fnuz", ocp)
    ok = (ocp & 0x7F) != 0x7F
    halved = got[ok].view(torch.float8_e4m3fnuz).double() * 2
    assert torch.equal(halved, ocp[ok].view(torch.float8_e4m3fn).double())


@pytest.mark.parametrize("lib", _libs("fp4x8_fnuz"))
def test_fp4x8_fnuz(lib):
    """Eight E2M1 nibbles (no -0) -> FNUZ bytes of the values times 2^-7,
    the even K in the first word, the odd in the second: exact."""
    cvt = _fn(lib, "fp4x8_fnuz")
    neg0 = [v for v in range(256) if (v & 0xF) == 8 or (v >> 4) == 8]
    w = H.all_byte_words(exclude=neg0)
    got = _run(f"f4f_{lib}", lambda w: list(cvt(w)), 2, w)
    vals = got.contiguous().view(torch.uint8).view(torch.float8_e4m3fnuz).double()
    ref = H.e2m1_bytes(w)[:, [0, 2, 4, 6, 1, 3, 5, 7]] * 2.0**-7
    H.assert_equal(vals.view(-1, 8), ref, "fp4x8_fnuz", w[:, None])


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
