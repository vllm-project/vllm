# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4.1's AMD monokernel device helpers against the float64
definitions in `tests/models/common/specialized_decode/reference.py`, one
helper at a time, from a test kernel that calls the helper itself on every
lane.

Roundings, packings, scale codes and data movement are bit-exact: there is a
closed form for each, so a tolerance would only hide a wrong one.

This is the primitive tier (T1) of #68016, a debugging aid that reaches into
this model's internals on purpose. It is not the numerical contract; that is
`DecodeNumerics` (T2), at the decoder layer's public entry point.

The model carries two copies of these helpers, one on the MoE and seam side
and one on the attention side. `attention/device.py` does take a few names
from `common/gfx942.py`, but every helper in the tables below it defines
again for itself, so a bug fixed in one copy is not fixed in the other. Each
copy is therefore imported under its own name and checked separately, and a
test runs over only the copies that define what it checks. Collapsing the two
belongs in the model, not here; when that lands, the tables collapse with
it.

The machinery is in `tests/models/common/specialized_decode/`: `harness.py`
runs a device helper, `inputs.py` builds what makes a rounding rule
observable. Another model's helpers belong in a file under its own
`tests/models/<model>/` directory, over its own imports."""

from collections.abc import Callable
from typing import Any

import pytest
import torch

from tests.models.common.specialized_decode import reference as R
from vllm.platforms import current_platform

# Every gate this module can be skipped by comes before the imports it would
# make pointless, cheapest first: the model's device modules are FlyDSL and
# only ever load on the hardware they are written for.
if not current_platform.is_rocm():
    pytest.skip("the mono kernels need ROCm", allow_module_level=True)

from vllm.platforms.rocm import on_mi3xx  # noqa: E402

if not on_mi3xx():
    pytest.skip("the mono kernels are MI300/MI350 only", allow_module_level=True)
pytest.importorskip("flydsl")
pytest.importorskip("aiter")

import flydsl.expr as fx  # noqa: E402

from tests.models.common.specialized_decode import harness as H  # noqa: E402
from tests.models.common.specialized_decode import inputs as I  # noqa: E402
from vllm.models.deepseek_v41.amd.mono.attention import device as attn  # noqa: E402
from vllm.models.deepseek_v41.amd.mono.common import gfx942 as gfx  # noqa: E402
from vllm.models.deepseek_v41.amd.mono.common import mx, ops  # noqa: E402
from vllm.models.deepseek_v41.amd.mono.common.arch import GFX942  # noqa: E402
from vllm.models.deepseek_v41.amd.mono.stages import moe  # noqa: E402

_PKG = "vllm.models.deepseek_v41.amd.mono"
DEV, F32, I32 = I.DEV, I.F32, I.I32

# The two copies, under the id each one is checked as. "dsv41" is the MoE and
# seam side, spread over common/ and stages/; "dsv41_attn" is
# attention/device.py. A table holds only the copies defining the helpers that
# table is about, so parametrising over it skips the copy that has none.
_BF16 = {
    "dsv41": (ops.bf16_round, ops.bf16_pair, ops.bf_lo, ops.bf_hi),
    "dsv41_attn": (attn.bf16_round, attn.bf16_pair, attn.bf_lo, attn.bf_hi),
}
_FP8_PACK = {
    "dsv41": (mx.clamp_fp8, ops.fp8_pack4),
    "dsv41_attn": (attn.clamp_fp8, attn.fp8_pack4),
}
_FP8X8_BF16 = {"dsv41_attn": attn.fp8x8_bf16}
_OCP_TO_FNUZ = {"dsv41": gfx.ocp_to_fnuz}
_FP4X8_FNUZ = {"dsv41": gfx.fp4x8_fnuz}
_SCALE_INDEX = {"dsv41": mx.scale_index}


def _ceil_rule(value: torch.Tensor, fmax: float) -> torch.Tensor:
    """What `code_ceil` is defined to do: the E8M0 round-up code of `value`,
    over the range a scale code may take. It does not divide by `fmax`."""
    return R.ceil_code(value).clamp(1, 254)


# (name, the device helper, the float64 rule it has to hold). `mx_code` and
# `vllm_mx_code` are the same rule under two names, one per copy.
_CodeRule = Callable[[torch.Tensor, float], torch.Tensor]
_SCALE_CODES: dict[str, tuple[tuple[str, H.DeviceFn, _CodeRule], ...]] = {
    "dsv41": (
        ("code_ceil", mx.code_ceil, _ceil_rule),
        ("vllm_mx_code", moe.vllm_mx_code, R.mx_code_vllm),
    ),
    "dsv41_attn": (("mx_code", attn.mx_code, R.mx_code_vllm),),
}
# Checked alongside the codes, on the codes themselves. The MoE copy has no
# `pow2`, so that column of its row is None.
_POW2: dict[str, tuple[H.DeviceFn | None, H.DeviceFn]] = {
    "dsv41": (None, mx.rcp_pow2),
    "dsv41_attn": (attn.pow2, attn.rcp_pow2),
}
_FP8_MAX: dict[str, float] = {"dsv41": mx.FP8_MAX, "dsv41_attn": attn.FP8_MAX}

# Two columns only say something while they are two functions. If a copy ever
# imports the other's helper instead of defining its own, the [dsv41] and
# [dsv41_attn] ids would go on reporting two passes for one function, and this
# is where that shows up.
for _table in (_BF16, _FP8_PACK, _POW2):
    _ids = [{id(f) for f in _table[c] if f} for c in ("dsv41", "dsv41_attn")]
    assert not _ids[0] & _ids[1], "the two copies now share a helper"


def _run(key: str, fn: H.DeviceFn, n_out: int, *cols: torch.Tensor) -> torch.Tensor:
    """`fn` over the rows of the int32 / float32 columns `cols` -> int32
    [rows, n_out]."""
    return H.run(key, fn, n_out, *cols, package=_PKG)


# ---------------------------------------------------------------- bf16
@pytest.mark.parametrize("lib", _BF16)
def test_bf16_round_and_pack(lib: str) -> None:
    """bf16_round, bf16_pair, bf_lo, bf_hi on every bf16 pattern next to its
    rounding boundaries (ties, carries into the exponent, overflow to inf,
    subnormals) and 2^20 random words: RNE, bit for bit."""
    bf16_round, bf16_pair, bf_lo, bf_hi = _BF16[lib]

    def body(a: Any, b: Any) -> list[Any]:
        p = fx.Float32(bf16_pair(H.f32(a), H.f32(b))).bitcast(fx.Int32)
        return [H.i32(bf16_round(H.f32(a))), p, H.i32(bf_lo(p)), H.i32(bf_hi(p))]

    hi = torch.arange(1 << 16, dtype=torch.int64)[:, None] << 16
    lo = torch.tensor([0, 1, 0x7FFF, 0x8000, 0x8001, 0xFFFF])
    g = torch.Generator().manual_seed(0)
    rnd = torch.randint(-(1 << 31), 1 << 31, (1 << 20,), generator=g)
    a = torch.cat([(hi | lo).flatten(), rnd]).to(I32).view(F32).to(DEV)
    # NaN is dropped rather than merely awkward to compare: how a NaN payload
    # survives a truncation to 7 mantissa bits is the ISA's business, and
    # float64 has no definition of it to check against. Whether a NaN may
    # reach these helpers at all is a question for the stage that feeds them.
    a = torch.where(torch.isnan(a), torch.zeros_like(a), a)
    b = a.roll(1)
    got = _run(f"bf16_{lib}", body, 4, a, b)
    ra, rb = I.bits(R.bf16_rne(a)), I.bits(R.bf16_rne(b))
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
@pytest.mark.parametrize("lib", _FP8_PACK)
def test_fp8_clamp_and_pack(lib: str) -> None:
    """fp8_pack4(clamp_fp8(x)): the saturating E4M3 RNE of x, byte for byte
    (FNUZ with a maximum of 224 where the kernels target gfx942)."""
    clamp_fp8, fp8_pack4 = _FP8_PACK[lib]
    values = I.fp8_domain(GFX942)
    values = torch.cat([values, values.new_zeros(-values.numel() % 4)])
    got = _run(
        f"fp8_{lib}",
        lambda *words: [fp8_pack4(*[clamp_fp8(H.f32(w)) for w in words])],
        1,
        *values.view(-1, 4).T,
    )
    fp8_dtype = torch.float8_e4m3fnuz if GFX942 else torch.float8_e4m3fn
    ref = R.to_e4m3_rne(values, GFX942).to(fp8_dtype).view(torch.uint8)
    I.assert_bit_exact(got.view(torch.uint8).flatten(), ref, "fp8", values)
    fmax = 224.0 if GFX942 else 448.0
    # In units of the smallest subnormal, so `.5` is a tie the RNE has to
    # break and `< 8` keeps it inside the subnormal range.
    subnormal_ulps = values.double().abs() / (2.0**-10 if GFX942 else 2.0**-9)
    ties = subnormal_ulps == subnormal_ulps.floor() + 0.5
    assert (ties & (subnormal_ulps < 8)).any()
    assert (values.abs() > fmax).any()
    assert (ref.view(fp8_dtype).double().abs() == fmax).any()


# ---------------------------------------------------------------- scale codes
@pytest.mark.parametrize("lib", _SCALE_CODES)
def test_scale_codes(lib: str) -> None:
    """code_ceil and vllm_mx_code / mx_code on every f32 exponent with the
    mantissas around a power of two, the values whose quotient by the fp8
    maximum is a power of two and their neighbours, 0 and the subnormals;
    pow2 / rcp_pow2 on every code they take. Bit-exact."""
    rules = _SCALE_CODES[lib]
    pow2, rcp_pow2 = _POW2[lib]

    def body(value: Any, code: Any) -> list[Any]:
        out = [fx.Int32(fn(H.f32(value))) for _, fn, _ in rules]
        out.append(H.i32(pow2(code)) if pow2 else code)
        out.append(H.i32(rcp_pow2(code)))
        return out

    k = torch.arange(-149, 128, dtype=torch.float64, device=DEV)
    pow2s = torch.exp2(k)
    # 224 * 2^k is 448 * 2^(k - 1), so these are the edges of both maxima.
    edges = 448 * pow2s
    v = torch.cat([I.edge_floats(), I.neighbours(edges[edges < 3e38].float())])
    v = v[torch.isfinite(v) & (v >= 0)]
    # 1 to 253, which is where `pow2` and `rcp_pow2` claim to be exact. They
    # are the bit tricks `code << 23` and `(254 - code) << 23`, and the three
    # codes outside that range are where a bit trick parts company with the
    # E8M0 definition: pow2(0) is +0.0 for 2^-127, rcp_pow2(254) is +0.0 for
    # 2^-127, and 255 is an infinity for both. Nothing here covers them, and
    # nothing in the model reaches them either: both helpers are fed
    # `code_ceil(amax / FP8_MAX)` or `mx_code(amax)`, and dividing a finite
    # float32 by 224 caps the code at 248. `code_ceil` on its own does clamp
    # to 254, which `_ceil_rule` holds it to on the `v` column above.
    code = torch.arange(v.numel(), device=DEV, dtype=I32) % 253 + 1
    got = _run(f"codes_{lib}", body, len(rules) + 2, v, code)
    for j, (name, _, rule) in enumerate(rules):
        I.assert_bit_exact(got[:, j], rule(v, _FP8_MAX[lib]), name, v)
    if pow2:
        assert torch.equal(got[:, -2].view(F32).double(), R.e8m0(code))
    assert torch.equal(got[:, -1].view(F32).double(), 1 / R.e8m0(code))
    assert (v == 0).any() and ((v > 0) & (v < R.FLT_MIN)).any()


# ---------------------------------------------------------------- conversions
@pytest.mark.parametrize("lib", _FP8X8_BF16)
def test_fp8x8_bf16(lib: str) -> None:
    """fp8x8_bf16(w0, w1, scale): every E4M3 byte (FNUZ on gfx942) in every
    position times a power-of-two scale -> bf16, exact."""
    fp8x8_bf16 = _FP8X8_BF16[lib]
    b = I.fp8_bytes(GFX942)
    words = torch.stack([b.roll(j) for j in range(8)], 1).contiguous().view(I32)
    codes = torch.arange(110, 145, device=DEV)
    w = words.repeat(codes.numel(), 1)
    c = codes.repeat_interleave(words.shape[0])
    got = _run(
        f"f8bf_{lib}",
        lambda w0, w1, sc: [
            fx.Vector(fp8x8_bf16(w0, w1, H.f32(sc << 23)))[j] for j in range(4)
        ],
        4,
        w,
        c.to(I32),
    )
    ref = R.dequant_fp8(w.contiguous().view(torch.uint8), c[:, None], (1, 8), GFX942)
    I.assert_bit_exact(got.view(torch.bfloat16).double(), ref, "fp8x8_bf16", c)


@pytest.mark.parametrize("lib", _OCP_TO_FNUZ)
def test_ocp_to_fnuz(lib: str) -> None:
    """Every OCP byte in every position -> the FNUZ byte of half its value:
    the same byte, but OCP's -0 (0x80, the FNUZ NaN) to 0x00."""
    ocp_to_fnuz = _OCP_TO_FNUZ[lib]
    w = I.all_byte_words()
    got = _run(f"o2f_{lib}", lambda w: [ocp_to_fnuz(w)], 1, w)
    got = got.view(torch.uint8).flatten()
    ocp = w.view(torch.uint8)
    I.assert_bit_exact(got, torch.where(ocp == 0x80, 0, ocp), "ocp_to_fnuz", ocp)
    ok = (ocp & 0x7F) != 0x7F
    halved = got[ok].view(torch.float8_e4m3fnuz).double() * 2
    assert torch.equal(halved, ocp[ok].view(torch.float8_e4m3fn).double())


@pytest.mark.parametrize("lib", _FP4X8_FNUZ)
def test_fp4x8_fnuz(lib: str) -> None:
    """Eight E2M1 nibbles (no -0) -> FNUZ bytes of the values times 2^-7,
    the even K in the first word, the odd in the second: exact."""
    fp4x8_fnuz = _FP4X8_FNUZ[lib]
    neg0 = tuple(v for v in range(256) if (v & 0xF) == 8 or (v >> 4) == 8)
    w = I.all_byte_words(exclude=neg0)
    got = _run(f"f4f_{lib}", lambda w: list(fp4x8_fnuz(w)), 2, w)
    vals = got.contiguous().view(torch.uint8).view(torch.float8_e4m3fnuz).double()
    ref = I.e2m1_bytes(w)[:, [0, 2, 4, 6, 1, 3, 5, 7]] * 2.0**-7
    I.assert_bit_exact(vals.view(-1, 8), ref, "fp4x8_fnuz", w[:, None])


# ---------------------------------------------------------------- layouts
@pytest.mark.parametrize("lib", _SCALE_INDEX)
def test_scale_index(lib: str) -> None:
    """scale_index(row, col, cols) on every scale of a 64 x 256 tile is where
    AITER's shuffle_scale `view(rows / 32, 2, 16, cols / 8, 2, 4)
    .permute(0, 3, 5, 2, 4, 1)` puts that scale.

    The only helper here that runs on the host: it is integer arithmetic on
    whatever it is handed, so torch evaluates it directly and no kernel is
    needed. The model calls it with FlyDSL expressions instead, over the same
    arithmetic."""
    scale_index = _SCALE_INDEX[lib]
    rows, cols = 64, 256
    count = torch.arange(rows * cols)
    shuffled = count.view(rows // 32, 2, 16, cols // 8, 2, 4)
    shuffled = shuffled.permute(0, 3, 5, 2, 4, 1).flatten()
    where = torch.empty_like(count)
    where[shuffled] = torch.arange(rows * cols)
    r, c = count // cols, count % cols
    assert torch.equal(scale_index(r, c, cols), where)
