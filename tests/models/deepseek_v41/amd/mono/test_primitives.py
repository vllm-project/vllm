# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4.1's AMD monokernel device helpers against the float64
definitions in `tests/models/common/specialized_decode/reference.py`, one
helper at a time, from a test kernel that calls the helper itself on every
lane.

Roundings, packings, scale codes and data movement are bit-exact: there is a
closed form for each, so a tolerance would only hide a wrong one. Hardware
transcendentals, helpers built on them, and MFMAs are held to the error
bounds in `bounds.py` instead.

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
observable, `bounds.py` bounds the helpers that cannot be exact. Another
model's helpers belong in a file under its own
`tests/models/<model>/` directory, over its own imports."""

import math
from collections.abc import Callable
from typing import Any, NamedTuple

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

from tests.models.common.specialized_decode import bounds as B  # noqa: E402
from tests.models.common.specialized_decode import harness as H  # noqa: E402
from tests.models.common.specialized_decode import inputs as I  # noqa: E402
from vllm.models.deepseek_v41.amd.mono.attention import device as attn  # noqa: E402
from vllm.models.deepseek_v41.amd.mono.common import gfx942 as gfx  # noqa: E402
from vllm.models.deepseek_v41.amd.mono.common import mx, ops  # noqa: E402
from vllm.models.deepseek_v41.amd.mono.common.arch import GFX942  # noqa: E402
from vllm.models.deepseek_v41.amd.mono.stages import moe, seam  # noqa: E402

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


# ---------------------------------------------------------------- transcendentals
# Each helper's roundings, written out through a `RoundingModel`. Plain f32
# ops, `/` and `fmath.sqrt` are correctly rounded (one `B.F32_ROUNDOFF`); a
# product with a rounded constant is two roundings.
def _sigmoid_rounding(rounding: B.RoundingModel, logit: torch.Tensor) -> torch.Tensor:
    """`stages/seam.py` sigmoid: 1 / (1 + hw_exp2(z * RN_f32(-log2 e)))."""
    exponent = rounding(logit * -math.log2(math.e), 2 * B.F32_ROUNDOFF)
    denominator = rounding(1 + rounding.transcendental(torch.exp2(exponent)))
    return rounding(1 / denominator)


def _score_rounding(rounding: B.RoundingModel, logit: torch.Tensor) -> torch.Tensor:
    """`stages/moe.py` _score, squared:
    hw_log2(1 + hw_exp2(x * RN_f32(log2 e))) * RN_f32(ln 2), or x above 20.
    The score is compared squared, so the final sqrt contributes 2^-23."""
    exponent = rounding(logit * math.log2(math.e), 2 * B.F32_ROUNDOFF)
    grown = rounding.transcendental(torch.exp2(exponent))
    # No flush term: the argument is >= 1, so the result is never denormal.
    log2 = rounding(torch.log2(rounding(1 + grown)), B.TRANSCENDENTAL_RELATIVE)
    softplus = rounding(log2 * math.log(2), 2 * B.F32_ROUNDOFF)
    return rounding(torch.where(logit > 20, logit, softplus), 2 * B.F32_ROUNDOFF)


class _Transcendental(NamedTuple):
    """A helper that cannot be exact, and how it is checked."""

    # (count, seed) -> one input column per helper argument.
    inputs: Callable[[int, int], list[torch.Tensor]]
    # Float64 definition.
    reference: Callable[..., torch.Tensor]
    # The helper's roundings, or None for a bare instruction (held to 1 ulp).
    rounding: Callable[[B.RoundingModel, torch.Tensor], torch.Tensor] | None
    # Implementations to check, by test id.
    copies: dict[str, H.DeviceFn]
    # Applied to both result and reference before comparing.
    measured: Callable[[torch.Tensor], torch.Tensor] | None = None


# Exponent range of the log-uniform half of `I.mixed_floats`.
_SMALL = (-30.0, 6.0)
_TRANSCENDENTALS: dict[str, _Transcendental] = {
    "hw_exp2": _Transcendental(
        # Below -126 the result is denormal, which exercises the flush.
        inputs=lambda count, seed: [I.mixed_floats(count, -150, 127, _SMALL, seed)],
        reference=torch.exp2,
        rounding=None,
        copies={"dsv41": ops.hw_exp2, "dsv41_attn": attn.hw_exp2},
    ),
    "hw_rcp": _Transcendental(
        inputs=lambda count, seed: [I.log_uniform(count, -126, 126, seed)],
        reference=torch.reciprocal,
        rounding=None,
        copies={"dsv41": ops.hw_rcp, "dsv41_attn": attn.hw_rcp},
    ),
    "hw_rsq": _Transcendental(
        inputs=lambda count, seed: [
            I.log_uniform(count, -126, 127, seed, signed=False)
        ],
        reference=torch.rsqrt,
        rounding=None,
        copies={"dsv41": ops.hw_rsq, "dsv41_attn": attn.hw_rsq},
    ),
    "sigmoid": _Transcendental(
        inputs=lambda count, seed: [I.mixed_floats(count, -80, 80, _SMALL, seed)],
        reference=torch.sigmoid,
        rounding=_sigmoid_rounding,
        copies={"dsv41": seam.sigmoid},
    ),
    "_score": _Transcendental(
        # 20 and its neighbours: where softplus switches to returning x.
        inputs=lambda count, seed: [
            torch.cat(
                [
                    I.mixed_floats(count - 3, -40, 40, _SMALL, seed),
                    I.neighbours(torch.full((1,), 20.0, device=DEV)),
                ]
            )
        ],
        reference=R.sqrt_softplus_score,
        rounding=_score_rounding,
        copies={"dsv41": moe._score},
        measured=torch.square,
    ),
}


def _transcendental_cases() -> list[Any]:
    return [
        pytest.param(lib, name, id=f"{lib}-{name}")
        for name, case in _TRANSCENDENTALS.items()
        for lib in case.copies
    ]


def _transcendental_error(lib: str, name: str, seed: int, count: int) -> float:
    """The largest error of helper `name` of `lib` over `count` inputs, as a
    fraction of its bound."""
    case = _TRANSCENDENTALS[name]
    helper = case.copies[lib]
    columns = case.inputs(count, seed)
    got = _run(
        f"t_{lib}_{name}", lambda *w: [H.i32(helper(*map(H.f32, w)))], 1, *columns
    )
    result = got.view(F32)[:, 0].double()
    exact = [column.double() for column in columns]
    reference = case.reference(*exact)
    if case.measured is not None:
        result, reference = case.measured(result), case.measured(reference)
    if case.rounding is None:
        return (
            ((result - reference).abs() / B.transcendental_bound(reference))
            .max()
            .item()
        )
    rounding = B.RoundingModel()
    modelled = case.rounding(rounding, *exact)
    # The bound is the rounding error plus the gap between the helper's exact
    # formula and the reference, which are different expressions.
    rounding_bound = rounding.bound(modelled)
    formula_gap = (modelled.detach() - reference).abs()
    # Keep the formula gap negligible (checked per element) so it can't hide
    # a wrong rounding model. Measured on gfx950: ~2e-9.
    worst = (formula_gap / rounding_bound).nan_to_num().max()
    assert worst <= 1e-3, (
        f"{name}: the modelled formula is {worst:.3g} of the rounding bound "
        f"from the reference, which is too much of it to stay negligible"
    )
    bound = formula_gap + rounding_bound
    # Exact results pass even where the bound is zero.
    error = torch.where(result == reference, 0.0, (result - reference).abs() / bound)
    return error.max().item()


@pytest.mark.parametrize(("lib", "name"), _transcendental_cases())
def test_transcendentals(lib: str, name: str) -> None:
    """Hardware transcendentals within 1 ulp of float64 torch, and helpers
    built on them within their rounding bound, over 2^22 inputs."""
    error = _transcendental_error(lib, name, seed=0, count=1 << 22)
    assert error <= 1, f"{name}: {error:.3g} of its bound"


_DIV_RN: dict[str, H.DeviceFn] = {"dsv41": ops.div_rn, "dsv41_attn": attn.div_rn}


@pytest.mark.parametrize("lib", _DIV_RN)
def test_div_rn(lib: str) -> None:
    """div_rn(x, d, RN(1 / d)) returns the correctly rounded quotient."""
    div_rn = _DIV_RN[lib]
    count = 1 << 22
    # One draw split in two, so numerator and denominator differ.
    drawn = I.log_uniform(2 * count, -60, 60, seed=3)
    numerator, denominator = drawn[:count], drawn[count:]
    # Rounding the f64 quotient to f32 is exact here (53 >= 2 * 24 + 2 bits).
    reciprocal = 1 / denominator
    got = _run(
        f"div_{lib}",
        lambda *w: [H.i32(div_rn(*map(H.f32, w)))],
        1,
        numerator,
        denominator,
        reciprocal,
    )
    quotient = R.f32_rne(numerator.double() / denominator.double())
    I.assert_bit_exact(got[:, 0], I.bits(quotient), "div_rn", numerator)


# ---------------------------------------------------------------- reductions
_BUTTERFLY: dict[str, H.DeviceFn] = {
    "dsv41": ops.butterfly,
    "dsv41_attn": attn.butterfly,
}
_BUTTERFLY_OPS: dict[str, Callable[[Any, Any], Any]] = {
    "add": lambda a, b: a + b,
    "max": fx.max,
}


@pytest.mark.parametrize("op", ["add", "max"])
@pytest.mark.parametrize("offsets", [(32, 16, 8, 4, 2, 1), (16, 8, 4, 2, 1)])
@pytest.mark.parametrize("lib", _BUTTERFLY)
def test_butterfly(lib: str, offsets: tuple[int, ...], op: str) -> None:
    """butterfly(v, offsets, op) reduces in exactly the documented order."""
    butterfly = _BUTTERFLY[lib]
    combine = _BUTTERFLY_OPS[op]
    values = I.log_uniform(R.WAVE * 4096, -20, 20, seed=4)
    got = _run(
        f"bfly_{lib}_{''.join(map(str, offsets))}_{op}",
        lambda w: [H.i32(butterfly(H.f32(w), offsets, combine))],
        1,
        values,
    )
    folded = R.wave_butterfly(values, offsets, op)
    I.assert_bit_exact(got[:, 0], folded.view(I32), "butterfly", values)
    if op == "add":
        # The inputs must be sensitive to the order for the test to check it.
        other_order = R.wave_butterfly(values, offsets[::-1], op)
        assert not torch.equal(folded, other_order), "inputs blind to the order"


_SUM_PARTIALS: dict[str, H.DeviceFn] = {"dsv41": ops.sum_partials}


@pytest.mark.parametrize("tp", [2, 4, 8])
@pytest.mark.parametrize("lib", _SUM_PARTIALS)
def test_sum_partials(lib: str, tp: int) -> None:
    """sum_partials sums the ranks' bf16 pair partials in float32 in rank
    order, bit for bit. The order check needs `tp >= 3`."""
    sum_partials = _SUM_PARTIALS[lib]
    partials = I.wide_bf16((1 << 16, tp, 2), (-8, 8), seed=5)

    def body(*words: Any) -> list[Any]:
        polled = [[words[rank]] for rank in range(tp)]
        return [
            H.i32(x) for x in sum_partials(lambda reqs: polled, None, lambda s: s, tp)
        ]

    got = _run(f"sp_{lib}_{tp}", body, 2, partials.view(-1, tp * 2).view(I32))
    in_order = partials[:, 0].float()
    for rank in range(1, tp):
        in_order = in_order + partials[:, rank].float()
    I.assert_bit_exact(got.view(F32), in_order, "sum_partials", partials.float().sum(1))
    backwards = partials.float().flip(1)
    reversed_order = backwards[:, 0]
    for rank in range(1, tp):
        reversed_order = reversed_order + backwards[:, rank]
    assert tp == 2 or not torch.equal(in_order, reversed_order), (
        "inputs blind to the rank order"
    )


# ---------------------------------------------------------------- MFMA
def _mfma_error(
    lanes: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    acc: torch.Tensor,
    block: int | None = None,
) -> float:
    """Error of the MFMA result `lanes` for `D = C + A B^T`, as a fraction of
    `B.mfma_accumulation_bound`."""
    product = I.from_acc_lanes(lanes.view(F32).double())
    reference = acc + a @ b.transpose(1, 2)
    bound = B.mfma_accumulation_bound(a, b, acc, block)
    return ((product - reference).abs() / bound).max().item()


_MFMA_BF16: dict[str, H.DeviceFn] = {
    "dsv41": ops.mfma_bf16,
    "dsv41_attn": attn.mfma_bf16,
    # gfx942 emulation from two 16 x 16 x 16 MFMAs.
    "dsv41-k32": gfx.mfma_bf16_k32,
}


@pytest.mark.parametrize("variant", _MFMA_BF16)
def test_mfma_bf16(variant: str) -> None:
    """The 16 x 16 x 32 bf16 MFMA against the float64 product."""
    mfma = _MFMA_BF16[variant]
    a, b, acc = I.bf16_mfma_operands(256, seed=7)

    def body(*words: Any) -> list[Any]:
        operands = [
            fx.Vector.from_elements(list(words[i : i + 4]), fx.Int32) for i in (0, 4)
        ]
        accumulator = fx.Vector.from_elements([H.f32(w) for w in words[8:]], fx.Float32)
        d = mfma(
            operands[0].bitcast(fx.BFloat16),
            operands[1].bitcast(fx.BFloat16),
            accumulator,
        )
        return [H.i32(d[i]) for i in range(4)]

    lanes = [I.mfma_lanes(t, 8).contiguous().view(I32) for t in (a, b)]
    # Kernel names cannot contain "-".
    got = _run(f"mfma_{variant.replace('-', '_')}", body, 4, *lanes, I.acc_lanes(acc))
    error = _mfma_error(got, a.double(), b.double(), acc.double())
    assert error <= 1, f"{error:.3g} of the fp32 accumulation bound"


class _ScaledMfma(NamedTuple):
    """A scaled MFMA implementation and the cases to test it on."""

    mfma: H.DeviceFn
    # (A, B) operand formats used by callers.
    formats: tuple[tuple[int, int], ...]
    # Scale-word byte for A and for B.
    selectors: tuple[tuple[int, int], ...]
    # False if the wrapper hard-codes byte 0 and takes no selector arguments.
    takes_selectors: bool


_MFMA_SCALED: dict[str, _ScaledMfma] = {
    "dsv41": _ScaledMfma(
        mfma=mx.mfma_scaled,
        formats=((mx.FP8, mx.FP8), (mx.FP4, mx.FP8), (mx.FP4, mx.FP4)),
        selectors=((0, 0), (1, 2), (3, 1)),
        takes_selectors=True,
    ),
    "dsv41_attn": _ScaledMfma(
        mfma=attn.mfma_scaled,
        formats=((attn.FP8, attn.FP8),),
        selectors=((0, 0),),
        takes_selectors=False,
    ),
}

# The same check for the bounded helpers.
for _copies in (
    *(case.copies for case in _TRANSCENDENTALS.values()),
    _DIV_RN,
    _BUTTERFLY,
    _MFMA_BF16,
    {lib: spec.mfma for lib, spec in _MFMA_SCALED.items()},
):
    if "dsv41_attn" in _copies:
        assert _copies["dsv41"] is not _copies["dsv41_attn"], (
            "the two copies now share a helper"
        )


def _scaled_cases() -> list[Any]:
    out = []
    for lib, spec in _MFMA_SCALED.items():
        for formats in spec.formats:
            names = "x".join("fp4" if fmt else "fp8" for fmt in formats)
            for selector in spec.selectors:
                out.append(
                    pytest.param(lib, formats, selector, id=f"{lib}-{names}-{selector}")
                )
    return out


def _scaled_mfma_error(
    lib: str,
    formats: tuple[int, int],
    selector: tuple[int, int],
    seed: int,
    tiles: int,
) -> tuple[float, float]:
    """Run the scaled MFMA of `lib` on random operands. Returns the error as a
    fraction of the full bound and of the fp32-only bound."""
    spec = _MFMA_SCALED[lib]
    operands = I.mx_mfma_operands(tiles, formats, seed)
    a_words, b_words = operands.a_words, operands.b_words
    a_count, b_count = a_words.shape[1], b_words.shape[1]
    extra = selector if spec.takes_selectors else ()

    def body(*words: Any) -> list[Any]:
        a = fx.Vector.from_elements(list(words[:a_count]), fx.Int32)
        b = fx.Vector.from_elements(list(words[a_count : a_count + b_count]), fx.Int32)
        accumulator = fx.Vector.from_elements(
            [H.f32(w) for w in words[a_count + b_count : -2]], fx.Float32
        )
        d = spec.mfma(a, b, accumulator, words[-2], words[-1], *formats, *extra)
        return [H.i32(d[i]) for i in range(4)]

    got = _run(
        f"mfs_{lib}_{formats[0]}{formats[1]}_{selector[0]}{selector[1]}",
        body,
        4,
        a_words,
        b_words,
        I.acc_lanes(operands.acc),
        operands.scale_words[0],
        operands.scale_words[1],
    )

    def scaled(values: torch.Tensor, side: int) -> torch.Tensor:
        """`values` `[tiles, 16, 128]` times its selected K-block scale."""
        code = operands.scale_codes[side, :, selector[side]]
        scale = R.e8m0(code.view(tiles, 4, 16).transpose(1, 2))
        return values * scale.repeat_interleave(32, -1)

    a = scaled(operands.a_values, 0)
    b = scaled(operands.b_values, 1)
    # FP4 x FP4 products span at most 12^2 = 144, so the block sum loses
    # nothing and needs no block term.
    block = None if formats == (I.MX_FP4, I.MX_FP4) else 32
    acc = operands.acc.double()
    return _mfma_error(got, a, b, acc, block), _mfma_error(got, a, b, acc)


@pytest.mark.skipif(GFX942, reason="the scaled f8f6f4 MFMA is gfx950 only")
@pytest.mark.parametrize(("lib", "formats", "selector"), _scaled_cases())
def test_mfma_scaled(
    lib: str, formats: tuple[int, int], selector: tuple[int, int]
) -> None:
    """The 16 x 16 x 128 scaled MFMA (f8f6f4) against the float64 product of
    the dequantized operands.

    The FP8 bound is dominated by the empirical `B._FP8_BLOCK_PRECISION`, so
    this checks the instruction is wired correctly rather than its precision.
    """
    error, fp32_only = _scaled_mfma_error(lib, formats, selector, seed=8, tiles=256)
    assert error <= 1, f"{error:.3g} of the bound"
    if formats != (I.MX_FP4, I.MX_FP4):
        # The block term must be needed: the error should far exceed the
        # fp32-only bound (25-49x on gfx950).
        assert fp32_only > 10, (
            f"the error is only {fp32_only:.3g}x an fp32-only accumulation "
            f"bound, so this instruction may now be accumulating a K block in "
            f"fp32; if so the 2^-14 block envelope is no longer justified and "
            f"`B._FP8_BLOCK_PRECISION` should go rather than be loosened"
        )


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
