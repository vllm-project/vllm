# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""What the formats and the layer math a specialized decode uses are defined
to do, in float64.

Any model's decode tests can use these: a test names the rule it expects
(`bf16_rne`, `to_e4m3_rne`, an E8M0 code rule, an RMSNorm, a softmax) and
compares its kernel against it, rather than inheriting whatever rounding some
other implementation happens to do.

A definition must not be derived from the implementation it checks, nor reach
for a helper that implementation also reaches for, or a shared bug cancels and
the test passes. That is about where a definition comes from, not about which
library writes it down.

Two things the layer math takes as arguments rather than deciding here:

- **Summation order.** Float32 addition does not associate, so the order a
  reduction runs in is part of what a kernel computes. `ordered_sum` takes it;
  so do the definitions built on it. Hard-coding one kernel's order would make
  this file that kernel's transcript.
- **Where the rounding falls.** Everything below is float64 throughout. A
  caller that wants a kernel's intermediate bf16 or MXFP8 rounding asks for it
  (`bf16_rne`, `mx_quantize`) at the point its kernel rounds. How far from a
  definition here a kernel may then land is a tolerance question, not a
  definition question.

Formats are all that the primitive tier (T1) of #68016 needs; the layer math
is the yardstick the `DecodeNumerics` contract (T2) measures against, shared
rather than written once per model so that it measures every model the same
way. The error metrics land here next."""

import math
from collections.abc import Callable, Sequence

import torch
import torch.nn.functional as F

F64 = torch.float64
# The smallest positive normal float32, which the E8M0 code rules floor at,
# under the name the device code gives it (attention/device.py, stages/moe.py).
FLT_MIN = 2.0**-126
_E2M1_MAGNITUDES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
# Indexed by the 4-bit E2M1 code: the sign is bit 3, so the negatives follow
# the positives in the same order.
E2M1_VALUES = torch.tensor(
    _E2M1_MAGNITUDES + tuple(-v for v in _E2M1_MAGNITUDES), dtype=F64
)
_FP8_BYTE_DTYPES = (
    torch.uint8,
    torch.int8,
    torch.float8_e4m3fn,
    torch.float8_e4m3fnuz,
)


# ---------------------------------------------------------------- formats
def e8m0(codes: torch.Tensor) -> torch.Tensor:
    return torch.exp2(codes.double() - 127)


def _expand(scale: torch.Tensor, block: tuple[int, int]) -> torch.Tensor:
    return scale.repeat_interleave(block[0], -2).repeat_interleave(block[1], -1)


def dequant_fp8(
    quantized: torch.Tensor,
    scales: torch.Tensor,
    block: tuple[int, int],
    fnuz: bool,
) -> torch.Tensor:
    """E4M3 [..., N, K] (OCP, or FNUZ: the same bytes at half the value) with
    E8M0 codes [..., N / block[0], K / block[1]] -> float64 [..., N, K]."""
    assert quantized.dtype in _FP8_BYTE_DTYPES, (
        f"`quantized` holds fp8 bytes, got {quantized.dtype}"
    )
    assert not scales.is_floating_point(), (
        f"`scales` holds E8M0 codes, got {scales.dtype}"
    )
    fp8_dtype = torch.float8_e4m3fnuz if fnuz else torch.float8_e4m3fn
    values = quantized.view(torch.uint8).view(fp8_dtype).double()
    return values * _expand(e8m0(scales), block)


def dequant_mxfp4(codes: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """fp4x2 bytes [..., K / 2] (the low nibble the lower K) with E8M0 codes
    [..., K / 32] -> float64 [..., K]."""
    # uint8 only, not the wider byte set `quantize_e4m3` accepts: the nibble
    # split below is `& 15` and `>> 4`, and a float8 dtype has neither, so it
    # would raise out of torch rather than here. int8 would quietly work --
    # its arithmetic shift lands on a negative index that wraps to the same
    # table entry -- but it is nobody's layout for packed fp4 pairs, and
    # accepting it would leave that coincidence load-bearing.
    assert codes.dtype == torch.uint8, f"`codes` holds fp4x2 bytes, got {codes.dtype}"
    assert not scales.is_floating_point(), (
        f"`scales` holds E8M0 codes, got {scales.dtype}"
    )
    table = E2M1_VALUES.to(codes.device)
    nibbles = torch.stack([table[(codes & 15).long()], table[(codes >> 4).long()]], -1)
    return nibbles.flatten(-2) * e8m0(scales).repeat_interleave(32, -1)


# ---------------------------------------------------------------- rounding
def round_float(
    x: torch.Tensor,
    mantissa_bits: int,
    min_exponent: int,
    fmax: float | None = None,
) -> torch.Tensor:
    """`x` rounded to nearest-even onto a binary format with `mantissa_bits`
    mantissa bits and normal exponents >= `min_exponent` (subnormals below),
    saturating at `fmax` if given, else overflowing to inf at 2^(emax+1)."""
    x = x.double()
    if fmax is not None:
        x = x.clamp(-fmax, fmax)
    exponent = (torch.frexp(x.abs()).exponent.long() - 1).clamp(min=min_exponent)
    # float64, or exp2 of an integer tensor gives the quantum in the default
    # dtype and 2^(exponent - mantissa_bits) is past float32's range for a
    # large x.
    quantum = torch.exp2((exponent - mantissa_bits).double())
    rounded = torch.where(
        torch.isfinite(x) & (x != 0), torch.round(x / quantum) * quantum, x
    )
    if fmax is None:
        rounded = torch.where(
            rounded.abs() >= 2.0**128, rounded * float("inf"), rounded
        )
    return rounded


def bf16_rne(x: torch.Tensor) -> torch.Tensor:
    return round_float(x, 7, -126)


def f32_rne(x: torch.Tensor) -> torch.Tensor:
    return round_float(x, 23, -126)


def to_e4m3_rne(x: torch.Tensor, fnuz: bool) -> torch.Tensor:
    """The E4M3 value of `x` clamped to the format maximum (448 OCP, 224
    FNUZ), RNE."""
    return round_float(x, 3, -7 if fnuz else -6, 224.0 if fnuz else 448.0)


def _ceil_log2(value: torch.Tensor) -> torch.Tensor:
    mantissa, exponent = torch.frexp(value.double())
    return torch.where(
        value > 0, exponent.long() - (mantissa == 0.5).long(), -(1 << 20)
    )


def ceil_code(value: torch.Tensor) -> torch.Tensor:
    """The E8M0 round-up rule on an f32 `value` >= 0: its exponent field, plus
    one if its mantissa field is not zero (the smallest power of two >= the
    value for a normal one; 1 for a subnormal, 0 for zero)."""
    value = value.double()
    return torch.where(value < FLT_MIN, (value > 0).long(), _ceil_log2(value) + 127)


def mx_code_vllm(amax: torch.Tensor, fmax: float) -> torch.Tensor:
    """VLLM's `mxfp8_e4m3_quantize`: ceil(log2(RN_f32(max(amax, FLT_MIN)
    / fmax))) + 127, clamped to [0, 254] (0 for an all-zero group)."""
    scaled = f32_rne(amax.double().clamp(min=FLT_MIN) / fmax)
    return (_ceil_log2(scaled) + 127).clamp(0, 254)


# ---------------------------------------------------------------- route
def sqrt_softplus_score(logits: torch.Tensor) -> torch.Tensor:
    """The sqrt-softplus router score of `logits`, which is what DeepSeek-V4.1
    routes on. `F.softplus` returns the logit itself past 20, which is the
    same shortcut the kernels take, so the two agree there by construction
    rather than inside a tolerance.

    Named after the formula rather than the model: a router that scores some
    other way belongs next to this one, not behind a flag on it.
    """
    return torch.sqrt(F.softplus(logits.double()))


# ---------------------------------------------------------------- operands
def unshuffle_weight(
    weight: torch.Tensor, rows: int = 16, cols: int = 16
) -> torch.Tensor:
    """A weight held in AITER's `(rows, cols)` preshuffle, back in row-major
    [..., N, K] order so that a dequantizer can read it.

    The preshuffle walks a group of `rows` rows one K block at a time --
    `[N / rows][K / cols][row][col]`, the order an MFMA's operand loads want --
    so undoing it is that transpose the other way round. `cols` counts stored
    bytes, which for a packed fp4x2 weight is half its K.

    Written here rather than imported because AITER ships the forward
    permutation and no inverse, and these tests run on a CPU where AITER is
    not importable at all. The durable home for it is AITER's own
    `shuffle_weight`; if one lands there, this should become its caller --
    an inverse that only exists here is a third place the layout is written
    down, and the one nobody updates.
    """
    n, k = weight.shape[-2:]
    assert n % rows == 0 and k % cols == 0, (
        f"a ({rows}, {cols}) shuffle needs whole blocks, got [{n}, {k}]"
    )
    blocks = weight.reshape(-1, n // rows, k // cols, rows, cols)
    return blocks.transpose(-3, -2).reshape(weight.shape)


def unshuffle_scale(scales: torch.Tensor) -> torch.Tensor:
    """E8M0 codes [..., rows, cols] held in AITER's `shuffle_scale` order,
    back in row-major order.

    `shuffle_scale` is `view(rows / 32, 2, 16, cols / 8, 2, 4).permute(0, 3,
    5, 2, 4, 1)`: with `row = 32 r + 16 a + b` and `col = 8 c + 4 d + e`, the
    codes are written out in `(r, c, e, b, d, a)` order, which puts a 32-row
    block's scales -- both 16-row halves, all of K -- in one contiguous run.
    This is that permutation inverted.
    """
    rows, cols = scales.shape[-2:]
    assert rows % 32 == 0 and cols % 8 == 0, (
        f"`shuffle_scale` needs [..., 32k, 8k] codes, got [{rows}, {cols}]"
    )
    stored = scales.reshape(-1, rows // 32, cols // 8, 4, 16, 2, 2)
    return stored.permute(0, 1, 6, 4, 2, 5, 3).reshape(scales.shape)


def mx_quantize(
    x: torch.Tensor, fmax: float, group: int = 32, fnuz: bool = False
) -> torch.Tensor:
    """`x` [..., K] through one MX block quantization and back out: the E8M0
    code `mx_code_vllm` gives each group of `group` values from that group's
    amax, then E4M3 round-to-nearest-even of the values divided by it.

    Float64 in, float64 out: this is the operand a kernel's GEMM actually
    multiplies, not the bytes it stores. A definition that needs to know what
    an activation lost on its way into a kernel calls this at the point the
    kernel quantizes, and stays in float64 either side of it.

    `fmax` is what the scale code aims the group's amax at, and `fnuz` picks
    the E4M3 the values then round onto. Aiming past the format's own maximum
    saturates every group whose amax lands in the top of its range -- a clip
    of up to 2x that produces numbers rather than an error -- so the two have
    to agree. Below it is headroom, which is the caller's business.
    """
    e4m3_max = 224.0 if fnuz else 448.0
    assert fmax <= e4m3_max, (
        f"the scale code aims at `fmax` and the rounding saturates at the "
        f"E4M3 maximum, so `fmax` is at most {e4m3_max} for "
        f"{'FNUZ' if fnuz else 'OCP'}, got {fmax}"
    )
    assert x.shape[-1] % group == 0, (
        f"`x`'s last axis is whole groups of {group}, got {x.shape[-1]}"
    )
    grouped = x.double().unflatten(-1, (-1, group))
    scale = e8m0(mx_code_vllm(grouped.abs().amax(-1, keepdim=True), fmax))
    return (to_e4m3_rne(grouped / scale, fnuz) * scale).flatten(-2)


# ---------------------------------------------------------------- layers
def ordered_sum(
    terms: torch.Tensor,
    dim: int = -1,
    order: Sequence[int] | None = None,
    round_to: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> torch.Tensor:
    """`terms` summed along `dim` one term at a time, in `order` (every index
    in turn by default), each running total -- the first term included -- put
    through `round_to` if given.

    Float32 addition does not associate, so the order is part of what a kernel
    computes rather than a detail of how: a router's K parts summed in part
    order, a token's experts in its top-k order, an all-reduce's partials in
    rank order. It is an argument because it is the caller's kernel's choice,
    not this file's.
    """
    terms = terms.double()
    count = terms.shape[dim]
    order = list(range(count)) if order is None else list(order)
    # A reordering, not a selection: an order that drops a term or names one
    # twice would still return a well-formed total, and a reference that
    # quietly answers the wrong question is worse than one that stops.
    assert sorted(order) == list(range(count)), (
        f"`order` is a permutation of the {count} terms along dim {dim}, got {order}"
    )
    total = None
    for i in order:
        term = terms.select(dim, i)
        total = term if total is None else total + term
        if round_to is not None:
            total = round_to(total)
    assert total is not None, "`order` names at least one term"
    return total


def rms_norm(
    x: torch.Tensor, weight: torch.Tensor | None = None, eps: float = 0.0
) -> torch.Tensor:
    """`x` [..., H] over the root mean square of its own row, times `weight`
    if given: `x * rsqrt(mean(x^2) + eps)`.

    Mean of the squares, not sum: `eps` sits next to a quantity that does not
    grow with H, which is why the same `eps` means the same thing at every
    hidden size.

    Written out rather than deferred to `F.rms_norm`, which the layers under
    test reach for themselves -- a yardstick that calls the same helper as
    the thing it measures cancels that helper's bugs instead of catching
    them. Four lines is the price of the definition being independent.
    """
    x = x.double()
    normed = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps)
    return normed if weight is None else normed * weight.double()


def sinkhorn(logits: torch.Tensor, eps: float, repeat: int) -> torch.Tensor:
    """`logits` [..., M, M] softmaxed along its rows, then pushed toward
    doubly stochastic: columns once, then rows and columns `repeat - 1` more
    times, `eps` added to the softmax output and to every denominator.

    `eps` in the denominators is part of the definition, not a guard on it:
    it leaves the result near doubly stochastic rather than exactly so, and
    a kernel that dropped it would converge somewhere else.
    """
    # `range(repeat - 1)` is empty for 0 and for every negative value, so
    # without this a caller asking for no passes at all silently gets one.
    assert repeat >= 1, (
        f"`repeat` counts column passes, so it is at least 1, got {repeat}"
    )
    comb = torch.softmax(logits.double(), dim=-1) + eps
    comb = comb / (comb.sum(-2, keepdim=True) + eps)
    for _ in range(repeat - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + eps)
        comb = comb / (comb.sum(-2, keepdim=True) + eps)
    return comb


def mhc_post(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_mix: torch.Tensor,
    comb_mix: torch.Tensor,
) -> torch.Tensor:
    """The mHC post fold: a sublayer's output `x` [T, H] folded into the
    residual streams `residual` [T, M, H], stream j becoming
    `sum_i comb[t, i, j] r[t, i] + post[t, j] x[t]`.

    `post_mix` is [T, M] and `comb_mix` [T, M, M] -- the mixes the *previous*
    seam handed out, which is what makes the seam a delayed one, and the
    shapes `mhc_mixes` publishes them in, so a seam's output is the next
    seam's input with nothing reshaped in between.
    """
    residual, x = residual.double(), x.double()
    m = residual.shape[-2]
    assert post_mix.shape[-1] == m and tuple(comb_mix.shape[-2:]) == (m, m), (
        f"`residual` carries {m} streams, so `post_mix` is [..., {m}] and "
        f"`comb_mix` [..., {m}, {m}], got {tuple(post_mix.shape)} and "
        f"{tuple(comb_mix.shape)}"
    )
    mixed = torch.einsum("...ij,...ih->...jh", comb_mix.double(), residual)
    return mixed + post_mix.double().unsqueeze(-1) * x.unsqueeze(-2)


def mhc_collapse(residual: torch.Tensor, pre_mix: torch.Tensor) -> torch.Tensor:
    """The streams `residual` [T, M, H] collapsed to the one input a sublayer
    sees, [T, H], by the pre mix [T, M]: `sum_i pre[t, i] r[t, i]`."""
    return (pre_mix.double().unsqueeze(-1) * residual.double()).sum(-2)


def mhc_mixes(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    *,
    rms_eps: float,
    pre_eps: float,
    sinkhorn_eps: float,
    post_mult: float,
    sinkhorn_repeat: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The mixes a seam hands the next one, read off the folded residual
    `residual` [T, M, H] by the gate projection `fn` [M (M + 2), M H] of the
    flattened streams, scaled by their own RMS.

    The projection's `M (M + 2)` rows split three ways -- pre, post, then the
    M x M comb -- each row group taking its own entry of `hc_scale` and its
    own slice of `hc_base`. -> pre [T, M], post [T, M], comb [T, M, M].

    The pre mix has `pre_eps` added rather than multiplied, so no stream can
    be switched off entirely; the post mix is a sigmoid times `post_mult`, so
    it is bounded; the comb mix goes through `sinkhorn`.
    """
    m = residual.shape[-2]
    rows = m * (m + 2)
    assert fn.shape[0] == rows, (
        f"the gate projection has M (M + 2) = {rows} rows for {m} streams, "
        f"got {fn.shape[0]}"
    )
    assert hc_scale.numel() == 3 and hc_base.numel() == rows, (
        f"one `hc_scale` per row group and one `hc_base` per row: 3 and "
        f"{rows}, got {hc_scale.numel()} and {hc_base.numel()}"
    )
    flat = residual.double().flatten(-2)
    scale, base = hc_scale.double(), hc_base.double()
    mixes = (flat @ fn.double().t()) * torch.rsqrt(
        flat.square().mean(-1, keepdim=True) + rms_eps
    )
    pre = torch.sigmoid(mixes[..., :m] * scale[0] + base[:m]) + pre_eps
    post = torch.sigmoid(mixes[..., m : 2 * m] * scale[1] + base[m : 2 * m]) * post_mult
    comb = mixes[..., 2 * m :].unflatten(-1, (m, m)) * scale[2] + base[2 * m :].view(
        m, m
    )
    return pre, post, sinkhorn(comb, sinkhorn_eps, sinkhorn_repeat)


def mhc_delayed_seam(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_mix: torch.Tensor,
    comb_mix: torch.Tensor,
    pre_mix: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    norm_weight: torch.Tensor | None = None,
    *,
    norm_eps: float,
    rms_eps: float,
    pre_eps: float,
    sinkhorn_eps: float,
    post_mult: float,
    sinkhorn_repeat: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """One delayed mHC seam: fold `x` into the residual with the mixes the
    previous seam left (`mhc_post`), collapse the folded streams with the pre
    mix it also left (`mhc_collapse`), RMSNorm that into the next sublayer's
    input, and read this seam's own mixes off the same folded residual.

    Delayed is the whole point: the mixes a seam computes are not the ones it
    applies. It applies the previous seam's and publishes its own, so the
    projection that reads the residual does not have to finish before the
    sublayer input can leave. -> (residual [T, M, H], layer input [T, H],
    post [T, M], comb [T, M, M], pre [T, M]).

    The three mixes come back in the order the next seam takes them, so
    `seam(y, folded, *mixes, fn, ...)` is the chain. Returning them in any
    other order would make the obvious forwarding silently misroute a mix.
    """
    folded = mhc_post(x, residual, post_mix, comb_mix)
    layer_input = rms_norm(mhc_collapse(folded, pre_mix), norm_weight, norm_eps)
    pre, post, comb = mhc_mixes(
        folded,
        fn,
        hc_scale,
        hc_base,
        rms_eps=rms_eps,
        pre_eps=pre_eps,
        sinkhorn_eps=sinkhorn_eps,
        post_mult=post_mult,
        sinkhorn_repeat=sinkhorn_repeat,
    )
    return folded, layer_input, post, comb, pre


def rope_pairs(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, inverse: bool = False
) -> torch.Tensor:
    """GPT-J rotary embedding of `x` [..., 2 P] with `cos` and `sin`
    [..., P]: the pair `(e, o)` becomes `(e cos - o sin, e sin + o cos)`, or
    `(e cos + o sin, o cos - e sin)` if `inverse`.

    Adjacent pairs, not split halves: dimensions `2 p` and `2 p + 1` rotate
    together. A model whose tables are laid out for the split-half convention
    rotates `(x[p], x[P + p])` instead and is a different function, not the
    same one in a different order.

    Each pair is a plane rotation, so the inverse is the transpose and the
    round trip is exact up to float64 round-off, and `cos ** 2 + sin ** 2`
    being one is what makes a pair's norm survive -- neither is assumed here.
    """
    pairs = cos.shape[-1]
    assert x.shape[-1] == 2 * pairs, (
        f"`cos` holds {pairs} pairs, so `x` holds {2 * pairs} values, got {x.shape[-1]}"
    )
    x, cos, sin = x.double(), cos.double(), sin.double()
    even, odd = x[..., 0::2], x[..., 1::2]
    if inverse:
        rotated = (even * cos + odd * sin, odd * cos - even * sin)
    else:
        rotated = (even * cos - odd * sin, even * sin + odd * cos)
    return torch.stack(rotated, -1).flatten(-2)


# Base-two softmax is the same function as base-e -- the factor cancels
# between numerator and denominator -- computed the way a kernel computes it,
# where exp2 is one instruction and exp is a sequence ending in one.
LOG2E = math.log2(math.e)


def mla_decode(
    q: torch.Tensor,
    keys: torch.Tensor,
    scale: float,
    sink: torch.Tensor | None = None,
    chunks: Sequence[int] | None = None,
    order: Sequence[int] | None = None,
) -> torch.Tensor:
    """One token's MLA decode: queries `q` [H, D] against the latent KV
    records `keys` [K, D], which are their own values, scores `scale q . k`
    softmaxed over the keys. -> [H, D].

    `scale` is in natural units (`D ** -0.5` as a rule); the log2(e) that
    turns the exponential into an exp2 belongs to how the softmax is taken,
    so it is applied here rather than asked of the caller.

    `sink` [H] is a per-head logit, in those same natural units, that joins
    the denominator with no value of its own: a head can put weight on
    nothing, and its output shrinks instead of being forced to average
    something. The weights over the keys then sum to less than one.

    `chunks` gives the key counts of the flash-decoding partials, each taking
    its own running maximum and rescaled against the final one as it is
    combined -- in `order` (`ordered_sum`'s), after the sink term, which is
    where the arithmetic depends on the split and so is an argument.
    """
    q, keys = q.double(), keys.double()
    sizes = [keys.shape[0]] if chunks is None else list(chunks)
    assert sizes and min(sizes) > 0 and sum(sizes) == keys.shape[0], (
        f"`chunks` splits all {keys.shape[0]} keys into non-empty parts, got {sizes}"
    )
    order = list(range(len(sizes))) if order is None else list(order)
    # Same reason as `ordered_sum`: a chunk left out of `order` drops from
    # the numerator and the denominator together, so the result stays a
    # finite, normalized-looking attention output and nothing complains.
    assert sorted(order) == list(range(len(sizes))), (
        f"`order` is a permutation of the {len(sizes)} chunks, got {order}"
    )
    logits = (q @ keys.t()) * (scale * LOG2E)
    parts = logits.split(sizes, -1)
    values = keys.split(sizes, 0)
    maxes = torch.stack([part.amax(-1) for part in parts])
    top = maxes.amax(0)
    if sink is not None:
        top = torch.maximum(top, sink.double() * LOG2E)
        denominator = torch.exp2(sink.double() * LOG2E - top)
    else:
        denominator = torch.zeros_like(top)
    numerator = torch.zeros_like(q)
    for c in order:
        weights = torch.exp2(parts[c] - maxes[c].unsqueeze(-1))
        rescale = torch.exp2(maxes[c] - top)
        denominator = denominator + weights.sum(-1) * rescale
        numerator = numerator + (weights @ values[c]) * rescale.unsqueeze(-1)
    return numerator / denominator.unsqueeze(-1)


def moe_route(
    logits: torch.Tensor,
    bias: torch.Tensor | None = None,
    topk: int = 1,
    renormalize: bool = True,
    scale: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """VLLM's `topk_hash_softplus_sqrt` route, without the hash: score every
    expert with `sqrt_softplus_score`, pick the `topk` highest of the scores
    plus `bias`, and weight each pick by its score *without* the bias --
    renormalized to sum to one if `renormalize`, then times `scale`.
    -> (ids [T, topk] int64, weights [T, topk]).

    The bias picks the experts and then steps out of the arithmetic, which is
    what lets it be tuned for load balance without moving the output. Ties go
    to the lower expert id: `bias` is shared across tokens, so two experts can
    draw exactly, and a kernel that broke the tie the other way would route
    the token somewhere else -- a definition, not a detail.
    """
    assert 1 <= topk <= logits.shape[-1], (
        f"`topk` is at most the {logits.shape[-1]} experts, got {topk}"
    )
    scores = sqrt_softplus_score(logits)
    choice = scores if bias is None else scores + bias.double()
    ids = torch.argsort(choice, dim=-1, descending=True, stable=True)[..., :topk]
    weights = scores.gather(-1, ids)
    if renormalize:
        weights = weights / weights.sum(-1, keepdim=True)
    return ids, weights * scale


def swiglu_clamp(gate: torch.Tensor, up: torch.Tensor, limit: float) -> torch.Tensor:
    """VLLM's `silu_and_mul_with_clamp`: `silu(min(gate, limit)) *
    clamp(up, -limit, limit)`.

    One-sided on the gate and two-sided on the up: silu is bounded below
    already, so only the gate's upper tail needs capping, and the cap is on
    the gate before the silu rather than on the product after it.
    """
    gate = gate.double().clamp(max=limit)
    return gate * torch.sigmoid(gate) * up.double().clamp(-limit, limit)


def moe_expert(
    x: torch.Tensor,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    limit: float,
) -> torch.Tensor:
    """One expert on `x` [..., H]: the gate and up projections [I, H], a
    clamped SwiGLU between them, then the down projection [H, I].

    Gate and up are separate arguments because the layout that interleaves
    them is a storage decision (`unshuffle_weight`), not part of the math.
    """
    x = x.double()
    hidden = swiglu_clamp(
        x @ gate_weight.double().t(), x @ up_weight.double().t(), limit
    )
    return hidden @ down_weight.double().t()


def moe_combine(
    outputs: torch.Tensor,
    weights: torch.Tensor,
    order: Sequence[int] | None = None,
    round_to: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> torch.Tensor:
    """A token's expert outputs [T, topk, H], each scaled by its routing
    weight [T, topk] and summed in `order` -- by default the top-k order the
    route returned, which is the order a kernel accumulates in and so part of
    the answer rather than of the implementation. `round_to` is applied to
    every running total (`ordered_sum`), which is how a kernel that keeps its
    accumulator in bf16 is described.
    """
    terms = outputs.double() * weights.double().unsqueeze(-1)
    return ordered_sum(terms, dim=-2, order=order, round_to=round_to)
