# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""How far from the float64 definition a kernel is allowed to land, derived
rather than measured. Plain torch, no device code.

A tolerance taken from running the kernel it checks only records what that
kernel did on the day it was run: it moves when the kernel moves, and it never
says whether the kernel was right. The bounds here are derived instead, from
the accuracy the ISA documents and from the arithmetic the helper performs, so
the number a test asserts against exists before the kernel does.

`RoundingModel` covers a helper assembled out of f32 operations, and
`mfma_accumulation_bound` the matrix instructions. What the error is measured
from is `reference.py`; the inputs that make it visible are in `inputs.py`.

One term is an exception and is marked as one: the FP8 MFMA's block-max drop,
whose size the ISA does not state. It is an envelope over the behaviour, kept
here with the derived terms so that every model inherits the same envelope
instead of each fitting its own.

Shared rather than written once per model on purpose. The `DecodeNumerics`
contract (T2 of #68016) compares one model's decode against another's only if
the two were bounded the same way, and a bound each model derives for itself
drifts the moment one of them rounds a constant differently.
"""

import torch

from tests.models.common.specialized_decode import reference as R

# One f32 round-to-nearest's relative error: half the spacing at 1, which is
# the unit roundoff rather than an ulp. `n * F32_ROUNDOFF` below is therefore
# "n roundings", not "n ulps".
F32_ROUNDOFF = 2.0**-24

# CDNA3 and CDNA4 ISA 12.8 (VOP1) gives V_EXP_F32, V_LOG_F32 and V_RSQ_F32 as
# "1ULP accuracy, denormals are flushed". One ulp is two roundoffs at the worst
# point of a binade, and the flush costs a result that cannot be a normal the
# whole of itself. V_RCP_F32 is the same bare, and only reaches "< 0.5ULP"
# after the two fixup FMAs of a Newton-Raphson step.
TRANSCENDENTAL_RELATIVE = 2 * F32_ROUNDOFF
TRANSCENDENTAL_FLUSH = R.FLT_MIN


def transcendental_bound(reference: torch.Tensor) -> torch.Tensor:
    """How far a bare hardware transcendental may land from `reference`, the
    exact result: one f32 ulp of it, except where it is too small to be a
    normal float32 and the flush may take all of it.

    The two halves are the two halves of the ISA sentence, and the split is at
    the value the hardware flushes at rather than an ulp either side of it: an
    ulp of a subnormal is `2^-149`, so measuring the flush in ulps would report
    a correct result as millions of them.

    The split sits one relative step *above* `FLT_MIN`, not at it. A result
    that close to the smallest normal is allowed to come back one ulp low,
    which is a subnormal, which the hardware then flushes -- so the whole
    result is in play there too, and a bound of one ulp would fail a chip
    doing exactly what the ISA says. Nothing wider: the region is `2^-23` of
    one binade.
    """
    return torch.where(
        reference.abs() < TRANSCENDENTAL_FLUSH * (1 + TRANSCENDENTAL_RELATIVE),
        TRANSCENDENTAL_FLUSH,
        R.ulp(reference, torch.float32),
    )


# The one figure in this file that is not derived. A gfx950 FP8 MFMA does not
# sum a K block as an fp32 sum of the separate products: a product far enough
# below the block's largest one does not reach the sum at all, and the ISA
# documents the block-scaled instruction (CDNA4 ISA 7.2) without saying how far
# below that is. 2^-14 is an envelope over the behaviour rather than a
# published number -- on gfx950 the measured error of `test_mfma_scaled` sits
# at 0.20-0.24 of the bound it produces, so it is loose by about 4x, not fitted
# to a run. The one test that rests on it says the same.
_FP8_BLOCK_PRECISION = 2.0**-14


class RoundingModel:
    """A helper's formula in float64 with every f32 rounding written as
    `value -> value (1 + relative) + absolute`, the perturbations starting at
    zero and carrying a gradient.

    Build the formula through the instance -- each call stands for one of the
    helper's roundings -- and `bound` reads the first-order error bound
    `sum_i |d result / d relative_i| * relative_i + |d result / d absolute_i| *
    absolute_i` off the result by autograd. The derivative is exactly the
    weight that rounding carries into the answer, which is the part a bound
    written out by hand gets wrong on anything with a cancellation in it.

    Second-order terms in the perturbations are dropped: they are down by
    another 2^-24, and the roundings are not independent anyway, so carrying
    them would buy no honesty.

    Elementwise formulas only. The bound is read off `result.sum()`, which is
    the usual autograd trick for getting a per-element derivative in one pass,
    and it is only that if no element of the result depends on another
    element's rounding. A formula that reduces across the draw -- a sum, a max,
    a matmul -- would have the sum fold those cross terms together and report
    a bound that is not the one any element obeys. `bound` asserts the shapes
    rather than leaving that to be remembered.
    """

    def __init__(self) -> None:
        # (perturbation, how large it may be), one or two per rounding.
        self._perturbations: list[tuple[torch.Tensor, float]] = []

    def __call__(
        self,
        value: torch.Tensor,
        relative: float = F32_ROUNDOFF,
        absolute: float = 0.0,
    ) -> torch.Tensor:
        """`value` as an operation of relative accuracy `relative` and
        absolute accuracy `absolute` would have returned it.

        The defaults are a correctly rounded f32 op. A product with a constant
        the compiler had to round first carries two roundings and no absolute
        term; a hardware transcendental has its own method below.
        """
        relative_term = torch.zeros_like(value, requires_grad=True)
        self._perturbations.append((relative_term, relative))
        value = value * (1 + relative_term)
        # An operation with no absolute error contributes `0 * |gradient|`, so
        # the term is left out rather than carried: at 2^22 float64 inputs it
        # is tens of megabytes of autograd state per rounding that cannot
        # change the answer.
        if absolute:
            absolute_term = torch.zeros_like(value, requires_grad=True)
            self._perturbations.append((absolute_term, absolute))
            value = value + absolute_term
        return value

    def transcendental(self, value: torch.Tensor) -> torch.Tensor:
        """`value` as one of the 1-ulp, flush-to-zero hardware transcendentals
        would have returned it."""
        return self(value, TRANSCENDENTAL_RELATIVE, TRANSCENDENTAL_FLUSH)

    def bound(self, result: torch.Tensor) -> torch.Tensor:
        """How far `result`, built through this instance, may be from the same
        formula evaluated exactly."""
        assert self._perturbations, "the formula went through no rounding"
        # See the class docstring: summing to a scalar gives a per-element
        # derivative only for an elementwise formula, and every perturbation
        # keeping the result's shape is what says the formula was one.
        assert all(term.shape == result.shape for term, _ in self._perturbations), (
            "elementwise formulas only: a rounding whose shape differs from "
            "the result's means the formula reduced across elements"
        )
        gradients = torch.autograd.grad(
            result.sum(), [term for term, _ in self._perturbations]
        )
        total = torch.zeros_like(gradients[0])
        for (_, size), gradient in zip(self._perturbations, gradients):
            total = total + size * gradient.abs()
        return total


def _block_max_products(a: torch.Tensor, b: torch.Tensor, block: int) -> torch.Tensor:
    """Sum over the K blocks of `max_k |a_k b_k|`, for `a` [..., M, K] times
    `b` [..., N, K] transposed: the scale of what an FP8 MFMA drops, since it
    sums a block at the precision of the block's largest product."""
    a, b = a.double().abs(), b.double().abs()
    total = torch.zeros(*a.shape[:-1], b.shape[-2], dtype=R.F64, device=a.device)
    for k in range(0, a.shape[-1], block):
        a_block, b_block = a[..., k : k + block], b[..., k : k + block]
        total = total + (a_block.unsqueeze(-2) * b_block.unsqueeze(-3)).amax(-1)
    return total


def mfma_accumulation_bound(
    a: torch.Tensor,
    b: torch.Tensor,
    acc: torch.Tensor,
    k: int,
    block: int | None = None,
) -> torch.Tensor:
    """How far `D = C + A B^T` may be from the float64 product, for an MFMA
    accumulating `k` products per output in fp32:
    `(k + 1) 2^-24 (|C| + |A| |B|^T)` -- one rounding per accumulate, plus the
    one that put C into the accumulator.

    `2^-24` is the unit roundoff, so each accumulate is assumed to round to
    nearest and the products themselves to be exact: an fp8 or bf16 pair
    multiplies into a product every f32 holds exactly, so the only rounding in
    the chain is the add. A part that truncated instead would need `2^-23`
    here, and the term is linear in `k` rather than `sqrt(k)` because the
    worst case is taken, not a random walk.

    With `block` -- an FP8 MFMA, which sums each K block of `block` products
    at the precision of the largest of them rather than at fp32 -- that is
    joined by `block 2^-14 sum_blocks max_k |a_k b_k|`, where the 2^-14 is an
    envelope on what the block sum drops rather than a documented figure (see
    `_FP8_BLOCK_PRECISION`). It is the larger of the two terms by far, so a
    test using it is checking the instruction is wired up right, not tracking
    a 1-ulp drift.

    `a` [..., M, K] and `b` [..., N, K] are float64 and already dequantized,
    `acc` is [..., M, N]. No operand of the instruction is looked at, only
    what the instruction was asked to compute.
    """
    # `k` is how many products reach one output, which for every MFMA here is
    # the K the operands carry. Taken as an argument because a caller has to
    # state it -- asserted because a wrong one silently rescales a tolerance
    # instead of failing.
    assert k == a.shape[-1], f"{k} accumulations over K={a.shape[-1]}"
    bound = (k + 1) * F32_ROUNDOFF * (acc.abs() + a.abs() @ b.abs().transpose(-1, -2))
    if block is not None:
        bound = bound + block * _FP8_BLOCK_PRECISION * _block_max_products(a, b, block)
    return bound
