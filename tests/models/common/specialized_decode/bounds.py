# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Error bounds for helpers that cannot be bit-exact.

The bounds are derived from the ISA's documented accuracy and from the
arithmetic each helper performs, not measured from the kernel they check.
They are shared so that every model's tests are bounded the same way.
The one exception is `_FP8_BLOCK_PRECISION`, an empirical envelope.

ISA references:
- CDNA3: https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-mi300-cdna3-instruction-set-architecture.pdf
- CDNA4: https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-cdna4-instruction-set-architecture.pdf
"""

import torch

from tests.models.common.specialized_decode import reference as R

# Unit roundoff of float32 (half an ulp at 1): the relative error of one
# round-to-nearest operation.
F32_ROUNDOFF = 2.0**-24

# The ISA documents V_EXP_F32, V_LOG_F32, V_RSQ_F32 and V_RCP_F32 as accurate
# to 1 ulp, with denormal results flushed to zero.
TRANSCENDENTAL_RELATIVE = 2 * F32_ROUNDOFF
TRANSCENDENTAL_FLUSH = R.FLT_MIN


def transcendental_bound(reference: torch.Tensor) -> torch.Tensor:
    """Allowed error of a hardware transcendental: 1 ulp of `reference`, or
    the whole result where it may be flushed to zero.

    The flush region starts just above `FLT_MIN`, because a result there may
    round down to a denormal and then be flushed.
    """
    return torch.where(
        reference.abs() < TRANSCENDENTAL_FLUSH * (1 + TRANSCENDENTAL_RELATIVE),
        TRANSCENDENTAL_FLUSH,
        R.ulp(reference, torch.float32),
    )


# Empirical, not derived: the gfx950 FP8 MFMA sums each K block at the
# precision of the block's largest product, and the ISA does not say how much
# that drops. The measured error of `test_mfma_scaled` is 0.20-0.24 of the
# resulting bound.
_FP8_BLOCK_PRECISION = 2.0**-14


class RoundingModel:
    """First-order rounding error of a float32 formula, computed by autograd.

    Evaluate the formula in float64 and wrap each float32 operation in a call
    to the instance, which models it as `value * (1 + relative) + absolute`.
    `bound` then returns `sum_i |d result / d e_i| * |e_i|` over those error
    terms. Second-order terms are ignored.

    Only elementwise formulas are supported, since the gradient is taken of
    `result.sum()`.
    """

    def __init__(self) -> None:
        # (error term, its maximum magnitude)
        self._perturbations: list[tuple[torch.Tensor, float]] = []

    def __call__(
        self,
        value: torch.Tensor,
        relative: float = F32_ROUNDOFF,
        absolute: float = 0.0,
    ) -> torch.Tensor:
        """Model one operation with the given relative and absolute error.
        The defaults are a correctly rounded float32 operation."""
        relative_term = torch.zeros_like(value, requires_grad=True)
        self._perturbations.append((relative_term, relative))
        value = value * (1 + relative_term)
        # Skipped when zero to save autograd memory on large inputs.
        if absolute:
            absolute_term = torch.zeros_like(value, requires_grad=True)
            self._perturbations.append((absolute_term, absolute))
            value = value + absolute_term
        return value

    def transcendental(self, value: torch.Tensor) -> torch.Tensor:
        """Model a 1-ulp, flush-to-zero hardware transcendental."""
        return self(value, TRANSCENDENTAL_RELATIVE, TRANSCENDENTAL_FLUSH)

    def bound(self, result: torch.Tensor) -> torch.Tensor:
        """Maximum error of `result` from the exact formula."""
        assert self._perturbations, "the formula went through no rounding"
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
    """Sum over K blocks of `max_k |a_k b_k|`, for `a` `[..., M, K]` and
    `b` `[..., N, K]`."""
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
    block: int | None = None,
) -> torch.Tensor:
    """Worst-case error of an MFMA computing `D = C + A B^T` with fp32
    accumulation over K: `(K + 1) 2^-24 (|C| + |A| |B|^T)`.

    Products of bf16 or fp8 operands are exact in fp32, so only the adds
    round. For FP8 MFMAs, pass `block` to add the block-sum term
    `block * _FP8_BLOCK_PRECISION * sum_blocks max_k |a_k b_k|`.

    Args:
        a: Dequantized float64 operand, `[..., M, K]`.
        b: Dequantized float64 operand, `[..., N, K]`.
        acc: Accumulator input C, `[..., M, N]`.
        block: FP8 K-block size, or None for an fp32 accumulation.

    """
    bound = (
        (a.shape[-1] + 1)
        * F32_ROUNDOFF
        * (acc.abs() + a.abs() @ b.abs().transpose(-1, -2))
    )
    if block is not None:
        bound = bound + block * _FP8_BLOCK_PRECISION * _block_max_products(a, b, block)
    return bound
