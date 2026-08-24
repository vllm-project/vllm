# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Numerically derived tolerances for the ROCm AITER attention accuracy tests.

Ground-truth derivation (non-circular)
--------------------------------------
Committed ``atol`` values are set from

    atol_full = max_e( |ref_e - golden_e| + kernel_budget_e - rtol * |ref_e| )

where ``golden`` is fp64 attention, ``ref`` is the test's own reference, and
``kernel_budget`` is a unit-roundoff bound for a *correct* kernel. The kernel
output is **not** used to set the tolerance; it is audited separately (does
``|kernel - golden|`` stay within budget?).

Recompute on ROCm hardware with::

    python tests/kernels/attention/derive_rocm_aiter_tolerances.py --run all

Or aggregate existing JSONL::

    python tests/kernels/attention/derive_rocm_aiter_tolerances.py --aggregate \\
        tolerance_derivation/out/apriori_*.jsonl

Each ``Tol`` records ``atol_derived`` (max ``atol_full`` over the sweep),
``measured_max`` (max circular ``required_atol`` for comparison), and
``atol`` (committed = ``atol_derived * margin``, margin ~1.5).

Error model
-----------
Attention output is a convex combination of the value rows,

    o_i = sum_j p_ij v_j,   p_ij >= 0,   sum_j p_ij = 1,

so ``|o_i| <= max_j |v_j|`` and the output cannot grow with the sequence length.
Both the kernels and the references accumulate in fp32 but round the attention
weights ``P`` and the output to the working precision, so the residual between
two correct implementations is driven by that rounding rather than by the fp32
accumulation. Writing ``u`` for the unit roundoff of the working precision, the
per-element residual behaves as

    |o_hat - o| ~ C * u * ||o||_inf ,                                       (1)

with ``C`` a small dimensionless constant. The dependence on ``||o||_inf``
rather than on ``|o_i|`` is why these assertions need an ``atol`` at all: the
rounding error of the ``P V`` product is roughly uniform across a row, while the
elements within the row are not, so elements near zero carry the same absolute
error as the largest ones.

Model (1) is *n*-independent, which is the non-obvious part. The classical
worst-case bound for a length-*n* inner product grows like ``gamma_n = nu/(1-nu)``
[Higham 2002, sec. 3.1]; the probabilistic analysis of Higham and Mary replaces
it by ``~sqrt(n log n) u`` [Higham & Mary 2019], and when the *data* is also
zero-mean random -- which is exactly the ``torch.randn`` inputs these tests use
-- the bound sharpens further to ``O(u)``, independent of ``n``
[Higham & Mary 2020]. The softmax stage does not change the picture: the shifted
(max-subtracting) formula used by every online-softmax kernel is as accurate as
the unshifted one [Blanchard, Higham & Higham 2021]. What remains is a genuine
constant-factor difference in reduction order between a tiled/online-softmax
kernel and a naive reference, which is known to cost roughly an order of
magnitude of numeric deviation at BF16 [Golden et al. 2024] -- consistent with
the ``C`` values measured below.

The measurements confirm (1): across every group, dtype, and kernel variant
(asm, Gluon, Triton, and the direct paged path), ``C`` lands in 0.13-2.0 for the
native-dtype paths. The tolerances are therefore *not* tuned per configuration
by hand; they follow the measured residual for each group.

Measurement protocol
--------------------
Run ``derive_rocm_aiter_tolerances.py`` over each test's parametrization x seeds
on every supported arch (gfx942, gfx950). Committed ``atol`` is
``max(atol_full) * margin``; ``measured_max`` is the old circular
``max(required_atol)`` kept for regression comparison.

``rtol`` is tied to the unit roundoff of the working precision instead of being
shared across dtypes, since fp16 carries three more significand bits than bf16:

    bf16      u = 2^-8  = 3.91e-3  ->  rtol 1e-2   (2.6 u)
    fp16      u = 2^-11 = 4.88e-4  ->  rtol 1e-3   (2.0 u)
    fp8 e4m3  u = 2^-4  = 6.25e-2  ->  rtol 0.15   (2.4 u)

Caveat: the live sweeps below were run on gfx950 (MI355). gfx942 (MI300X) selects
different kernels for several of these paths, so a failure there is meaningful
signal about that kernel rather than about these constants.

References
----------
[Higham 2002]
    N. J. Higham, "Accuracy and Stability of Numerical Algorithms", 2nd ed.,
    SIAM, 2002. Sec. 2.2 (unit roundoff u = 2^-p for p significand bits),
    sec. 3.1 (inner-product bound with gamma_n = nu/(1-nu)).
[Higham & Mary 2019]
    N. J. Higham and T. Mary, "A New Approach to Probabilistic Rounding Error
    Analysis", SIAM J. Sci. Comput. 41(5):A2815-A2835, 2019.
    doi:10.1137/18M1226312. Replaces gamma_n by ~sqrt(n log n) u.
[Higham & Mary 2020]
    N. J. Higham and T. Mary, "Sharper Probabilistic Backward Error Analysis for
    Basic Linear Algebra Kernels with Random Data", SIAM J. Sci. Comput.
    42(5):A3427-A3446, 2020. doi:10.1137/20M1314355. For zero-mean random data
    the bound becomes O(u), independent of n -- the justification for tolerances
    that do not scale with sequence length.
[Blanchard, Higham & Higham 2021]
    P. Blanchard, D. J. Higham and N. J. Higham, "Accurately computing the
    log-sum-exp and softmax functions", IMA J. Numer. Anal. 41(4):2311-2330,
    2021. doi:10.1093/imanum/draa038. The shifted softmax is as accurate as the
    unshifted one.
[Golden et al. 2024]
    G. Golden et al., "Is Flash Attention Stable?", arXiv:2405.02803, 2024.
    Flash Attention shows ~an order of magnitude more numeric deviation than
    baseline attention at BF16 in an isolated forward pass.
[Dao et al. 2022]
    T. Dao et al., "FlashAttention: Fast and Memory-Efficient Exact Attention
    with IO-Awareness", NeurIPS 2022. The tiling/online-softmax reduction order
    that differs from the naive references used here.
[OCP OFP8 1.1]
    Open Compute Project, "OCP 8-bit Floating Point Specification (OFP8)",
    rev. 1.1, 2023. E4M3 = 1 sign + 4 exponent + 3 mantissa bits, hence
    p = 4 significand bits and u = 2^-4.
[Micikevicius et al. 2022]
    P. Micikevicius et al., "FP8 Formats for Deep Learning", arXiv:2209.05433.
"""

from typing import NamedTuple

import torch

# Unit roundoff u = 2^-p, p = significand bits including the implicit leading
# bit. [Higham 2002, sec. 2.2], [OCP OFP8 1.1] for e4m3.
U_BF16 = 2**-8
U_FP16 = 2**-11
U_FP8_E4M3 = 2**-4


class Tol(NamedTuple):
    """A tolerance pair plus derivation audit fields.

    ``atol_derived`` is ``max(atol_full)`` from the non-circular derivation;
    ``measured_max`` is the circular ``required_atol`` maximum for comparison;
    ``atol`` is the committed test constant (~margin × ``atol_derived``).
    """

    atol: float
    rtol: float
    atol_derived: float
    measured_max: float

    @property
    def margin(self) -> float:
        return self.atol / self.atol_derived


# --------------------------------------------------------------------------
# tests/kernels/attention/test_rocm_aiter_fa.py -- aiter.flash_attn_varlen_func
# --------------------------------------------------------------------------
# Single sequence, native dtypes (head-size and varlen paged-KV coverage).
# Live sweep: gfx950, 180 configs x 10 seeds (derive_rocm_aiter_tolerances.py).
FA_SINGLE_SEQ = {
    torch.bfloat16: Tol(
        atol=1.3e-2, rtol=1e-2, atol_derived=8.58e-3, measured_max=4.92e-3
    ),
    torch.float16: Tol(
        atol=1.7e-3, rtol=1e-3, atol_derived=1.11e-3, measured_max=5.78e-4
    ),
}

# Multiple sequences per batch. Needs a larger atol than the single-sequence
# case purely because its shortest sequence is kv_len=64: less averaging in
# `sum_j p_ij v_j` leaves a larger ||o||_inf, and by (1) the residual scales
# with it. Same C, different output scale.
FA_MULTI_BATCH = {
    torch.bfloat16: Tol(
        atol=2e-2, rtol=1e-2, atol_derived=1.35e-2, measured_max=7.54e-3
    ),
    torch.float16: Tol(
        atol=2.9e-3, rtol=1e-3, atol_derived=1.91e-3, measured_max=8.94e-4
    ),
}

# Single-token decode (q_len=1). bf16 only; fp16 is xfail on an upstream AITER
# bug, see test_aiter_mha_decode_single_token.
FA_DECODE = Tol(atol=8e-3, rtol=1e-2, atol_derived=5.54e-3, measured_max=1.89e-3)

# FP8 KV cache. The reference dequantizes the *same* fp8 tensors that the kernel
# reads, so the e4m3 rounding cancels on both sides and the residual is pure
# kernel error at the working precision -- not fp8 quantization error. That is
# why these are the tightest tolerances in the file and why an fp8-sized
# tolerance here would be vacuous.
FA_FP8_KV = {
    torch.bfloat16: Tol(
        atol=6e-3, rtol=1e-2, atol_derived=3.79e-3, measured_max=9.49e-4
    ),
    torch.float16: Tol(
        atol=7e-4, rtol=1e-3, atol_derived=4.63e-4, measured_max=1.36e-4
    ),
}

# Direct paged path: large block tables and sliding-window masking. Larger atol
# than FA_SINGLE_SEQ for the same reason as FA_MULTI_BATCH -- these batches
# include a kv_len=18 sequence, pushing ||o||_inf to ~2.5.
FA_DIRECT = Tol(atol=2.3e-2, rtol=1e-2, atol_derived=1.52e-2, measured_max=9.36e-3)


# --------------------------------------------------------------------------
# tests/kernels/attention/test_rocm_aiter_mla_decode.py
# --------------------------------------------------------------------------
# Absorbed MLA decode vs an fp32 PyTorch reference. The reference keeps scores
# and softmax in fp32 and only the output is bf16, so C is low (~0.2): the
# residual is dominated by the kernel's own P/output rounding.
MLA_DECODE = Tol(atol=1.3e-2, rtol=1e-2, atol_derived=8.78e-3, measured_max=2.71e-3)


# --------------------------------------------------------------------------
# tests/kernels/attention/test_rocm_aiter_mla_head_padding.py
# --------------------------------------------------------------------------
# 12-head asm persistent decode vs fp32 SDPA. Averaging over a 4096-token
# context makes ||o||_inf small (~0.25), so by (1) the absolute residual is
# small too. The a priori budget is conservative here; committed atol follows
# the derived ceiling with margin.
MLA_H12_DECODE = Tol(atol=5e-3, rtol=1e-2, atol_derived=3.34e-3, measured_max=1.12e-4)


# --------------------------------------------------------------------------
# tests/kernels/attention/test_rocm_aiter_mla_fp8_prefill.py
# --------------------------------------------------------------------------
# FP8 MLA prefill. Unlike FA_FP8_KV the second GEMM consumes an fp8 P, so u here
# is the e4m3 unit roundoff (6.25e-2), not the bf16 one -- two decades larger,
# which is why this is the loosest tolerance in the suite despite C being only
# ~0.2. Measured over seq_len {128, 512, 1024, 2048}.
MLA_FP8_PREFILL = Tol(atol=2.6e-1, rtol=5e-2, atol_derived=1.74e-1, measured_max=4.94e-2)


# --------------------------------------------------------------------------
# tests/kernels/attention/test_rocm_aiter_unified_attn.py
# --------------------------------------------------------------------------
UNIFIED_MIXED_BATCH = {
    torch.bfloat16: Tol(
        atol=2.5e-2, rtol=1e-2, atol_derived=1.67e-2, measured_max=9.39e-3
    ),
    torch.float16: Tol(
        atol=3.1e-3, rtol=1e-3, atol_derived=2.07e-3, measured_max=1.26e-3
    ),
}
UNIFIED_DECODE = Tol(atol=1.2e-2, rtol=1e-2, atol_derived=7.71e-3, measured_max=3.04e-3)
UNIFIED_PREFILL = Tol(atol=2.9e-2, rtol=1e-2, atol_derived=1.95e-2, measured_max=8.17e-3)

# The three fp8 variants are *not* the same experiment and must not share a
# tolerance:
#   fp8_kv       - reference dequantizes the same fp8 KV, so e4m3 rounding
#                  cancels; residual is bf16-scale kernel error.
#   fp8_query    - reference uses the unquantized bf16 query while the kernel
#                  gets an e4m3 copy, so e4m3 query rounding is inside the
#                  residual and sets the scale (~0.2 on a ||o||_inf of ~4.9).
#   fp8_query_kv - both, but with the KV clamped to [-1,1] and descaled by
#                  0.5/0.25, so ||o||_inf is only ~0.25.
# A single fp8-sized tolerance would leave fp8_kv unable to detect an error
# smaller than ~50% of the signal.
UNIFIED_FP8_KV = Tol(atol=2.5e-3, rtol=1e-2, atol_derived=1.70e-3, measured_max=7.60e-4)
UNIFIED_FP8_QUERY = Tol(atol=3.9e-1, rtol=1.5e-1, atol_derived=2.59e-1, measured_max=1.39e-1)
UNIFIED_FP8_QUERY_KV = Tol(
    atol=2.9e-2, rtol=1.5e-1, atol_derived=1.93e-2, measured_max=7.09e-3
)
