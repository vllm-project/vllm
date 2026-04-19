# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Tests for `convert_rs_fp8x4_e4m3`.

The helper wraps PTX `cvt.rs.satfinite.e4m3x4.f32` (SM_100a+, PTX 9.1),
which does unbiased stochastic rounding + saturating cast from fp32 to
fp8_e4m3fn, 4 elements per asm call.

These tests guard against two specific failure modes we hit during
development:

1. **Byte-order bug**: Triton's `inline_asm_elementwise(pack=4)` is
   little-endian but PTX packs the leftmost source into the high byte.
   Using the natural `{$1, $2, $3, $4}` order silently reverses every
   group of 4 contiguous elements in the packed output — single-prompt
   smoke tests pass but gsm8k 5-shot collapses to ~0 (we observed
   0.0023 strict-match).  See triton-lang/triton#8822.

   The bracket test below (`test_sr_outputs_are_grid_neighbours`)
   catches this: with a reversed order the outputs do not stay within
   the two nearest E4M3 grid points of the inputs.

2. **Biased rounding**: a broken SR that just acts like RN would still
   produce valid outputs on the grid but would fail unbiasedness.  The
   empirical-mean test (`test_sr_is_unbiased`) catches that by
   averaging over many seeds and checking the result approximates the
   true input to within one fp8 ulp.
"""

import pytest
import torch
import triton
import triton.language as tl

from vllm.model_executor.layers.mamba.ops.mamba_ssm import convert_rs_fp8x4_e4m3
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not (
        current_platform.is_cuda()
        and current_platform.is_device_capability_family(100)
    ),
    reason=(
        "PTX cvt.rs.satfinite.e4m3x4.f32 requires compute capability 10.0"
        " (SM_100a+); SM_120 and below are not supported."
    ),
)


E4M3_MAX = 448.0  # torch.finfo(torch.float8_e4m3fn).max


# ---------------------------------------------------------------------------
# Test kernel: wrap the pure-Triton helper so we can call it from Python.
# ---------------------------------------------------------------------------


@triton.jit
def _fp8_sr_cvt_kernel(X, Y, rand_seed, N, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X + offs, mask=mask, other=0.0).to(tl.float32)
    rbits = tl.randint(rand_seed, offs)
    y = convert_rs_fp8x4_e4m3(x, rbits)
    tl.store(Y + offs, y, mask=mask)


def _sr_round_trip(x_fp32: torch.Tensor, seed: int) -> torch.Tensor:
    """Convert `x_fp32` to fp8_e4m3fn via the SR helper and read back as fp32."""
    assert x_fp32.numel() % 4 == 0, "helper uses pack=4, len must be multiple of 4"
    y = torch.empty_like(x_fp32, dtype=torch.float8_e4m3fn)
    block = triton.next_power_of_2(x_fp32.numel())
    _fp8_sr_cvt_kernel[(1,)](x_fp32, y, seed, x_fp32.numel(), BLOCK=block)
    return y.to(torch.float32)


# ---------------------------------------------------------------------------
# E4M3 grid utilities (pure Python, used to compute ground-truth brackets).
# ---------------------------------------------------------------------------


def _e4m3_grid(device: torch.device) -> torch.Tensor:
    """All finite fp8_e4m3fn grid points as fp32, sorted ascending."""
    # E4M3 uses 0x7F / 0xFF for NaN; all other 254 byte patterns are finite.
    bytes_u8 = torch.arange(256, dtype=torch.uint8, device=device)
    fp8 = bytes_u8.view(torch.float8_e4m3fn)
    grid = fp8.to(torch.float32)
    finite = torch.isfinite(grid)
    grid = grid[finite]
    grid, _ = torch.sort(grid)
    return grid.contiguous()


def _bracket(x: torch.Tensor, grid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the two adjacent grid points `(lo, hi)` such that `lo <= x <= hi`.

    Values with `|x| > E4M3_MAX` are clamped to `±E4M3_MAX` on both sides —
    that matches `.satfinite` semantics and is what the PTX instruction
    produces.
    """
    clamped = torch.clamp(x, -E4M3_MAX, E4M3_MAX)
    # searchsorted gives the first index with grid[i] >= clamped.
    hi_idx = torch.searchsorted(grid, clamped, right=False).clamp_max(grid.numel() - 1)
    hi = grid[hi_idx]
    lo_idx = (hi_idx - (hi > clamped).long()).clamp_min(0)
    lo = grid[lo_idx]
    return lo, hi


# ---------------------------------------------------------------------------
# 1. Bracket invariant: every SR output is one of the two E4M3 grid neighbours.
#    This is the test that actually catches the byte-order bug.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [123, 456, 789, 101112])
def test_sr_outputs_are_grid_neighbours(seed):
    device = torch.device("cuda")
    grid = _e4m3_grid(device)

    # Hand-picked off-grid values covering subnormal / small / medium / large
    # magnitudes, plus the ±448 saturation boundary and some on-grid values.
    x = torch.tensor(
        [
            0.0,     0.0,     1.0,     -1.0,       # on-grid sanity
            0.124,   0.126,   0.128,   0.130,      # small-mag bracket flip
            -0.501,  -0.499,  0.260,   -0.260,     # signs
            3.9,     4.1,     4.2,     -4.1,       # medium
            33.7,    -33.7,   63.0,    -63.0,      # coarser grid
            123.4,   -200.5,  420.0,   -447.9,     # near saturation
            500.0,   -500.0,  1e4,     -1e4,       # beyond +/-448: saturates
        ],
        dtype=torch.float32,
        device=device,
    )
    assert x.numel() % 4 == 0

    y = _sr_round_trip(x, seed=seed)
    lo, hi = _bracket(x, grid)

    # Every SR output must be exactly lo or exactly hi for its input.
    on_lo = torch.eq(y, lo)
    on_hi = torch.eq(y, hi)
    ok = on_lo | on_hi
    if not ok.all():
        bad = (~ok).nonzero(as_tuple=True)[0]
        msg_lines = [
            f"SR output escaped the E4M3 bracket for {bad.numel()} / {x.numel()} inputs:",
            "   idx           x         lo         hi         sr",
        ]
        for i in bad.tolist():
            msg_lines.append(
                f"   {i:3d}   {x[i].item():10.6g}"
                f"   {lo[i].item():10.6g}   {hi[i].item():10.6g}"
                f"   {y[i].item():10.6g}"
            )
        pytest.fail("\n".join(msg_lines))


# ---------------------------------------------------------------------------
# 2. Unbiasedness: averaging across many seeds converges to the true input
#    within one fp8 ulp.  This rejects a "lies-on-grid but always picks the
#    nearest" (i.e. RN-masquerading-as-SR) implementation.
# ---------------------------------------------------------------------------


def test_sr_is_unbiased():
    device = torch.device("cuda")
    n_samples = 512        # elements per SR call (pack=4 → must be multiple of 4)
    n_seeds = 256          # averaging budget
    # Pick values inside the representable range where the local ulp is
    # moderate; avoid saturation so the bias of `satfinite` clamping doesn't
    # skew the empirical mean.
    torch.manual_seed(0)
    x = (torch.rand(n_samples, device=device) * 200.0 - 100.0).to(torch.float32)
    # Ensure no value is so tiny it maps to the same grid point deterministically.
    x = torch.where(x.abs() < 0.01, x + 0.5, x)

    running = torch.zeros_like(x)
    for seed in range(n_seeds):
        running += _sr_round_trip(x, seed=seed)
    mean = running / n_seeds

    # Per-element tolerance = one fp8 ulp at that magnitude.
    grid = _e4m3_grid(device)
    lo, hi = _bracket(x, grid)
    ulp = (hi - lo).clamp_min(torch.finfo(torch.float32).tiny)

    # The sample mean of n_seeds draws of a Bernoulli-style distribution
    # concentrates around the true mean with stderr ~ ulp / (2 sqrt(n_seeds)).
    # We allow 0.5 ulp absolute slack, which is >> 3 stderr for n_seeds=256.
    err = (mean - x).abs()
    tol = 0.5 * ulp
    assert torch.all(err <= tol), (
        f"SR mean across {n_seeds} seeds not unbiased within 0.5 ulp.\n"
        f"max |mean - x| = {err.max().item():.4g},  "
        f"corresponding ulp = {ulp[err.argmax()].item():.4g}"
    )


# ---------------------------------------------------------------------------
# 3. On-grid stability: inputs that are already exact fp8 grid points must
#    round back to themselves deterministically for every seed.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [1, 42, 2026])
def test_sr_is_identity_on_grid(seed):
    device = torch.device("cuda")
    grid = _e4m3_grid(device)
    # Trim to a multiple of 4 for pack=4.
    n = (grid.numel() // 4) * 4
    x = grid[:n].clone()

    y = _sr_round_trip(x, seed=seed)
    diff = (y - x).abs()
    assert torch.all(diff == 0), (
        f"SR modified on-grid input (max diff = {diff.max().item():.4g}). "
        "An on-grid value has no adjacent neighbour inside the bracket and "
        "must round to itself every time."
    )
