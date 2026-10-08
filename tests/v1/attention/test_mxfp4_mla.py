# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Validate the MXFP4 MLA primitives.

Two classes of check:

* **Invariants** the kernel design depends on -- RoundUp never clamps, the low
  nibble holds the lower element index, the row is 272 bytes, the aliasing
  scale view lands on the right bytes. These run on CPU.
* **Agreement with AITER's ``per_1x32_f4_quant``**, the reference implementation
  for OCP MXFP4 E8M0 that the HIP and Triton production paths mirror. Skipped
  when AITER is unavailable.
"""

from __future__ import annotations

import pytest
import torch

from vllm.v1.attention.ops import mxfp4_mla as mx

LATENT = 512
GROUP = 32


def _latent(rows: int = 64, dim: int = LATENT, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    # RMSNormed activations: roughly unit-variance with occasional outliers.
    x = torch.randn(rows, dim, generator=g, dtype=torch.float32)
    x[:, ::97] *= 8.0
    return x


# --------------------------------------------------------------------------
# Format invariants
# --------------------------------------------------------------------------


def test_row_is_272_bytes():
    assert mx.row_bytes(LATENT) == 272
    assert mx.scale_region_offset(LATENT) == 256
    # 4.25 bits/value is the OCP MXFP4 ratio: 17 bytes per 32-value block.
    assert mx.row_bytes(LATENT) * 8 / LATENT == 4.25
    assert mx.row_bytes(LATENT) == (LATENT // GROUP) * 17


def test_ceil_pow2_exponent_is_exact_at_powers_of_two():
    # The reason frexp is used instead of ceil(log2(v)): exact powers of two
    # must not round up to the next exponent.
    v = torch.tensor([2.0**k for k in range(-20, 21)], dtype=torch.float32)
    got = mx.ceil_pow2_exponent(v)
    want = torch.arange(-20, 21, dtype=got.dtype)
    assert torch.equal(got, want)

    # Anything strictly above a power of two must round up to the next
    # exponent. Use nextafter rather than a decimal literal: 2.0000001 is not
    # representable in float32 and rounds back to exactly 2.0.
    base = torch.tensor([1.0, 2.0, 4.0], dtype=torch.float32)
    just_above = torch.nextafter(base, torch.full_like(base, float("inf")))
    assert torch.equal(
        mx.ceil_pow2_exponent(just_above), torch.tensor([1, 2, 3], dtype=got.dtype)
    )


def test_roundup_never_clamps():
    """The whole reason RoundUp replaced floor(log2(absmax)) - 2.

    After dividing by the shared scale the group peak must land in (3, 6], so
    no element ever saturates at the top E2M1 code.
    """
    x = _latent(rows=256, seed=1)
    scales = mx.compute_e8m0_scales(x, GROUP)
    scale = mx.decode_e8m0_scales(scales).unsqueeze(-1)

    grouped = x.reshape(x.shape[0], -1, GROUP)
    peak = (grouped / scale).abs().amax(dim=-1)
    nonzero = grouped.abs().amax(dim=-1) > 0

    assert (peak[nonzero] <= mx.E2M1_MAX + 1e-6).all(), peak[nonzero].max()
    assert (peak[nonzero] > 3.0 - 1e-6).all(), peak[nonzero].min()


def test_floor_formula_would_clamp():
    """Guard the claim that the retired formula clamps ~half of groups."""
    x = _latent(rows=256, seed=2)
    grouped = x.reshape(x.shape[0], -1, GROUP)
    absmax = grouped.abs().amax(dim=-1)

    floor_exp = torch.floor(torch.log2(absmax)) - 2
    peak = absmax / torch.exp2(floor_exp)
    clamped = (peak > mx.E2M1_MAX).float().mean().item()
    assert 0.3 < clamped < 0.7, f"expected ~half clamped, got {clamped:.3f}"


def test_low_nibble_holds_lower_index():
    """The Triton unpack reads the low nibble as the lower latent index."""
    from vllm.v1.attention.ops.ultraquant.reference import _pack_nibbles_last_dim

    codes = torch.tensor([[1, 2, 3, 4]], dtype=torch.uint8)
    assert _pack_nibbles_last_dim(codes).tolist() == [[1 | (2 << 4), 3 | (4 << 4)]]


def test_representable_values_survive_exactly():
    """Values already on the E2M1 grid at scale 1 must round-trip bit-exactly."""
    values = torch.tensor(mx.E2M1_VALUES, dtype=torch.float32)
    x = torch.cat([values, -values]).repeat(2)[:GROUP].reshape(1, GROUP)
    # Force scale == 1 by making the group peak exactly 6.0.
    out = mx.quantize_dequantize(x, GROUP).to(torch.float32)
    assert torch.equal(out, x), (x, out)


def test_signed_zero_matches_the_reference_encoding():
    """Small negatives encode as -0.0 (code 8), which is what AITER emits.

    Both codes dequantize to zero and ``dot_scaled`` cannot distinguish them,
    but preserving the sign is what makes the AITER comparison bit-exact.
    """
    x = torch.zeros(1, GROUP, dtype=torch.float32)
    x[0, 0] = 6.0  # pin the group scale to 1.0
    x[0, 1] = 0.01  # rounds to +0
    x[0, 2] = -0.01  # rounds to -0
    scales = mx.compute_e8m0_scales(x, GROUP)
    codes = mx.quantize_to_e2m1_codes(x, scales, GROUP)

    assert codes[0, 1].item() == 0, "positive underflow should be code 0"
    assert codes[0, 2].item() == 8, "negative underflow should be code 8 (-0.0)"

    # A negative-zero input is not "negative": -0.0 < 0 is False.
    x2 = torch.tensor([[-0.0] * GROUP], dtype=torch.float32)
    assert (
        mx.quantize_to_e2m1_codes(x2, mx.compute_e8m0_scales(x2, GROUP), GROUP) == 0
    ).all()

    # Either encoding dequantizes to zero.
    out = mx.dequantize_e2m1_codes(codes, scales, GROUP, out_dtype=torch.float32)
    assert out[0, 1].item() == 0.0 and out[0, 2].item() == 0.0


def test_zero_group_is_exact_and_deterministic():
    x = torch.zeros(1, GROUP, dtype=torch.float32)
    packed, scales = mx.quantize_pack(x, GROUP)
    assert (packed == 0).all()
    assert scales.item() == mx.E8M0_BIAS  # pinned to 2**0
    assert (mx.unpack_dequantize(packed, scales, GROUP).to(torch.float32) == 0).all()


def test_signs_preserved():
    x = _latent(rows=32, seed=3)
    out = mx.quantize_dequantize(x, GROUP).to(torch.float32)
    nonzero = out != 0
    assert (torch.sign(out[nonzero]) == torch.sign(x[nonzero])).all()


def test_relative_error_is_bounded():
    """Sanity-bound the quantization error, and show per-group scaling earns its keep.

    The absolute number is not a specification -- 4-bit with 8 magnitudes and
    power-of-two scales lands a few percent relative MSE on outlier-bearing
    data, and this fixture plants 8x outliers deliberately. What is worth
    asserting is that it stays in that regime, and that per-group scaling is
    dramatically better than a single scale for the whole tensor, which is the
    reason the row carries 16 inline scale bytes at all.
    """
    x = _latent(rows=512, seed=4)

    out = mx.quantize_dequantize(x, GROUP).to(torch.float32)
    rel_grouped = ((out - x).pow(2).sum() / x.pow(2).sum()).item()
    assert rel_grouped < 0.05, f"relative MSE {rel_grouped:.4f} is out of regime"

    # One scale for the whole tensor: the FP8 path's approach, applied to FP4.
    flat = x.reshape(1, -1)
    per_tensor = mx.quantize_dequantize(flat, flat.shape[-1]).reshape(x.shape)
    rel_tensor = ((per_tensor - x).pow(2).sum() / x.pow(2).sum()).item()

    assert rel_tensor > 4 * rel_grouped, (
        f"per-group {rel_grouped:.4f} vs per-tensor {rel_tensor:.4f}: "
        "per-group scaling should be far better"
    )


# --------------------------------------------------------------------------
# The aliasing scale view -- the mechanism that keeps byte math out of the kernel
# --------------------------------------------------------------------------


def test_scale_view_aliases_the_row_scale_region():
    num_blocks, block_size = 3, 5
    row = mx.row_bytes(LATENT)
    cache = torch.zeros(num_blocks, block_size, row, dtype=torch.uint8)

    view = mx.scale_view(cache, LATENT, GROUP)
    assert view.shape == (num_blocks, block_size, LATENT // GROUP)

    # Writing through the view must land in the row's scale region, and nowhere
    # in its data region.
    view[1, 2, :] = torch.arange(LATENT // GROUP, dtype=torch.uint8)
    assert torch.equal(
        cache[1, 2, 256:], torch.arange(LATENT // GROUP, dtype=torch.uint8)
    )
    assert (cache[1, 2, :256] == 0).all()
    assert (cache[0] == 0).all() and (cache[2] == 0).all()


def test_scale_view_rejects_wrong_stride():
    cache = torch.zeros(2, 4, 512, dtype=torch.uint8)
    with pytest.raises(ValueError, match="row stride"):
        mx.scale_view(cache, LATENT, GROUP)


def test_paged_roundtrip_through_the_row_layout():
    """Write latents into a paged 272-byte-row cache and read them back."""
    num_blocks, block_size = 2, 8
    row = mx.row_bytes(LATENT)
    cache = torch.zeros(num_blocks, block_size, row, dtype=torch.uint8)
    scales_view = mx.scale_view(cache, LATENT, GROUP)

    x = _latent(rows=num_blocks * block_size, seed=5)
    packed, scales = mx.quantize_pack(x, GROUP)
    cache[..., :256] = packed.reshape(num_blocks, block_size, 256)
    scales_view[...] = scales.reshape(num_blocks, block_size, LATENT // GROUP)

    got = mx.unpack_dequantize(
        cache[..., :256].reshape(-1, 256),
        scales_view.reshape(-1, LATENT // GROUP),
        GROUP,
        out_dtype=torch.float32,
    )
    assert torch.equal(got, mx.quantize_dequantize(x, GROUP))


# --------------------------------------------------------------------------
# Agreement with AITER, the MXFP4 reference implementation
# --------------------------------------------------------------------------


def _aiter_quant():
    # Import the parent package first. AITER's ``utility`` subpackage is a
    # namespace package, and importing ``aiter.ops.quant`` directly while
    # ``aiter`` is absent from sys.modules fails in the namespace path
    # recalculation with KeyError: 'aiter'.
    if not torch.cuda.is_available():
        pytest.skip("AITER quantization needs a GPU")
    pytest.importorskip("aiter", reason="AITER not available")
    return pytest.importorskip("aiter.ops.quant", reason="AITER not available")


def _as_bytes(t: torch.Tensor) -> torch.Tensor:
    """Reinterpret a sub-byte/fp8 typed tensor as its raw ``uint8`` storage."""
    return t.contiguous().cpu().view(torch.uint8)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_matches_aiter_per_1x32_f4_quant(seed: int):
    """Our scales and codes must match AITER's reference bit-for-bit.

    AITER's ``per_1x32_f4_quant`` is documented as the torch reference that the
    HIP (``quant_mxfp4_hip``) and Triton (``per_1x32_f4_quant_triton``) paths
    mirror, all sharing the E8M0 block layout. Its default round mode is
    ``RoundUp``, which is what :mod:`mxfp4_mla` implements.
    """
    quant = _aiter_quant()
    x = _latent(rows=64, seed=seed)

    ref_packed, ref_scales = quant.per_1x32_f4_quant(x, pack_dim=-1)
    packed, scales = mx.quantize_pack(x, GROUP)

    # AITER hands back typed tensors -- ``float4_e2m1fn_x2`` for the pairs and
    # ``float8_e8m0fnu`` for the scales. Bitcast, do not convert: ``.to(uint8)``
    # on an e8m0 tensor casts the *value* (0.5 -> 0) rather than exposing the
    # stored biased exponent byte.
    assert torch.equal(scales.cpu(), _as_bytes(ref_scales).reshape(scales.shape)), (
        "E8M0 shared exponents disagree with AITER RoundUp"
    )
    assert torch.equal(packed.cpu(), _as_bytes(ref_packed).reshape(packed.shape)), (
        "packed E2M1 codes disagree with AITER"
    )


def test_matches_aiter_dequantized_values():
    """End-to-end: dequantizing AITER's own output must reproduce ours."""
    quant = _aiter_quant()
    x = _latent(rows=64, seed=7)

    ref_packed, ref_scales = quant.per_1x32_f4_quant(x, pack_dim=-1)
    ours = mx.quantize_dequantize(x, GROUP).to(torch.float32)
    theirs = mx.unpack_dequantize(
        _as_bytes(ref_packed).reshape(x.shape[0], -1),
        _as_bytes(ref_scales).reshape(x.shape[0], -1),
        GROUP,
        out_dtype=torch.float32,
    )
    assert torch.equal(ours, theirs)


def test_aiter_pack_dim_matches_our_row_layout():
    """``pack_dim=-1`` is the layout our row uses, and the one dot_scaled wants.

    AITER documents ``pack_dim=-1`` as the ``tl.dot_scaled`` **LHS** form,
    ``A(M, K) -> fp4=(M, K//2), scale=(M, K//32)``. That is exactly a slot-major
    row: 256 data bytes and 16 scale bytes per 512-wide latent, which is why the
    read kernel can feed the KV tile in as the lhs with no transpose.
    """
    quant = _aiter_quant()
    x = _latent(rows=8, dim=LATENT, seed=8)
    packed, scales = quant.per_1x32_f4_quant(x, pack_dim=-1)
    assert tuple(packed.shape) == (8, LATENT // 2) == (8, 256)
    assert tuple(scales.shape) == (8, LATENT // GROUP) == (8, 16)
