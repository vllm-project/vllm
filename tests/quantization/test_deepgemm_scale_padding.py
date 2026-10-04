# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression tests for issue #49783.

DeepGEMM 2.6+ added a device-side assertion that every FP32 scale value in
the TMA-aligned scale buffer is an exact power of two
((bits & 0x807fffff) == 0).  vLLM was allocating the TMA-aligned backing
store with torch.empty_strided(), leaving the alignment-padding rows
(indices [m, tma_aligned_m) in each column) uninitialised.  Those garbage
values fail the assertion and cause a CUDA context error.

These tests verify that:
1. The TMA-aligned scale buffers from per_token_group_quant_fp8 have
   every element (including padding) satisfying the power-of-two constraint.
2. The same holds for rms_norm_per_block_quant.
"""

import struct

import pytest
import torch


def _is_power_of_two_fp32(t: torch.Tensor) -> torch.Tensor:
    """Return a boolean tensor: True iff each float is an exact power of 2.

    For a finite, positive float32 value the IEEE bit layout is:
        [sign(1)] [exponent(8)] [mantissa(23)]
    A power of two has mantissa == 0, so (bits & 0x807fffff) == 0.
    We also accept 0.0 (which is 2^{-inf}) as a valid fill value.
    """
    raw = t.view(torch.int32)
    return (raw & 0x807FFFFF) == 0


def _check_all_pow2(t: torch.Tensor, label: str) -> None:
    """Assert that every element of *t* is an exact power of two."""
    flat = t.contiguous().view(-1)
    mask = _is_power_of_two_fp32(flat)
    bad = (~mask).sum().item()
    assert bad == 0, (
        f"{label}: {bad} out of {flat.numel()} values are not powers of two. "
        f"First few bad values: {flat[~mask][:8].tolist()}"
    )


# -------------------------------------------------------------------
# Helper: read back the *full* underlying storage of a strided tensor
# -------------------------------------------------------------------


def _storage_as_float32(t: torch.Tensor) -> torch.Tensor:
    """Return a 1-D float32 tensor covering every byte of t's storage."""
    assert t.dtype == torch.float32
    # storage_offset + numel * stride might miss padding; use storage size
    storage = t.untyped_storage()
    n_floats = storage.nbytes() // 4
    return torch.frombuffer(storage, dtype=torch.float32, count=n_floats)


# ===================================================================
# Tests for per_token_group_quant_fp8  (fp8_utils.py)
# ===================================================================


@pytest.mark.parametrize("m", [1, 2, 3, 5, 7, 8, 15, 16, 17])
@pytest.mark.parametrize("hidden", [128, 256])
def test_per_token_group_fp8_tma_scale_padding(m: int, hidden: int):
    """TMA-aligned scale backing store must be entirely power-of-two values."""
    from vllm.model_executor.layers.quantization.utils.fp8_utils import (
        per_token_group_quant_fp8,
    )

    group_size = 128
    if hidden % group_size != 0:
        pytest.skip(f"hidden={hidden} not divisible by group_size={group_size}")

    x = torch.randn(m, hidden, device="cpu", dtype=torch.bfloat16)
    _xq, x_s = per_token_group_quant_fp8(
        x,
        group_size=group_size,
        column_major_scales=True,
        tma_aligned_scales=True,
    )

    # Check the logical view
    assert x_s.shape == (m, hidden // group_size), f"Unexpected scale shape {x_s.shape}"

    # Check that the full underlying backing store (including padding rows)
    # contains only valid power-of-two FP32 values.
    storage_vals = _storage_as_float32(x_s)
    _check_all_pow2(storage_vals, f"per_token_group_quant_fp8 m={m}")


@pytest.mark.parametrize("batch,m", [(2, 3), (4, 5)])
@pytest.mark.parametrize("hidden", [128, 256])
def test_per_token_group_fp8_tma_scale_padding_3d(batch: int, m: int, hidden: int):
    """Same check for 3-D input (batch, m, hidden)."""
    from vllm.model_executor.layers.quantization.utils.fp8_utils import (
        per_token_group_quant_fp8,
    )

    group_size = 128
    if hidden % group_size != 0:
        pytest.skip(f"hidden={hidden} not divisible by group_size={group_size}")

    x = torch.randn(batch, m, hidden, device="cpu", dtype=torch.bfloat16)
    _xq, x_s = per_token_group_quant_fp8(
        x,
        group_size=group_size,
        column_major_scales=True,
        tma_aligned_scales=True,
    )

    assert x_s.shape == (batch, m, hidden // group_size), (
        f"Unexpected scale shape {x_s.shape}"
    )

    storage_vals = _storage_as_float32(x_s)
    _check_all_pow2(storage_vals, f"per_token_group_quant_fp8 3D batch={batch},m={m}")


# ===================================================================
# Tests for rms_norm_per_block_quant  (_custom_ops.py)
# ===================================================================


@pytest.mark.parametrize("m", [1, 2, 3, 5, 7, 8, 15, 16, 17])
@pytest.mark.parametrize("hidden", [128, 256])
def test_rms_norm_per_block_quant_tma_scale_padding(m: int, hidden: int):
    """TMA-aligned scale backing store from rms_norm_per_block_quant must
    be entirely power-of-two values."""
    from vllm._custom_ops import rms_norm_per_block_quant

    group_size = 128
    if hidden % group_size != 0:
        pytest.skip(f"hidden={hidden} not divisible by group_size={group_size}")

    x = torch.randn(m, hidden, device="cpu", dtype=torch.bfloat16)
    weight = torch.ones(hidden, device="cpu", dtype=torch.bfloat16)

    _out, scales = rms_norm_per_block_quant(
        input=x,
        weight=weight,
        epsilon=1e-5,
        quant_dtype=torch.float8_e4m3fn,
        group_size=[1, group_size],
        is_scale_transposed=True,
        tma_alignment=4,
    )

    # Logical shape should be (m, hidden // group_size)
    assert scales.shape == (m, hidden // group_size), (
        f"Unexpected scale shape {scales.shape}"
    )

    storage_vals = _storage_as_float32(scales)
    _check_all_pow2(storage_vals, f"rms_norm_per_block_quant m={m}")


# ===================================================================
# Unit-level check: verify 1.0 satisfies DeepGEMM power-of-two check
# ===================================================================


def test_fp32_one_is_power_of_two():
    """Sanity check: 1.0 must satisfy (bits & 0x807fffff) == 0."""
    bits = struct.unpack("<I", struct.pack("<f", 1.0))[0]
    assert (bits & 0x807FFFFF) == 0, f"1.0 bits={bits:#010x} fails pow2 check"


def test_fp32_random_is_not_power_of_two():
    """Sanity check: a typical non-pow2 float fails the constraint."""
    # 0.1 is not a power of two
    bits = struct.unpack("<I", struct.pack("<f", 0.1))[0]
    assert (bits & 0x807FFFFF) != 0, "0.1 unexpectedly passes pow2 check"


def test_tma_padding_rows_are_ones():
    """Directly verify that the backing store has 1.0 in padding positions."""
    from vllm.model_executor.layers.quantization.utils.fp8_utils import (
        per_token_group_quant_fp8,
    )

    # m=3 with tma_aligned_m=4: one padding row per column
    m = 3
    group_size = 128
    sf_k = 2  # 256 / 128
    hidden = sf_k * group_size  # 256

    x = torch.randn(m, hidden, device="cpu", dtype=torch.bfloat16)
    _xq, x_s = per_token_group_quant_fp8(
        x,
        group_size=group_size,
        column_major_scales=True,
        tma_aligned_scales=True,
    )

    # tma_aligned_m = 4 for float32 (align to 16//4=4)
    tma_aligned_m = 4
    storage = _storage_as_float32(x_s)
    # storage has sf_k * tma_aligned_m = 2 * 4 = 8 elements
    assert storage.numel() == sf_k * tma_aligned_m, (
        f"Expected {sf_k * tma_aligned_m} storage elements, got {storage.numel()}"
    )

    # Padding positions are indices 3 and 7 (last in each column of 4)
    for col in range(sf_k):
        pad_idx = col * tma_aligned_m + m  # = col*4 + 3
        pad_val = storage[pad_idx].item()
        assert pad_val == 1.0, (
            f"Padding at col={col}, idx={pad_idx}: expected 1.0, got {pad_val}"
        )
