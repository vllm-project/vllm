# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the NVFP4 Marlin power-of-2 scale factor helper.

The helper must return the same factor as the previous mask-and-gather
implementation while allocating no tensor-sized temporaries.
"""

import pytest
import torch

from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
    _nvfp4_compute_scale_factor,
)

DTYPES = [torch.float16, torch.bfloat16]
SHAPES = [(64,), (128, 256), (8, 64, 128)]
UPPER = 448 * (2**7)


def _reference_scale_factor(
    marlin_scales: torch.Tensor, a_dtype: torch.dtype | None = None
) -> float:
    """The previous implementation, kept verbatim as the oracle."""
    if a_dtype is not None and a_dtype == torch.half:
        return 1.0
    ws_float = marlin_scales.float() * (2**7)
    nonzero_mask = ws_float > 0
    if nonzero_mask.any():
        max_val = ws_float[nonzero_mask].max()
        if max_val < UPPER:
            sf = (UPPER / max_val).log2().floor().exp2()
            return sf.item()
    return 1.0


def _cases(shape: tuple[int, ...], dtype: torch.dtype) -> dict[str, torch.Tensor]:
    gen = torch.Generator().manual_seed(0)
    small = torch.rand(shape, generator=gen) * 0.01
    with_zeros = small.clone()
    with_zeros.flatten()[::3] = 0.0
    with_negatives = small.clone()
    with_negatives.flatten()[::5] *= -1.0
    normalized = torch.rand(shape, generator=gen) * 400.0 + 48.0
    return {
        "small": small.to(dtype),
        "with_zeros": with_zeros.to(dtype),
        "with_negatives": with_negatives.to(dtype),
        "all_zeros": torch.zeros(shape, dtype=dtype),
        "all_negative": (-small - 0.001).to(dtype),
        "already_normalized": normalized.to(dtype),
        "single_large": torch.full(shape, 0.5, dtype=dtype),
    }


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("a_dtype", [None, torch.float16, torch.bfloat16])
def test_scale_factor_matches_reference(
    dtype: torch.dtype, shape: tuple[int, ...], a_dtype: torch.dtype | None
):
    for name, scales in _cases(shape, dtype).items():
        expected = _reference_scale_factor(scales, a_dtype)
        actual = _nvfp4_compute_scale_factor(scales, a_dtype)
        assert actual == expected, name
        assert actual >= 1.0
        assert actual == 2.0 ** round(torch.tensor(actual).log2().item())


def test_scale_factor_rescales_into_range():
    scales = torch.full((32, 64), 2**-10, dtype=torch.bfloat16)
    sf = _nvfp4_compute_scale_factor(scales)
    rescaled = scales.float().max() * (2**7) * sf
    assert 2.0 <= rescaled < UPPER


def test_scale_factor_empty_tensor():
    assert _nvfp4_compute_scale_factor(torch.empty(0, dtype=torch.bfloat16)) == 1.0


def test_scale_factor_rejects_nan():
    scales = torch.rand(16, 16, dtype=torch.bfloat16)
    scales[3, 3] = float("nan")
    with pytest.raises(ValueError, match="NaN"):
        _nvfp4_compute_scale_factor(scales)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_scale_factor_allocates_no_tensor_temporaries():
    device = torch.device("cuda")
    scales = torch.rand((288, 512, 256), dtype=torch.bfloat16, device=device) * 0.01
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    baseline = torch.cuda.memory_allocated(device)
    sf = _nvfp4_compute_scale_factor(scales)
    torch.cuda.synchronize(device)
    transient = torch.cuda.max_memory_allocated(device) - baseline
    assert sf > 1.0
    # The previous implementation needed ~4.5x the tensor size here.
    assert transient < scales.numel() * scales.element_size() // 64
