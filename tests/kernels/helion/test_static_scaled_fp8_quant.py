# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the dynamic_per_token_scaled_fp8_quant helion kernel

Run `pytest tests/kernels/helion/test_static_scaled_fp8_quant.py`.
"""

import pytest
import torch

from tests.kernels.quant_utils import FP8_DTYPE
from vllm.kernels.helion.static_scaled_fp8_quant import (
    static_scaled_fp8_quant,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    scaled_quantize,
)
from vllm.platforms import current_platform
from vllm.utils.import_utils import has_helion
from vllm.utils.torch_utils import set_random_seed

if not has_helion():
    pytest.skip(
        "Helion is not installed. Install with: pip install vllm[helion]",
        allow_module_level=True,
    )

DTYPES = [torch.bfloat16, torch.float]
HIDDEN_SIZES = [17, 1024, 1025, 1026, 5137, 8193]
NUM_TOKENS = [1, 7, 4096]
SEEDS = [0]

# Test static FP8 quantization with 2D group scales
GROUP_SHAPES_2D = [
    (-1, -1),  # Per-tensor
    (-1, 1),  # Per-channel
    (1, -1),  # Per-token
    (-1, 128),  # Per-head quantization
    (1, 128),  # DeepSeek-style per-token-per-group (group_m=1, group_n=128)
    (128, 128),  # DeepSeek-style block quantization
    (1, 64),  # Smaller group size
    (1, 16),  # Small group (scalar path in kernel)
    (4, 256),  # Non-trivial both dimensions
]
# Use sizes divisible by all group shapes
NUM_TOKENS_GROUP = [128, 512]
HIDDEN_SIZES_GROUP = [256, 1024, 2048]


@pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA not available")
@pytest.mark.parametrize("num_tokens", NUM_TOKENS_GROUP)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES_GROUP)
@pytest.mark.parametrize("group_shape", GROUP_SHAPES_2D)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
def test_static_fp8_quant_group_2d(
    num_tokens: int,
    hidden_size: int,
    group_shape: tuple[int, int],
    dtype: torch.dtype,
    seed: int,
) -> None:
    """Test static FP8 quantization with 2D group scales using scaled_quantize."""
    # Normalize group_shape (-1 means full extent)
    norm_group_m = num_tokens if group_shape[0] == -1 else group_shape[0]
    norm_group_n = hidden_size if group_shape[1] == -1 else group_shape[1]

    # Skip if sizes are not divisible by group shape
    if num_tokens % norm_group_m != 0 or hidden_size % norm_group_n != 0:
        pytest.skip(
            f"Skipping: ({num_tokens}, {hidden_size}) not divisible by "
            f"group_shape ({group_shape[0]}, {group_shape[1]})"
        )

    set_random_seed(seed)

    x = torch.rand(num_tokens, hidden_size, dtype=dtype, device="cuda")
    ref_out, scale = scaled_quantize(
        x, group_shape, current_platform.fp8_dtype(), compute_dtype=torch.float32
    )
    ops_out = torch.empty(ref_out.shape, device=ref_out.device, dtype=ref_out.dtype)
    static_scaled_fp8_quant(x, scale, ops_out, group_shape)

    torch.testing.assert_close(ref_out.float(), ops_out.float(), rtol=1.2e-1, atol=1e-5)


@pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA not available")
@pytest.mark.parametrize("num_tokens", NUM_TOKENS_GROUP)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES_GROUP)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("group_shape", [(1, -1), (-1, 1)])  # per-token, per-channel
def test_static_fp8_quant_1d_scale(
    num_tokens: int,
    hidden_size: int,
    dtype: torch.dtype,
    seed: int,
    group_shape: tuple[int, int],
) -> None:
    """Test static FP8 quantization with 1D scale (per-token or per-channel)."""
    set_random_seed(seed)

    x = torch.rand(num_tokens, hidden_size, dtype=dtype, device="cuda")
    ref_out, scale_2d = scaled_quantize(
        x, group_shape, FP8_DTYPE, compute_dtype=torch.float32
    )

    # Flatten scale to 1D for testing 1D scale path
    scale_1d = scale_2d.flatten()
    ops_out = torch.empty(ref_out.shape, device=ref_out.device, dtype=ref_out.dtype)
    static_scaled_fp8_quant(x, scale_1d, ops_out, group_shape)

    torch.testing.assert_close(ref_out.float(), ops_out.float(), rtol=0.12, atol=0.0)
