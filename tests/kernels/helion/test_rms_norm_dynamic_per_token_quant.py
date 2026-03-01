# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the rms_norm_dynamic_per_token_quant helion kernel

Run `pytest tests/kernels/helion/test_rms_norm_dynamic_per_token_quant.py`.
"""

import pytest
import torch

import vllm._custom_ops as ops
from vllm.kernels.helion.rms_norm_dynamic_per_token_quant import (
    rms_norm_dynamic_per_token_quant,
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
QUANT_DTYPES = [torch.int8, current_platform.fp8_dtype()]
VEC_HIDDEN_SIZES = [1024, 1025, 1027, 1029]
# Avoid combinatorial explosion with full Cartesian product
NUM_TOKENS_HIDDEN_SIZES = [
    *[(1, i) for i in [1, 64, *VEC_HIDDEN_SIZES, 5120, 5137]],
    *[(2048, i) for i in [1, 64, *VEC_HIDDEN_SIZES, 5137]],
    *[(4096, i) for i in [1, 64, 5137]],
]

ADD_RESIDUAL = [False, True]
SCALE_UBS = [True, False]
SEEDS = [0]

EPS = 1e-6


@pytest.mark.parametrize("num_tokens, hidden_size", NUM_TOKENS_HIDDEN_SIZES)
@pytest.mark.parametrize("add_residual", ADD_RESIDUAL)
@pytest.mark.parametrize("has_scale_ub", SCALE_UBS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("quant_dtype", QUANT_DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
def test_rms_norm(
    num_tokens: int,
    hidden_size: int,
    add_residual: bool,
    has_scale_ub: bool,
    dtype: torch.dtype,
    quant_dtype: torch.dtype,
    seed: int,
) -> None:
    set_random_seed(seed)

    if has_scale_ub and quant_dtype != current_platform.fp8_dtype():
        # skip
        return

    scale = 1 / (hidden_size)
    x = torch.randn(num_tokens, hidden_size, dtype=dtype, device="cuda") * scale
    weight = torch.normal(
        mean=1.0, std=1.0, size=(hidden_size,), dtype=dtype, device=x.device
    )
    residual = torch.randn_like(x) * scale if add_residual else None
    scale_ub = (
        torch.mean(x).to(dtype=torch.float32, device="cuda") if has_scale_ub else None
    )

    ref_residual = residual.clone() if residual is not None else None
    ref_out, ref_scales = ops.rms_norm_dynamic_per_token_quant(
        x, weight, EPS, quant_dtype, scale_ub, ref_residual
    )

    ops_out = torch.empty(x.shape, device=x.device, dtype=quant_dtype)
    ops_scales = torch.empty((x.shape[0], 1), device=x.device, dtype=torch.float32)
    ops_residual = residual.clone() if residual is not None else None
    rms_norm_dynamic_per_token_quant(
        ops_out, x, weight, ops_scales, EPS, scale_ub, ops_residual
    )

    torch.testing.assert_close(ref_scales, ops_scales)
    # allow 1 ULP difference
    assert (
        ref_out.view(torch.uint8).to(torch.int16)
        - ops_out.view(torch.uint8).to(torch.int16)
    ).abs().max() <= 1

    if add_residual:
        torch.testing.assert_close(ref_residual, ops_residual)
