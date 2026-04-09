# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch.nn.functional as F

import vllm._custom_ops as ops
from tests.kernels.utils import opcheck
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    per_token_group_quant_fp8,
)
from vllm.model_executor.layers.quantization.utils.int8_utils import (
    per_token_group_quant_int8,
)
from vllm.platforms import current_platform

DTYPES = [torch.float16, torch.bfloat16]
QUANT_DTYPES = [current_platform.fp8_dtype(), torch.int8]
VEC_HIDDEN_SIZES = [1024, 1025, 1027, 1029]
NUM_TOKENS_HIDDEN_SIZES = [
    *[(1, i) for i in [64, *VEC_HIDDEN_SIZES, 2048, 5120]],
    *[(16, i) for i in [64, *VEC_HIDDEN_SIZES, 5120]],
    *[(128, i) for i in [64, *VEC_HIDDEN_SIZES]],
    *[(512, i) for i in [64, 5120]],
]
SCALE_UBS = [False]
GROUP_SIZES = [64, 128]
IS_SCALE_TRANSPOSED = [False, True]
SEEDS = [0]
CUDA_DEVICES = [i for i in range(1 if torch.accelerator.device_count() == 1 else 2)]


def ref_silu_and_mul_per_block_quant(
    x: torch.Tensor,
    quant_dtype: torch.dtype,
    group_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference implementation: unfused SiLU+Mul then group quantization."""
    hidden = x.shape[-1] // 2
    gate, up = x.split(hidden, dim=-1)
    silu_out = F.silu(gate) * up

    if quant_dtype == current_platform.fp8_dtype():
        return per_token_group_quant_fp8(
            silu_out, group_size=group_size, use_ue8m0=False
        )
    elif quant_dtype == torch.int8:
        return per_token_group_quant_int8(silu_out, group_size=group_size)
    else:
        raise ValueError(f"Unsupported quant_dtype: {quant_dtype}")


@pytest.mark.parametrize("num_tokens, hidden_size", NUM_TOKENS_HIDDEN_SIZES)
@pytest.mark.parametrize("has_scale_ub", SCALE_UBS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("quant_dtype", QUANT_DTYPES)
@pytest.mark.parametrize("group_size", GROUP_SIZES)
@pytest.mark.parametrize("is_scale_transposed", IS_SCALE_TRANSPOSED)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device_idx", CUDA_DEVICES)
@torch.inference_mode()
def test_silu_and_mul_per_block_quant(
    default_vllm_config,
    num_tokens: int,
    hidden_size: int,
    has_scale_ub: bool,
    dtype: torch.dtype,
    quant_dtype: torch.dtype,
    group_size: int,
    is_scale_transposed: bool,
    seed: int,
    device_idx: str,
) -> None:
    """Test SiLU+Mul+Block Quantization kernel correctness."""
    torch.accelerator.set_device_index(device_idx)
    device = f"cuda:{device_idx}"
    torch.random.manual_seed(seed)
    torch.set_default_device(device)

    if hidden_size % group_size != 0:
        return

    if has_scale_ub:
        pytest.skip("Scale upper bound not yet supported")

    scale = 1 / hidden_size
    x = torch.randn(num_tokens, hidden_size * 2, dtype=dtype, device=device) * scale

    # Reference implementation
    ref_out, ref_scales = ref_silu_and_mul_per_block_quant(x, quant_dtype, group_size)

    # Fused kernel implementation
    ops_out, ops_scales = ops.silu_and_mul_per_block_quant(
        x, group_size, quant_dtype, None, is_scale_transposed
    )

    # Check for NaN/Inf
    assert not torch.isnan(ops_out.float()).any(), "Kernel output contains NaN"
    assert not torch.isinf(ops_out.float()).any(), "Kernel output contains Inf"
    assert not torch.isnan(ops_scales).any(), "Kernel scales contain NaN"
    assert not torch.isinf(ops_scales).any(), "Kernel scales contain Inf"

    # Check dtypes
    assert ref_out.dtype == quant_dtype
    assert ops_out.dtype == quant_dtype

    # Check scales match
    torch.testing.assert_close(ref_scales, ops_scales, rtol=1e-5, atol=1e-5)

    # Check output correctness via dequantized values
    ref_scales_expanded = ref_scales.repeat_interleave(group_size, dim=1)
    ops_scales_expanded = ops_scales.repeat_interleave(group_size, dim=1)
    ref_deq = ref_out.to(dtype=torch.float32) * ref_scales_expanded
    ops_deq = ops_out.to(dtype=torch.float32) * ops_scales_expanded
    torch.testing.assert_close(ref_deq, ops_deq, atol=5e-2, rtol=5e-2)

    # opcheck
    output = torch.empty(num_tokens, hidden_size, device=device, dtype=quant_dtype)
    num_groups = hidden_size // group_size
    if is_scale_transposed:
        scales = torch.empty(num_groups, num_tokens, device=device, dtype=torch.float32)
    else:
        scales = torch.empty(num_tokens, num_groups, device=device, dtype=torch.float32)
    opcheck(
        torch.ops._C.silu_and_mul_per_block_quant,
        (output, x, scales, group_size, None, is_scale_transposed),
    )


@pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.parametrize("hidden_size", [4096])
@pytest.mark.parametrize("num_tokens", [128])
@pytest.mark.parametrize("group_size", [128])
def test_silu_block_quant_shapes(
    default_vllm_config,
    dtype: torch.dtype,
    hidden_size: int,
    num_tokens: int,
    group_size: int,
):
    """Test that output shapes are correct."""
    torch.set_default_device("cuda")
    x = torch.randn(num_tokens, hidden_size * 2, dtype=dtype, device="cuda")

    # Row-major scales
    out, scales = ops.silu_and_mul_per_block_quant(
        x,
        group_size=group_size,
        quant_dtype=current_platform.fp8_dtype(),
        is_scale_transposed=False,
    )
    assert out.shape == (num_tokens, hidden_size)
    assert scales.shape == (num_tokens, hidden_size // group_size)

    # Column-major scales (logical shape same after .t() in _custom_ops)
    out, scales = ops.silu_and_mul_per_block_quant(
        x,
        group_size=group_size,
        quant_dtype=current_platform.fp8_dtype(),
        is_scale_transposed=True,
    )
    assert out.shape == (num_tokens, hidden_size)
    assert scales.shape == (num_tokens, hidden_size // group_size)


@pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.parametrize("batch_size", [1, 16, 256])
@pytest.mark.parametrize("hidden_size", [1024, 5120, 14336])
def test_silu_block_quant_edge_cases(
    default_vllm_config, dtype: torch.dtype, batch_size: int, hidden_size: int
):
    """Test edge cases: single token, large batch, large hidden size."""
    torch.set_default_device("cuda")
    x = torch.randn(batch_size, hidden_size * 2, dtype=dtype, device="cuda")

    out, scales = ops.silu_and_mul_per_block_quant(
        x,
        group_size=128,
        quant_dtype=current_platform.fp8_dtype(),
        is_scale_transposed=False,
    )

    assert out.shape == (batch_size, hidden_size)
    assert out.dtype == current_platform.fp8_dtype()
    assert scales.dtype == torch.float32
    assert not torch.isnan(out.float()).any()
    assert not torch.isnan(scales).any()
    assert not torch.isinf(scales).any()
