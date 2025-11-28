# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

import vllm._custom_ops as ops
from tests.kernels.utils import opcheck
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.platforms import current_platform

DTYPES = [torch.bfloat16, torch.float16]
QUANT_DTYPES = [current_platform.fp8_dtype()]
NUM_TOKENS = [1, 17, 86, 1234, 3045]  # Arbitrary values for testing
HIDDEN_SIZES = [16, 48, 128, 1562, 4096]  # Arbitrary values for testing
SEEDS = [0]
CUDA_DEVICES = [f"cuda:{i}" for i in range(1 if torch.cuda.device_count() == 1 else 2)]


def ref_impl(
    silu_and_mul: SiluAndMul, x: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    silu_and_mul_out = silu_and_mul.forward_native(x)
    out, scales = ops.scaled_fp8_quant(silu_and_mul_out, scale)
    return out


def ops_impl(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    out_shape = (x.shape[0], x.shape[1] // 2)
    out = torch.empty(out_shape, dtype=current_platform.fp8_dtype(), device=x.device)
    torch.ops._C.silu_and_mul_quant(out, x, scale)
    return out


@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("quant_dtype", QUANT_DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@torch.inference_mode()
def test_silu_and_mul(
    num_tokens: int,
    hidden_size: int,
    dtype: torch.dtype,
    quant_dtype: torch.dtype,
    seed: int,
    device: str,
) -> None:
    torch.random.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.set_default_device(device)

    layer = SiluAndMul()

    # Make inputs
    scale = torch.randn((1), device=device, dtype=torch.float32)
    x = torch.randn(num_tokens, hidden_size, dtype=dtype)

    ref_out = ref_impl(layer, x, scale)
    ops_out = ops_impl(x, scale)

    assert ref_out.dtype == quant_dtype
    assert ops_out.dtype == quant_dtype
    assert ref_out.shape == ops_out.shape
    assert torch.allclose(
        ref_out.to(dtype=torch.float32), ops_out.to(dtype=torch.float32)
    )
    opcheck(torch.ops._C.silu_and_mul_quant, (ops_out, x, scale))
