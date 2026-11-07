# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import math

import pytest
import torch

from vllm import _custom_ops as ops
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed

if not current_platform.has_device_capability(100):
    pytest.skip(
        reason="Flashinfer FP8 gemms requires compute capability of 10.0 or above.",
        allow_module_level=True,
    )

DTYPES = [torch.float16, torch.bfloat16]
# m, n, k
SHAPES = [(128, 128, 64), (128, 128, 128), (256, 128, 64), (128, 256, 128)]
PAD_SHAPES = [(150, 128, 64), (128, 128, 96)]
SHAPES.extend(PAD_SHAPES)

SEEDS = [42]
CUDA_DEVICES = ["cuda:0"]


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("use_bias", [True, False])
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@pytest.mark.parametrize("autotune", [False, True])
def test_flashinfer_fp8_gemm(
    dtype: torch.dtype,
    shape: tuple[int, int, int],
    use_bias: bool,
    seed: int,
    device: str,
    autotune: bool,
) -> None:
    _test_flashinfer_fp8_gemm(dtype, shape, use_bias, seed, device, autotune, ())


@pytest.mark.parametrize("batch_shape", [(2,), (1, 2)])
def test_flashinfer_fp8_gemm_output_shape(batch_shape: tuple[int, ...]) -> None:
    """Scaled-MM restores leading dimensions of the requested output shape."""
    _test_flashinfer_fp8_gemm(
        dtype=torch.bfloat16,
        shape=SHAPES[0],
        use_bias=True,
        seed=SEEDS[0],
        device=CUDA_DEVICES[0],
        autotune=False,
        batch_shape=batch_shape,
    )


@torch.inference_mode()
def _test_flashinfer_fp8_gemm(
    dtype: torch.dtype,
    shape: tuple[int, int, int],
    use_bias: bool,
    seed: int,
    device: str,
    autotune: bool,
    batch_shape: tuple[int, ...],
) -> None:
    from vllm.model_executor.kernels.linear.scaled_mm.flashinfer import (
        FlashInferFP8ScaledMMLinearKernel,
    )

    set_random_seed(seed)
    m, n, k = shape
    output_shape = [*batch_shape, m // math.prod(batch_shape), n]
    a = torch.randn((m, k), dtype=dtype, device=device)
    b = torch.randn((n, k), dtype=dtype, device=device) / k

    a_fp8, a_scale = ops.scaled_fp8_quant(a)
    b_fp8, b_scale = ops.scaled_fp8_quant(b)

    expected_out = torch.mm(
        a_scale * a_fp8.to(dtype=torch.float32),
        b_scale * b_fp8.to(dtype=torch.float32).t(),
    ).to(dtype=dtype)

    if use_bias:
        bias = torch.randn((n,), dtype=dtype, device=device)
        expected_out = expected_out + bias
    else:
        bias = None

    import flashinfer

    kernel = object.__new__(FlashInferFP8ScaledMMLinearKernel)
    with flashinfer.autotune(autotune):
        out = kernel.apply_scaled_mm(
            A=a_fp8,
            B=b_fp8.t(),
            As=a_scale,
            Bs=b_scale,
            out_dtype=dtype,
            bias=bias,
            output_shape=output_shape,
        )

    torch.testing.assert_close(
        out, expected_out.view(*output_shape), atol=1e-2, rtol=1e-2
    )
