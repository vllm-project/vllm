# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Correctness tests for the online MXFP4 weight quantization backends.

Each backend (Triton, aiter, FlashInfer, XPU) quantizes a synthetic bf16/fp16
tensor to MXFP4, the result is dequantized back to bf16/fp16, and compared
against the pure-torch reference in `reference_mxfp4.py`.
"""

import pytest
import torch

from vllm.platforms import current_platform

from .reference_mxfp4 import dq_mxfp4_torch, qdq_mxfp4_torch


def _skip_reason_if_unavailable(backend: str, dtype: torch.dtype) -> str | None:
    """Return a skip reason if `backend` cannot run on the current host."""
    if backend == "triton":
        if not (current_platform.is_cuda() or current_platform.is_rocm()):
            return "Triton MXFP4 kernel requires a CUDA or ROCm GPU."
        return None
    if backend == "aiter":
        from vllm._aiter_ops import is_aiter_found_and_supported

        if not is_aiter_found_and_supported():
            return "aiter is not available/supported on this platform."
        if dtype != torch.bfloat16:
            return "aiter's dynamic_mxfp4_quant only supports bfloat16 input."
        return None
    if backend == "flashinfer":
        from vllm.utils.flashinfer import has_flashinfer

        if not (current_platform.is_cuda() and has_flashinfer()):
            return "FlashInfer is not available on this platform."
        return None
    if backend == "xpu":
        if not current_platform.is_xpu():
            return "not on XPU platform."
        return None
    raise ValueError(f"Unknown backend {backend}")


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("backend", ["triton", "aiter", "flashinfer", "xpu"])
def test_mxfp4_quantization_correctness(backend: str, dtype: torch.dtype):
    skip_reason = _skip_reason_if_unavailable(backend, dtype)
    if skip_reason is not None:
        pytest.skip(skip_reason)

    torch.manual_seed(3)

    num_rows = 64
    hidden_size = 32 * 32  # multiple 32-element MXFP4 blocks
    device = current_platform.device_type

    x = (torch.rand(num_rows, hidden_size, dtype=dtype, device=device) - 0.5) * 2
    # Vary the magnitude block-to-block so several scale exponents are
    # exercised, rather than a single one for the whole tensor.
    scalings = [2.3, 0.03, 7.3, 0.1, 0.004, 17.3, 1e4, 1e-4]
    for i in range(hidden_size // 32):
        x[:, i * 32 : (i + 1) * 32] *= scalings[i % len(scalings)]

    if backend == "triton":
        from vllm.model_executor.layers.quantization.utils.mxfp4_utils import (
            downcast_to_mxfp,
        )

        x_fp4, x_scale, _ = downcast_to_mxfp(x, axis=-1)
    elif backend == "aiter":
        from vllm.model_executor.layers.quantization.quark.utils import (
            quark_quantize_weight_to_mxfp4,
        )

        x_fp4, x_scale = quark_quantize_weight_to_mxfp4(x)
    elif backend == "flashinfer":
        # TODO: enable this test with flashinfer
        pytest.skip("flashinfer mxfp4 quantization match to reference is untested")
        # from vllm.utils.flashinfer import flashinfer_mxfp4_quantize

        # x_fp4, x_scale = flashinfer_mxfp4_quantize(x, backend="cute-dsl")
    elif backend == "xpu":
        # TODO: enable this test on XPU
        pytest.skip("xpu mxfp4 quantization match to reference is untested")
        # from vllm.model_executor.layers.quantization.utils.mxfp4_utils import (
        #     xpu_mxfp4_quantize,
        # )

        # x_fp4, x_scale = xpu_mxfp4_quantize(x)
    else:
        raise ValueError(f"Unknown backend {backend}")

    result = dq_mxfp4_torch(x_fp4, x_scale, x.dtype)
    reference = qdq_mxfp4_torch(x, scale_calculation_mode="even")

    assert torch.equal(result, reference)
