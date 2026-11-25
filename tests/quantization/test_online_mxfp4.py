# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Correctness tests for the online MXFP4 weight quantization backends.

Each backend (Triton, aiter, XPU) quantizes a synthetic bf16/fp16
tensor to MXFP4, the result is dequantized back to bf16/fp16, and compared
against the pure-torch reference in `reference_mxfp4.py`.
"""

import pytest
import torch

from vllm._aiter_ops import is_aiter_found_and_supported, rocm_aiter_ops
from vllm.model_executor.layers.quantization.quark.schemes.quark_ocp_mx import (
    QuarkOCP_MX,
)
from vllm.model_executor.layers.quantization.quark.utils import (
    quark_quantize_weight_to_mxfp4,
)
from vllm.model_executor.layers.quantization.utils.mxfp4_utils import (
    downcast_to_mxfp,
    quant_dequant_mxfp4,
)
from vllm.platforms import current_platform

from .reference_mxfp4 import dq_mxfp4_torch, qdq_mxfp4_torch


def fix_negative_zeros(tensor: torch.Tensor) -> torch.Tensor:
    """Normalize FP4 e2m1 negative-zero codewords (0b1000) to positive zero
    (0b0000) in a packed uint8 tensor (two e2m1 values per byte, low nibble
    first).

    -0.0 and +0.0 dequantize to the same float value, but different MXFP4
    quantization backends disagree on which one they emit for values that
    round to zero magnitude. This normalizes both packed representations so
    tensors from different backends can be compared with `torch.equal`.
    """
    assert tensor.dtype == torch.uint8, (
        f"Expected a torch.uint8 tensor, got {tensor.dtype}"
    )

    low_nibble = tensor & 0x0F
    high_nibble = tensor & 0xF0

    low_nibble = torch.where(
        low_nibble == 0x08, torch.zeros_like(low_nibble), low_nibble
    )
    high_nibble = torch.where(
        high_nibble == 0x80, torch.zeros_like(high_nibble), high_nibble
    )

    return low_nibble | high_nibble


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
    if backend == "xpu":
        if not current_platform.is_xpu():
            return "not on XPU platform."
        return None
    if backend == "quark":
        try:
            import quark.torch.kernel.mx  # noqa: F401
        except ImportError:
            return "amd-quark is not installed."
        return None
    raise ValueError(f"Unknown backend {backend}")


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("backend", ["triton", "aiter", "xpu", "quark"])
def test_mxfp4_quantization_correctness(backend: str, dtype: torch.dtype):
    """Tests that the different implementations of mxfp4_quantize
    in mxfp4_utils.py all match.
    """
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
        x_fp4, x_scale, _ = downcast_to_mxfp(x, axis=-1)
    elif backend == "aiter":
        x_fp4, x_scale = quark_quantize_weight_to_mxfp4(x)
    elif backend == "xpu":
        # TODO: enable this test on XPU
        pytest.skip("xpu mxfp4 quantization match to reference is untested")
        # from vllm.model_executor.layers.quantization.utils.mxfp4_utils import (
        #     xpu_mxfp4_quantize,
        # )

        # x_fp4, x_scale = xpu_mxfp4_quantize(x)
    elif backend == "quark":
        result = quant_dequant_mxfp4(x, scale_calculation_mode="even")
    else:
        raise ValueError(f"Unknown backend {backend}")

    if backend != "quark":
        result = dq_mxfp4_torch(x_fp4, x_scale, x.dtype)
    reference = qdq_mxfp4_torch(x, scale_calculation_mode="even")

    assert torch.equal(result, reference)


def test_quark_ocp_mx_dense_rejects_unaligned_input_partition():
    scheme = object.__new__(QuarkOCP_MX)
    layer = torch.nn.Module()
    layer.prefix = "model.layers.0.mlp.shared_expert.down_proj"

    with pytest.raises(
        ValueError,
        match=(
            "input size per partition of 80, which must be divisible by "
            "the OCP MX group size 32"
        ),
    ):
        scheme.create_weights(
            layer=layer,
            output_partition_sizes=[2560],
            input_size_per_partition=80,
            params_dtype=torch.bfloat16,
            weight_loader=lambda *_: None,
        )


@pytest.mark.skipif(
    not is_aiter_found_and_supported(),
    reason="rocm_aiter_mxfp4_gemm_dynamic_quant is registered only with AITER.",
)
def test_mxfp4_gemm_dynamic_quant_fake_dtype():
    """torch.compile sizes the op's output from its fake; a fake that reports
    another dtype than the real op makes the compiled graph read past it."""
    from torch._subclasses.fake_tensor import FakeTensorMode

    with FakeTensorMode(), torch.device(current_platform.device_type):
        x = torch.empty(16, 64, dtype=torch.bfloat16)
        weight = torch.empty(32, 32, dtype=torch.uint8)
        weight_scale = torch.empty(32, 2, dtype=torch.uint8)
        y = rocm_aiter_ops.mxfp4_gemm_dynamic_quant(
            x, weight, weight_scale, True, torch.bfloat16, hadamard_rotate=True
        )
    assert y.dtype == torch.bfloat16
    assert y.shape == (16, 32)
