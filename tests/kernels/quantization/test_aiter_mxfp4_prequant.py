# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""AITER MXFP4 linear kernel consumes a pre-quantized activation.

A pre-quantized (fp4, e8m0) pair fed through AiterMxfp4LinearKernel.apply_weights
must be bit-identical to letting the kernel quantize the same BF16 input. This
is the ROCm/MXFP4 half of the QuantizedActivation contract.
"""

import importlib.util

import pytest
import torch
from torch.nn.parameter import Parameter

from vllm.platforms import current_platform

aiter_available = importlib.util.find_spec("aiter") is not None

pytestmark = pytest.mark.skipif(
    not (current_platform.is_rocm() and aiter_available),
    reason="AITER MXFP4 prequant is only available on ROCm with aiter",
)

# Kimi-K3 TP8 o_proj plus a small square so the skip is not one-shape-only.
K3_N, K3_K = 7168, 1536
SMALL_N, SMALL_K = 256, 256
MS = (1, 3, 8, 16, 17, 32, 64, 128)


def _kernel_and_layer(n: int, k: int, device: str = "cuda"):
    from aiter.ops.triton.quant import dynamic_mxfp4_quant

    from vllm.model_executor.kernels.linear.mxfp4.aiter import AiterMxfp4LinearKernel
    from vllm.model_executor.kernels.linear.mxfp4.base import MxFp4LinearLayerConfig
    from vllm.model_executor.layers.quantization.utils.quant_utils import kMxfp4Dynamic

    torch.set_default_dtype(torch.bfloat16)
    kernel = AiterMxfp4LinearKernel(
        MxFp4LinearLayerConfig(activation_quant_key=kMxfp4Dynamic)
    )
    if kernel.use_asm_gemm:
        pytest.skip("ASM gemm_a4w4 declines prequant (scale layout is not in QuantKey)")

    weight = (torch.randn(n, k, device=device, dtype=torch.float32) * 0.02).to(
        torch.bfloat16
    )
    w_q, w_s = dynamic_mxfp4_quant(weight.contiguous())
    layer = torch.nn.Module()
    layer.weight = Parameter(w_q, requires_grad=False)
    layer.weight_scale = Parameter(w_s, requires_grad=False)
    kernel.process_weights_after_loading(layer)
    return kernel, layer


def test_expose_input_quant_key_on_triton_branch() -> None:
    from vllm.model_executor.layers.quantization.utils.quant_utils import kMxfp4Dynamic

    kernel, layer = _kernel_and_layer(SMALL_N, SMALL_K)
    assert kernel.input_quant_key() == kMxfp4Dynamic
    assert getattr(layer, "input_quant_key", None) == kMxfp4Dynamic


@pytest.mark.parametrize("n,k", [(K3_N, K3_K), (SMALL_N, SMALL_K)])
@pytest.mark.parametrize("m", MS)
def test_prequant_matches_internal_quant(n: int, k: int, m: int) -> None:
    from aiter.ops.triton.quant import dynamic_mxfp4_quant

    from vllm.model_executor.layers.fusion.quant_activation import QuantizedActivation
    from vllm.model_executor.layers.quantization.utils.quant_utils import kMxfp4Dynamic

    torch.manual_seed(0)
    kernel, layer = _kernel_and_layer(n, k)
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)

    y_ref = kernel.apply_weights(layer, x)

    x_q, x_s = dynamic_mxfp4_quant(x)
    assert x_s.dtype == torch.uint8
    assert x_s.shape == (m, k // 32)
    # dynamic_mxfp4_quant allocates column-major e8m0: empty((K/32, M)).T
    assert x_s.stride() == (1, m)

    qa = QuantizedActivation(
        data=x_q,
        scale=x_s,
        orig_dtype=x.dtype,
        orig_shape=x.shape,
        quant_key=kMxfp4Dynamic,
    )
    y_pre = kernel.apply_weights(layer, qa)
    assert torch.equal(y_ref, y_pre), (
        f"prequant != in-kernel quant at M={m} N={n} K={k}"
    )


def test_key_mismatch_raises() -> None:
    from aiter.ops.triton.quant import dynamic_mxfp4_quant

    from vllm.model_executor.layers.fusion.quant_activation import QuantizedActivation
    from vllm.model_executor.layers.quantization.utils.quant_utils import kNvfp4Dynamic

    kernel, layer = _kernel_and_layer(SMALL_N, SMALL_K)
    x = torch.randn(8, SMALL_K, device="cuda", dtype=torch.bfloat16)
    x_q, x_s = dynamic_mxfp4_quant(x)
    qa = QuantizedActivation(
        data=x_q,
        scale=x_s,
        orig_dtype=x.dtype,
        orig_shape=x.shape,
        quant_key=kNvfp4Dynamic,
    )
    with pytest.raises(AssertionError, match="QuantizedActivation key"):
        kernel.apply_weights(layer, qa)


def test_asm_branch_declines_prequant() -> None:
    from vllm.model_executor.layers.fusion.quant_activation import (
        QuantizedActivation,
        as_quantized_activation,
    )
    from vllm.model_executor.layers.quantization.utils.quant_utils import kMxfp4Dynamic

    kernel, layer = _kernel_and_layer(SMALL_N, SMALL_K)
    kernel.use_asm_gemm = True
    assert kernel.input_quant_key() is None

    qa = QuantizedActivation(
        data=torch.empty(8, SMALL_K // 2, dtype=torch.uint8, device="cuda"),
        scale=torch.empty(8, SMALL_K // 32, dtype=torch.uint8, device="cuda"),
        orig_dtype=torch.bfloat16,
        orig_shape=torch.Size([8, SMALL_K]),
        quant_key=kMxfp4Dynamic,
    )
    with pytest.raises(AssertionError, match="QuantizedActivation key"):
        as_quantized_activation(qa, kernel.input_quant_key())
    # Plain tensors still go through in-kernel quant; do not call apply_weights
    # here because the stub layer's weights are in the Triton layout.
    assert isinstance(layer.weight, torch.nn.Parameter)
