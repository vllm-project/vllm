# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import cast

import pytest
import torch

import vllm.envs as envs
from tests.kernels.quantization.nvfp4_utils import quant_nvfp4_tensor
from vllm._custom_ops import cutlass_scaled_fp4_mm, scaled_fp4_quant
# yapf conflicts with isort for this block
# yapf: disable
from vllm.compilation.activation_quant_fusion import (
    FUSED_OPS, SILU_MUL_OP, ActivationQuantFusionPass)
# yapf: enable
from vllm.compilation.fusion import QUANT_OPS
from vllm.compilation.noop_elimination import NoOpEliminationPass
from vllm.config import CompilationConfig, PassConfig, VllmConfig
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    GroupShape, kFp8StaticTensorSym, kNvfp4Quant)
from vllm.model_executor.layers.quantization.utils.w8a8_utils import (
    Fp8LinearOp, cutlass_fp8_supported)
from vllm.platforms import current_platform

from ..utils import override_cutlass_fp8_supported
from .backend import TestBackend

FP8_DTYPE = current_platform.fp8_dtype()
FP4_DTYPE = torch.uint8


def is_nvfp4_supported():
    return current_platform.has_device_capability(100)


class TestSiluMulFp8QuantModel(torch.nn.Module):

    def __init__(self, hidden_size: int, cuda_force_torch: bool, **kwargs):
        super().__init__()
        self.silu_and_mul = SiluAndMul()
        self.wscale = torch.rand(1, dtype=torch.float32)
        self.scale = torch.rand(1, dtype=torch.float32)

        self.w = torch.rand(hidden_size, hidden_size).to(dtype=FP8_DTYPE).t()

        with override_cutlass_fp8_supported(not cuda_force_torch):
            self.fp8_linear = Fp8LinearOp(
                act_quant_static=True,
                act_quant_group_shape=GroupShape.PER_TENSOR,
            )

    def forward(self, x):
        y = self.silu_and_mul(x)
        x2 = self.fp8_linear.apply(y,
                                   self.w,
                                   self.wscale,
                                   input_scale=self.wscale)
        return x2

    def ops_in_model_before(self):
        return [SILU_MUL_OP, QUANT_OPS[kFp8StaticTensorSym]]

    def ops_in_model_after(self):
        return [FUSED_OPS[kFp8StaticTensorSym]]


class TestSiluMulNvfp4QuantModel(torch.nn.Module):

    def __init__(self, hidden_size: int, x: torch.Tensor, **kwargs):
        super().__init__()
        self.silu_and_mul = SiluAndMul()

        # create nvfp4 weight
        w = torch.rand((hidden_size, hidden_size))
        self.w, self.w_block_scale, self.w_global_scale = quant_nvfp4_tensor(w)

        # get global scale offline
        _, _, self.y_global_scale = quant_nvfp4_tensor(self.silu_and_mul(x))

        self.alpha = 1.0 / (self.w_global_scale * self.y_global_scale)

    def forward(self, x):
        y = self.silu_and_mul(x)
        y_quant, y_block_scale = scaled_fp4_quant(y, self.y_global_scale)
        out = cutlass_scaled_fp4_mm(a=y_quant,
                                    b=self.w,
                                    block_scale_a=y_block_scale,
                                    block_scale_b=self.w_block_scale,
                                    alpha=self.alpha,
                                    out_dtype=y.dtype)
        return out

    def ops_in_model_before(self):
        return [SILU_MUL_OP, QUANT_OPS[kNvfp4Quant]]

    def ops_in_model_after(self):
        return [FUSED_OPS[kNvfp4Quant]]


@pytest.mark.parametrize("num_tokens", [32, 64])
@pytest.mark.parametrize("hidden_size", [128, 256])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "model_class",
    cast(list[type], [TestSiluMulFp8QuantModel, TestSiluMulNvfp4QuantModel]
         if is_nvfp4_supported() else [TestSiluMulFp8QuantModel]))
# cuda_force_torch used to test torch code path on platforms that
# cutlass_fp8_supported() == True.
@pytest.mark.parametrize("cuda_force_torch",
                         [True, False] if cutlass_fp8_supported() else [True])
@pytest.mark.skipif(envs.VLLM_TARGET_DEVICE not in ["cuda", "rocm"],
                    reason="Only test on CUDA and ROCm")
def test_fusion_silu_and_mul_quant(num_tokens, hidden_size, dtype, model_class,
                                   cuda_force_torch):
    if model_class == TestSiluMulNvfp4QuantModel and cuda_force_torch:
        pytest.skip("Duplicate tests for NVFP4")

    torch.set_default_device("cuda")
    torch.set_default_dtype(dtype)

    x = torch.rand(num_tokens, hidden_size * 2)

    # Reshape pass is needed for the fusion pass to work
    config = VllmConfig()
    config.compilation_config = CompilationConfig(
        pass_config=PassConfig(enable_fusion=True, enable_noop=True))
    fusion_pass = ActivationQuantFusionPass(config)

    backend = TestBackend(NoOpEliminationPass(config), fusion_pass)
    model = model_class(hidden_size=hidden_size,
                        cuda_force_torch=cuda_force_torch,
                        x=x)

    # First dimension dynamic
    torch._dynamo.mark_dynamic(x, 0)

    result = model(x)

    model2 = torch.compile(model, backend=backend)
    result2 = model2(x)

    # Check that it gives the same answer
    if model_class == TestSiluMulFp8QuantModel:
        atol, rtol = 1e-3, 1e-3
    elif model_class == TestSiluMulNvfp4QuantModel:
        atol, rtol = 1e-1, 1e-1

    torch.testing.assert_close(result[0].to(dtype=dtype),
                               result2[0].to(dtype=dtype),
                               atol=atol,
                               rtol=rtol)

    # In pre-nodes, quant op should be present and fused kernels should not
    backend.check_before_ops(model.ops_in_model_before())

    # In post-nodes, fused kernels should be present and quant op should not
    backend.check_after_ops(model.ops_in_model_after())
