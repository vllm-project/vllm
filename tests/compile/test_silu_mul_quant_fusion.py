# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

import vllm.envs as envs
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
    Fp8LinearOp)
from vllm.platforms import current_platform

from .backend import TestBackend

FP8_DTYPE = current_platform.fp8_dtype()
FP4_DTYPE = torch.uint8


def is_nvfp4_supported():
    return current_platform.has_device_capability(100)


class TestSiluMulFp8QuantModel(torch.nn.Module):

    def __init__(self, hidden_size: int, force_fp8_e4m3fnuz: bool, **kwargs):
        super().__init__()
        self.silu_and_mul = SiluAndMul()
        self.wscale = torch.rand(1, dtype=torch.float32)
        self.scale = torch.rand(1, dtype=torch.float32)

        self.w = torch.rand(hidden_size, hidden_size).to(dtype=FP8_DTYPE).t()

        self.fp8_linear = Fp8LinearOp(
            force_fp8_e4m3fnuz=force_fp8_e4m3fnuz,
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

    def __init__(self, hidden_size: int, **kwargs):
        super().__init__()
        self.silu_and_mul = SiluAndMul()
        self.w = torch.randint(256, (hidden_size, hidden_size // 2),
                               dtype=FP4_DTYPE)
        self.wscale = torch.randn(hidden_size,
                                  hidden_size // 16).to(dtype=FP8_DTYPE)
        self.wscale2 = torch.rand(1, dtype=torch.float32)
        self.scale = torch.rand(1, dtype=torch.float32)

    def forward(self, x):
        y = self.silu_and_mul(x)
        y_quant, y_block_scale = scaled_fp4_quant(y, 1 / self.scale)
        out = cutlass_scaled_fp4_mm(a=y_quant,
                                    b=self.w,
                                    block_scale_a=y_block_scale,
                                    block_scale_b=self.wscale,
                                    alpha=self.scale * self.wscale2,
                                    out_dtype=y.dtype)
        return out

    def ops_in_model_before(self):
        return [SILU_MUL_OP, QUANT_OPS[kNvfp4Quant]]

    def ops_in_model_after(self):
        return [FUSED_OPS[kNvfp4Quant]]


@pytest.mark.parametrize("num_tokens", [64])
@pytest.mark.parametrize("hidden_size", [128])
@pytest.mark.parametrize(
    "model_class", [TestSiluMulFp8QuantModel, TestSiluMulNvfp4QuantModel]
    if is_nvfp4_supported() else [TestSiluMulFp8QuantModel])
@pytest.mark.parametrize("force_fp8_e4m3fnuz", [True, False])
@pytest.mark.skipif(envs.VLLM_TARGET_DEVICE not in ["cuda", "rocm"],
                    reason="Only test on CUDA and ROCm")
def test_fusion_silu_and_mul_quant(num_tokens, hidden_size, model_class,
                                   force_fp8_e4m3fnuz):
    if model_class == TestSiluMulNvfp4QuantModel and force_fp8_e4m3fnuz:
        pytest.skip("Duplicate tests for NVFP4")

    torch.set_default_device("cuda")
    torch.set_default_dtype(torch.float16)

    # Reshape pass is needed for the fusion pass to work
    config = VllmConfig()
    config.compilation_config = CompilationConfig(
        pass_config=PassConfig(enable_fusion=True, enable_noop=True))
    fusion_pass = ActivationQuantFusionPass(config)

    backend = TestBackend(NoOpEliminationPass(config), fusion_pass)
    model = model_class(hidden_size=hidden_size,
                        force_fp8_e4m3fnuz=force_fp8_e4m3fnuz)

    # First dimension dynamic
    x = torch.rand(num_tokens, hidden_size * 2)
    torch._dynamo.mark_dynamic(x, 0)

    result = model(x)

    model2 = torch.compile(model, backend=backend)
    result2 = model2(x)

    # Check that it gives the same answer
    torch.testing.assert_close(result[0].to(dtype=torch.float16),
                               result2[0].to(dtype=torch.float16),
                               atol=1e-3,
                               rtol=1e-3)

    # In pre-nodes, quant op should be present and fused kernels should not
    backend.check_before_ops(model.ops_in_model_before())

    # In post-nodes, fused kernels should be present and quant op should not
    backend.check_after_ops(model.ops_in_model_after())
