# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

import vllm.envs as envs
from vllm.compilation.activation_quant_fusion import ActivationQuantFusionPass
from vllm.compilation.fx_utils import find_auto_fn, find_auto_fn_maybe
from vllm.compilation.noop_elimination import NoOpEliminationPass
from vllm.config import CompilationConfig, PassConfig, VllmConfig
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    GroupShape)
from vllm.model_executor.layers.quantization.utils.w8a8_utils import (
    CUTLASS_FP8_SUPPORTED, Fp8LinearOp)
from vllm.platforms import current_platform

from .backend import TestBackend


class TestModel(torch.nn.Module):

    def __init__(self, hidden_size: int, cutlass_fp8_enabled: bool, *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.silu_and_mul = SiluAndMul()
        self.wscale = torch.rand(1, dtype=torch.float32)
        self.scale = torch.rand(1, dtype=torch.float32)

        self.w = (torch.rand(
            hidden_size,
            hidden_size).to(dtype=current_platform.fp8_dtype()).t())

        self.fp8_linear = Fp8LinearOp(
            cutlass_fp8_supported=cutlass_fp8_enabled,
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


@pytest.mark.parametrize("num_tokens", [256])
@pytest.mark.parametrize("hidden_size", [64])
@pytest.mark.parametrize("cutlass_fp8_enabled",
                         [True, False] if CUTLASS_FP8_SUPPORTED else [False])
@pytest.mark.skipif(envs.VLLM_TARGET_DEVICE not in ["cuda", "rocm"],
                    reason="Only test on CUDA and ROCm")
def test_fusion_silu_and_mul_quant(num_tokens, hidden_size,
                                   cutlass_fp8_enabled):
    torch.set_default_device("cuda")
    torch.set_default_dtype(torch.float16)

    # Reshape pass is needed for the fusion pass to work
    config = VllmConfig()
    config.compilation_config = CompilationConfig(
        pass_config=PassConfig(enable_fusion=True, enable_noop=True))
    fusion_pass = ActivationQuantFusionPass(config)

    backend = TestBackend(NoOpEliminationPass(config), fusion_pass)
    model = TestModel(hidden_size, cutlass_fp8_enabled)

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

    # Check substitution worked
    pre_nodes = backend.graph_pre_pass.nodes
    post_nodes = backend.graph_post_pass.nodes

    silu_and_mul_quant = torch.ops._C.silu_and_mul_quant.default
    fp8_quant = torch.ops._C.static_scaled_fp8_quant.default

    # In pre-nodes, fp8 quant should be present and fused kernels should not
    assert find_auto_fn_maybe(pre_nodes, silu_and_mul_quant) is None
    find_auto_fn(pre_nodes, fp8_quant)

    # In post-nodes, fused kernels should be present and fp8 quant should not
    find_auto_fn(post_nodes, silu_and_mul_quant)
    assert find_auto_fn_maybe(post_nodes, fp8_quant) is None
