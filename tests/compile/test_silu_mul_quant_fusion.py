# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

import vllm.envs as envs
from vllm._custom_ops import scaled_fp8_quant
from vllm.compilation.activation_quant_fusion import ActivationQuantFusionPass
from vllm.compilation.fx_utils import find_auto_fn, find_auto_fn_maybe
from vllm.config import CompilationConfig, VllmConfig
from vllm.model_executor.layers.activation import SiluAndMul

from .backend import TestBackend


class TestModel(torch.nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.silu_and_mul = SiluAndMul()
        self.scale = torch.rand(1, dtype=torch.float32)

    def forward(self, x):
        y = self.silu_and_mul(x)
        x2 = scaled_fp8_quant(y, self.scale)
        return x2


@pytest.mark.parametrize("num_tokens", [256])
@pytest.mark.parametrize("hidden_size", [64])
@pytest.mark.skipif(envs.VLLM_TARGET_DEVICE != "cuda",
                    reason="Only test on CUDA")
def test_fusion_silu_and_mul_quant(num_tokens, hidden_size):
    torch.set_default_device("cuda")
    torch.set_default_dtype(torch.float16)

    # Reshape pass is needed for the fusion pass to work
    config = VllmConfig()
    config.compilation_config = CompilationConfig(
        pass_config=CompilationConfig.PassConfig(enable_fusion=True,
                                                 enable_reshape=True))
    fusion_pass = ActivationQuantFusionPass(config)

    backend = TestBackend(fusion_pass)
    model = TestModel()

    # First dimension dynamic
    x = torch.rand(num_tokens, hidden_size)
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
