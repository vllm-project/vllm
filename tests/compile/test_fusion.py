import pytest
import torch
from compressed_tensors.quantization import FP8_DTYPE

import vllm.envs as envs
from vllm.compilation.fusion import (FusionPass, find_auto_fn,
                                     find_auto_fn_maybe)
from vllm.compilation.reshapes import RedundantReshapesPass
from vllm.config import CompilationConfig
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.quantization.utils.w8a8_utils import (
    apply_fp8_linear)

from .backend import TestBackend


class TestModel(torch.nn.Module):

    def __init__(self, hidden_size: int, eps: float, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.norm = [RMSNorm(hidden_size, eps) for _ in range(3)]
        self.scale = [torch.rand(1, dtype=torch.float32) for _ in range(4)]
        self.w = [
            torch.rand(hidden_size, hidden_size).to(dtype=FP8_DTYPE).t()
            for _ in range(2)
        ]

    def forward(self, x):
        resid = torch.relu(x)
        y = self.norm[0](x)

        x2 = apply_fp8_linear(y, self.w[0], self.scale[0], self.scale[1])
        # make sure resid is used for replacement to work
        y2, resid = self.norm[1](x2, resid)

        x3 = apply_fp8_linear(y2, self.w[1], self.scale[2], self.scale[3])
        y3, resid = self.norm[2](x3, resid)  # use resid here
        return y3


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("hidden_size", [64, 3392, 4096])
@pytest.mark.parametrize("num_tokens", [7, 256, 533, 2048, 2049])
@pytest.mark.parametrize("eps", [1e-5, 1e-6])
@pytest.mark.skipif(envs.VLLM_TARGET_DEVICE != "cuda",
                    reason="Only test on CUDA")
def test_fusion_rmsnorm_quant(dtype, hidden_size, num_tokens, eps):
    torch.set_default_device("cuda")
    torch.set_default_dtype(torch.float16)

    if eps != 1e-5:
        pytest.skip("Only test eps=1e-5 for now")

    # Reshape pass is needed for the fusion pass to work
    config = CompilationConfig.PassConfig(enable_fusion=True,
                                          enable_reshape=True)
    reshape_pass = RedundantReshapesPass(config)
    fusion_pass = FusionPass.instance(config)

    backend = TestBackend(reshape_pass, fusion_pass)
    model = TestModel(hidden_size, eps)

    # First dimension dynamic
    x = torch.rand(num_tokens, hidden_size)
    torch._dynamo.mark_dynamic(x, 0)

    result = model(x)

    model2 = torch.compile(model, backend=backend)
    result2 = model2(x)

    # Check that it gives the same answer
    torch.testing.assert_close(result, result2, atol=1e-3, rtol=1e-3)

    # Check substitution worked
    pre_nodes = backend.graph_pre_pass.nodes
    post_nodes = backend.graph_post_pass.nodes

    rms_quant = torch.ops._C.rms_norm_static_fp8_quant.default
    add_rms_quant = torch.ops._C.fused_add_rms_norm_static_fp8_quant.default
    fp8_quant = torch.ops._C.static_scaled_fp8_quant.default

    # In pre-nodes, fp8 quant should be present and fused kernels should not
    assert find_auto_fn_maybe(pre_nodes, rms_quant) is None
    assert find_auto_fn_maybe(pre_nodes, add_rms_quant) is None
    find_auto_fn(pre_nodes, fp8_quant)

    # In post-nodes, fused kernels should be present and fp8 quant should not
    find_auto_fn(post_nodes, rms_quant)
    find_auto_fn(post_nodes, add_rms_quant)
    assert find_auto_fn_maybe(post_nodes, fp8_quant) is None
