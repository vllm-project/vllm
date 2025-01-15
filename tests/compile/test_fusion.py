import pytest
import torch
from compressed_tensors.quantization import FP8_DTYPE

import vllm.envs as envs
from vllm.compilation.fusion import (FUSED_OPS, QUANT_OPS, FusedRMSQuantKey,
                                     FusionPass, QuantKey)
from vllm.compilation.fx_utils import find_auto_fn, find_auto_fn_maybe
from vllm.compilation.reshapes import RedundantReshapesPass
from vllm.config import CompilationConfig
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.quantization.utils.w8a8_utils import (
    apply_fp8_linear)

from .backend import TestBackend


class TestModel(torch.nn.Module):

    def __init__(self, hidden_size: int, eps: float, static: bool, *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.norm = [RMSNorm(hidden_size, eps) for _ in range(3)]
        self.wscale = [torch.rand(1, dtype=torch.float32) for _ in range(2)]
        if static:
            self.scale = [torch.rand(1, dtype=torch.float32) for _ in range(2)]
        else:
            self.scale = [None for _ in range(2)]
        self.w = [
            torch.rand(hidden_size, hidden_size).to(dtype=FP8_DTYPE).t()
            for _ in range(2)
        ]

    def forward(self, x):
        resid = torch.sqrt(x)
        y = self.norm[0](x)

        x2 = apply_fp8_linear(y,
                              self.w[0],
                              self.wscale[0],
                              self.scale[0],
                              use_per_token_if_dynamic=True)
        # make sure resid is used for replacement to work
        y2, resid = self.norm[1](x2, resid)

        x3 = apply_fp8_linear(y2,
                              self.w[1],
                              self.wscale[1],
                              self.scale[1],
                              use_per_token_if_dynamic=True)
        y3, resid = self.norm[2](x3, resid)  # use resid here
        return y3


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("hidden_size", [64, 3392, 4096])
@pytest.mark.parametrize("num_tokens", [7, 256, 533, 2048, 2049])
@pytest.mark.parametrize("eps", [1e-5, 1e-6])
@pytest.mark.parametrize("static", [True, False])
@pytest.mark.skipif(envs.VLLM_TARGET_DEVICE != "cuda",
                    reason="Only test on CUDA")
def test_fusion_rmsnorm_quant(dtype, hidden_size, num_tokens, eps, static):
    torch.set_default_device("cuda")
    torch.set_default_dtype(dtype)
    torch.manual_seed(1)

    # Reshape pass is needed for the fusion pass to work
    config = CompilationConfig.PassConfig(enable_fusion=True,
                                          enable_reshape=True)
    reshape_pass = RedundantReshapesPass(config)
    fusion_pass = FusionPass.instance(config)

    backend = TestBackend(reshape_pass, fusion_pass)
    model = TestModel(hidden_size, eps, static)

    # First dimension dynamic
    x = torch.rand(num_tokens, hidden_size)
    torch._dynamo.mark_dynamic(x, 0)

    result = model(x)

    model2 = torch.compile(model, backend=backend)
    result2 = model2(x)

    # Higher tol for dynamic, even higher for bfloat16
    if static:
        ATOL, RTOL = (1e-3, 1e-3)
    elif dtype == torch.float16:
        ATOL, RTOL = (2e-3, 2e-3)
    else:
        ATOL, RTOL = (1e-2, 1e-2)

    torch.testing.assert_close(result, result2, atol=ATOL, rtol=RTOL)

    # Check substitution worked
    pre_nodes = backend.graph_pre_pass.nodes
    post_nodes = backend.graph_post_pass.nodes

    # static is per-tensor, dynamic is per-token
    key = QuantKey(dtype=FP8_DTYPE,
                   static=static,
                   per_tensor=static,
                   symmetric=True)
    rms_quant = FUSED_OPS[FusedRMSQuantKey(key, False)]
    add_rms_quant = FUSED_OPS[FusedRMSQuantKey(key, True)]
    fp8_quant = QUANT_OPS[key]

    # In pre-nodes, fp8 quant should be present and fused kernels should not
    assert find_auto_fn_maybe(pre_nodes, rms_quant) is None
    assert find_auto_fn_maybe(pre_nodes, add_rms_quant) is None
    find_auto_fn(pre_nodes, fp8_quant)

    # In post-nodes, fused kernels should be present and fp8 quant should not
    find_auto_fn(post_nodes, rms_quant)
    find_auto_fn(post_nodes, add_rms_quant)
    assert find_auto_fn_maybe(post_nodes, fp8_quant) is None
