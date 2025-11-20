# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from importlib.util import find_spec

import pytest
import torch

from vllm._aiter_ops import rocm_aiter_ops
from vllm.compilation.noop_elimination import NoOpEliminationPass
from vllm.compilation.post_cleanup import PostCleanupPass
from vllm.compilation.rocm_aiter_fusion import (
    RocmAiterRMSNormFp8GroupQuantFusionPass,
)
from vllm.config import (
    CompilationConfig,
    CompilationMode,
    ModelConfig,
    PassConfig,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.model_executor.layers.quantization.utils.fp8_utils import W8A8BlockFp8LinearOp
from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape
from vllm.model_executor.layers.quantization.utils.w8a8_utils import (
    maybe_create_device_identity,
)
from vllm.platforms import current_platform

from .backend import TestBackend

FP8_DTYPE = current_platform.fp8_dtype()

# Check if aiter package is installed
aiter_available = find_spec("aiter") is not None


class TestAiterRmsnormGroupFp8QuantModel(torch.nn.Module):
    def __init__(self, hidden_size: int, eps: float):
        super().__init__()
        self.w8a8_block_fp8_linear = W8A8BlockFp8LinearOp(
            weight_group_shape=GroupShape(128, 128),
            act_quant_group_shape=GroupShape(1, 128),
            cutlass_block_fp8_supported=False,
            use_aiter_and_is_supported=True,
        )
        self.w = [
            torch.rand(hidden_size, hidden_size).to(dtype=FP8_DTYPE).t()
            for _ in range(3)
        ]

        scale_hidden_size = (hidden_size + 128 - 1) // 128
        self.wscale = [
            torch.rand((scale_hidden_size, scale_hidden_size), dtype=torch.float32)
            for _ in range(3)
        ]

        self.norm_weight = [torch.ones(hidden_size) for _ in range(4)]
        self.eps = eps

    def forward(self, x):
        # avoid having graph input be an arg to a pattern directly
        x = resid = torch.relu(x)
        y = rocm_aiter_ops.rms_norm(x, self.norm_weight[0], self.eps)

        x2 = self.w8a8_block_fp8_linear.apply(y, self.w[0], self.wscale[0])
        # make sure resid is used for replacement to work
        y2, resid = rocm_aiter_ops.rms_norm2d_with_add(
            x2, resid, self.norm_weight[1], self.eps
        )

        x3 = self.w8a8_block_fp8_linear.apply(y2, self.w[1], self.wscale[1])

        y3, resid = rocm_aiter_ops.rms_norm2d_with_add(
            x3, resid, self.norm_weight[2], self.eps
        )

        x4 = self.w8a8_block_fp8_linear.apply(y3, self.w[2], self.wscale[2])

        y4, resid = rocm_aiter_ops.rms_norm2d_with_add(
            x4, resid, self.norm_weight[3], self.eps
        )
        return y4

    def ops_in_model_before(self):
        return [
            torch.ops.vllm.rocm_aiter_rms_norm,
            torch.ops.vllm.rocm_aiter_group_fp8_quant,
        ]

    def ops_in_model_after(self):
        return [
            torch.ops.vllm.rocm_aiter_rmsnorm_fp8_group_quant,
            torch.ops.vllm.rocm_aiter_rmsnorm_with_add_fp8_group_quant,
        ]


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("hidden_size", [128, 256])
@pytest.mark.parametrize("num_tokens", [257])
@pytest.mark.parametrize("eps", [1e-5, 1e-6])
@pytest.mark.skipif(
    not (current_platform.is_rocm() and aiter_available),
    reason="AITER ops are only available on ROCm with aiter package installed",
)
def test_fusion_aiter_rmsnorm_group_fp8_quant(
    dtype,
    hidden_size,
    num_tokens,
    eps,
):
    torch.set_default_device("cuda")
    torch.set_default_dtype(dtype)
    torch.manual_seed(1)
    maybe_create_device_identity()  # needed for certain non-cutlass fp8 paths

    custom_ops = []
    custom_ops.append("+rms_norm")
    vllm_config = VllmConfig(
        model_config=ModelConfig(dtype=dtype),
        compilation_config=CompilationConfig(
            mode=CompilationMode.VLLM_COMPILE,
            custom_ops=custom_ops,
            pass_config=PassConfig(enable_fusion=True, enable_noop=True),
        ),
    )
    with set_current_vllm_config(vllm_config):
        # Reshape pass is needed for the fusion pass to work
        noop_pass = NoOpEliminationPass(vllm_config)
        fusion_pass = RocmAiterRMSNormFp8GroupQuantFusionPass(vllm_config)
        cleanup_pass = PostCleanupPass(vllm_config)

        backend = TestBackend(noop_pass, fusion_pass, cleanup_pass)
        backend2 = TestBackend(noop_pass, cleanup_pass)
        model = TestAiterRmsnormGroupFp8QuantModel(hidden_size, eps)

        # First dimension dynamic
        x = torch.rand(num_tokens, hidden_size)
        torch._dynamo.mark_dynamic(x, 0)

        model_fused = torch.compile(model, backend=backend)
        result_fused = model_fused(x)

        model_unfused = torch.compile(model, backend=backend2)
        result_unfused = model_unfused(x)

        if dtype == torch.float16:
            ATOL, RTOL = (2e-3, 2e-3)
        else:
            ATOL, RTOL = (1e-2, 1e-2)

        torch.testing.assert_close(result_fused, result_unfused, atol=ATOL, rtol=RTOL)

        assert fusion_pass.matched_count == 3
        backend.check_before_ops(model.ops_in_model_before())
        backend.check_after_ops(model.ops_in_model_after())
