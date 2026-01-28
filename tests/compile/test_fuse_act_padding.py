# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import pytest
import torch

import vllm.config
from vllm._aiter_ops import is_aiter_found_and_supported, rocm_aiter_ops
from vllm.compilation.noop_elimination import NoOpEliminationPass
from vllm.compilation.post_cleanup import PostCleanupPass
from vllm.compilation.rocm_aiter_fusion import RocmAiterTritonAddRMSNormPadFusionPass
from vllm.config import (
    CompilationConfig,
    CompilationMode,
    ModelConfig,
    PassConfig,
    VllmConfig,
)
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.utils import rocm_unquantized_gemm

from .backend import TestBackend


class TestModel(torch.nn.Module):
    def __init__(
        self, hidden_size: int, num_local_experts: int, x_pad_to_multiple: int
    ):
        super().__init__()
        self.norm = RMSNorm(hidden_size, eps=1e-5)
        self.router = torch.nn.Linear(hidden_size, num_local_experts)
        self.x_pad_to_multiple = x_pad_to_multiple
        self.pad_dim = x_pad_to_multiple - (hidden_size % x_pad_to_multiple)

    def forward(self, x, residual):
        x, residual = self.norm(x, residual)
        router_logits = rocm_unquantized_gemm(
            self, x, self.router.weight, self.router.bias
        )
        x = torch.nn.functional.pad(x, (0, self.pad_dim), mode="constant", value=0.0)
        return x, residual, router_logits

    def ops_in_model_before(self):
        return [
            rocm_aiter_ops.get_rmsnorm_fused_add_op(),
            torch.ops.aten.constant_pad_nd,
        ]

    def ops_in_model_after(self):
        return [rocm_aiter_ops.get_triton_add_rmsnorm_pad_op()]


@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("hidden_size", [2880])
@pytest.mark.parametrize("num_local_experts", [128])
@pytest.mark.parametrize("x_pad_to_multiple", [256])
@pytest.mark.skipif(
    not is_aiter_found_and_supported(),
    reason="Only test on ROCm with AITER installed and supported",
)
def test_fuse_act_padding(
    dtype: torch.dtype,
    hidden_size: int,
    num_local_experts: int,
    x_pad_to_multiple: int,
    monkeypatch: pytest.MonkeyPatch,
):
    vllm_config = VllmConfig(
        model_config=ModelConfig(dtype=dtype),
        compilation_config=CompilationConfig(
            mode=CompilationMode.VLLM_COMPILE,
            custom_ops=["+rms_norm"],
            pass_config=PassConfig(fuse_act_padding=True, eliminate_noops=True),
        ),
    )

    with vllm.config.set_current_vllm_config(vllm_config), monkeypatch.context() as m:
        torch.set_default_device("cuda")
        torch.set_default_dtype(dtype)
        torch.manual_seed(1)

        m.setenv("VLLM_ROCM_USE_AITER", "1")
        rocm_aiter_ops.refresh_env_variables()

        fusion_pass = RocmAiterTritonAddRMSNormPadFusionPass(vllm_config)
        passes = [
            NoOpEliminationPass(vllm_config),
            fusion_pass,
            PostCleanupPass(vllm_config),
        ]
        backend = TestBackend(*passes)
        model = TestModel(hidden_size, num_local_experts, x_pad_to_multiple)

        x = torch.rand(1, hidden_size)
        residual = torch.rand(1, hidden_size)
        torch._dynamo.mark_dynamic(x, 0)
        torch._dynamo.mark_dynamic(residual, 0)

        x_unfused, residual_unfused, router_logits_unfused = model(x, residual)

        model_fused = torch.compile(model, backend=backend)
        x_fused, residual_fused, router_logits_fused = model_fused(x, residual)

        torch.testing.assert_close(x_fused, x_unfused)
        torch.testing.assert_close(residual_fused, residual_unfused)
        torch.testing.assert_close(router_logits_fused, router_logits_unfused)

        assert fusion_pass.matched_count == 1

        backend.check_before_ops(model.ops_in_model_before())
        backend.check_after_ops(model.ops_in_model_after())
