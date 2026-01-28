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
        self,
        num_layers: int,
        hidden_size: int,
        num_local_experts: int,
        x_pad_to_multiple: int,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.x_pad_to_multiple = x_pad_to_multiple
        self.pad_dim = x_pad_to_multiple - (hidden_size % x_pad_to_multiple)

        self.norm = [RMSNorm(hidden_size, eps=1e-5) for _ in range(num_layers)]
        self.router = [
            torch.nn.Linear(hidden_size, num_local_experts) for _ in range(4)
        ]

    def forward(self, x):
        # avoid having graph input be an arg to a pattern directly
        x = resid = torch.relu(x)
        all_router_logits = []
        for layer in range(self.num_layers):
            x = x[:, : self.hidden_size]
            x, resid = self.norm[layer](x, resid)
            router_logits = rocm_unquantized_gemm(
                self, x, self.router[layer].weight, self.router[layer].bias
            )
            x = torch.nn.functional.pad(
                x, (0, self.pad_dim), mode="constant", value=0.0
            )
            all_router_logits.append(router_logits)

        return x, resid, *all_router_logits

    def ops_in_model_before(self):
        return [
            rocm_aiter_ops.get_rmsnorm_fused_add_op(),
            torch.ops.aten.constant_pad_nd,
        ]

    def ops_in_model_after(self):
        return [rocm_aiter_ops.get_triton_add_rmsnorm_pad_op()]


@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("num_layers", [3])
@pytest.mark.parametrize("hidden_size", [2880])
@pytest.mark.parametrize("num_local_experts", [128])
@pytest.mark.parametrize("x_pad_to_multiple", [256])
@pytest.mark.skipif(
    not is_aiter_found_and_supported(),
    reason="Only test on ROCm with AITER installed and supported",
)
def test_fuse_act_padding(
    dtype: torch.dtype,
    num_layers: int,
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
        model = TestModel(num_layers, hidden_size, num_local_experts, x_pad_to_multiple)

        x = torch.rand(1, hidden_size)
        torch._dynamo.mark_dynamic(x, 0)

        outputs_unfused = model(x)

        model_fused = torch.compile(model, backend=backend)
        outputs_fused = model_fused(x)

        torch.testing.assert_close(outputs_unfused, outputs_fused)

        assert fusion_pass.matched_count == num_layers

        backend.check_before_ops(model.ops_in_model_before())
        backend.check_after_ops(model.ops_in_model_after())
