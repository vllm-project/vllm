# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

import vllm.ir.ops
from tests.compile.backend import TestBackend
from vllm.compilation.passes.fusion.add_rms_fusion import (
    AddRMSNormFusionPass,
    RMSNormReshapeFusionPass,
)
from vllm.compilation.passes.fx_utils import find_op_nodes, is_func
from vllm.compilation.passes.utility.noop_elimination import NoOpEliminationPass
from vllm.compilation.passes.utility.post_cleanup import PostCleanupPass
from vllm.config import CompilationConfig, CompilationMode, VllmConfig
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda_alike(), reason="Requires CUDA or ROCm"
)


class RMSNormModel(torch.nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = vllm.ir.ops.rms_norm(x, self.weight, 1e-6)
        return rms.reshape(-1, rms.shape[-1])


class AddRMSNormModel(torch.nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(hidden_size))

    def forward(
        self, x: torch.Tensor, residual: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        residual = residual + x
        rms = vllm.ir.ops.rms_norm(residual, self.weight, 1e-6)
        return rms.reshape(-1, rms.shape[-1]), residual

    def ops_in_model_before(self):
        return [torch.ops.aten.add, torch.ops.vllm_ir.rms_norm]

    def ops_in_model_after(self):
        return [torch.ops.vllm_ir.fused_add_rms_norm]


def _run_fusion_test(model, config, passes, *inputs):
    backend = TestBackend(NoOpEliminationPass(config), *passes, PostCleanupPass(config))
    outputs_unfused = model(*inputs)
    outputs_fused = torch.compile(model, backend=backend)(*inputs)
    torch.testing.assert_close(outputs_fused, outputs_unfused)
    return backend


def _is_reshape(node):
    return isinstance(node, torch.fx.Node) and is_func(
        node, torch.ops.aten.reshape.default
    )


@pytest.fixture
def vllm_config():
    config = VllmConfig(
        compilation_config=CompilationConfig(mode=CompilationMode.VLLM_COMPILE)
    )
    with vllm.config.set_current_vllm_config(config):
        torch.set_default_device("cuda")
        torch.set_default_dtype(torch.bfloat16)
        torch.manual_seed(0)
        yield config


def test_rmsnorm_reshape_fusion(vllm_config):
    fusion_pass = RMSNormReshapeFusionPass(vllm_config)
    model = RMSNormModel(hidden_size=32)
    x = torch.randn(2, 7, 32)
    backend = _run_fusion_test(model, vllm_config, [fusion_pass], x)

    assert fusion_pass.matched_count == 1
    (rms_node,) = find_op_nodes(torch.ops.vllm_ir.rms_norm, backend.graph_post_pass)
    assert _is_reshape(rms_node.args[0])


def test_add_rmsnorm_reshape_fusion(vllm_config):
    add_fusion = AddRMSNormFusionPass(vllm_config)
    reshape_fusion = RMSNormReshapeFusionPass(vllm_config)
    model = AddRMSNormModel(hidden_size=32)
    x = torch.randn(2, 7, 32)
    residual = torch.randn_like(x)
    backend = _run_fusion_test(
        model, vllm_config, [add_fusion, reshape_fusion], x, residual
    )

    assert add_fusion.matched_count == 1
    assert reshape_fusion.matched_count == 1
    backend.check_before_ops(model.ops_in_model_before())
    backend.check_after_ops(model.ops_in_model_after())
