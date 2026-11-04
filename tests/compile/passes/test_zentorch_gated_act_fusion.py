# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pickle

import pytest
import torch

from tests.compile.backend import TestBackend
from vllm.compilation.passes.utility.noop_elimination import NoOpEliminationPass
from vllm.compilation.passes.utility.post_cleanup import PostCleanupPass
from vllm.config import (
    CompilationConfig,
    CompilationMode,
    PassConfig,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.model_executor.kernels.linear.zentorch_utils import has_zentorch_op
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.utils import dispatch_cpu_unquantized_gemm
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_zen_cpu()
    or not has_zentorch_op(["zentorch_linear_gated_act"]),
    reason="Requires an AMD Zen CPU with zentorch_linear_gated_act",
)

DA8W8_OP_NAME = "zentorch::zentorch_dynamic_qlinear"


@pytest.fixture(autouse=True)
def _clear_zendnn_weight_cache():
    # ZenDNN caches reordered weights by data pointer; a freed weight's address
    # reused by the next test's weight would otherwise hit a stale entry.
    import zentorch

    zentorch._C.clear_weight_cache()
    yield


class GateUpSiluMulModel(torch.nn.Module):
    """bf16 gate/up projection through vLLM's zentorch CPU linear."""

    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.gate_up = torch.nn.Module()
        self.gate_up.weight = torch.nn.Parameter(
            torch.randn(2 * intermediate_size, hidden_size, dtype=torch.bfloat16)
            / hidden_size**0.5,
            requires_grad=False,
        )
        dispatch_cpu_unquantized_gemm(self.gate_up, remove_weight=False)
        self.act = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up = self.gate_up.cpu_linear(x, self.gate_up.weight, None)
        return self.act(gate_up)


class QGateUpSiluMulModel(torch.nn.Module):
    """DA8W8 gate/up projection, called as ZentorchInt8ScaledMMLinearKernel
    (with zentorch_op_name) or ZentorchWNA16LinearKernel (without) does."""

    def __init__(
        self, hidden_size: int, intermediate_size: int, zentorch_op_name: str | None
    ) -> None:
        super().__init__()
        self.weight = torch.randint(
            -8, 8, (2 * intermediate_size, hidden_size), dtype=torch.int8
        )
        self.weight_scales = torch.rand(2 * intermediate_size) / (8 * hidden_size**0.5)
        self.kwargs = (
            {} if zentorch_op_name is None else {"zentorch_op_name": zentorch_op_name}
        )
        self.act = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up = torch.ops.zentorch.zentorch_dynamic_qlinear(
            x, self.weight, self.weight_scales, None, **self.kwargs
        )
        return self.act(gate_up)


def _gate_up_op(model: torch.nn.Module) -> torch._ops.OpOverload:
    if isinstance(model, QGateUpSiluMulModel):
        return torch.ops.zentorch.zentorch_dynamic_qlinear.default
    return torch.ops.zentorch.zentorch_linear_unary.default


MODELS = [
    pytest.param(GateUpSiluMulModel, {}, id="bf16"),
    pytest.param(
        QGateUpSiluMulModel, {"zentorch_op_name": DA8W8_OP_NAME}, id="qlinear-w8a8-call"
    ),
    pytest.param(
        QGateUpSiluMulModel, {"zentorch_op_name": None}, id="qlinear-wna16-call"
    ),
]


@pytest.mark.parametrize("num_tokens", [7, 256])
@pytest.mark.parametrize("hidden_size,intermediate_size", [(256, 128)])
@pytest.mark.parametrize("model_class,model_kwargs", MODELS)
@pytest.mark.parametrize("weight_prepack", [True, False])
@pytest.mark.parametrize("enable_silu_mul_custom_op", [True, False])
def test_zentorch_gated_act_fusion(
    num_tokens: int,
    hidden_size: int,
    intermediate_size: int,
    model_class: type[torch.nn.Module],
    model_kwargs: dict,
    weight_prepack: bool,
    enable_silu_mul_custom_op: bool,
    monkeypatch: pytest.MonkeyPatch,
):
    if model_class is QGateUpSiluMulModel and weight_prepack:
        pytest.skip("VLLM_ZENTORCH_WEIGHT_PREPACK only affects bf16 linears")
    from vllm.compilation.passes.fusion.zentorch_gated_act_fusion import (
        GATED_ACT_OP,
        ZentorchGatedActFusionPass,
    )

    monkeypatch.setenv("VLLM_ZENTORCH_WEIGHT_PREPACK", "1" if weight_prepack else "0")
    torch.manual_seed(0)
    torch.set_default_dtype(torch.bfloat16)
    x = torch.rand(num_tokens, hidden_size)

    custom_ops = ["none"]
    if enable_silu_mul_custom_op:
        custom_ops.append("+silu_and_mul")
    config = VllmConfig(
        compilation_config=CompilationConfig(
            mode=CompilationMode.VLLM_COMPILE,
            custom_ops=custom_ops,
            pass_config=PassConfig(fuse_zentorch_gated_act=True, eliminate_noops=True),
        ),
    )

    with set_current_vllm_config(config):
        fusion_pass = ZentorchGatedActFusionPass(config)
        backend = TestBackend(
            NoOpEliminationPass(config), fusion_pass, PostCleanupPass(config)
        )
        model = model_class(hidden_size, intermediate_size, **model_kwargs)

        torch._dynamo.mark_dynamic(x, 0)
        result = model(x)
        result2 = torch.compile(model, backend=backend)(x)

        torch.testing.assert_close(result, result2, atol=2e-2, rtol=2e-2)
        assert fusion_pass.matched_count == 1
        backend.check_before_ops([_gate_up_op(model)])
        assert backend.op_count(GATED_ACT_OP) == 1


@pytest.mark.parametrize("weight_prepack", [True, False])
@pytest.mark.parametrize("enable_fusion", [True, False])
def test_zen_cpu_platform_installs_pass(
    weight_prepack: bool, enable_fusion: bool, monkeypatch: pytest.MonkeyPatch
):
    """The CPU platform compiles in DYNAMO_TRACE_ONCE mode with plain Inductor,
    so the pass must reach Inductor through inductor_compile_config."""
    from torch._inductor.utils import run_and_get_code

    from vllm.compilation.passes.fusion.zentorch_gated_act_fusion import (
        ZenCpuPostGradPass,
    )

    monkeypatch.setenv("VLLM_ZENTORCH_WEIGHT_PREPACK", "1" if weight_prepack else "0")
    torch.manual_seed(0)
    torch.set_default_dtype(torch.bfloat16)

    config = VllmConfig(
        compilation_config=CompilationConfig(
            pass_config=PassConfig(fuse_zentorch_gated_act=enable_fusion)
        )
    )
    compilation_config = config.compilation_config
    assert compilation_config.mode == CompilationMode.DYNAMO_TRACE_ONCE
    post_pass = compilation_config.inductor_compile_config.get(
        "post_grad_custom_post_pass"
    )
    if enable_fusion:
        assert isinstance(post_pass, ZenCpuPostGradPass)
        # The config is pickled to the engine and worker processes.
        restored = pickle.loads(pickle.dumps(post_pass))
        assert restored.uuid() == post_pass.uuid()
    else:
        assert post_pass is None

    with set_current_vllm_config(config):
        model = GateUpSiluMulModel(256, 128)
    x = torch.rand(16, 256)
    result = model(x)

    compiled = torch.compile(
        model,
        backend="inductor",
        options=dict(compilation_config.inductor_compile_config),
    )
    # A cache hit would skip the pass, so force a fresh Inductor compile.
    with torch._inductor.config.patch(force_disable_caches=True):
        result2, codes = run_and_get_code(compiled, x)

    torch.testing.assert_close(result, result2, atol=2e-2, rtol=2e-2)
    fused = any("zentorch_linear_gated_act" in code for code in codes)
    assert fused == enable_fusion
    if enable_fusion:
        assert post_pass._passes[0].matched_count == 1
