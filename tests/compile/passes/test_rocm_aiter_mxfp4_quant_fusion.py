# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the MXFP4 activation quant folds in the ROCm AITER fusion passes:
SiLU-mul + mxfp4 quant (bitwise equal to the compiled native silu_and_mul
followed by the quant) and RMSNorm /
fused_add_rms_norm + mxfp4 quant. The first is on by default
(VLLM_ROCM_USE_AITER_MXFP4_SILU_QUANT_FUSION), the second is opt-in
(VLLM_ROCM_USE_AITER_MXFP4_RMSNORM_QUANT_FUSION).
"""

import pytest
import torch

from tests.compile.backend import TestBackend
from vllm._aiter_ops import IS_AITER_FOUND, rocm_aiter_ops
from vllm.compilation.passes.utility.noop_elimination import NoOpEliminationPass
from vllm.compilation.passes.utility.post_cleanup import PostCleanupPass
from vllm.config import (
    CompilationConfig,
    CompilationMode,
    ModelConfig,
    PassConfig,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.platforms import current_platform

EPS = 1e-6

pytestmark = pytest.mark.skipif(
    not (current_platform.is_rocm() and IS_AITER_FOUND),
    reason="ROCm AITER only",
)


class SiluMulMxfp4Model(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.silu_and_mul = SiluAndMul()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        y = self.silu_and_mul(x)
        q, s = rocm_aiter_ops.get_dynamic_mxfp4_quant_op()(y)
        return q, s


class RMSNormMxfp4Model(torch.nn.Module):
    def __init__(self, hidden_size: int, residual: bool) -> None:
        super().__init__()
        self.residual = residual
        self.weight = torch.nn.Parameter(torch.rand(hidden_size) + 0.5)

    def forward(self, x: torch.Tensor, r: torch.Tensor):
        if self.residual:
            y, r = torch.ops.vllm_ir.fused_add_rms_norm(x, r, self.weight, EPS)
        else:
            y = torch.ops.vllm_ir.rms_norm(x, self.weight, EPS)
        q, s = rocm_aiter_ops.get_dynamic_mxfp4_quant_op()(y)
        return q, s, r


def _setup(monkeypatch: pytest.MonkeyPatch, env: str, enabled: bool) -> None:
    if not current_platform.supports_mx():
        pytest.skip("MXFP4 is not supported on this GPU")
    monkeypatch.setenv("VLLM_ROCM_USE_AITER", "1")
    monkeypatch.setenv(env, "1" if enabled else "0")
    rocm_aiter_ops.refresh_env_variables()
    if not rocm_aiter_ops._fused_mxfp4_quant_supported():
        pytest.skip("installed AITER lacks round_to_input_dtype / transpose_scale")
    torch._dynamo.reset()
    torch.set_default_device("cuda")
    torch.set_default_dtype(torch.bfloat16)


def _compile(make_model, pass_cls, custom_ops, *args):
    config = VllmConfig(
        model_config=ModelConfig(dtype=torch.bfloat16),
        compilation_config=CompilationConfig(
            mode=CompilationMode.VLLM_COMPILE,
            custom_ops=custom_ops,
            pass_config=PassConfig(
                fuse_act_quant=True, fuse_norm_quant=True, eliminate_noops=True
            ),
        ),
    )
    with set_current_vllm_config(config):
        fusion_pass = pass_cls(config)
        backend = TestBackend(
            NoOpEliminationPass(config), fusion_pass, PostCleanupPass(config)
        )
        # custom ops pick native vs custom forward at init, so build it here
        model = make_model()
        out = torch.compile(model, backend=backend)(*args)
    return out, fusion_pass, backend


@pytest.mark.parametrize("num_tokens", [1, 7, 64, 257])
@pytest.mark.parametrize("hidden_size", [256, 2048])
@pytest.mark.parametrize("enable_silu_mul_custom_op", [True, False])
def test_silu_mul_mxfp4_quant_fusion(
    num_tokens: int,
    hidden_size: int,
    enable_silu_mul_custom_op: bool,
    monkeypatch: pytest.MonkeyPatch,
):
    from vllm.compilation.passes.fusion.rocm_aiter_fusion import (
        RocmAiterSiluMulFp8GroupQuantFusionPass,
    )

    env = "VLLM_ROCM_USE_AITER_MXFP4_SILU_QUANT_FUSION"
    custom_ops = ["none"] + (["+silu_and_mul"] if enable_silu_mul_custom_op else [])
    quant_op = rocm_aiter_ops.get_dynamic_mxfp4_quant_op()
    fused_op = rocm_aiter_ops.get_act_mul_fused_mxfp4_quant_op()
    torch.manual_seed(0)
    x = torch.randn(num_tokens, 2 * hidden_size, dtype=torch.bfloat16, device="cuda")

    results = {}
    for enabled in (False, True):
        with monkeypatch.context() as m:
            _setup(m, env, enabled)
            xi = x.clone()
            torch._dynamo.mark_dynamic(xi, 0)
            results[enabled], fusion_pass, backend = _compile(
                SiluMulMxfp4Model,
                RocmAiterSiluMulFp8GroupQuantFusionPass,
                custom_ops,
                xi,
            )
            if enabled:
                assert fusion_pass.matched_count == 1
                backend.check_after_ops([fused_op])
                backend.check_before_ops([quant_op])
            else:
                assert fusion_pass.matched_count == 0
                assert backend.op_count(fused_op) == 0
                assert backend.op_count(quant_op) == 1

    (q_ref, s_ref), (q, s) = results[False], results[True]
    assert s.stride() == s_ref.stride()
    if enable_silu_mul_custom_op:
        # the _C silu_and_mul kernel rounds silu to the input dtype before the
        # multiply, the fused kernel (like the compiled native op) rounds once
        assert (q == q_ref).float().mean().item() > 0.95
        assert (s == s_ref).float().mean().item() > 0.95
    else:
        # against the compiled native silu_and_mul the fold must not change a
        # single byte of the fp4 payload or the e8m0 scales
        assert torch.equal(q, q_ref)
        assert torch.equal(s, s_ref)


@pytest.mark.parametrize("num_tokens", [1, 7, 64, 257])
@pytest.mark.parametrize("hidden_size", [256, 4096])
@pytest.mark.parametrize("residual", [True, False])
def test_rmsnorm_mxfp4_quant_fusion(
    num_tokens: int,
    hidden_size: int,
    residual: bool,
    monkeypatch: pytest.MonkeyPatch,
):
    from vllm.compilation.passes.fusion.rocm_aiter_fusion import (
        RocmAiterRMSNormQuantFusionPass,
    )

    env = "VLLM_ROCM_USE_AITER_MXFP4_RMSNORM_QUANT_FUSION"
    fused_op = (
        rocm_aiter_ops.get_fused_add_rmsnorm_mxfp4_quant_op()
        if residual
        else rocm_aiter_ops.get_rmsnorm_fused_mxfp4_quant_op()
    )
    torch.manual_seed(0)
    x = torch.randn(num_tokens, hidden_size, dtype=torch.bfloat16, device="cuda")
    r = torch.randn(num_tokens, hidden_size, dtype=torch.bfloat16, device="cuda")

    def make_model():
        # same norm weight on both runs
        torch.manual_seed(1)
        return RMSNormMxfp4Model(hidden_size, residual)

    results = {}
    for enabled in (False, True):
        with monkeypatch.context() as m:
            _setup(m, env, enabled)
            xi, ri = x.clone(), r.clone()
            torch._dynamo.mark_dynamic(xi, 0)
            torch._dynamo.mark_dynamic(ri, 0)
            results[enabled], fusion_pass, backend = _compile(
                make_model,
                RocmAiterRMSNormQuantFusionPass,
                ["none", "+rms_norm"],
                xi,
                ri,
            )
            if enabled:
                assert fusion_pass.matched_count == 1
                backend.check_after_ops([fused_op])
            else:
                assert fusion_pass.matched_count == 0
                assert backend.op_count(fused_op) == 0

    (q_ref, s_ref, r_ref), (q, s, r_out) = results[False], results[True]
    assert s.stride() == s_ref.stride()
    torch.testing.assert_close(r_out, r_ref)
    # not bitwise: the fused kernel reduces the norm in its own order, which
    # moves a small share of values across an e2m1 / e8m0 rounding boundary
    assert (q == q_ref).float().mean().item() > 0.95
    assert (s == s_ref).float().mean().item() > 0.95
