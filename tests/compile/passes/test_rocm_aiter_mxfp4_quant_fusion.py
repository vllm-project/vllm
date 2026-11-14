# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the MXFP4 activation quant folds in the ROCm AITER fusion passes:
SiLU-mul + mxfp4 quant (bitwise equal to the compiled native silu_and_mul
followed by the quant) and RMSNorm / fused_add_rms_norm + mxfp4 quant. The
first runs with fuse_act_quant, the second needs fuse_norm_quant plus the
opt-in fuse_norm_mxfp4_quant. The SiLU-mul fold also covers linears on the ASM
layout (HIP quant, shuffled scales), bitwise equal there too.
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


class SiluMulMxfp4AsmModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.silu_and_mul = SiluAndMul()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        y = self.silu_and_mul(x)
        q, s = rocm_aiter_ops.get_dynamic_mxfp4_asm_quant_op()(y)
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


def _setup(monkeypatch: pytest.MonkeyPatch) -> None:
    if not current_platform.supports_mx():
        pytest.skip("MXFP4 is not supported on this GPU")
    monkeypatch.setenv("VLLM_ROCM_USE_AITER", "1")
    rocm_aiter_ops.refresh_env_variables()
    if not rocm_aiter_ops._fused_mxfp4_quant_supported():
        pytest.skip("installed AITER lacks round_to_input_dtype / transpose_scale")
    torch._dynamo.reset()
    torch.set_default_device("cuda")
    torch.set_default_dtype(torch.bfloat16)


def _config(custom_ops, mode=CompilationMode.VLLM_COMPILE, **pass_flags):
    return VllmConfig(
        model_config=ModelConfig(dtype=torch.bfloat16),
        compilation_config=CompilationConfig(
            mode=mode,
            custom_ops=custom_ops,
            pass_config=PassConfig(eliminate_noops=True, **pass_flags),
        ),
    )


def _compile(config, make_passes, make_model, *args):
    with set_current_vllm_config(config):
        fusion_passes = make_passes(config)
        backend = TestBackend(
            NoOpEliminationPass(config), *fusion_passes, PostCleanupPass(config)
        )
        # custom ops pick native vs custom forward at init, so build it here
        model = make_model()
        out = torch.compile(model, backend=backend)(*args)
    return out, fusion_passes, backend


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

    def make_passes(config):
        # PostGradPassManager only adds this pass under fuse_act_quant
        if not config.compilation_config.pass_config.fuse_act_quant:
            return []
        return [RocmAiterSiluMulFp8GroupQuantFusionPass(config)]

    custom_ops = ["none"] + (["+silu_and_mul"] if enable_silu_mul_custom_op else [])
    quant_op = rocm_aiter_ops.get_dynamic_mxfp4_quant_op()
    fused_op = rocm_aiter_ops.get_act_mul_fused_mxfp4_quant_op()
    torch.manual_seed(0)
    x = torch.randn(num_tokens, 2 * hidden_size, dtype=torch.bfloat16, device="cuda")

    results = {}
    for enabled in (False, True):
        with monkeypatch.context() as m:
            _setup(m)
            xi = x.clone()
            torch._dynamo.mark_dynamic(xi, 0)
            config = _config(custom_ops, fuse_act_quant=enabled, fuse_norm_quant=False)
            results[enabled], fusion_passes, backend = _compile(
                config, make_passes, SiluMulMxfp4Model, xi
            )
            if enabled:
                assert len(fusion_passes) == 1
                assert fusion_passes[0].matched_count == 1
                backend.check_after_ops([fused_op])
                backend.check_before_ops([quant_op])
            else:
                assert not fusion_passes
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


def _asm_scales(s: torch.Tensor, m: int, n: int) -> torch.Tensor:
    # the (m, n // 32) e8m0 scales out of the ASM layout buffer: row-major in
    # the first m rows below 32 rows, the aiter shuffle_scale layout from 32
    s = s.view(torch.uint8)
    if m >= 32:
        sm, sn = s.shape
        s = s.view(sm // 32, sn // 8, 4, 16, 2, 2).permute(0, 5, 3, 1, 4, 2)
        s = s.reshape(sm, sn)
    return s[:m, : n // 32]


@pytest.mark.parametrize("num_tokens", [1, 7, 31, 32, 64, 65, 257])
@pytest.mark.parametrize("hidden_size", [256, 2048])
@pytest.mark.parametrize("enable_silu_mul_custom_op", [True, False])
def test_silu_mul_mxfp4_asm_quant_fusion(
    num_tokens: int,
    hidden_size: int,
    enable_silu_mul_custom_op: bool,
    monkeypatch: pytest.MonkeyPatch,
):
    from vllm.compilation.passes.fusion.rocm_aiter_fusion import (
        RocmAiterSiluMulFp8GroupQuantFusionPass,
    )

    def make_passes(config):
        if not config.compilation_config.pass_config.fuse_act_quant:
            return []
        return [RocmAiterSiluMulFp8GroupQuantFusionPass(config)]

    custom_ops = ["none"] + (["+silu_and_mul"] if enable_silu_mul_custom_op else [])
    quant_op = rocm_aiter_ops.get_dynamic_mxfp4_asm_quant_op()
    fused_op = rocm_aiter_ops.get_act_mul_fused_mxfp4_asm_quant_op()
    torch.manual_seed(0)
    x = torch.randn(num_tokens, 2 * hidden_size, dtype=torch.bfloat16, device="cuda")

    results = {}
    for enabled in (False, True):
        with monkeypatch.context() as m:
            _setup(m)
            if not rocm_aiter_ops.is_mxfp4_asm_quant_fusion_supported():
                pytest.skip("installed AITER lacks scaling_mode='ceil'")
            xi = x.clone()
            torch._dynamo.mark_dynamic(xi, 0)
            config = _config(custom_ops, fuse_act_quant=enabled, fuse_norm_quant=False)
            results[enabled], fusion_passes, backend = _compile(
                config, make_passes, SiluMulMxfp4AsmModel, xi
            )
            if enabled:
                assert fusion_passes[0].matched_count == 1
                backend.check_after_ops([fused_op])
                backend.check_before_ops([quant_op])
            else:
                assert backend.op_count(fused_op) == 0
                assert backend.op_count(quant_op) == 1

    (q_ref, s_ref), (q, s) = results[False], results[True]
    # same padded buffer as per_1x32_f4_quant_hip(shuffle=True), which is what
    # gemm_with_dynamic_quant hands the preshuffle / a4w4 GEMM
    padded = ((num_tokens + 255) // 256 * 256, hidden_size // 32)
    assert s.shape == s_ref.shape == padded
    assert q.dtype == q_ref.dtype and s.dtype == s_ref.dtype
    s, s_ref = (_asm_scales(t, num_tokens, hidden_size) for t in (s, s_ref))
    q, q_ref = q.view(torch.uint8), q_ref.view(torch.uint8)
    if enable_silu_mul_custom_op:
        assert (q == q_ref).float().mean().item() > 0.95
        assert (s == s_ref).float().mean().item() > 0.95
    else:
        # scaling_mode="ceil" is the HIP quant's RoundUp scale, so against the
        # compiled native silu_and_mul + HIP quant not a byte moves
        assert torch.equal(q, q_ref)
        assert torch.equal(s, s_ref)


@pytest.mark.parametrize("num_tokens", [1, 16, 32, 64, 65, 256])
def test_mxfp4_asm_linear_split_quant_matches_single_op(
    num_tokens: int, monkeypatch: pytest.MonkeyPatch
):
    # with the fold on, an ASM-layout linear runs the HIP quant as its own op
    # and hands the GEMM pre-quantized input, that must not change the output
    from torch.nn.parameter import Parameter

    from vllm.model_executor.kernels.linear.mxfp4.aiter import (
        AiterMxfp4LinearKernel,
    )
    from vllm.model_executor.kernels.linear.mxfp4.base import (
        MxFp4LinearLayerConfig,
    )
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        kMxfp4Dynamic,
    )

    _setup(monkeypatch)
    if not rocm_aiter_ops.is_mxfp4_asm_quant_fusion_supported():
        pytest.skip("installed AITER lacks scaling_mode='ceil'")
    if not rocm_aiter_ops.is_asm_fp4_gemm_supported():
        pytest.skip("ASM FP4 GEMM is gfx950 only")
    monkeypatch.setattr(rocm_aiter_ops, "is_asm_fp4_gemm_tuned", lambda n, k: True)
    monkeypatch.setattr(
        rocm_aiter_ops, "is_triton_gemm_afp4wfp4_presh_ws_tuned", lambda n, k: True
    )
    from aiter import per_1x32_f4_quant_hip

    # Llama-3.1-8B down_proj, aiter ships preshuffle and a4w4 configs for it
    n, k = 4096, 14336
    torch.manual_seed(0)
    w, ws = per_1x32_f4_quant_hip(torch.randn(n, k) * 0.05, shuffle=False)
    x = torch.randn(num_tokens, k) * 2

    outs = {}
    for mode in (CompilationMode.NONE, CompilationMode.VLLM_COMPILE):
        config = _config(["none"], mode=mode, fuse_act_quant=True)
        with set_current_vllm_config(config):
            kernel = AiterMxfp4LinearKernel(
                MxFp4LinearLayerConfig(activation_quant_key=kMxfp4Dynamic)
            )
        layer = torch.nn.Module()
        layer.weight = Parameter(w.view(torch.uint8).clone(), requires_grad=False)
        layer.weight_scale = Parameter(
            ws.view(torch.uint8)[:n].clone(), requires_grad=False
        )
        kernel.process_weights_after_loading(layer)
        assert kernel.use_asm_gemm
        assert not kernel.split_act_quant
        assert kernel.split_asm_act_quant is (mode == CompilationMode.VLLM_COMPILE)
        outs[mode] = kernel.apply_weights(layer, x)
    assert torch.equal(outs[CompilationMode.NONE], outs[CompilationMode.VLLM_COMPILE])


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
            _setup(m)
            xi, ri = x.clone(), r.clone()
            torch._dynamo.mark_dynamic(xi, 0)
            torch._dynamo.mark_dynamic(ri, 0)
            # fuse_norm_quant alone must leave the mxfp4 quant alone
            config = _config(
                ["none", "+rms_norm"],
                fuse_act_quant=False,
                fuse_norm_quant=True,
                fuse_norm_mxfp4_quant=enabled,
            )
            results[enabled], (fusion_pass,), backend = _compile(
                config,
                lambda c: [RocmAiterRMSNormQuantFusionPass(c)],
                make_model,
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


@pytest.mark.parametrize(
    "mode, flags, expected, expected_asm",
    [
        (CompilationMode.VLLM_COMPILE, {"fuse_act_quant": True}, True, True),
        (CompilationMode.VLLM_COMPILE, {"fuse_norm_quant": True}, False, False),
        (
            CompilationMode.VLLM_COMPILE,
            {"fuse_norm_quant": True, "fuse_norm_mxfp4_quant": True},
            True,
            False,
        ),
        (CompilationMode.VLLM_COMPILE, {}, False, False),
        (CompilationMode.NONE, {"fuse_act_quant": True}, False, False),
    ],
)
def test_mxfp4_linear_splits_quant_only_for_a_fold(
    mode, flags, expected, expected_asm, monkeypatch: pytest.MonkeyPatch
):
    from vllm.model_executor.kernels.linear.mxfp4.aiter import (
        AiterMxfp4LinearKernel,
    )
    from vllm.model_executor.kernels.linear.mxfp4.base import (
        MxFp4LinearLayerConfig,
    )
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        kMxfp4Dynamic,
    )

    _setup(monkeypatch)
    pass_flags = {"fuse_act_quant": False, "fuse_norm_quant": False, **flags}
    config = _config(["none"], mode=mode, **pass_flags)
    with set_current_vllm_config(config):
        kernel = AiterMxfp4LinearKernel(
            MxFp4LinearLayerConfig(activation_quant_key=kMxfp4Dynamic)
        )
    assert kernel.fold_act_quant is expected
    assert kernel.fold_asm_act_quant is expected_asm
