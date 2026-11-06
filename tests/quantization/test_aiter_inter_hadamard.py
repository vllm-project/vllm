# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Rotated Quark MXFP4 MoE on AITER's fused W4A4 path: the hidden states are
rotated in vLLM and the SwiGLU output inside AITER (`inter_hadamard`)."""

import inspect

import pytest
import torch

from vllm._aiter_ops import rocm_aiter_ops
from vllm.config.model import ModelConfig
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    RoutingMethodType,
)
from vllm.model_executor.layers.fused_moe.experts.ocp_mx_emulation_moe import (
    InputHadamardOCPMXEmulationExperts,
)
from vllm.model_executor.layers.fused_moe.experts.rocm_aiter_moe import (
    AiterMxfp4Experts,
)
from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import (
    Mxfp4MoeBackend,
    convert_gpt_oss_weight_to_mxfp4_moe_kernel_format,
    make_mxfp4_moe_kernel,
    make_mxfp4_moe_quant_config,
)
from vllm.model_executor.layers.quantization.input_hadamard import (
    block_hadamard_rotate,
    hadamard_matrix,
)
from vllm.model_executor.layers.quantization.quark.quark_moe import (
    QuarkOCP_MX_MoEMethod,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kMxfp4Dynamic,
    kMxfp4Static,
)
from vllm.platforms import current_platform
from vllm.v1.worker.workspace import init_workspace_manager

from .test_moe_input_hadamard import _bf16_reference, _quantize_like_quark, _rel

hadamard = pytest.importorskip("aiter.ops.hadamard")


def _on_gfx950() -> bool:
    if not current_platform.is_rocm():
        return False
    from vllm.platforms.rocm import on_gfx950

    return on_gfx950()


@pytest.fixture
def aiter_moe_enabled(monkeypatch):
    monkeypatch.setenv("VLLM_ROCM_USE_AITER", "1")
    monkeypatch.setenv("VLLM_ROCM_USE_AITER_MOE", "1")
    rocm_aiter_ops.refresh_env_variables()
    yield
    monkeypatch.undo()
    rocm_aiter_ops.refresh_env_variables()


def _moe_config(num_experts: int, hidden: int, intermediate: int) -> FusedMoEConfig:
    return FusedMoEConfig(
        num_experts=num_experts,
        experts_per_token=1,
        hidden_dim=hidden,
        intermediate_size=intermediate,
        num_local_experts=num_experts,
        num_logical_experts=num_experts,
        activation=MoEActivation.SWIGLUOAI_UNINTERLEAVE,
        device=current_platform.device_type,
        routing_method=RoutingMethodType.Renormalize,
        moe_parallel_config=FusedMoEParallelConfig.make_no_parallel(),
        in_dtype=torch.bfloat16,
        swiglu_limit=7.0,
        swiglu_alpha=1.702,
        swiglu_beta=1.0,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
def test_aiter_block_hadamard_matches_vllm():
    x = torch.randn(3, 256, device="cuda", dtype=torch.bfloat16)
    got = hadamard.block_hadamard_rotate(x, 128)
    ref = block_hadamard_rotate(x, hadamard_matrix(128, x.device))
    torch.testing.assert_close(got, ref)


def test_second_gemm_rotation_preserves_product():
    """(y H)(w H)^T equals y w^T. w is the second GEMM, y is the SwiGLU output."""
    y = torch.randn(4, 256, dtype=torch.float32)
    w = torch.randn(64, 256, dtype=torch.float32)
    h = hadamard_matrix(128, y.device)
    y_rot = block_hadamard_rotate(y, h)
    w_rot = block_hadamard_rotate(w, h)
    torch.testing.assert_close(y_rot @ w_rot.T, y @ w.T, atol=1e-4, rtol=1e-4)


def test_inter_hadamard_reaches_fused_moe_2stages():
    import aiter.fused_moe as fused_moe

    for fn in (
        fused_moe.fused_moe,
        fused_moe.fused_moe_fake,
        fused_moe.fused_moe_2stages,
    ):
        assert "inter_hadamard" in inspect.signature(fn).parameters, fn.__name__
    assert "inter_hadamard=inter_hadamard" in inspect.getsource(fused_moe.fused_moe)
    source = inspect.getsource(fused_moe.fused_moe_2stages)
    rotate_at = source.index("block_hadamard_rotate")
    assert source.index("a2 = _stage1_call()") < rotate_at
    assert source.index("fused_dynamic_mxfp4_quant_moe_sort", rotate_at) > rotate_at
    assert rocm_aiter_ops.fused_moe_supports_inter_hadamard()


@pytest.mark.skipif(not _on_gfx950(), reason="AITER MXFP4 W4A4 needs gfx950.")
def test_rotated_quark_moe_keeps_aiter_w4a4(default_vllm_config, aiter_moe_enabled):
    default_vllm_config.model_config = ModelConfig()
    method = QuarkOCP_MX_MoEMethod(
        _moe_config(4, 256, 256), kMxfp4Static, kMxfp4Dynamic, rotation_size=128
    )
    assert method.mxfp4_backend is Mxfp4MoeBackend.AITER_MXFP4_MXFP4
    assert method.experts_cls is AiterMxfp4Experts


def test_rotated_quark_moe_without_aiter_uses_emulation(
    default_vllm_config, monkeypatch
):
    monkeypatch.setenv("VLLM_ROCM_USE_AITER", "0")
    rocm_aiter_ops.refresh_env_variables()
    default_vllm_config.model_config = ModelConfig()
    method = QuarkOCP_MX_MoEMethod(
        _moe_config(4, 256, 256), kMxfp4Static, kMxfp4Dynamic, rotation_size=128
    )
    assert method.mxfp4_backend is Mxfp4MoeBackend.EMULATION
    assert method.experts_cls is InputHadamardOCPMXEmulationExperts


@pytest.mark.skipif(not _on_gfx950(), reason="AITER MXFP4 W4A4 needs gfx950.")
@pytest.mark.parametrize("tokens", [8, 512])
def test_rotated_aiter_mxfp4_moe_matches_rotated_bf16(
    default_vllm_config, aiter_moe_enabled, tokens
):
    """Fewer than 256 tokens run AITER's bf16-activation stage, more run the
    MXFP4 activation stage, so both places that consume the rotated SwiGLU
    output are covered."""
    default_vllm_config.model_config = ModelConfig()
    torch.manual_seed(0)
    device = torch.device(current_platform.device_type)
    num_experts, hidden, intermediate = 4, 512, 256
    backend = Mxfp4MoeBackend.AITER_MXFP4_MXFP4
    h = hadamard_matrix(128, device)
    with torch.device(device):
        x = torch.randn(tokens, hidden, dtype=torch.bfloat16)
        w13 = torch.randn(num_experts, 2 * intermediate, hidden, dtype=torch.bfloat16)
        w2 = torch.randn(num_experts, hidden, intermediate, dtype=torch.bfloat16)
        w13_rot = block_hadamard_rotate(w13 * 0.02, h)
        w2_rot = block_hadamard_rotate(w2 * 0.02, h)
        w13_q, w13_s = _quantize_like_quark(w13_rot)
        w2_q, w2_s = _quantize_like_quark(w2_rot)
        topk_ids = torch.arange(tokens, dtype=torch.int32).remainder(num_experts)
        topk_ids = topk_ids[:, None]
        topk_weights = torch.ones(tokens, 1, dtype=torch.float32)

    layer = torch.nn.Module()
    layer.w13_weight = torch.nn.Parameter(w13_q, requires_grad=False)
    layer.w2_weight = torch.nn.Parameter(w2_q, requires_grad=False)
    w13_k, w2_k, s13, s2, _, _ = convert_gpt_oss_weight_to_mxfp4_moe_kernel_format(
        mxfp4_backend=backend,
        layer=layer,
        w13_weight=layer.w13_weight,
        w2_weight=layer.w2_weight,
        w13_weight_scale=w13_s,
        w2_weight_scale=w2_s,
    )
    quant = make_mxfp4_moe_quant_config(
        mxfp4_backend=backend,
        w1_scale=s13,
        w2_scale=s2,
        gemm1_alpha=1.702,
        gemm1_beta=1.0,
        swiglu_limit=7.0,
    )
    quant.input_rotation_size = 128
    init_workspace_manager(device)
    kernel = make_mxfp4_moe_kernel(
        quant,
        _moe_config(num_experts, hidden, intermediate),
        AiterMxfp4Experts,
        backend,
    )
    got = kernel.apply(
        hidden_states=x,
        w1=w13_k,
        w2=w2_k,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        activation=MoEActivation.SWIGLUOAI_UNINTERLEAVE,
        global_num_experts=num_experts,
        expert_map=None,
        apply_router_weight_on_input=False,
    )

    rotated_ref = _bf16_reference(x, w13_rot, w2_rot, topk_ids, rotate=True)
    skipped = _bf16_reference(x, w13_rot, w2_rot, topk_ids, rotate=False)
    assert _rel(got, rotated_ref) < 0.35
    assert _rel(got, rotated_ref) < 0.5 * _rel(got, skipped)
