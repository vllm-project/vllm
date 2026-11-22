# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MiniMax support in the AITER Triton MXFP4 W4A16 MoE backend.

MiniMax-M2 and MiniMax-M3 route with an ungrouped sigmoid router and a
per-expert correction bias, which `get_routing_method_type` classifies as
`RoutingMethodType.MiniMax2`. MiniMax-M3 also stores its fused gate/up weights
packed ([all gates; all ups]) and declares `SWIGLUOAI_UNINTERLEAVE`.
`AiterW4A16ExpertsMonolithic` rejected both, and on gfx942 it is the only
MXFP4 kernel of the `aiter` backend whose device gate accepts the card, so
`--moe-backend aiter` could not serve these checkpoints there.

The kernel already runs both: aiter's flat top-k is that router
(`score_mode="sigmoid"` plus a bias), and the oracle interleaves gate/up for
this backend when it loads a packed checkpoint, which leaves the SwiGLU-OAI
epilogue the backend runs for gpt-oss. This module covers the gates and the
result: a packed layer through the oracle's conversion and the kernel against
the layer computed in fp32 from the same dequantized weights.
"""

import types

import pytest
import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm._aiter_ops import is_aiter_found_and_supported, rocm_aiter_ops
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    RoutingMethodType,
    get_routing_method_type,
)
from vllm.model_executor.layers.fused_moe.experts.aiter_mxfp4_w4a16_moe import (
    AiterW4A16ExpertsMonolithic,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import kMxfp4Static

EXPERTS, TOP_K, HIDDEN, INTERMEDIATE = 32, 4, 512, 384
# MiniMax-M3 / M3.1-Flash: swiglu_alpha, swiglu_beta, swiglu_limit.
ALPHA, BETA, LIMIT = 1.702, 1.0, 7.0
FP4_VALUES = [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6]

requires_aiter = pytest.mark.skipif(
    not is_aiter_found_and_supported(),
    reason="only runs on ROCm with a supported AITER install",
)


def minimax_routing_method() -> RoutingMethodType:
    """MiniMax: sigmoid + correction bias, renormalized, no expert groups."""
    return get_routing_method_type(
        scoring_func="sigmoid",
        top_k=TOP_K,
        renormalize=True,
        num_expert_group=None,
        has_e_score_bias=True,
    )


def minimax_moe_config(activation: MoEActivation) -> FusedMoEConfig:
    return FusedMoEConfig(
        num_experts=EXPERTS,
        experts_per_token=TOP_K,
        hidden_dim=HIDDEN,
        intermediate_size=INTERMEDIATE,
        num_local_experts=EXPERTS,
        num_logical_experts=EXPERTS,
        moe_parallel_config=FusedMoEParallelConfig.make_no_parallel(),
        activation=activation,
        in_dtype=torch.bfloat16,
        device=torch.device("cuda"),
        routing_method=minimax_routing_method(),
        router_logits_dtype=torch.float32,
    )


def test_ungrouped_sigmoid_bias_router_is_minimax2() -> None:
    """Guard the premise: MiniMax's router is MiniMax2-classified."""
    assert minimax_routing_method() == RoutingMethodType.MiniMax2


def test_aiter_w4a16_gates_accept_minimax() -> None:
    """The two predicates that rejected MiniMax; runs on any host."""
    assert AiterW4A16ExpertsMonolithic._supports_routing_method(
        RoutingMethodType.MiniMax2, kMxfp4Static, None
    )
    assert AiterW4A16ExpertsMonolithic._supports_activation(
        MoEActivation.SWIGLUOAI_UNINTERLEAVE
    )


@requires_aiter
@pytest.mark.parametrize(
    "activation",
    [MoEActivation.SWIGLUOAI_UNINTERLEAVE, MoEActivation.SILU],
    ids=["minimax_m3", "minimax_m2"],
)
def test_aiter_w4a16_supports_minimax_config(
    activation: MoEActivation, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The same gates through the real config path, device check included."""
    monkeypatch.setenv("VLLM_ROCM_USE_AITER", "1")
    rocm_aiter_ops.refresh_env_variables()

    supported, reason = AiterW4A16ExpertsMonolithic.is_supported_config(
        AiterW4A16ExpertsMonolithic,
        minimax_moe_config(activation),
        kMxfp4Static,
        None,
        mk.FusedMoEActivationFormat.Standard,
    )

    assert supported, reason


def _quantize_mxfp4(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """[E, N, K] bf16 to packed MXFP4 [E, N, K/2] and e8m0 scales [E, N, K/32]."""
    from aiter.ops.triton.quant import dynamic_mxfp4_quant

    packed, scales = zip(*(dynamic_mxfp4_quant(expert) for expert in weight))
    return torch.stack(packed).view(torch.uint8), torch.stack(scales).view(torch.uint8)


def _dequantize_mxfp4(packed: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    nibbles = torch.stack((packed & 0xF, packed >> 4), dim=-1).flatten(-2).long()
    values = torch.tensor(FP4_VALUES, device=packed.device)[nibbles]
    exponent = (scale.to(torch.int32) << 23).view(torch.float32)
    return values * exponent.repeat_interleave(32, dim=-1)


@requires_aiter
@pytest.mark.parametrize("routed_scaling_factor", [1.0, 2.0])
@pytest.mark.parametrize("num_tokens", [1, 37])
@torch.inference_mode()
def test_aiter_w4a16_matches_minimax_m3_reference(
    num_tokens: int, routed_scaling_factor: float, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A packed MiniMax-M3 layer through the oracle and the kernel equals the
    layer in fp32: flat sigmoid top-k on the biased score, weights from the
    unbiased score, SwiGLU-OAI on [all gates; all ups].

    The up half is three times the gate half, so reading the packed weights
    as interleaved, dropping the `(up + 1)` residual, or routing by softmax
    each move the output far outside the tolerance.
    """
    import vllm.distributed.parallel_state as ps
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import (
        Mxfp4MoeBackend,
        convert_weight_to_mxfp4_moe_kernel_format,
        make_mxfp4_moe_kernel,
        make_mxfp4_moe_quant_config,
    )
    from vllm.v1.worker.workspace import init_workspace_manager

    init_workspace_manager(torch.accelerator.current_device_index())
    monkeypatch.setattr(ps, "_TP", types.SimpleNamespace(world_size=1))
    monkeypatch.setattr(rocm_aiter_ops, "_AITER_ENABLED", True)

    torch.manual_seed(0)
    device, dtype = "cuda", torch.bfloat16
    w13 = torch.randn(EXPERTS, 2 * INTERMEDIATE, HIDDEN, dtype=dtype, device=device)
    w13 *= 0.05
    w13[:, INTERMEDIATE:] *= 3.0
    w2 = torch.randn(EXPERTS, HIDDEN, INTERMEDIATE, dtype=dtype, device=device) * 0.05
    w13_packed, w13_scale = _quantize_mxfp4(w13)
    w2_packed, w2_scale = _quantize_mxfp4(w2)
    x = torch.randn(num_tokens, HIDDEN, dtype=dtype, device=device)
    logits = torch.randn(num_tokens, EXPERTS, device=device) * 2.0
    bias = torch.randn(EXPERTS, device=device) * 0.5

    scores = torch.sigmoid(logits)
    ids = (scores + bias).topk(TOP_K, dim=-1).indices
    weights = scores.gather(1, ids)
    weights = weights / weights.sum(dim=-1, keepdim=True) * routed_scaling_factor
    w13_ref = _dequantize_mxfp4(w13_packed, w13_scale)
    w2_ref = _dequantize_mxfp4(w2_packed, w2_scale)
    gate, up = torch.einsum("bsnk,bk->bsn", w13_ref[ids], x.float()).chunk(2, dim=-1)
    gate, up = gate.clamp(max=LIMIT), up.clamp(min=-LIMIT, max=LIMIT)
    act = gate * torch.sigmoid(ALPHA * gate) * (up + BETA)
    ref = torch.einsum("bshn,bsn,bs->bh", w2_ref[ids], act, weights)

    backend = Mxfp4MoeBackend.AITER_TRITON_MXFP4_BF16
    activation = MoEActivation.SWIGLUOAI_UNINTERLEAVE
    w13_conv, w2_conv, w13_precision, w2_precision, _, _ = (
        convert_weight_to_mxfp4_moe_kernel_format(
            mxfp4_backend=backend,
            layer=types.SimpleNamespace(activation=activation),
            w13_weight=w13_packed,
            w2_weight=w2_packed,
            w13_weight_scale=w13_scale,
            w2_weight_scale=w2_scale,
            activation=activation,
        )
    )
    quant_config = make_mxfp4_moe_quant_config(
        mxfp4_backend=backend,
        w1_scale=w13_precision,
        w2_scale=w2_precision,
        gemm1_alpha=ALPHA,
        gemm1_beta=BETA,
        swiglu_limit=LIMIT,
    )
    assert quant_config is not None
    with set_current_vllm_config(VllmConfig()):
        kernel = make_mxfp4_moe_kernel(
            moe_quant_config=quant_config,
            moe_config=minimax_moe_config(activation),
            experts_cls=AiterW4A16ExpertsMonolithic,
            mxfp4_backend=backend,
        )
        out = kernel.apply_monolithic(
            hidden_states=x,
            w1=w13_conv,
            w2=w2_conv,
            router_logits=logits,
            activation=activation,
            global_num_experts=EXPERTS,
            expert_map=None,
            apply_router_weight_on_input=False,
            e_score_correction_bias=bias,
            routed_scaling_factor=routed_scaling_factor,
            routing_sink=None,
        )

    assert isinstance(out, torch.Tensor)
    relative_error = (out.float() - ref).norm() / ref.norm()
    assert relative_error < 0.02, f"relative L2 error {relative_error:.4f}"
