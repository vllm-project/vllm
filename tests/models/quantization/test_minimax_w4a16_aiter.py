# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MiniMax MXFP4 layers on the AITER Triton MXFP4 W4A16 MoE backend.

MiniMax-M2 and MiniMax-M3 route with an ungrouped sigmoid router and a
correction bias (`RoutingMethodType.MiniMax2`), and MiniMax-M3 stores gate/up
packed ([all gates; all ups], `SWIGLUOAI_UNINTERLEAVE`).
"""

import types

import pytest
import torch

import vllm.distributed.parallel_state as ps
from vllm._aiter_ops import is_aiter_found_and_supported, rocm_aiter_ops
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    get_routing_method_type,
)
from vllm.model_executor.layers.fused_moe.experts.aiter_mxfp4_w4a16_moe import (
    AiterW4A16ExpertsMonolithic,
)

EXPERTS, TOP_K, HIDDEN, INTERMEDIATE, TOKENS = 32, 4, 512, 384, 37
ROUTED_SCALING_FACTOR = 2.0
FP4_VALUES = [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6]


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


requires_aiter = pytest.mark.skipif(
    not is_aiter_found_and_supported(),
    reason="only runs on ROCm with a supported AITER install",
)


def _run_layer(
    monkeypatch: pytest.MonkeyPatch,
    activation: MoEActivation,
    quant_swiglu: tuple[float | None, float | None, float | None],
    model_swiglu: tuple[float | None, float | None, float | None],
    expected_swiglu: tuple[float, float, float | None],
) -> tuple[torch.Tensor, torch.Tensor]:
    """A packed MiniMax layer through the oracle's conversion and the kernel,
    and the same layer in fp32 from the dequantized weights.

    Each `*_swiglu` is (alpha, beta, clamp limit): what the quantization
    method forwards, what the model declares, and what the layer must use.
    """
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
    # The up half is three times the gate half, so a gate/up mix-up shows.
    w13[:, INTERMEDIATE:] *= 3.0
    w2 = torch.randn(EXPERTS, HIDDEN, INTERMEDIATE, dtype=dtype, device=device) * 0.05
    w13_packed, w13_scale = _quantize_mxfp4(w13)
    w2_packed, w2_scale = _quantize_mxfp4(w2)
    x = torch.randn(TOKENS, HIDDEN, dtype=dtype, device=device)
    logits = torch.randn(TOKENS, EXPERTS, device=device) * 2.0
    bias = torch.randn(EXPERTS, device=device) * 0.5

    # MiniMax's router: select on the biased score, weight by the unbiased one.
    scores = torch.sigmoid(logits)
    ids = (scores + bias).topk(TOP_K, dim=-1).indices
    weights = scores.gather(1, ids)
    weights = weights / weights.sum(dim=-1, keepdim=True) * ROUTED_SCALING_FACTOR
    alpha, beta, limit = expected_swiglu
    w13_ref = _dequantize_mxfp4(w13_packed, w13_scale)
    w2_ref = _dequantize_mxfp4(w2_packed, w2_scale)
    gate, up = torch.einsum("bsnk,bk->bsn", w13_ref[ids], x.float()).chunk(2, dim=-1)
    if limit is not None:
        gate, up = gate.clamp(max=limit), up.clamp(min=-limit, max=limit)
    act = gate * torch.sigmoid(alpha * gate) * (up + beta)
    ref = torch.einsum("bshn,bsn,bs->bh", w2_ref[ids], act, weights)

    backend = Mxfp4MoeBackend.AITER_TRITON_MXFP4_BF16
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
        gemm1_alpha=quant_swiglu[0],
        gemm1_beta=quant_swiglu[1],
        swiglu_limit=quant_swiglu[2],
    )
    assert quant_config is not None
    moe_config = FusedMoEConfig(
        num_experts=EXPERTS,
        experts_per_token=TOP_K,
        hidden_dim=HIDDEN,
        intermediate_size=INTERMEDIATE,
        num_local_experts=EXPERTS,
        num_logical_experts=EXPERTS,
        moe_parallel_config=FusedMoEParallelConfig.make_no_parallel(),
        activation=activation,
        in_dtype=dtype,
        device=torch.device("cuda"),
        routing_method=get_routing_method_type("sigmoid", TOP_K, True, None, True),
        router_logits_dtype=torch.float32,
        swiglu_alpha=model_swiglu[0],
        swiglu_beta=model_swiglu[1],
        swiglu_limit=model_swiglu[2],
    )
    with set_current_vllm_config(VllmConfig()):
        kernel = make_mxfp4_moe_kernel(
            moe_quant_config=quant_config,
            moe_config=moe_config,
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
            routed_scaling_factor=ROUTED_SCALING_FACTOR,
            routing_sink=None,
        )
    assert isinstance(out, torch.Tensor)
    return out.float(), ref


UNSET = (None, None, None)


@requires_aiter
@pytest.mark.parametrize(
    "activation,quant_swiglu,model_swiglu,expected_swiglu",
    [
        pytest.param(
            MoEActivation.SWIGLUOAI_UNINTERLEAVE,
            (1.702, 1.0, 7.0),
            UNSET,
            (1.702, 1.0, 7.0),
            id="minimax_m3",
        ),
        # A quantization method that forwards only the clamp limit.
        pytest.param(
            MoEActivation.SWIGLUOAI_UNINTERLEAVE,
            (None, None, 7.0),
            (1.702, 1.0, None),
            (1.702, 1.0, 7.0),
            id="swiglu_from_model_config",
        ),
        pytest.param(
            MoEActivation.SWIGLUOAI_UNINTERLEAVE,
            (1.702, None, 7.0),
            UNSET,
            (1.702, 0.0, 7.0),
            id="swiglu_without_residual",
        ),
        pytest.param(
            MoEActivation.SILU, UNSET, UNSET, (1.0, 0.0, None), id="minimax_m2"
        ),
    ],
)
@torch.inference_mode()
def test_aiter_w4a16_matches_minimax_reference(
    activation: MoEActivation,
    quant_swiglu: tuple[float | None, float | None, float | None],
    model_swiglu: tuple[float | None, float | None, float | None],
    expected_swiglu: tuple[float, float, float | None],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reading the packed weights as interleaved, a wrong alpha, a dropped or
    added `(up + 1)` residual, or routing by softmax each move the output far
    outside the tolerance."""
    out, ref = _run_layer(
        monkeypatch, activation, quant_swiglu, model_swiglu, expected_swiglu
    )
    relative_error = (out - ref).norm() / ref.norm()
    assert relative_error < 0.02, f"relative L2 error {relative_error:.4f}"


@requires_aiter
@torch.inference_mode()
def test_aiter_w4a16_swigluoai_uninterleave_requires_clamp_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(AssertionError, match="requires gemm1_clamp_limit"):
        _run_layer(
            monkeypatch,
            MoEActivation.SWIGLUOAI_UNINTERLEAVE,
            (1.702, 1.0, None),
            UNSET,
            (1.702, 1.0, None),
        )
