# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Routing support for the AITER Triton MXFP4 W4A16 MoE backend.

MiMo-V2.6 uses an ungrouped sigmoid router with a per-expert correction bias
(``scoring_func=sigmoid``, ``topk_method=noaux_tc``, ``n_group == 1``), which
``get_routing_method_type`` classifies as ``RoutingMethodType.DeepSeekV3``.
``AiterW4A16ExpertsMonolithic`` used to reject that router ("kernel does not
support routing method ..."), and on gfx942 it is the only MXFP4 kernel whose
device gate accepts the card, so the checkpoint could not be served natively at
all.

aiter's *flat* top-k handles that router natively (``score_mode="sigmoid"`` plus
a correction bias), so the backend only has to ask for it. This module covers
both halves: the gate accepts the router family, and the routing the wrapper
requests reproduces the model's own math.
"""

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

MIMO_EXPERTS = 256  # MiMo-V2.6-Flash
MIMO_PRO_EXPERTS = 384  # MiMo-V2.6-Pro
MIMO_TOP_K = 8


def mimo_routing_method() -> RoutingMethodType:
    """MiMo-V2.6: sigmoid + correction bias, a single expert group."""
    return get_routing_method_type(
        scoring_func="sigmoid",
        top_k=MIMO_TOP_K,
        renormalize=True,
        num_expert_group=1,
        has_e_score_bias=True,
        routed_scaling_factor=None,
    )


def mimo_moe_config(num_experts: int = MIMO_EXPERTS) -> FusedMoEConfig:
    return FusedMoEConfig(
        num_experts=num_experts,
        experts_per_token=MIMO_TOP_K,
        hidden_dim=4096,
        intermediate_size=2048,
        num_local_experts=num_experts,
        num_logical_experts=num_experts,
        moe_parallel_config=FusedMoEParallelConfig.make_no_parallel(),
        activation=MoEActivation.SILU,
        in_dtype=torch.bfloat16,
        device=torch.device("cuda"),
        routing_method=mimo_routing_method(),
        router_logits_dtype=torch.float32,
    )


def test_ungrouped_sigmoid_router_is_deepseekv3() -> None:
    """Guard the premise of the fix: MiMo's router is DeepSeekV3-classified."""
    assert mimo_routing_method() == RoutingMethodType.DeepSeekV3


def test_aiter_w4a16_supports_mimo_routing() -> None:
    """The gate accepts MiMo's router family — the reported construction abort.

    Only the predicate the fix touches is exercised, so this runs on any host;
    the full config path is covered by the test below.
    """
    assert AiterW4A16ExpertsMonolithic._supports_routing_method(
        RoutingMethodType.DeepSeekV3, kMxfp4Static, None
    )


@pytest.mark.skipif(
    not is_aiter_found_and_supported(),
    reason="only runs on ROCm with a supported AITER install",
)
def test_aiter_w4a16_supports_mimo_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """The same gate through the real config path, device check included."""
    monkeypatch.setenv("VLLM_ROCM_USE_AITER", "1")
    rocm_aiter_ops.refresh_env_variables()

    supported, reason = AiterW4A16ExpertsMonolithic.is_supported_config(
        AiterW4A16ExpertsMonolithic,
        mimo_moe_config(),
        kMxfp4Static,
        None,
        mk.FusedMoEActivationFormat.Standard,
    )

    assert supported, reason


@pytest.mark.parametrize("num_experts", [MIMO_EXPERTS, MIMO_PRO_EXPERTS])
@pytest.mark.skipif(
    not is_aiter_found_and_supported(),
    reason="only runs on ROCm with a supported AITER install",
)
def test_flat_sigmoid_routing_matches_mimo_reference(num_experts: int) -> None:
    """The routing the wrapper requests must equal MiMo's own router math.

    MiMo's router is ungrouped: sigmoid scores, selection on
    ``sigmoid(logits) + bias``, weights = the *unbiased* sigmoid scores, then
    renormalized. Both checkpoints use it — 256 experts is Flash, 384 is Pro.

    The ``sigmoid`` score mode comes from ROCm/aiter#4688 (aiter >= 0.1.20).
    """
    aiter_routing = pytest.importorskip(
        "aiter.ops.triton.moe.moe_routing.routing"
    ).routing

    torch.manual_seed(0)
    top_k, num_tokens = MIMO_TOP_K, 16
    logits = torch.randn(num_tokens, num_experts, device="cuda") * 2.0
    bias = torch.randn(num_experts, device="cuda") * 0.5

    # Reference: select on the biased score, return the unbiased score, renorm.
    scores = torch.sigmoid(logits.float())
    ref_ids = (scores + bias).topk(top_k, dim=-1).indices
    ref_weights = scores.gather(1, ref_ids)
    ref_weights = ref_weights / ref_weights.sum(dim=-1, keepdim=True)

    def routed(score_mode: str) -> tuple[torch.Tensor, torch.Tensor]:
        routing_data, topk_indx, _ = aiter_routing(
            logits,
            top_k,
            score_mode=score_mode,
            bias=bias,
            renorm=True,
            routed_scaling_factor=1.0,
            use_grouped_topk=False,
        )
        # `topk_indx[i]` is the token-slot index (token * top_k + slot) of the
        # i-th entry of the expert-sorted array whose weights are gate_scal.
        hist = routing_data.expt_hist.to(torch.long).cpu()
        expert_of_slot = torch.repeat_interleave(torch.arange(hist.numel()), hist)
        slot = topk_indx.reshape(-1).to(torch.long).cpu()
        weights = routing_data.gate_scal.reshape(-1).to(torch.float32).cpu()
        ids = torch.full((num_tokens, top_k), -1, dtype=torch.long)
        out = torch.zeros(num_tokens, top_k)
        for i in range(slot.numel()):
            token, position = int(slot[i]) // top_k, int(slot[i]) % top_k
            ids[token, position] = int(expert_of_slot[i])
            out[token, position] = float(weights[i])
        return ids, out

    got_ids, got_weights = routed("sigmoid")
    assert torch.equal(
        torch.sort(got_ids, dim=-1).values, torch.sort(ref_ids.cpu(), dim=-1).values
    )
    assert torch.allclose(got_weights, ref_weights.cpu(), atol=1e-3)

    # Control: the correction bias must steer selection only. Ignoring it would
    # pick from the plain sigmoid scores instead.
    plain_ids = scores.topk(top_k, dim=-1).indices
    assert not torch.equal(
        torch.sort(plain_ids, dim=-1).values, torch.sort(ref_ids.cpu(), dim=-1).values
    )
