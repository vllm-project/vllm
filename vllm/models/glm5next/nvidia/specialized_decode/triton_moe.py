# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Block-FP8 Triton MoE with finalize fused into the decode boundary."""

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.experts.triton_moe import TritonExperts
from vllm.model_executor.layers.fused_moe.modular_kernel import FusedMoEKernel
from vllm.model_executor.layers.fused_moe.moe_output import (
    MoEOutput,
    UnfinalizedMoEOutput,
)

from .shared_expert import SharedExpert, uses_ue8m0_activation

logger = init_logger(__name__)


def build_triton_moe(layers, policy):
    shared = sorted(policy.tokens("shared_expert", "cute"))
    stream = torch.cuda.Stream()
    stages = {}
    for layer in layers:
        if layer._mlp_is_moe:
            run = _build_layer(layer.mlp, shared, stream)
            if run is not None:
                stages[layer.layer_idx] = run
    logger.info(
        "Specialized decode defers Triton MoE finalize on %d layers", len(stages)
    )
    return stages


def _build_layer(moe, shared_tokens, stream):
    routed = moe.experts.routed_experts
    kernel = routed.quant_method.moe_kernel
    if (
        kernel is None
        or type(kernel.fused_experts) is not TritonExperts
        or not kernel.supports_deferred_moe_finalize()
        or routed.apply_router_weight_on_input
        or moe.shared_experts is None
    ):
        return None
    stock = kernel.fused_experts
    experts = TritonExperts(
        stock.moe_config,
        stock.quant_config,
        activation_key=stock.activation_key,
        defer_finalize=True,
    )
    shared = moe.shared_experts
    gate_up, down = shared.gate_up_proj, shared.down_proj
    shared_gemv = None
    if (
        shared_tokens
        and down.weight.shape[1] == 512
        and getattr(gate_up, "weight_scale_inv", None) is not None
        and getattr(down, "weight_scale_inv", None) is not None
    ):
        shared_gemv = SharedExpert(
            gate_up.weight,
            gate_up.weight_scale_inv,
            down.weight,
            down.weight_scale_inv,
            shared.swiglu_limit,
            uses_ue8m0_activation(gate_up),
        )
        shared_gemv.prepare(shared_tokens)
    kernel = FusedMoEKernel(kernel.prepare_finalize, experts)

    def run(x, router):
        main = torch.cuda.current_stream(x.device)
        stream.wait_stream(main)
        with torch.cuda.stream(stream):
            if shared_gemv is not None and x.shape[0] in shared_gemv.compiled:
                shared = shared_gemv(x)
            else:
                shared = moe.shared_experts(x)
        weights, ids = moe.experts.router.select_experts(
            hidden_states=x,
            router_logits=router(moe, x),
            topk_indices_dtype=routed.quant_method.topk_indices_dtype,
        )
        output = kernel.apply(
            x,
            routed.w13_weight,
            routed.w2_weight,
            weights,
            ids,
            activation=routed.activation,
            global_num_experts=routed.global_num_experts,
            expert_map=routed.expert_map,
            apply_router_weight_on_input=False,
        )
        main.wait_stream(stream)
        assert isinstance(output, UnfinalizedMoEOutput)
        return MoEOutput(routed=output, shared_output=shared)

    return run
