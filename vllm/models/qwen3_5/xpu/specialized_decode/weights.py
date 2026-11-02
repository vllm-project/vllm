# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qualify and prepare the model parameters read by specialized MoE decode."""

from typing import TYPE_CHECKING, cast

import torch

from vllm.model_executor.kernels.linear.scaled_mm.xpu import XPUW8A16FP8LinearKernel
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.experts.xpu_moe import XPUExpertsFp8
from vllm.model_executor.layers.fused_moe.router.zero_expert_router import (
    ZeroExpertRouter,
)
from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner
from vllm.model_executor.utils import replace_parameter

if TYPE_CHECKING:
    from ..model import Qwen3_5SparseMoeBlock


def get_weights(
    block: "Qwen3_5SparseMoeBlock",
) -> tuple[torch.Tensor, ...] | None:
    runner, shared = block.experts, block.shared_expert
    if not isinstance(runner, MoERunner) or shared is None:
        return None
    config = runner.moe_config
    parallel = config.moe_parallel_config
    if (
        block.is_sequence_parallel
        or block.replicate_shared_expert
        or block.is_fused_shared_expert_enabled
        or runner.routed_input_transform is not None
        or runner.routed_output_transform is not None
        or runner.enable_dbo
        or isinstance(runner.router, ZeroExpertRouter)
        or runner.activation != MoEActivation.SILU
        or runner.routed_scaling_factor != 1.0
        or runner.routed_experts.apply_router_weight_on_input
        or config.has_bias
        or config.is_lora_enabled
        or parallel.enable_eplb
        or parallel.use_ep
        or parallel.ep_size != 1
        or runner.expert_map is not None
        or parallel.dp_size != 1
        or parallel.pcp_size != 1
        or parallel.is_sequence_parallel
        or not isinstance(shared.act_fn, SiluAndMul)
    ):
        return None
    if any(
        layer.bias is not None
        for layer in (
            shared.gate_up_proj,
            shared.down_proj,
            block.shared_expert_gate,
        )
    ):
        return None
    kernel = runner.routed_experts.quant_method.moe_kernel
    experts = kernel.fused_experts if kernel is not None else None
    if not isinstance(experts, XPUExpertsFp8):
        return None
    gu, down, gate = (
        shared.gate_up_proj,
        shared.down_proj,
        block.shared_expert_gate,
    )
    # The fused shared projections retain FP16/BF16 activations.
    if not all(
        isinstance(
            getattr(layer.quant_method, "fp8_linear", None),
            XPUW8A16FP8LinearKernel,
        )
        for layer in (gu, down)
    ):
        return None
    tensors = (
        runner.routed_experts.w13_weight,
        runner.routed_experts.w2_weight,
        gu.weight,
        down.weight,
        runner.routed_experts.w13_weight_scale,
        runner.routed_experts.w2_weight_scale,
        getattr(gu, "weight_scale", None),
        getattr(down, "weight_scale", None),
        gate.weight,
    )
    if not all(isinstance(t, torch.Tensor) for t in tensors):
        return None
    w13, w2, shared_w13, shared_w2, s13, s2, shared_s13, shared_s2, gate_weight = cast(
        tuple[torch.Tensor, ...], tensors
    )
    if w13.ndim != 3 or w2.ndim != 3:
        return None
    if not (
        w13.dtype
        == w2.dtype
        == shared_w13.dtype
        == shared_w2.dtype
        == torch.float8_e4m3fn
        and s13.shape == s2.shape == (w13.shape[0],)
        and shared_s13.numel() == shared_s2.numel() == 1
        and gate_weight.dtype == runner.moe_config.in_dtype
        and shared_w13.shape == w13.shape[1:]
        and shared_w2.shape == w2.shape[1:]
        and w2.shape[0] == w13.shape[0]
        and w13.shape[2] == 2 * w2.shape[1]
        and w2.shape[2] == w13.shape[1]
        and 0 < runner.moe_config.experts_per_token <= w13.shape[0] <= 512
        and gate_weight.shape in ((w13.shape[1],), (1, w13.shape[1]))
    ):
        return None
    matrices = (w13, w2, shared_w13, shared_w2)
    scales = (s13, s2, shared_s13, shared_s2)
    weights = (*matrices, *scales, gate_weight)
    if w13.device.type != "xpu" or not all(t.device == w13.device for t in weights):
        return None
    # Reject offset views whose existing contiguous storage is unaligned.
    if any(t.is_contiguous() and t.data_ptr() % 64 for t in matrices):
        return None
    if (gate_weight.is_contiguous() and gate_weight.data_ptr() % 32) or any(
        t.dtype != torch.float32 for t in scales
    ):
        return None
    return weights


@torch.no_grad()
def prepare_weights(block: "Qwen3_5SparseMoeBlock") -> tuple[torch.Tensor, ...] | None:
    """Use model-owned contiguous parameters, preserved by in-place reload."""
    if get_weights(block) is None:
        return None
    shared = block.shared_expert
    assert shared is not None
    routed = block.experts.routed_experts
    for layer, names in (
        (routed, ("w13_weight", "w2_weight", "w13_weight_scale", "w2_weight_scale")),
        (shared.gate_up_proj, ("weight", "weight_scale")),
        (shared.down_proj, ("weight", "weight_scale")),
        (block.shared_expert_gate, ("weight",)),
    ):
        for name in names:
            parameter = getattr(layer, name)
            if not parameter.is_contiguous():
                replace_parameter(layer, name, parameter.contiguous())
    return get_weights(block)
