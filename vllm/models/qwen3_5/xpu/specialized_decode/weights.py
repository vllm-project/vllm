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
) -> tuple[torch.Tensor, ...] | str:
    """Return eligible parameters or the first static fallback reason."""
    runner, shared = block.experts, block.shared_expert
    if not isinstance(runner, MoERunner) or shared is None:
        return "requires MoERunner and a shared expert"
    config = runner.moe_config
    parallel = config.moe_parallel_config
    checks = (
        (
            block.is_sequence_parallel or parallel.is_sequence_parallel,
            "sequence parallelism is unsupported",
        ),
        (block.replicate_shared_expert, "replicated shared experts are unsupported"),
        (
            block.is_fused_shared_expert_enabled,
            "shared expert fusion is already enabled",
        ),
        (
            runner.routed_input_transform is not None
            or runner.routed_output_transform is not None,
            "routed input/output transforms are unsupported",
        ),
        (runner.enable_dbo, "dual batch overlap is unsupported"),
        (
            isinstance(runner.router, ZeroExpertRouter),
            "zero-expert routing is unsupported",
        ),
        (
            runner.activation != MoEActivation.SILU
            or not isinstance(shared.act_fn, SiluAndMul),
            "routed and shared experts require SiLU activation",
        ),
        (runner.routed_scaling_factor != 1.0, "routed scaling must be 1.0"),
        (
            runner.routed_experts.apply_router_weight_on_input,
            "input router weighting is unsupported",
        ),
        (config.has_bias, "expert biases are unsupported"),
        (config.is_lora_enabled, "LoRA is unsupported"),
        (parallel.enable_eplb, "expert parallel load balancing is unsupported"),
        (
            parallel.use_ep or parallel.ep_size != 1 or runner.expert_map is not None,
            "expert parallelism and expert maps are unsupported",
        ),
        (parallel.dp_size != 1, "data parallelism is unsupported"),
        (parallel.pcp_size != 1, "prefill context parallelism is unsupported"),
    )
    reason = next((reason for failed, reason in checks if failed), None)
    if reason is not None:
        return reason
    if any(
        layer.bias is not None
        for layer in (
            shared.gate_up_proj,
            shared.down_proj,
            block.shared_expert_gate,
        )
    ):
        return "shared projection and gate biases are unsupported"
    kernel = runner.routed_experts.quant_method.moe_kernel
    experts = kernel.fused_experts if kernel is not None else None
    if not isinstance(experts, XPUExpertsFp8):
        return "routed experts require XPUExpertsFp8"
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
        return "shared projections require XPUW8A16FP8LinearKernel"
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
        return "required weight or scale tensors are missing"
    w13, w2, shared_w13, shared_w2, s13, s2, shared_s13, shared_s2, gate_weight = cast(
        tuple[torch.Tensor, ...], tensors
    )
    if w13.ndim != 3 or w2.ndim != 3:
        return "routed weights must be rank three"
    if shared_w13.ndim != 2 or shared_w2.ndim != 2:
        return "shared weights must be rank two"
    prepared = w13.shape[1:] == shared_w13.shape and w2.shape[1:] == shared_w2.shape
    if not (
        w13.dtype
        == w2.dtype
        == shared_w13.dtype
        == shared_w2.dtype
        == torch.float8_e4m3fn
        and s13.shape == s2.shape == (w13.shape[0],)
        and shared_s13.numel() == shared_s2.numel() == 1
        and gate_weight.dtype == runner.moe_config.in_dtype
        and w13.shape[1:] == (shared_w13.shape if prepared else shared_w13.shape[::-1])
        and w2.shape[1:] == (shared_w2.shape if prepared else shared_w2.shape[::-1])
        and w2.shape[0] == w13.shape[0]
        and shared_w13.shape[1] == 2 * shared_w2.shape[0]
        and shared_w2.shape[1] == shared_w13.shape[0]
        and 0 < runner.moe_config.experts_per_token <= w13.shape[0] <= 512
        and gate_weight.shape in ((shared_w13.shape[0],), (1, shared_w13.shape[0]))
    ):
        return "weight dtypes, dimensions or scale shapes are incompatible"
    matrices = (w13, w2, shared_w13, shared_w2)
    scales = (s13, s2, shared_s13, shared_s2)
    weights = (*matrices, *scales, gate_weight)
    if w13.device.type != "xpu" or not all(t.device == w13.device for t in weights):
        return "weights and scales must share one XPU device"
    # Reject offset views whose existing contiguous storage is unaligned.
    if any(t.is_contiguous() and t.data_ptr() % 64 for t in matrices):
        return "contiguous weight matrices must be 64-byte aligned"
    if gate_weight.is_contiguous() and gate_weight.data_ptr() % 32:
        return "contiguous shared gate weights must be 32-byte aligned"
    if any(t.dtype != torch.float32 for t in scales):
        return "weight scales must be FP32"
    return weights


@torch.no_grad()
def prepare_weights(block: "Qwen3_5SparseMoeBlock") -> tuple[torch.Tensor, ...] | str:
    """Finalize model-owned layouts before binding an executor or capturing graphs."""
    weights = get_weights(block)
    if isinstance(weights, str):
        return weights
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
    runner = cast(MoERunner, block.experts)
    kernel = runner.routed_experts.quant_method.moe_kernel
    assert kernel is not None
    experts = cast(XPUExpertsFp8, kernel.fused_experts)
    experts.initialize_weights(
        routed.w13_weight,
        routed.w2_weight,
        runner.moe_config.experts_per_token,
        runner.activation,
        runner.expert_map,
    )
    weights = get_weights(block)
    if not isinstance(weights, str) and (
        weights[0].shape[1:] != weights[2].shape
        or weights[1].shape[1:] != weights[3].shape
    ):
        return "routed provider layout is unsupported"
    return weights
