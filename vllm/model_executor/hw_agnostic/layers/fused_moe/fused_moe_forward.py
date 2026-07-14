# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The fused MoE forward pass: quantize → [dispatch] → experts → [combine].

This is the single transport the hw-agnostic tree supports:

- ``dp_size == 1``: no all2all. ``expert_map`` masks remote experts inside
  the Triton kernel, so every rank sees the full token set and computes
  only its local experts.
- ``dp_size > 1``: AllGather / ReduceScatter over ``torch.distributed`` via
  ``get_ep_group()``. Input is quantized *before* the gather so fp8 halves
  the transferred bytes.

``MoERunner._validate_supported_settings`` rejects any other transport, so
there is no P/F selection to make; a single function covers both branches.
The experts kernel applies the router weights and reduces the top-k outputs
internally (see ``TritonExperts.apply``), so its ``(num_tokens, hidden)``
output needs no further weight-and-reduce step here.
"""

from math import prod

import torch

import vllm.envs as envs
from vllm.distributed import get_ep_group
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.model_executor.hw_agnostic.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.hw_agnostic.layers.fused_moe.modular_kernel import (
    FusedMoEExpertsModular,
)
from vllm.model_executor.hw_agnostic.layers.fused_moe.utils import (
    _resize_cache,
    moe_kernel_quantize_input,
)
from vllm.v1.worker.workspace import current_workspace_manager


def _allocate_buffers(
    experts: FusedMoEExpertsModular,
    out_dtype: torch.dtype,
    M: int,
    N: int,
    K: int,
    top_k: int,
    global_num_experts: int,
    local_num_experts: int,
    activation: MoEActivation,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Allocate the two GEMM workspaces and the fused output from the shared
    chunked-M workspace pool.

    ``workspace13`` and the final output alias the same storage: by the time
    the second GEMM writes the output, the first GEMM's scratch is free.
    """
    workspace_dtype = experts.workspace_dtype(out_dtype)
    workspace13_shape, workspace2_shape, fused_out_shape = experts.workspace_shapes(
        M, N, K, top_k, global_num_experts, local_num_experts, None, activation
    )

    max_shape_size = max(prod(workspace13_shape), prod(fused_out_shape))
    common_workspace, workspace2 = current_workspace_manager().get_simultaneous(
        ((max_shape_size,), workspace_dtype),
        (workspace2_shape, workspace_dtype),
    )
    workspace13 = _resize_cache(common_workspace, workspace13_shape)
    fused_out = _resize_cache(common_workspace, fused_out_shape)
    return workspace13, workspace2, fused_out


def fused_moe_forward(
    experts: FusedMoEExpertsModular,
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: MoEActivation,
    global_num_experts: int,
    expert_map: torch.Tensor | None,
    apply_router_weight_on_input: bool,
) -> torch.Tensor:
    """Run the fused MoE layer with the given experts kernel.

    Args:
        experts: The experts compute kernel (e.g. ``TritonExperts``).
        hidden_states: ``(num_tokens, hidden)`` input activations.
        w1: Gate/up expert weights.
        w2: Down expert weights.
        topk_weights: Router weights, ``(num_tokens, top_k)``.
        topk_ids: Selected expert ids, ``(num_tokens, top_k)``.
        activation: Activation applied between the two GEMMs.
        global_num_experts: Total experts across the EP shard (``-1`` → local).
        expert_map: Global→local expert map; masks remote experts when set.
        apply_router_weight_on_input: Apply router weights before the experts
            (only valid for ``top_k == 1``).

    Returns:
        ``(num_tokens, hidden)`` MoE output.
    """
    quant_config = experts.quant_config
    moe_parallel_config = experts.moe_config.moe_parallel_config
    dp_ep = moe_parallel_config.dp_size > 1
    is_sequence_parallel = moe_parallel_config.is_sequence_parallel

    output = torch.empty_like(hidden_states)
    local_num_experts = w1.shape[0]
    if global_num_experts == -1:
        global_num_experts = local_num_experts

    if apply_router_weight_on_input:
        assert topk_ids.size(1) == 1, (
            "apply_router_weight_on_input is only implemented for topk=1"
        )
        hidden_states = hidden_states * topk_weights.to(hidden_states.dtype)

    # Drop cudagraph/DP padding rows by forcing their expert ids to -1 so the
    # experts kernel skips them. Gated by VLLM_MOE_SKIP_PADDING (off by
    # default) since it relies on the kernel treating -1 as a skip sentinel.
    if envs.VLLM_MOE_SKIP_PADDING and is_forward_context_available():
        is_padding = get_forward_context().is_padding
        if is_padding is not None:
            n = topk_ids.shape[0]
            topk_ids = torch.where(is_padding[:n].unsqueeze(1), -1, topk_ids)

    # Quantize before dispatch (fp8 halves the transferred bytes). Experts
    # that fuse input-quant into their GEMM (e.g. MXFP4 W4A16) keep the
    # activations unquantized.
    if experts.expects_unquantized_inputs:
        a1q, a1q_scale = hidden_states, None
    else:
        a1q, a1q_scale = moe_kernel_quantize_input(
            hidden_states,
            quant_config.a1_scale,
            quant_dtype=quant_config.quant_dtype,
            per_act_token_quant=quant_config.per_act_token_quant,
            block_shape=quant_config.block_shape,
        )

    if dp_ep:
        # Static (scalar) scales are replicated on every rank — skip gathering.
        scales = None if (a1q_scale is None or a1q_scale.ndim == 0) else [a1q_scale]
        res = get_ep_group().dispatch(
            a1q,
            topk_weights,
            topk_ids,
            is_sequence_parallel=is_sequence_parallel,
            extra_tensors=scales,
        )
        if scales is None:
            a1q, topk_weights, topk_ids = res
        else:
            a1q, topk_weights, topk_ids, gathered_scales = res
            a1q_scale = gathered_scales[0]

    _, M, N, K, top_k = experts.moe_problem_size(a1q, w1, w2, topk_ids)
    workspace13, workspace2, fused_out = _allocate_buffers(
        experts,
        hidden_states.dtype,
        M,
        N,
        K,
        top_k,
        global_num_experts,
        local_num_experts,
        activation,
    )

    experts.apply(
        output=fused_out,
        hidden_states=a1q,
        w1=w1,
        w2=w2,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        activation=activation,
        global_num_experts=global_num_experts,
        expert_map=expert_map,
        a1q_scale=a1q_scale,
        a2_scale=experts.a2_scale,
        workspace13=workspace13,
        workspace2=workspace2,
        expert_tokens_meta=None,
        apply_router_weight_on_input=apply_router_weight_on_input,
    )

    # ``fused_out`` is a view into the shared workspace pool, which the next
    # MoE layer overwrites, so copy the result into ``output``.
    if dp_ep:
        fused_out = get_ep_group().combine(
            fused_out, is_sequence_parallel=is_sequence_parallel
        )
    output.copy_(fused_out)
    return output
