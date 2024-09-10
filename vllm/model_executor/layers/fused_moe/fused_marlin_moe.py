"""Fused MoE utilities for GPTQ."""
import functools
from typing import Any, Dict, Optional

import torch

from vllm import _custom_ops as ops
from vllm.model_executor.layers.fused_moe.fused_moe import (
    fused_topk, moe_align_block_size, try_get_optimal_moe_config)


def single_marlin_moe(
        hidden_states: torch.Tensor,
        w: torch.Tensor,
        scales: torch.Tensor,
        gating_output: torch.Tensor,
        g_idx: torch.Tensor,
        perm: torch.Tensor,
        topk: int,
        renormalize: bool,
        override_config: Optional[Dict[str, Any]] = None) -> torch.Tensor:
    """
    This function computes the multiplication of hidden_states with expert
    weights used in Marlin MoE, using weights w and top-k gating mechanism.
    Its purpose is testing and debugging the fused MoE kernel.

    Parameters:
    - hidden_states (torch.Tensor): The input tensor to the Marlin Mul.
    - w (torch.Tensor): The set of expert weights.
    - scales (torch.Tensor): The quantization scales.
    - gating_output (torch.Tensor): The output of the gating operation
        (before softmax).
    - g_idx (torch.Tensor): The act_order indices.
    - perm (torch.Tensor): The act_order input permutation.
    - topk (int): The number of top-k experts to select.
    - renormalize (bool): If True, renormalize the top-k weights to sum to 1.
    - override_config (Optional[Dict[str, Any]]): Optional override
        for the kernel configuration.

    Returns:
    - torch.Tensor: The output tensor after applying the MoE layer.
    """
    # Check constraints.
    assert hidden_states.shape[0] == gating_output.shape[0], (
        "Number of tokens mismatch")
    assert hidden_states.shape[1] == w.shape[1] * 16, "Hidden size mismatch"
    assert gating_output.shape[1] == w.shape[0], "Number of experts mismatch"
    assert hidden_states.is_contiguous(), "Hidden_states must be contiguous"
    assert w.is_contiguous(), "Expert weights must be contiguous"
    assert hidden_states.dtype == torch.float16

    M, K = hidden_states.shape
    E = w.shape[0]
    N = w.shape[2] // 2

    topk_weights, topk_ids = fused_topk(hidden_states, gating_output, topk,
                                        renormalize)

    # This might not be an optimal config for a single MMM
    get_config_func = functools.partial(try_get_optimal_moe_config,
                                        w.shape,
                                        w.shape,
                                        topk_ids.shape[1],
                                        None,
                                        override_config=override_config,
                                        is_marlin=True)
    config = get_config_func(M)

    block_size_m = config['BLOCK_SIZE_M']

    sorted_token_ids, _, _ = moe_align_block_size(topk_ids, block_size_m, E)

    max_workspace_size = (N // 64) * 16
    workspace = torch.zeros(max_workspace_size,
                            dtype=torch.int,
                            device="cuda",
                            requires_grad=False)

    intermediate_cache = torch.ops._moe_C.marlin_gemm_moe(
        hidden_states, w, sorted_token_ids, topk_weights, topk_ids, scales,
        g_idx, perm, workspace, M, N, K, True, E, topk, block_size_m, True,
        False)

    return torch.sum(intermediate_cache.view(*intermediate_cache.shape), dim=1)


def fused_marlin_moe(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    gating_output: torch.Tensor,
    g_idx1: torch.Tensor,
    g_idx2: torch.Tensor,
    perm1: torch.Tensor,
    perm2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    override_config: Optional[Dict[str, Any]] = None,
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    This function computes a Mixture of Experts (MoE) layer using two sets of
    weights, w1 and w2, and top-k gating mechanism.

    Parameters:
    - hidden_states (torch.Tensor): The input tensor to the MoE layer.
    - w1 (torch.Tensor): The first set of expert weights.
    - w2 (torch.Tensor): The second set of expert weights.
    - gating_output (torch.Tensor): The output of the gating operation
        (before softmax).
    - g_idx1 (torch.Tensor): The first set of act_order indices.
    - g_idx2 (torch.Tensor): The second set of act_order indices.
    - perm1 (torch.Tensor): The first act_order input permutation.
    - perm2 (torch.Tensor): The second act_order input permutation.
    - topk_weights (torch.Tensor): Top-k weights.
    - topk_ids (torch.Tensor): Indices of topk-k elements.
    - renormalize (bool): If True, renormalize the top-k weights to sum to 1.
    - override_config (Optional[Dict[str, Any]]): Optional override
        for the kernel configuration.
    - w1_scale (Optional[torch.Tensor]): Optional scale to be used for
        w1.
    - w2_scale (Optional[torch.Tensor]): Optional scale to be used for
        w2.

    Returns:
    - torch.Tensor: The output tensor after applying the MoE layer.
    """
    # Check constraints.
    assert hidden_states.shape[0] == gating_output.shape[
        0], "Number of tokens mismatch"
    assert hidden_states.shape[
        1] == w1.shape[1] * 16, "Hidden size mismatch w1"
    assert hidden_states.shape[
        1] == w2.shape[2] // 2, "Hidden size mismatch w2"
    assert gating_output.shape[1] == w1.shape[0], "Number of experts mismatch"
    assert hidden_states.is_contiguous(), "Hidden_states must be contiguous"
    assert w1.is_contiguous(), "Expert weights1 must be contiguous"
    assert w2.is_contiguous(), "Expert weights2 must be contiguous"
    assert hidden_states.dtype == torch.float16

    M, K = hidden_states.shape
    E = w1.shape[0]
    N = w2.shape[1] * 16
    topk = topk_ids.shape[1]

    get_config_func = functools.partial(
        try_get_optimal_moe_config,
        w1.shape,
        w2.shape,
        topk_ids.shape[1],
        None,
        override_config=override_config,
        is_marlin=True,
    )
    config = get_config_func(M)

    block_size_m = config["BLOCK_SIZE_M"]

    sorted_token_ids, _, _ = moe_align_block_size(topk_ids, block_size_m, E)

    max_workspace_size = ((M + 255) // 256) * (max(2 * N, K) // 64) * 16
    workspace = torch.zeros(max_workspace_size,
                            dtype=torch.int,
                            device="cuda",
                            requires_grad=False)

    intermediate_cache2 = torch.empty(
        (M * topk_ids.shape[1], N),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )

    intermediate_cache1 = torch.ops._moe_C.marlin_gemm_moe(
        hidden_states,
        w1,
        sorted_token_ids,
        topk_weights,
        topk_ids,
        w1_scale,
        g_idx1,
        perm1,
        workspace,
        M,
        2 * N,
        K,
        True,
        E,
        topk,
        block_size_m,
        True,
        False,
    )

    ops.silu_and_mul(intermediate_cache2, intermediate_cache1.view(-1, 2 * N))

    intermediate_cache3 = torch.ops._moe_C.marlin_gemm_moe(
        intermediate_cache2,
        w2,
        sorted_token_ids,
        topk_weights,
        topk_ids,
        w2_scale,
        g_idx2,
        perm2,
        workspace,
        M,
        K,
        N,
        True,
        E,
        topk,
        block_size_m,
        False,
        True,
    )

    return torch.sum(intermediate_cache3.view(*intermediate_cache3.shape),
                     dim=1)
