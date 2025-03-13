# SPDX-License-Identifier: Apache-2.0
"""Fused MoE utilities for GPTQ."""
import functools
from typing import Optional

import torch

from vllm.model_executor.layers.fused_moe.fused_moe import (
    fused_topk, moe_align_block_size, try_get_optimal_moe_config)
from vllm.platforms import current_platform
from vllm.scalar_type import scalar_types
from vllm.utils import direct_register_custom_op


def get_scalar_type(num_bits: int, has_zp: bool):
    if has_zp:
        assert num_bits == 4
        return scalar_types.uint4
    else:
        return scalar_types.uint4b8 if num_bits == 4 else scalar_types.uint8b128


def single_marlin_moe(
    hidden_states: torch.Tensor,
    w: torch.Tensor,
    scales: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    g_idx: Optional[torch.Tensor] = None,
    sort_indices: Optional[torch.Tensor] = None,
    w_zeros: Optional[torch.Tensor] = None,
    num_bits: int = 8,
    is_k_full: bool = True,
) -> torch.Tensor:
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
    - g_idx (Optional[torch.Tensor]): Optional act_order indices.
    - sort_indices (Optional[torch.Tensor]): Optional act_order input
      permutation.
    - topk (int): The number of top-k experts to select.
    - renormalize (bool): If True, renormalize the top-k weights to sum to 1.
    - w_zeros (Optional[torch.Tensor]): Optional zero points to be used for w.
    - num_bits (bool): The number of bits in expert weights quantization.

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
    assert num_bits in [4, 8]

    M, K = hidden_states.shape
    E = w.shape[0]
    N = w.shape[2] // (num_bits // 2)

    topk_weights, topk_ids = fused_topk(hidden_states, gating_output, topk,
                                        renormalize)

    # This might not be an optimal config for a single MMM
    get_config_func = functools.partial(try_get_optimal_moe_config,
                                        w.shape,
                                        w.shape,
                                        topk_ids.shape[1],
                                        None,
                                        is_marlin=True)
    config = get_config_func(M)

    block_size_m = config['BLOCK_SIZE_M']

    sorted_token_ids, _, _ = moe_align_block_size(topk_ids, block_size_m, E)

    max_workspace_size = (N // 64) * 16
    workspace = torch.zeros(max_workspace_size,
                            dtype=torch.int,
                            device=hidden_states.device,
                            requires_grad=False)

    has_zero_point = w_zeros is not None
    if w_zeros is None:
        w_zeros = torch.empty((0, 0),
                              dtype=hidden_states.dtype,
                              device=hidden_states.device,
                              requires_grad=False)

    if g_idx is None:
        g_idx = torch.empty((0, 0),
                            dtype=torch.int32,
                            device=hidden_states.device,
                            requires_grad=False)

    if sort_indices is None:
        sort_indices = torch.empty((0),
                                   dtype=torch.int32,
                                   device=hidden_states.device,
                                   requires_grad=False)

    scalar_type = get_scalar_type(num_bits, has_zero_point)

    intermediate_cache = torch.ops._moe_C.marlin_gemm_moe(
        hidden_states, w, sorted_token_ids, topk_weights, topk_ids, scales,
        w_zeros, g_idx, sort_indices, workspace, scalar_type.id, M, N, K,
        is_k_full, E, topk, block_size_m, True, False)

    return torch.sum(intermediate_cache.view(*intermediate_cache.shape), dim=1)


def single_marlin_moe_fake(
    hidden_states: torch.Tensor,
    w: torch.Tensor,
    scales: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    g_idx: Optional[torch.Tensor] = None,
    sort_indices: Optional[torch.Tensor] = None,
    w_zeros: Optional[torch.Tensor] = None,
    num_bits: int = 8,
    is_k_full: bool = True,
) -> torch.Tensor:
    return torch.empty_like(hidden_states)


direct_register_custom_op(
    op_name="single_marlin_moe",
    op_func=single_marlin_moe,
    mutates_args=[],
    fake_impl=single_marlin_moe_fake,
)


def fused_marlin_moe(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    gating_output: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    g_idx1: Optional[torch.Tensor] = None,
    g_idx2: Optional[torch.Tensor] = None,
    sort_indices1: Optional[torch.Tensor] = None,
    sort_indices2: Optional[torch.Tensor] = None,
    w1_zeros: Optional[torch.Tensor] = None,
    w2_zeros: Optional[torch.Tensor] = None,
    num_bits: int = 8,
    is_k_full: bool = True,
) -> torch.Tensor:
    """
    This function computes a Mixture of Experts (MoE) layer using two sets of
    weights, w1 and w2, and top-k gating mechanism.

    Parameters:
    - hidden_states (torch.Tensor): The input tensor to the MoE layer.
    - w1 (torch.Tensor): The first set of expert weights.
    - w2 (torch.Tensor): The second set of expert weights.
    - w1_scale (torch.Tensor): Scale to be used for w1.
    - w2_scale (torch.Tensor): Scale to be used for w2.
    - gating_output (torch.Tensor): The output of the gating operation
        (before softmax).
    - g_idx1 (Optional[torch.Tensor]): The first set of act_order indices.
    - g_idx2 (Optional[torch.Tensor]): The second set of act_order indices.
    - sort_indices1 (Optional[torch.Tensor]): The first act_order input
        permutation.
    - sort_indices2 (Optional[torch.Tensor]): The second act_order input
        permutation.
    - topk_weights (torch.Tensor): Top-k weights.
    - topk_ids (torch.Tensor): Indices of topk-k elements.
    - w1_zeros (Optional[torch.Tensor]): Optional zero points to be used for w1.
    - w2_zeros (Optional[torch.Tensor]): Optional zero points to be used for w2.
    - num_bits (bool): The number of bits in expert weights quantization.

    Returns:
    - torch.Tensor: The output tensor after applying the MoE layer.
    """
    # Check constraints.
    assert hidden_states.shape[0] == gating_output.shape[
        0], "Number of tokens mismatch"
    assert hidden_states.shape[
        1] == w1.shape[1] * 16, "Hidden size mismatch w1"
    assert hidden_states.shape[1] == w2.shape[2] // (
        num_bits // 2), "Hidden size mismatch w2"
    assert gating_output.shape[1] == w1.shape[0], "Number of experts mismatch"
    assert hidden_states.is_contiguous(), "Hidden_states must be contiguous"
    assert w1.is_contiguous(), "Expert weights1 must be contiguous"
    assert w2.is_contiguous(), "Expert weights2 must be contiguous"
    assert hidden_states.dtype == torch.float16
    assert num_bits in [4, 8]

    has_no_act_order = (g_idx1 is None and g_idx2 is None
                        and sort_indices1 is None and sort_indices2 is None)
    has_all_act_order = (g_idx1 is not None and g_idx2 is not None
                         and sort_indices1 is not None
                         and sort_indices2 is not None)
    assert has_no_act_order or has_all_act_order, (
        "g_idx and sorted_indices "
        "must be all not None or must be all None")

    has_no_zp = w1_zeros is None and w2_zeros is None
    has_all_zp = w1_zeros is not None and w2_zeros is not None
    assert has_no_zp or has_all_zp, ("zero points must be both not None or "
                                     "must be both None")

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
        is_marlin=True,
    )
    config = get_config_func(M)

    block_size_m = config["BLOCK_SIZE_M"]

    sorted_token_ids, _, _ = moe_align_block_size(topk_ids, block_size_m, E)

    max_workspace_size = (max(2 * N, K) // 64) * 16
    workspace = torch.zeros(max_workspace_size,
                            dtype=torch.int,
                            device=current_platform.device_type,
                            requires_grad=False)

    if has_no_zp:
        w1_zeros = torch.empty((0, 0),
                               dtype=hidden_states.dtype,
                               device=hidden_states.device,
                               requires_grad=False)
        w2_zeros = torch.empty((0, 0),
                               dtype=hidden_states.dtype,
                               device=hidden_states.device,
                               requires_grad=False)

    if has_no_act_order:
        g_idx1 = torch.empty((0, 0),
                             dtype=torch.int32,
                             device=hidden_states.device,
                             requires_grad=False)
        g_idx2 = torch.empty((0, 0),
                             dtype=torch.int32,
                             device=hidden_states.device,
                             requires_grad=False)
        sort_indices1 = torch.empty((0),
                                    dtype=torch.int32,
                                    device=hidden_states.device,
                                    requires_grad=False)
        sort_indices2 = torch.empty((0, 0),
                                    dtype=torch.int32,
                                    device=hidden_states.device,
                                    requires_grad=False)

    scalar_type1 = get_scalar_type(num_bits, has_all_zp)
    scalar_type2 = get_scalar_type(num_bits, has_all_zp)

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
        w1_zeros,
        g_idx1,
        sort_indices1,
        workspace,
        scalar_type1.id,
        M,
        2 * N,
        K,
        is_k_full,
        E,
        topk,
        block_size_m,
        True,
        False,
    )

    torch.ops._C.silu_and_mul(intermediate_cache2,
                              intermediate_cache1.view(-1, 2 * N))

    intermediate_cache3 = torch.ops._moe_C.marlin_gemm_moe(
        intermediate_cache2,
        w2,
        sorted_token_ids,
        topk_weights,
        topk_ids,
        w2_scale,
        w2_zeros,
        g_idx2,
        sort_indices2,
        workspace,
        scalar_type2.id,
        M,
        K,
        N,
        is_k_full,
        E,
        topk,
        block_size_m,
        False,
        True,
    )

    return torch.sum(intermediate_cache3.view(*intermediate_cache3.shape),
                     dim=1)


def fused_marlin_moe_fake(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    gating_output: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    g_idx1: Optional[torch.Tensor] = None,
    g_idx2: Optional[torch.Tensor] = None,
    sort_indices1: Optional[torch.Tensor] = None,
    sort_indices2: Optional[torch.Tensor] = None,
    w1_zeros: Optional[torch.Tensor] = None,
    w2_zeros: Optional[torch.Tensor] = None,
    num_bits: int = 8,
    is_k_full: bool = True,
) -> torch.Tensor:
    return torch.empty_like(hidden_states)


direct_register_custom_op(
    op_name="fused_marlin_moe",
    op_func=fused_marlin_moe,
    mutates_args=[],
    fake_impl=fused_marlin_moe_fake,
)
