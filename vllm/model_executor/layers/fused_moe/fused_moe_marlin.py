"""Fused MoE utilities for GPTQ."""
import functools
import torch

from typing import Any, Dict, Optional
from vllm import _custom_ops as ops

from .fused_moe import fused_topk, moe_align_block_size, try_get_optimal_moe_config


def fused_moe_marlin(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    gating_output: torch.Tensor,
    g_idx1: torch.Tensor,
    g_idx2: torch.Tensor,
    rand_perm1: torch.Tensor,
    rand_perm2: torch.Tensor,
    topk: int,
    renormalize: bool,
    override_config: Optional[Dict[str, Any]] = None,
    use_fp8: bool = False,
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
    - topk (int): The number of top-k experts to select.
    - renormalize (bool): If True, renormalize the top-k weights to sum to 1.
    - inplace (bool): If True, perform the operation in-place.
        Defaults to False.
    - override_config (Optional[Dict[str, Any]]): Optional override
        for the kernel configuration.
    - use_fp8 (bool): If True, use fp8 arithmetic to compute the inner
        products for w1 and w2. Defaults to False.
    - w1_scale (Optional[torch.Tensor]): Optional scale to be used for
        w1.
    - w2_scale (Optional[torch.Tensor]): Optional scale to be used for
        w2.

    Returns:
    - torch.Tensor: The output tensor after applying the MoE layer.
    """
    # Check constraints.
    assert hidden_states.shape[0] == gating_output.shape[0], "Number of tokens mismatch"
    assert hidden_states.shape[1] == w1.shape[1] * 16, "Hidden size mismatch w1"
    assert hidden_states.shape[1] == w2.shape[2] // 2, "Hidden size mismatch w2"
    assert gating_output.shape[1] == w1.shape[0], "Number of experts mismatch"
    assert hidden_states.is_contiguous(), "Hidden_states must be contiguous"
    assert w1.is_contiguous(), "Expert weights1 must be contiguous"
    assert w2.is_contiguous(), "Expert weights2 must be contiguous"
    assert hidden_states.dtype in [torch.float32, torch.float16, torch.bfloat16]
    M, K = hidden_states.shape
    E = w1.shape[0]
    N = w2.shape[1] * 16

    topk_weights, topk_ids = fused_topk(hidden_states, gating_output, topk, renormalize)

    get_config_func = functools.partial(
        try_get_optimal_moe_config,
        w1.shape,
        w2.shape,
        topk_ids.shape[1],
        "float8" if use_fp8 else None,
        override_config=override_config,
        is_marlin=True,
    )
    config = get_config_func(M)

    block_size_m = config["BLOCK_SIZE_M"]

    sorted_token_ids, _, _ = moe_align_block_size(topk_ids, block_size_m, E)

    max_workspace_size = ((M + 255) // 256) * (max(2 * N, K) // 64) * 16
    workspace = torch.zeros(
        max_workspace_size, dtype=torch.int, device="cuda", requires_grad=False
    )

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
        rand_perm1,
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
        rand_perm2,
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

    return torch.sum(intermediate_cache3.view(*intermediate_cache3.shape), dim=1)
