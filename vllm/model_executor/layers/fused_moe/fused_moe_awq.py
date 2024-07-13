"""Fused MoE utilities for AWQ."""
import torch

from vllm import _custom_ops as ops
from vllm.logger import init_logger

from .fused_moe import fused_moe, fused_topk, moe_align_block_size

logger = init_logger(__name__)


def fused_moe_awq(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    pack_factor: int,
    w1_scales: torch.Tensor,
    w2_scales: torch.Tensor,
    w1_qzeros: torch.Tensor,
    w2_qzeros: torch.Tensor,
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
    - pack_factor (int): Weight packing factor (int4 in int32 == 8)
    - w1_scales (torch.Tensor): scale to be used for w1.
    - w2_scales (torch.Tensor): scale to be used for w2.
    - w1_qzeros (torch.Tensor): zero point to be used for w1.
    - w2_qzeros (torch.Tensor): zero point to be used for w2.

    Returns:
    - torch.Tensor: The output tensor after applying the MoE layer.
    """

    # If large seq_len prefill, dequantize and use the fp16 MoE kernel.
    do_naive_dequant = hidden_states.shape[:-1].numel() >= 1024
    if do_naive_dequant:
        # TODO: why is this not contiguous alreayd?
        dequant_w1 = ops.awq_dequantize(w1, w1_scales, w1_qzeros, 0, 0,
                                        0).permute(0, 2, 1).contiguous()
        dequant_w2 = ops.awq_dequantize(w2, w2_scales, w2_qzeros, 0, 0,
                                        0).permute(0, 2, 1).contiguous()

        return fused_moe(hidden_states, dequant_w1, dequant_w2, gating_output,
                         topk, renormalize)

    topk_weights, topk_ids = fused_topk(hidden_states, gating_output, topk,
                                        renormalize)
    (sorted_token_ids, expert_ids,
     num_tokens_post_padded) = moe_align_block_size(topk_ids, 16, w1.shape[0])

    x = hidden_states.view(hidden_states.shape[0], 1, *hidden_states.shape[1:])

    gate_up = ops.awq_fused_moe(x, w1, w1_scales, w1_qzeros, topk_weights,
                                sorted_token_ids, expert_ids,
                                num_tokens_post_padded, False, pack_factor)

    out = torch.empty((gate_up.shape[:-1] + (gate_up.shape[-1] // 2, )),
                      dtype=hidden_states.dtype,
                      device=hidden_states.device)
    ops.silu_and_mul(out, gate_up)

    out = ops.awq_fused_moe(out, w2, w2_scales, w2_qzeros, topk_weights,
                            sorted_token_ids, expert_ids,
                            num_tokens_post_padded, True, pack_factor)

    return torch.sum(out, dim=1)
