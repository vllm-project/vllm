"""Fused MoE utilities for AWQ."""
import torch

from vllm import _custom_ops as ops
from vllm.logger import init_logger

from .fused_moe import fused_experts, moe_align_block_size

logger = init_logger(__name__)

NAIVE_THRESHOLD = 1024


def fused_experts_awq(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w1_scales: torch.Tensor,
    w2_scales: torch.Tensor,
    w1_qzeros: torch.Tensor,
    w2_qzeros: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    pack_factor: int,
) -> torch.Tensor:
    """
    This function computes an AWQ fused_expert.

    Parameters:
    - hidden_states (torch.Tensor): The input tensor to the MoE layer.
    - w1 (torch.Tensor): The first set of expert weights.
    - w2 (torch.Tensor): The second set of expert weights.    
    - w1_scales (torch.Tensor): scale to be used for w1.
    - w2_scales (torch.Tensor): scale to be used for w2.
    - w1_qzeros (torch.Tensor): zero point to be used for w1.
    - w2_qzeros (torch.Tensor): zero point to be used for w2.
    - pack_factor (int): Weight packing factor (int4 in int32 == 8)

    Returns:
    - torch.Tensor: The output tensor after applying the MoE layer.
    """

    # If large seq_len prefill, dequantize and use the fp16 MoE kernel.
    do_naive_dequant = hidden_states.shape[:-1].numel() >= NAIVE_THRESHOLD
    if do_naive_dequant:
        # TODO: why is this not contiguous already?
        dequant_w1 = ops.awq_dequantize(w1, w1_scales, w1_qzeros, 0, 0,
                                        0).permute(0, 2, 1).contiguous()
        dequant_w2 = ops.awq_dequantize(w2, w2_scales, w2_qzeros, 0, 0,
                                        0).permute(0, 2, 1).contiguous()

        return fused_experts(hidden_states, dequant_w1, dequant_w2,
                             topk_weights, topk_ids)

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
