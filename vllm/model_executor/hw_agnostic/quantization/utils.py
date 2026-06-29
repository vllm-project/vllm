# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Hw-agnostic FP8 weight/scale helpers (pure tensor math, import-safe)."""

import torch

from vllm import _custom_ops as ops
from vllm.logger import init_logger

logger = init_logger(__name__)


def per_tensor_dequantize(
    tensor: torch.Tensor, inv_scale: float | torch.Tensor
) -> torch.Tensor:
    return tensor.to(torch.float16) * inv_scale


def _all_close_1d(x: torch.Tensor) -> bool:
    assert x.dim() == 1
    return all(torch.allclose(x[0], x[i]) for i in range(x.shape[0]))


def process_fp8_weight_tensor_strategy_moe(
    weight: torch.Tensor,
    weight_scales: torch.Tensor,
    shard_size: int,
    num_experts: int,
    is_act_and_mul: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-expert MoE weight requantization to a single per-expert scale.

    For w13 (gate+up fused) checkpoints carry one scale per shard.
    Requantize each shard with the max of (w1_scale, w3_scale) so the
    kernel can run with a single scale per expert.
    """
    max_scales = weight_scales.max(dim=1).values

    if not is_act_and_mul:
        assert weight_scales.shape[1] == 1
        assert max_scales.shape == (num_experts,)
        return weight, max_scales

    for expert_id in range(num_experts):
        start = 0
        for shard_id in range(2):
            dq_weight = per_tensor_dequantize(
                weight[expert_id][start : start + shard_size, :],
                weight_scales[expert_id][shard_id],
            )
            weight[expert_id][start : start + shard_size, :], _ = ops.scaled_fp8_quant(
                dq_weight, max_scales[expert_id]
            )
            start += shard_size
    return weight, max_scales


def process_fp8_input_tensor_strategy_moe(
    w13_input_scale: torch.Tensor,
    w2_input_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Collapse per-expert input scales to a single scalar (the max)."""
    if not _all_close_1d(w13_input_scale) or not _all_close_1d(w2_input_scale):
        logger.info_once(
            "Found input_scales that are not equal across experts for "
            "an FP8 MoE layer; using the per-layer maximum."
        )
    return w13_input_scale.max(), w2_input_scale.max()
