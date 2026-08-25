# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compatibility helpers for the optional HPC BF16 MoE backend."""

import functools
import importlib.util

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)


@functools.cache
def has_hpc() -> bool:
    """Return whether the optional ``hpc`` package is importable."""
    if importlib.util.find_spec("hpc") is None:
        logger.warning_once(
            "HPC BF16 MoE requires the hpc package from "
            "https://gitlab-cn-beijing.siflow.cn/inference/hpc-ops"
        )
        return False
    return True


def hpc_fuse_moe_bf16(
    x: torch.Tensor,
    gate_up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_scale: torch.Tensor,
    rank_ep: int,
    num_expert_total: int,
    shared_output: torch.Tensor | None = None,
    output: torch.Tensor | None = None,
    workspace: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run HPC BF16 MoE with caller-owned output and scratch memory."""
    from hpc import fuse_moe_bf16

    return fuse_moe_bf16(
        x,
        gate_up_weight,
        down_weight,
        topk_ids,
        topk_scale,
        rank_ep=rank_ep,
        num_expert_total=num_expert_total,
        shared_output=shared_output,
        output=output,
        workspace=workspace,
    )


__all__ = ["has_hpc", "hpc_fuse_moe_bf16"]
