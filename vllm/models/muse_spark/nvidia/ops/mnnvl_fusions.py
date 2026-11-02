# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Workspaces for FlashInfer MNNVL collectives fused into Muse Spark boundaries."""

import functools
from collections.abc import Callable
from typing import Any

import torch
import torch.distributed as dist

from vllm.distributed import get_tp_group
from vllm.distributed.parallel_state import in_the_same_node_as
from vllm.logger import init_logger
from vllm.platforms import current_platform

logger = init_logger(__name__)


@functools.cache
def tp_group_spans_nodes() -> bool:
    """Whether the TP group crosses nodes; collective on the first call."""
    return not all(in_the_same_node_as(get_tp_group().cpu_group, source_rank=0))


def _all_tp_ranks(flag: bool) -> bool:
    value = torch.tensor(int(flag))
    dist.all_reduce(value, op=dist.ReduceOp.MIN, group=get_tp_group().cpu_group)
    return bool(value.item())


def _create_on_all_tp_ranks(name: str, create: Callable[[], Any]) -> Any | None:
    """Collectively create a workspace; every TP rank gets one or none."""
    tp_group = get_tp_group()
    eligible = (
        current_platform.is_cuda()
        and current_platform.is_device_capability_family(100)
        and tp_group.world_size > 1
    )
    if eligible:
        try:
            from flashinfer.comm.mnnvl import is_multicast_supported

            eligible = is_multicast_supported(torch.accelerator.current_device_index())
        except ImportError:
            eligible = False
    if not _all_tp_ranks(eligible):
        logger.info_once("Muse Spark fused %s is unavailable.", name)
        return None
    try:
        workspace = create()
    except (ImportError, RuntimeError, ValueError) as exc:
        logger.warning("Muse Spark fused %s failed: %s", name, exc)
        workspace = None
    if not _all_tp_ranks(workspace is not None):
        return None
    logger.info_once("Muse Spark fused %s enabled (TP%d).", name, tp_group.world_size)
    return workspace


def maybe_create_all_gather_dual_norm(
    hidden_size: int, max_tokens: int, *, enable_pdl: bool
) -> Any | None:
    """FlashInfer ``MNNVLCuteDSLAllGatherDualNormWorkspace`` for the TP group."""

    def create() -> Any:
        from flashinfer.comm.mnnvl_cutedsl_ag import (
            LLAllGatherDualNormTuning,
            MNNVLCuteDSLAllGatherDualNormWorkspace,
        )

        tp_group = get_tp_group()
        return MNNVLCuteDSLAllGatherDualNormWorkspace(
            tp_group.world_size,
            tp_group.rank_in_group,
            max_tokens,
            hidden_size,
            group=tp_group.device_group,
            tuning=LLAllGatherDualNormTuning(enable_pdl=enable_pdl),
        )

    return _create_on_all_tp_ranks("unpooler all-gather + residual boundary", create)


def maybe_create_all_gather_norm_quant(hidden_size: int, max_tokens: int) -> Any | None:
    """FlashInfer ``MNNVLCuteDSLAllGatherNormQuantWorkspace`` for the TP group."""

    def create() -> Any:
        from flashinfer.comm.mnnvl_cutedsl_ag import (
            MNNVLCuteDSLAllGatherNormQuantWorkspace,
        )

        tp_group = get_tp_group()
        return MNNVLCuteDSLAllGatherNormQuantWorkspace(
            tp_group.world_size,
            tp_group.rank_in_group,
            max_tokens,
            hidden_size,
            group=tp_group.device_group,
        )

    return _create_on_all_tp_ranks("pooler all-gather + norm + NVFP4 quant", create)


def maybe_create_moe_finalize_all_reduce(
    hidden_size: int, top_k: int, max_tokens: int, *, enable_pdl: bool
) -> Any | None:
    """FlashInfer ``MNNVLCuteDSLMoEFinalizeAllReduceWorkspace`` for the TP group."""

    def create() -> Any:
        from flashinfer.comm.mnnvl_cutedsl_moe_finalize import (
            MNNVLCuteDSLMoEFinalizeAllReduceWorkspace,
        )

        tp_group = get_tp_group()
        return MNNVLCuteDSLMoEFinalizeAllReduceWorkspace(
            tp_group.world_size,
            tp_group.rank_in_group,
            max_tokens,
            hidden_size,
            top_k,
            group=tp_group.device_group,
            enable_pdl=enable_pdl,
        )

    return _create_on_all_tp_ranks("MoE finalize + all-reduce", create)
