# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Muse Spark PTPC FP8 specialized decode on MI300X / MI325X (gfx942, TP8 = EP8):
persistent FlyDSL launches, one a decoder layer, for plain decode steps of
1-16 rows (`--monokernel` on gfx942). See README.md."""

import importlib

import torch
import torch.distributed as dist
from torch import nn

from vllm.distributed import get_tp_group
from vllm.logger import init_logger

logger = init_logger(__name__)


def attach(lm: nn.Module) -> None:
    """Attach the kernels to `lm.model` after the weights are final.

    Collective over the TP group: every rank calls it. A deployment that any
    rank cannot run raises on every rank, with every reason of this rank.
    """
    from ..mono import _ensure_writable_flydsl_cache
    from .refusals import deployment_refusals, layer_refusals

    model = lm.model
    if model.mono is not None:
        model.mono.refresh_weights(model)
        return
    group = get_tp_group()
    rank, world_size = group.rank_in_group, group.world_size
    reasons = deployment_refusals(lm, rank, world_size)
    if not reasons:
        reasons = layer_refusals(model)
    if not reasons:
        _ensure_writable_flydsl_cache()
        try:
            runtime = importlib.import_module(f"{__name__}.runtime")
        except ImportError as e:
            reasons.append(f"needs FlyDSL ({e})")
    verdict = torch.tensor([0 if reasons else 1], dtype=torch.int32)
    dist.all_reduce(verdict, op=dist.ReduceOp.MIN, group=group.cpu_group)
    if int(verdict.item()) == 0:
        raise ValueError(
            "--monokernel: Muse Spark cannot run the gfx942 decode kernels: "
            + ("; ".join(reasons) or "a peer rank refused the deployment")
        )
    widths = runtime.built_widths(lm.vllm_config)
    route_scale = float(
        lm.vllm_config.model_config.hf_text_config.routed_scaling_factor
    )
    model.mono = runtime.SpecializedDecode(
        model,
        rank=rank,
        group=group.cpu_group,
        device=torch.device("cuda", torch.accelerator.current_device_index()),
        widths=widths,
        route_scale=route_scale,
    )
    logger.info(
        "Muse Spark gfx942 decode attached: layers 1-60 of decode steps of %s "
        "rows run as FlyDSL launches",
        ",".join(map(str, widths)),
    )
