# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FlyDSL decode monokernel for MuseSpark1x on MI355X (``--monokernel``)."""

import importlib
from types import ModuleType

import torch
import torch.distributed as dist
from torch import nn

from vllm.distributed import get_tp_group
from vllm.logger import init_logger
from vllm.platforms import current_platform

from .decoder import (
    FIRST_LAYER,
    LAST_LAYER,
    NUM_CUS,
    MonoDims,
    MuseSparkMono,
    _routed_experts,
    config_refusals,
    deployment_refusals,
    layer_refusals,
)

logger = init_logger(__name__)


def _load_k3() -> tuple[ModuleType | None, str | None]:
    try:
        return importlib.import_module(f"{__name__}.kernels.k3"), None
    except ImportError as e:
        return None, f"FlyDSL K3 kernels unavailable ({e})"


def static_refusals(lm: nn.Module, tp_rank: int, tp_size: int) -> list[str]:
    """Every reason this deployment cannot run the monokernel; empty = supported."""
    reasons: list[str] = []
    if not current_platform.is_rocm():
        return ["not ROCm"]
    from vllm.platforms.rocm import on_gfx950

    if not on_gfx950():
        reasons.append("not gfx950")
    cus = torch.cuda.get_device_properties(
        torch.accelerator.current_device_index()
    ).multi_processor_count
    if cus != NUM_CUS:
        reasons.append(f"{cus} CUs (kernel: {NUM_CUS})")
    model = lm.model
    reasons += config_refusals(model.config)
    if reasons:
        return reasons
    experts = _routed_experts(model.layers[FIRST_LAYER])
    moe_parallel = getattr(
        getattr(experts, "moe_config", None), "moe_parallel_config", None
    )
    reasons += deployment_refusals(lm.vllm_config, tp_rank, tp_size, moe_parallel)
    if reasons:
        return reasons
    return layer_refusals(model, MonoDims.from_config(model.config, tp_size))


def maybe_attach_mono(lm: nn.Module) -> None:
    """Attach the monokernel to ``lm.model`` after the weights are final.

    Collective over the TP group: every rank calls it. A deployment that any
    rank cannot run raises on every rank, with every reason of this rank.
    """
    if not lm.vllm_config.kernel_config.monokernel:
        return
    model = lm.model
    if model.mono is not None:
        model.mono.refresh_weights(model)
        return
    group = get_tp_group()
    rank, world_size = group.rank_in_group, group.world_size
    reasons = static_refusals(lm, rank, world_size)
    k3 = None
    if not reasons:
        k3, missing = _load_k3()
        if missing is not None:
            reasons.append(missing)
    verdict = torch.tensor([0 if reasons else 1], dtype=torch.int32)
    dist.all_reduce(verdict, op=dist.ReduceOp.MIN, group=group.cpu_group)
    if int(verdict.item()) == 0:
        raise ValueError(
            "--monokernel: Muse Spark cannot run the monokernel: "
            + ("; ".join(reasons) or "a peer rank refused the deployment")
        )
    assert k3 is not None
    model.mono = MuseSparkMono(
        model,
        k3,
        rank=rank,
        group=group.cpu_group,
        device=torch.device("cuda", torch.accelerator.current_device_index()),
        dims=MonoDims.from_config(model.config, world_size),
    )
    logger.info(
        "Muse Spark monokernel attached: layers %d-%d of one-request verify "
        "steps run as FlyDSL K3 launches (MoE activations in bf16)",
        FIRST_LAYER,
        LAST_LAYER,
    )


__all__ = ["MuseSparkMono", "maybe_attach_mono"]
