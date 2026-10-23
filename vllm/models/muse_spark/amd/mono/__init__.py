# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FlyDSL decode monokernel for MuseSpark1x on MI355X (``--monokernel``)."""

import importlib
import os
from pathlib import Path

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
    MonoKernels,
    MuseSparkMono,
    _routed_experts,
    built_buckets,
    config_refusals,
    deployment_refusals,
    fused_attention_layers,
    layer_refusals,
)

logger = init_logger(__name__)


def _ensure_writable_flydsl_cache() -> None:
    """AITER points FlyDSL at its bundled, read-only cache; new kernels need a
    writable one."""
    cur = os.environ.get("FLYDSL_RUNTIME_CACHE_DIR")
    if cur and os.access(cur, os.W_OK):
        return
    path = Path.home() / ".flydsl" / "cache"
    path.mkdir(parents=True, exist_ok=True)
    os.environ["FLYDSL_RUNTIME_CACHE_DIR"] = str(path)


def _load_kernels() -> tuple[MonoKernels | None, str | None]:
    _ensure_writable_flydsl_cache()
    try:
        modules = {
            name: importlib.import_module(f"{__name__}.kernels.{name}")
            for name in ("k3", "k5", "k5m", "attn_frag")
        }
    except ImportError as e:
        return None, f"needs FlyDSL and AITER's FlyDSL helpers ({e})"
    return MonoKernels(**modules), None


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
    text_config = lm.vllm_config.model_config.hf_text_config
    reasons += config_refusals(text_config)
    if reasons:
        return reasons
    experts = _routed_experts(model.layers[FIRST_LAYER])
    moe_parallel = getattr(
        getattr(experts, "moe_config", None), "moe_parallel_config", None
    )
    reasons += deployment_refusals(lm.vllm_config, tp_rank, tp_size, moe_parallel)
    if reasons:
        return reasons
    return layer_refusals(model, MonoDims.from_config(text_config, tp_size))


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
    text_config = lm.vllm_config.model_config.hf_text_config
    reasons = static_refusals(lm, rank, world_size)
    kernels = None
    if not reasons:
        kernels, missing = _load_kernels()
        if missing is not None:
            reasons.append(missing)
    verdict = torch.tensor([0 if reasons else 1], dtype=torch.int32)
    dist.all_reduce(verdict, op=dist.ReduceOp.MIN, group=group.cpu_group)
    if int(verdict.item()) == 0:
        raise ValueError(
            "--monokernel: Muse Spark cannot run the monokernel: "
            + ("; ".join(reasons) or "a peer rank refused the deployment")
        )
    assert kernels is not None
    buckets = built_buckets(lm.vllm_config.scheduler_config.max_num_seqs)
    max_model_len = lm.vllm_config.model_config.max_model_len
    fused = fused_attention_layers(
        model.layers, max_model_len, kernels.attn_frag.VLLM_MAX_KEYS
    )
    if not fused:
        logger.info(
            "Muse Spark monokernel: verify attention unfused (max_model_len %d, "
            "fused builds cover %d)",
            max_model_len,
            kernels.attn_frag.VLLM_MAX_KEYS,
        )
    model.mono = MuseSparkMono(
        model,
        kernels,
        rank=rank,
        group=group.cpu_group,
        device=torch.device("cuda", torch.accelerator.current_device_index()),
        dims=MonoDims.from_config(text_config, world_size),
        buckets=buckets,
        fused_attention=fused,
    )
    logger.info(
        "Muse Spark monokernel attached: layers %d-%d of verify steps with "
        "up to %d rows run as FlyDSL launches (builds S=%s; MoE activations "
        "in bf16)",
        FIRST_LAYER,
        LAST_LAYER,
        buckets[-1],
        ",".join(map(str, buckets)),
    )


__all__ = ["MuseSparkMono", "maybe_attach_mono"]
