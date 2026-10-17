# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Refusals. ``configure`` runs once from core config post-init, before load:
it imports only the model's spec, adds the widths to the CUDA graph capture
sizes and raises with every reason the config is refused. ``agree`` runs a
later refusal stage (after load) on every TP rank and agrees on the verdict
before anything collective follows."""

import importlib
from collections.abc import Callable
from typing import TYPE_CHECKING

import torch

from vllm.config.compilation import CUDAGraphMode
from vllm.logger import init_logger
from vllm.models.common.mono.spec import Feature, MonoSpec, MonoUnsupported

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)


def enabled(vllm_config: "VllmConfig") -> bool:
    """``--monokernel``."""
    return vllm_config.kernel_config.monokernel


def load_spec(path: str) -> MonoSpec:
    """``"package.module:NAME"`` -> the spec, importing only that module."""
    module, name = path.split(":")
    return getattr(importlib.import_module(module), name)


def features_in_use(vllm_config: "VllmConfig") -> set[Feature]:
    pc = vllm_config.parallel_config
    oc = vllm_config.offload_config
    on = {
        Feature.PIPELINE_PARALLEL: pc.pipeline_parallel_size > 1,
        Feature.DATA_PARALLEL: pc.data_parallel_size > 1,
        Feature.EXPERT_PARALLEL: pc.enable_expert_parallel,
        Feature.EPLB: pc.enable_eplb,
        Feature.DECODE_CONTEXT_PARALLEL: pc.decode_context_parallel_size > 1,
        Feature.PREFILL_CONTEXT_PARALLEL: pc.prefill_context_parallel_size > 1,
        Feature.LORA: vllm_config.lora_config is not None,
        Feature.KV_CONNECTOR: vllm_config.kv_transfer_config is not None,
        Feature.ROUTED_EXPERT_CAPTURE: (
            vllm_config.aux_output_config.enable_return_routed_experts
        ),
        Feature.SPECULATIVE_DECODING: vllm_config.speculative_config is not None,
        Feature.SLEEP_MODE: vllm_config.model_config.enable_sleep_mode,
        Feature.CPU_WEIGHT_OFFLOAD: (
            oc.uva.cpu_offload_gb > 0 or oc.prefetch.offload_group_size > 0
        ),
        Feature.WEIGHT_UPDATES: vllm_config.weight_transfer_config is not None,
    }
    return {f for f, used in on.items() if used}


def refuse_config(spec: MonoSpec, vllm_config: "VllmConfig") -> list[str]:
    """Default-deny, the runner and the graph mode, then the spec's own."""
    reasons = [
        f"{f.value} is not supported"
        for f in sorted(features_in_use(vllm_config) - spec.supports)
    ]
    if not vllm_config.use_v2_model_runner:
        reasons.append("needs Model Runner V2")
    mode = vllm_config.compilation_config.cudagraph_mode
    if mode is not None and mode not in (
        CUDAGraphMode.NONE,
        CUDAGraphMode.FULL,
        CUDAGraphMode.FULL_DECODE_ONLY,
        CUDAGraphMode.FULL_AND_PIECEWISE,
    ):
        # PIECEWISE captures carry no attention metadata
        reasons.append(f"CUDA graph mode {mode.name}: decode needs FULL graphs")
    return reasons + spec.refuse_config(vllm_config)


def set_capture_sizes(spec: MonoSpec, vllm_config: "VllmConfig") -> None:
    """The capture sizes up to the largest width become exactly the widths, so
    a replay never pads a request's rows; larger sizes stay as configured (and
    capture vLLM's path)."""
    cc = vllm_config.compilation_config
    widths = spec.widths(vllm_config)
    sizes = cc.cudagraph_capture_sizes or []
    new = sorted(set(widths) | {s for s in sizes if s > max(widths)})
    if new != sorted(sizes):
        logger.info("MonoKernel %s: CUDA graph capture sizes %s", spec.name, new)
        cc.cudagraph_capture_sizes = new


def configure(vllm_config: "VllmConfig", spec_path: str) -> None:
    spec = load_spec(spec_path)
    reasons = refuse_config(spec, vllm_config)
    if reasons:
        raise MonoUnsupported(f"MonoKernel {spec.name} refused: " + "; ".join(reasons))
    set_capture_sizes(spec, vllm_config)
    if vllm_config.compilation_config.cudagraph_mode == CUDAGraphMode.NONE:
        logger.warning(
            "MonoKernel %s under eager mode: for debugging only, serve with FULL "
            "CUDA graphs",
            spec.name,
        )


def agree(refuse: Callable[[], list[str]], group, what: str) -> None:
    """``refuse``'s reasons on this rank, then every TP rank's verdict agreed: a
    rank that refused alone would leave the others spinning in-kernel."""
    reasons = refuse()
    flag = torch.tensor([int(not reasons)], dtype=torch.int32)
    torch.distributed.all_reduce(flag, op=torch.distributed.ReduceOp.MIN, group=group)
    if reasons:
        raise MonoUnsupported(f"{what}: " + "; ".join(reasons))
    if not flag.item():
        raise MonoUnsupported(f"{what}: another TP rank refused")
