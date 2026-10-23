# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Common validation logic for specialized decode kernels."""

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import torch
import torch.distributed as dist

from vllm.config.compilation import CUDAGraphMode
from vllm.logger import init_logger
from vllm.models.common.specialized_decode.spec import (
    DecodeSpec,
    DecodeUnsupported,
    Feature,
    KernelT,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)

_FULL_MODES = (
    CUDAGraphMode.NONE,
    CUDAGraphMode.FULL,
    CUDAGraphMode.FULL_DECODE_ONLY,
    CUDAGraphMode.FULL_AND_PIECEWISE,
)


def tp_group() -> tuple[int, int, object | None]:
    """This rank's (rank, world size, CPU group) in the TP group, and
    (0, 1, None) before distributed is initialized."""
    if not dist.is_available() or not dist.is_initialized():
        return 0, 1, None
    from vllm.distributed import get_tp_group

    tp = get_tp_group()
    return tp.rank_in_group, tp.world_size, tp.cpu_group


def tp_agree(ok: bool, group) -> bool:
    """Agree on a boolean across the TP group.

    Args:
        ok: This rank's local verdict.
        group: The TP group's CPU (gloo) process group, or None for one rank.

    """
    if group is None:
        return ok
    flag = torch.tensor([int(ok)], dtype=torch.int32)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN, group=group)
    return bool(flag.item())


def _enabled(vllm_config: "VllmConfig") -> bool:
    """Whether the engine is configured to run specialized decode."""
    return bool(vllm_config.kernel_config.specialized_decode)


def _features_in_use(vllm_config: "VllmConfig") -> set[Feature]:
    """Features in use by this config."""
    pc = vllm_config.parallel_config
    oc = vllm_config.offload_config
    on = {
        Feature.PIPELINE_PARALLEL: pc.pipeline_parallel_size > 1,
        Feature.DATA_PARALLEL: pc.data_parallel_size > 1,
        Feature.EXPERT_PARALLEL: pc.enable_expert_parallel,
        Feature.EPLB: pc.enable_eplb,
        Feature.DECODE_CONTEXT_PARALLEL: pc.decode_context_parallel_size > 1,
        Feature.UBATCHING: pc.use_ubatching,
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


def _refuse_config(spec: DecodeSpec[Any], vllm_config: "VllmConfig") -> list[str]:
    """The reasons the config cannot run the specialized decode kernel, if any."""
    reasons = [
        f"{f.value} is not supported"
        for f in sorted(_features_in_use(vllm_config) - spec.supports)
    ]
    if not vllm_config.use_v2_model_runner:
        reasons.append("needs Model Runner V2")
    mode = vllm_config.compilation_config.cudagraph_mode
    if mode is not None and mode not in _FULL_MODES:
        reasons.append(f"CUDA graph mode {mode.name}: decode needs FULL graphs")
    return reasons + list(spec.refuse_config(vllm_config))


def _refuse_model(spec: DecodeSpec[Any], model) -> list[str]:
    """The reasons the mode cannot run the specialized decode kernel, if any."""
    from vllm.compilation.wrapper import TorchCompileWithNoGuardsWrapper

    reasons = []
    if isinstance(model, TorchCompileWithNoGuardsWrapper):
        reasons.append(
            "model is torch.compile'd: specialized decode replaces the layer loop, "
            "so the model must not carry @support_torch_compile"
        )
    return reasons + list(spec.refuse_model(model))


def _select_spec(
    specs: Sequence[DecodeSpec[Any]], vllm_config: "VllmConfig"
) -> tuple[DecodeSpec[Any] | None, dict[str, list[str]]]:
    """Get the first compatible spec."""
    refusals: dict[str, list[str]] = {}
    for spec in specs:
        reasons = _refuse_config(spec, vllm_config)
        if not reasons:
            return spec, refusals
        refusals[spec.name] = reasons
    return None, refusals


def _set_capture_sizes(spec: DecodeSpec[Any], vllm_config: "VllmConfig") -> None:
    """Amend the capture sizes with the supported widths, so that no step is
    refused for its width alone."""
    cc = vllm_config.compilation_config
    widths = spec.widths(vllm_config)
    sizes = cc.cudagraph_capture_sizes or []
    new = sorted(set(widths) | {s for s in sizes if s > max(widths)})
    if new != sorted(sizes):
        logger.info(
            "Specialized decode %s: CUDA graph capture sizes %s", spec.name, new
        )
        cc.cudagraph_capture_sizes = new


def select_decode_spec(
    vllm_config: "VllmConfig",
    specs: DecodeSpec[KernelT] | Sequence[DecodeSpec[KernelT]],
) -> DecodeSpec[KernelT] | None:
    """Select the first compatible spec and, unless it declines them, set the
    capture sizes to its widths.

    Args:
        vllm_config: The engine's config, before the weights are loaded.
        specs: The model's variants, in the order they are preferred.

    """
    if not _enabled(vllm_config):
        return None
    ordered = (specs,) if isinstance(specs, DecodeSpec) else tuple(specs)
    spec, refusals = _select_spec(ordered, vllm_config)
    if spec is None:
        raise DecodeUnsupported(
            "Specialized decode refused: "
            + "; ".join(f"{n}: {', '.join(rs)}" for n, rs in refusals.items())
        )
    if spec.set_capture_sizes:
        _set_capture_sizes(spec, vllm_config)
    if vllm_config.compilation_config.cudagraph_mode == CUDAGraphMode.NONE:
        logger.warning(
            "Specialized decode %s under eager mode: for debugging only, serve "
            "with FULL CUDA graphs",
            spec.name,
        )
    return spec
