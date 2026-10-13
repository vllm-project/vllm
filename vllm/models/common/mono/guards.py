# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Refusals: the variant a config can serve, and TP agreement on every stage."""

import importlib
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import torch
import torch.distributed as dist

from vllm.config.compilation import CUDAGraphMode
from vllm.logger import init_logger
from vllm.models.common.mono.spec import Feature, MonoSpec, MonoUnsupported

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
    """Agrees a verdict every TP rank must take alike.

    A MonoKernel waits in-kernel on every peer rank, so a rank that turned mono
    off alone, or failed to build what the others launch, leaves the others
    spinning on the GPU for ever. Every rank must call this at the same point.

    Args:
        ok: This rank's local verdict.
        group: The TP group's CPU (gloo) process group, or None for one rank.

    Returns:
        True when every rank of ``group`` passed ``ok`` true.

    """
    if group is None:
        return ok
    flag = torch.tensor([int(ok)], dtype=torch.int32)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN, group=group)
    return bool(flag.item())


def _enabled(vllm_config: "VllmConfig") -> bool:
    """Whether ``--monokernel`` asked for the model's MonoKernels."""
    return bool(vllm_config.kernel_config.monokernel)


def _load_specs(path: str) -> tuple[MonoSpec[Any], ...]:
    """``"package.module:NAME"`` -> the spec variants, importing only that
    module. ``NAME`` is a ``MonoSpec`` or a tuple of them, in the order they
    are preferred."""
    module, name = path.split(":")
    specs = getattr(importlib.import_module(module), name)
    return (specs,) if isinstance(specs, MonoSpec) else tuple(specs)


def _features_in_use(vllm_config: "VllmConfig") -> set[Feature]:
    """The deployment's features a spec has to list in ``supports``."""
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


def _refuse_config(spec: MonoSpec[Any], vllm_config: "VllmConfig") -> list[str]:
    """Every reason this config cannot run ``spec``: default-deny on features,
    then the runner and the graph mode, then the spec's own."""
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


def _refuse_model(spec: MonoSpec[Any], model) -> list[str]:
    """Every reason the loaded model cannot run ``spec``. A MonoKernel replaces
    the layer loop, which a traced graph has already inlined, so a model
    carrying ``@support_torch_compile`` is refused."""
    from vllm.compilation.wrapper import TorchCompileWithNoGuardsWrapper

    reasons = []
    if isinstance(model, TorchCompileWithNoGuardsWrapper):
        reasons.append(
            "model is torch.compile'd: MonoKernels replace the layer loop, "
            "so the model must not carry @support_torch_compile"
        )
    return reasons + list(spec.refuse_model(model))


def _select_spec(
    specs: Sequence[MonoSpec[Any]], vllm_config: "VllmConfig"
) -> tuple[MonoSpec[Any] | None, dict[str, list[str]]]:
    """The first variant this config can serve, and every variant's refusals.

    Returns:
        (spec, refusals). ``spec`` is None when every variant refused, and
        ``refusals`` holds each variant's reasons, by spec name.

    """
    refusals: dict[str, list[str]] = {}
    for spec in specs:
        reasons = _refuse_config(spec, vllm_config)
        if not reasons:
            return spec, refusals
        refusals[spec.name] = reasons
    return None, refusals


def _refusal_message(refusals: dict[str, list[str]]) -> str:
    """Every variant's reasons as one message."""
    return "; ".join(
        f"{name}: {', '.join(reasons)}" for name, reasons in refusals.items()
    )


def _set_capture_sizes(spec: MonoSpec[Any], vllm_config: "VllmConfig") -> None:
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


def configure(vllm_config: "VllmConfig", specs_path: str) -> None:
    """Picks the variant, sets the capture sizes, or raises with every reason.
    A no-op with ``--monokernel`` off.

    Args:
        vllm_config: The engine's config, before the weights are loaded.
        specs_path: ``"package.module:MONO_SPECS"``.

    Raises:
        MonoUnsupported: Every variant refused this config.

    """
    if not _enabled(vllm_config):
        return
    spec, refusals = _select_spec(_load_specs(specs_path), vllm_config)
    if spec is None:
        raise MonoUnsupported("MonoKernel refused: " + _refusal_message(refusals))
    _set_capture_sizes(spec, vllm_config)
    if vllm_config.compilation_config.cudagraph_mode == CUDAGraphMode.NONE:
        logger.warning(
            "MonoKernel %s under eager mode: for debugging only, serve with FULL "
            "CUDA graphs",
            spec.name,
        )
