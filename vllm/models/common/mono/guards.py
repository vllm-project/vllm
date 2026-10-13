# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Refusals. ``configure``, the only name a model calls, runs once from core
config post-init, before load: it imports only the model's spec module, picks
the variant this config can serve, adds its widths to the CUDA graph capture
sizes and raises with every reason the config is refused. The rest is the
library's own and stays private to this package: ``_agree_across_ranks`` runs a
later refusal stage (after load) on every TP rank and agrees on the verdict
before anything collective follows.
"""

import importlib
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

from vllm.config.compilation import CUDAGraphMode
from vllm.logger import init_logger
from vllm.models.common.mono.peer import tp_agree
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


def _agree_across_ranks(refuse: Callable[[], list[str]], group, what: str) -> None:
    """``refuse``'s reasons on this rank, then every TP rank's verdict agreed: a
    rank that refused alone would leave the others spinning in-kernel.

    Raises:
        MonoUnsupported: This rank refused, or another rank did.

    """
    reasons = refuse()
    agreed = tp_agree(not reasons, group)
    if reasons:
        raise MonoUnsupported(f"{what}: " + "; ".join(reasons))
    if not agreed:
        raise MonoUnsupported(f"{what}: another TP rank refused")
