# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Config-stage refusal: every reason at once, default-deny on features."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

from vllm import envs
from vllm.logger import init_logger

from .spec import Feature, MonoSpec

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)

SWITCH = "VLLM_ROCM_MONO_DECODE"


def enabled() -> bool:
    return envs.VLLM_ROCM_MONO_DECODE


def load_spec(path: str) -> MonoSpec:
    """``"package.module:NAME"`` -> the spec object, importing only that module."""
    module, _, name = path.partition(":")
    spec = getattr(importlib.import_module(module), name)
    if not isinstance(spec, MonoSpec):
        raise TypeError(f"{path} is not a MonoSpec")
    return spec


def features_in_use(vllm_config: VllmConfig) -> set[Feature]:
    pc = vllm_config.parallel_config
    mc = vllm_config.model_config
    offload = vllm_config.offload_config
    spec_cfg = vllm_config.speculative_config
    used = {
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
        Feature.SPECULATIVE_DECODING: spec_cfg is not None,
        Feature.SLEEP_MODE: mc.enable_sleep_mode,
        Feature.CPU_WEIGHT_OFFLOAD: (
            offload.uva.cpu_offload_gb > 0 or offload.prefetch.offload_group_size > 0
        ),
        Feature.WEIGHT_UPDATES: vllm_config.weight_transfer_config is not None,
        Feature.PROMPT_EMBEDS: mc.enable_prompt_embeds,
        Feature.EAGLE3_AUX_LAYERS: (
            spec_cfg is not None
            and getattr(spec_cfg, "method", None) in ("eagle3", "dflash2")
        ),
    }
    return {f for f, on in used.items() if on}


def graph_reasons(vllm_config: VllmConfig, spec: MonoSpec) -> list[str]:
    from vllm.config.compilation import CUDAGraphMode

    cc = vllm_config.compilation_config
    reasons = []
    mode = cc.cudagraph_mode
    if mode is not None and mode != CUDAGraphMode.NONE:
        if mode.decode_mode() == CUDAGraphMode.PIECEWISE:
            reasons.append(
                "piecewise-only CUDA graphs carry no attention metadata "
                "(use FULL_AND_PIECEWISE, FULL_DECODE_ONLY or FULL)"
            )
        tokens = set(spec_tokens(spec))
        bad = [
            s
            for s in cc.cudagraph_capture_sizes or ()
            if s <= max(tokens) and s not in tokens
        ]
        if bad:
            reasons.append(
                f"CUDA graph capture sizes {bad} are not kernel step sizes "
                f"{sorted(tokens)}"
            )
    if envs.VLLM_USE_BREAKABLE_CUDAGRAPH:
        reasons.append("breakable CUDA graphs")
    return reasons


def spec_tokens(spec: MonoSpec) -> tuple[int, ...]:
    return tuple(spec.options.get("tokens", spec.widths))


def refuse_config(
    vllm_config: VllmConfig, spec: MonoSpec, compute_units: int | None
) -> list[str]:
    """Every reason this deployment cannot run ``spec`` (empty: supported)."""
    reasons = []
    if not vllm_config.use_v2_model_runner:
        reasons.append("needs Model Runner V2")
    reasons += [
        f.value for f in sorted(features_in_use(vllm_config) - spec.supports, key=str)
    ]
    reasons += graph_reasons(vllm_config, spec)
    if compute_units is not None and compute_units < spec.grid:
        reasons.append(
            f"needs {spec.grid} compute units for a resident grid, "
            f"the GPU has {compute_units}"
        )
    reasons += spec.refuse_config(vllm_config)
    return reasons


def configure(vllm_config: VllmConfig, spec_path: str) -> MonoSpec | None:
    """Config-stage check when the switch is on: raises with every reason."""
    if not enabled():
        return None
    from vllm.config.compilation import CUDAGraphMode
    from vllm.platforms import current_platform

    spec = load_spec(spec_path)
    compute_units = None
    if current_platform.is_rocm():
        import torch

        compute_units = current_platform.num_compute_units(
            torch.accelerator.current_device_index()
        )
    reasons = refuse_config(vllm_config, spec, compute_units)
    if reasons:
        raise ValueError(
            f"{SWITCH}=1 cannot run the {spec.name} MonoKernel: "
            + "; ".join(reasons)
            + "."
        )
    if vllm_config.compilation_config.cudagraph_mode == CUDAGraphMode.NONE:
        logger.warning("%s MonoKernel running eagerly: for debugging only.", spec.name)
    return spec
