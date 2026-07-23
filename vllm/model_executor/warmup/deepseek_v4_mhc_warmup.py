# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Warm up DeepSeek V4 mHC TileLang kernels before serving requests.

Caller-side entry point. The per-kernel dispatch / compile-key enumeration /
compile logic lives next to the kernel definitions in
``vllm/model_executor/kernels/mhc/warmup.py`` (kernel-owned warmup contract
per RFC #47456 / PR #47451).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from vllm.model_executor.kernels.mhc.warmup import (
    HC_HEAD_FUSED_KERNEL,
    MHC_FUSED_POST_PRE_KERNEL,
    MHC_PRE_KERNEL,
)
from vllm.tracing import instrument

if TYPE_CHECKING:
    from vllm.config import VllmConfig


def _find_first_mhc_layer(model: torch.nn.Module) -> torch.nn.Module | None:
    for module in model.modules():
        if module.__class__.__name__ != "DeepseekV4DecoderLayer":
            continue
        if all(
            hasattr(module, attr)
            for attr in (
                "hc_attn_fn",
                "hc_attn_scale",
                "hc_attn_base",
                "hc_ffn_fn",
                "hc_ffn_scale",
                "hc_ffn_base",
            )
        ):
            return module
    return None


def _find_deepseek_v4_model(model: torch.nn.Module) -> torch.nn.Module | None:
    for module in model.modules():
        if module.__class__.__name__ != "DeepseekV4Model":
            continue
        if all(
            hasattr(module, attr)
            for attr in ("hc_head_fn", "hc_head_scale", "hc_head_base")
        ):
            return module
    return None


@instrument(span_name="mHC warmup")
def deepseek_v4_mhc_warmup(
    model: torch.nn.Module,
    *,
    vllm_config: VllmConfig,
) -> None:
    """Pre-compile every mHC TileLang specialization the runtime may invoke.

    No-op for non-DeepSeek-V4 models and non-CUDA devices. Each wrapper's
    ``warmup()`` expands ``WarmupIntRange(1, max_tokens+1)`` to the actual
    compile-key set (deduplicated by the AST tracer) and calls ``.compile()``
    on the underlying TileLang kernels (compile-only, no launch).
    """
    config = getattr(model, "config", None)
    model_type = getattr(config, "model_type", None) if config is not None else None
    if model_type is not None and model_type != "deepseek_v4":
        return

    layer = _find_first_mhc_layer(model)
    if layer is None:
        return

    if layer.hc_attn_fn.device.type != "cuda":
        return

    hidden_size = int(layer.hidden_size)
    hc_mult = int(layer.hc_mult)
    # NVIDIA fuses RMSNorm into the TileLang kernels (norm_weight path);
    # AMD/XPU apply RMSNorm separately (norm_weight=None path).
    use_norm_weight = not hasattr(layer, "mhc_pre")

    MHC_PRE_KERNEL.warmup(
        vllm_config,
        hidden_size=hidden_size,
        hc_mult=hc_mult,
        use_norm_weight=use_norm_weight,
    )
    MHC_FUSED_POST_PRE_KERNEL.warmup(
        vllm_config,
        hidden_size=hidden_size,
        hc_mult=hc_mult,
        use_norm_weight=use_norm_weight,
    )

    if _find_deepseek_v4_model(model) is not None:
        HC_HEAD_FUSED_KERNEL.warmup(
            vllm_config,
            hidden_size=hidden_size,
            hc_mult=hc_mult,
            use_norm_weight=use_norm_weight,
        )

    torch.accelerator.synchronize()
