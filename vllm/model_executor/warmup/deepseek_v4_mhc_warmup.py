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
    MhcKernelConstants,
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


def _build_kernel_constants(layer: torch.nn.Module) -> MhcKernelConstants:
    """Read all model-level constants that appear in the TileLang cache_key.

    These values do not vary with num_tokens, so they are read once from the
    layer and threaded into every compile() call.  This ensures the warmup
    cache_key matches the runtime cache_key exactly, regardless of the
    model's configuration.

    Sources (DeepseekV4DecoderLayer):
        hc_post_alpha        — hardcoded 2.0 in all DSv4 variants
        hc_sinkhorn_iters     — from config.hc_sinkhorn_iters
        rms_norm_eps          — from config.rms_norm_eps
        hc_eps                — from config.hc_eps
        attn_norm.variance_epsilon — == rms_norm_eps (RMSNorm init)
    """
    return MhcKernelConstants(
        hc_post_mult_value=float(getattr(layer, "hc_post_alpha", 2.0)),
        sinkhorn_repeat=int(getattr(layer, "hc_sinkhorn_iters", 20)),
        rms_eps=float(getattr(layer, "rms_norm_eps", 1e-6)),
        hc_pre_eps=float(getattr(layer, "hc_eps", 1e-6)),
        hc_sinkhorn_eps=float(getattr(layer, "hc_eps", 1e-6)),
        norm_eps=float(layer.attn_norm.variance_epsilon),
    )


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
    # Broadcast (2D-residual first-layer) path exists only when the model
    # has hc_attn_fn_broadcast — NVIDIA sets it in _configure_fused_norm,
    # AMD/XPU do not.  This is independent of use_norm_weight: it reflects
    # whether the runtime code has a broadcast branch, not whether norm
    # is fused.
    has_broadcast = getattr(layer, "hc_attn_fn_broadcast", None) is not None
    is_broadcast_values = [False, True] if has_broadcast else [False]

    constants = _build_kernel_constants(layer)

    MHC_PRE_KERNEL.warmup(
        vllm_config,
        hidden_size=hidden_size,
        hc_mult=hc_mult,
        use_norm_weight=use_norm_weight,
        is_broadcast_values=is_broadcast_values,
        constants=constants,
    )
    MHC_FUSED_POST_PRE_KERNEL.warmup(
        vllm_config,
        hidden_size=hidden_size,
        hc_mult=hc_mult,
        use_norm_weight=use_norm_weight,
        constants=constants,
    )

    if _find_deepseek_v4_model(model) is not None:
        HC_HEAD_FUSED_KERNEL.warmup(
            vllm_config,
            hidden_size=hidden_size,
            hc_mult=hc_mult,
            use_norm_weight=use_norm_weight,
            constants=constants,
        )

    torch.accelerator.synchronize()
