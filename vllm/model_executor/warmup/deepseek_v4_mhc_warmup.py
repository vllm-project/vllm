# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Register DeepSeek V4 mHC TileLang kernels for startup warmup.

The per-kernel dispatch / compile-key enumeration / compile logic lives next
to the kernel definitions in
``vllm/model_executor/kernels/mhc/warmup.py`` (kernel-owned warmup contract
per RFC #47456 / PR #47451).
"""

from __future__ import annotations

import time
from collections.abc import Iterable
from typing import TYPE_CHECKING

import torch

from vllm.logger import init_logger
from vllm.model_executor.kernels.mhc.warmup import (
    HC_HEAD_FUSED_KERNEL,
    MHC_FUSED_POST_PRE_KERNEL,
    MHC_PRE_KERNEL,
    MhcKernelConstants,
)
from vllm.tracing import instrument

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)

_AUTO_WARMUP_MAX_TOKENS = 16_384
_DEFAULT_TOKEN_SIZE_CANDIDATES = (
    1,
    2,
    4,
    8,
    16,
    32,
    64,
    128,
    256,
    512,
    1024,
    2048,
    4096,
    8192,
    16_384,
)


def _normalize_token_sizes(
    token_sizes: Iterable[int],
    *,
    max_tokens: int,
) -> list[int]:
    return sorted({size for size in token_sizes if 1 <= size <= max_tokens})


def _select_custom_op_warmup_token_sizes(
    *,
    max_tokens: int,
    cudagraph_capture_sizes: list[int],
) -> list[int]:
    if max_tokens <= 0:
        return []

    max_auto_tokens = min(max_tokens, _AUTO_WARMUP_MAX_TOKENS)
    candidates = list(_DEFAULT_TOKEN_SIZE_CANDIDATES)
    candidates.extend(cudagraph_capture_sizes)
    candidates.append(max_auto_tokens)
    return _normalize_token_sizes(candidates, max_tokens=max_auto_tokens)


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


def _find_mhc_head_module(model: torch.nn.Module) -> torch.nn.Module | None:
    for module in model.modules():
        if all(
            hasattr(module, attr)
            for attr in ("hc_head_fn", "hc_head_scale", "hc_head_base")
        ):
            return module
    return None


def _warmup_custom_op_mhc_layer(
    layer: torch.nn.Module,
    token_sizes: list[int],
) -> None:
    max_tokens = max(token_sizes)
    hidden_size = int(layer.hidden_size)
    hc_mult = int(layer.hc_mult)
    device = layer.hc_attn_fn.device
    residual = torch.zeros(
        max_tokens,
        hc_mult,
        hidden_size,
        dtype=torch.bfloat16,
        device=device,
    )

    for size in token_sizes:
        residual_slice = residual[:size]
        for fn, scale, base in (
            (layer.hc_attn_fn, layer.hc_attn_scale, layer.hc_attn_base),
            (layer.hc_ffn_fn, layer.hc_ffn_scale, layer.hc_ffn_base),
        ):
            layer_input, post_mix, comb_mix = layer.hc_pre(
                residual_slice,
                fn,
                scale,
                base,
            )
            layer.hc_post(layer_input, residual_slice, post_mix, comb_mix)


def _warmup_custom_op_hc_head(
    model: torch.nn.Module,
    token_sizes: list[int],
) -> None:
    hc_head_op = getattr(model, "hc_head_op", None)
    if hc_head_op is None:
        return

    max_tokens = max(token_sizes)
    hidden_size = int(model.config.hidden_size)
    hc_mult = int(model.hc_mult)
    device = model.hc_head_fn.device
    hidden_states = torch.zeros(
        max_tokens,
        hc_mult,
        hidden_size,
        dtype=torch.bfloat16,
        device=device,
    )

    for size in token_sizes:
        hc_head_op(
            hidden_states[:size],
            model.hc_head_fn,
            model.hc_head_scale,
            model.hc_head_base,
            model.rms_norm_eps,
            model.hc_eps,
        )


@instrument(span_name="DeepSeek V4 mHC CustomOp warmup")
def deepseek_v4_mhc_custom_op_warmup(
    model: torch.nn.Module,
    *,
    max_tokens: int,
    cudagraph_capture_sizes: list[int] | None = None,
) -> None:
    """Preserve the existing AMD CustomOp warmup path.

    NVIDIA decoder layers do not expose ``hc_pre`` / ``hc_post`` and are
    handled by :func:`register_deepseek_v4_mhc_warmup` instead.
    """
    config = getattr(model, "config", None)
    model_type = getattr(config, "model_type", None) if config is not None else None
    if model_type is not None and model_type != "deepseek_v4":
        return

    layer = _find_first_mhc_layer(model)
    if layer is None or not all(hasattr(layer, attr) for attr in ("hc_pre", "hc_post")):
        return
    if layer.hc_attn_fn.device.type != "cuda":
        return

    token_sizes = _select_custom_op_warmup_token_sizes(
        max_tokens=max_tokens,
        cudagraph_capture_sizes=cudagraph_capture_sizes or [],
    )
    if not token_sizes:
        return

    started = time.perf_counter()
    logger.info(
        "Warming up DeepSeek V4 mHC CustomOps for token sizes: %s",
        token_sizes,
    )
    with torch.inference_mode():
        _warmup_custom_op_mhc_layer(layer, token_sizes)
        head = _find_mhc_head_module(model)
        if head is not None:
            _warmup_custom_op_hc_head(head, token_sizes)
        torch.accelerator.synchronize()
    logger.info(
        "DeepSeek V4 mHC CustomOp warmup finished in %.2f seconds.",
        time.perf_counter() - started,
    )


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


def register_deepseek_v4_mhc_warmup(
    model: torch.nn.Module,
    *,
    vllm_config: VllmConfig,
    include_broadcast: bool,
    include_head: bool,
) -> None:
    """Register every mHC TileLang specialization this pipeline rank may use.

    This is called while the model runner's :class:`JitWarmupRegistry` is
    active. The registry later expands ``WarmupIntRange`` to the actual
    compile-key set and compiles each key once.
    """
    if not vllm_config.kernel_config.enable_jit_warmup:
        return

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
    # NVIDIA always fuses RMSNorm. AMD may fuse it when TileLang is selected;
    # its layer records the effective choice in ``fuse_mhc_rmsnorm``.
    use_norm_weight = bool(
        getattr(layer, "fuse_mhc_rmsnorm", not hasattr(layer, "mhc_pre"))
    )
    is_broadcast_values = [False, True] if include_broadcast else [False]

    constants = _build_kernel_constants(layer)

    MHC_PRE_KERNEL.register_warmup(
        vllm_config,
        hidden_size=hidden_size,
        hc_mult=hc_mult,
        use_norm_weight=use_norm_weight,
        is_broadcast_values=is_broadcast_values,
        constants=constants,
    )
    MHC_FUSED_POST_PRE_KERNEL.register_warmup(
        vllm_config,
        hidden_size=hidden_size,
        hc_mult=hc_mult,
        use_norm_weight=use_norm_weight,
        constants=constants,
    )

    if include_head and _find_mhc_head_module(model) is not None:
        HC_HEAD_FUSED_KERNEL.register_warmup(
            vllm_config,
            hidden_size=hidden_size,
            hc_mult=hc_mult,
            use_norm_weight=use_norm_weight,
            constants=constants,
        )
