# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Warm up DeepSeek V4 mHC TileLang kernels before serving requests.

Ported from lucifer1004/vllm-jasl with the two env-var knobs removed
(`VLLM_ENABLE_DEEPSEEK_V4_MHC_WARMUP`, `VLLM_DEEPSEEK_V4_MHC_WARMUP_TOKEN_SIZES`).
Gating is intrinsic: non-DSv4 models and layers without hc_* attributes
return early, so the warmup is a no-op except where it's needed.
"""

import time
from collections.abc import Callable, Iterable
from typing import Any

import torch

from vllm.logger import init_logger
from vllm.tracing import instrument

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


def _select_mhc_warmup_token_sizes(
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
        has_mhc_parameters = all(
            hasattr(module, attr)
            for attr in (
                "hc_attn_fn",
                "hc_attn_scale",
                "hc_attn_base",
                "hc_ffn_fn",
                "hc_ffn_scale",
                "hc_ffn_base",
            )
        )
        has_custom_ops = all(
            callable(getattr(module, attr, None)) for attr in ("hc_pre", "hc_post")
        )
        has_nvidia_tilelang_ops = all(
            hasattr(module, attr) for attr in ("attn_norm", "ffn_norm")
        )
        if has_mhc_parameters and (has_custom_ops or has_nvidia_tilelang_ops):
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


def _get_tilelang_mhc_ops() -> tuple[Callable[..., Any], ...]:
    # Import lazily so non-NVIDIA platforms keep using their registered
    # CustomOps without importing the NVIDIA TileLang implementation.
    from vllm.model_executor.kernels.mhc.tilelang import (
        hc_head_fused_kernel_tilelang,
        mhc_fused_post_pre_tilelang,
        mhc_post_tilelang,
        mhc_pre_broadcast_tilelang,
        mhc_pre_tilelang,
    )

    return (
        mhc_pre_tilelang,
        mhc_pre_broadcast_tilelang,
        mhc_fused_post_pre_tilelang,
        mhc_post_tilelang,
        hc_head_fused_kernel_tilelang,
    )


def _warmup_nvidia_layer_mhc(
    layer: torch.nn.Module,
    residual: torch.Tensor,
    token_sizes: list[int],
) -> None:
    mhc_pre, mhc_pre_broadcast, mhc_fused_post_pre, mhc_post, _ = (
        _get_tilelang_mhc_ops()
    )

    fn_broadcast = getattr(layer, "hc_attn_fn_broadcast", None)
    if fn_broadcast is not None:
        broadcast_residual = torch.zeros(
            residual.shape[0],
            residual.shape[2],
            dtype=residual.dtype,
            device=residual.device,
        )
        for size in token_sizes:
            mhc_pre_broadcast(
                broadcast_residual[:size],
                layer.hc_attn_fn,
                layer.hc_attn_scale,
                layer.hc_attn_base,
                layer.rms_norm_eps,
                layer.hc_eps,
                layer.hc_eps,
                layer.hc_post_alpha,
                layer.hc_sinkhorn_iters,
                norm_weight=layer.attn_norm.weight.data,
                norm_eps=layer.attn_norm.variance_epsilon,
                fn_broadcast=fn_broadcast,
            )

    for size in token_sizes:
        residual_slice = residual[:size]
        post_mix, res_mix, layer_input = mhc_pre(
            residual_slice,
            layer.hc_attn_fn,
            layer.hc_attn_scale,
            layer.hc_attn_base,
            layer.rms_norm_eps,
            layer.hc_eps,
            layer.hc_eps,
            layer.hc_post_alpha,
            layer.hc_sinkhorn_iters,
            norm_weight=layer.attn_norm.weight.data,
            norm_eps=layer.attn_norm.variance_epsilon,
        )
        residual_cur, post_mix, res_mix, layer_input = mhc_fused_post_pre(
            layer_input,
            residual_slice,
            post_mix,
            res_mix,
            layer.hc_ffn_fn,
            layer.hc_ffn_scale,
            layer.hc_ffn_base,
            layer.rms_norm_eps,
            layer.hc_eps,
            layer.hc_eps,
            layer.hc_post_alpha,
            layer.hc_sinkhorn_iters,
            n_splits=1,
            tile_n=1,
            norm_weight=layer.ffn_norm.weight.data,
            norm_eps=layer.ffn_norm.variance_epsilon,
        )
        mhc_post(layer_input, residual_cur, post_mix, res_mix)


def _warmup_layer_mhc(
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

    hc_pre = getattr(layer, "hc_pre", None)
    hc_post = getattr(layer, "hc_post", None)
    if not callable(hc_pre) or not callable(hc_post):
        _warmup_nvidia_layer_mhc(layer, residual, token_sizes)
        return

    for size in token_sizes:
        residual_slice = residual[:size]
        for fn, scale, base in (
            (layer.hc_attn_fn, layer.hc_attn_scale, layer.hc_attn_base),
            (layer.hc_ffn_fn, layer.hc_ffn_scale, layer.hc_ffn_base),
        ):
            layer_input, post_mix, comb_mix = hc_pre(
                residual_slice,
                fn,
                scale,
                base,
            )
            hc_post(layer_input, residual_slice, post_mix, comb_mix)


def _warmup_hc_head(
    model: torch.nn.Module,
    token_sizes: list[int],
) -> None:
    hc_head_op = getattr(model, "hc_head_op", None)
    if hc_head_op is None:
        # The NVIDIA implementation calls the TileLang function directly,
        # while the AMD and XPU implementations attach an HCHeadOp.
        _, _, _, _, hc_head_op = _get_tilelang_mhc_ops()

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


@instrument(span_name="DeepSeek V4 mHC warmup")
def deepseek_v4_mhc_warmup(
    model: torch.nn.Module,
    *,
    max_tokens: int,
    cudagraph_capture_sizes: list[int] | None = None,
) -> None:
    # Cheap model-type gate before walking ``model.modules()``. The class
    # walk below is O(num_layers) and shows up in startup time on very
    # large checkpoints; bail out for any model that is not DeepSeek V4.
    config = getattr(model, "config", None)
    model_type = getattr(config, "model_type", None) if config is not None else None
    if model_type is not None and model_type != "deepseek_v4":
        return

    layer = _find_first_mhc_layer(model)
    if layer is None:
        return

    device = layer.hc_attn_fn.device
    if device.type != "cuda":
        return

    deepseek_model = _find_deepseek_v4_model(model)
    token_sizes = _select_mhc_warmup_token_sizes(
        max_tokens=max_tokens,
        cudagraph_capture_sizes=cudagraph_capture_sizes or [],
    )
    if not token_sizes:
        return

    started = time.perf_counter()
    logger.info(
        "Warming up DeepSeek V4 mHC TileLang kernels for token sizes: %s",
        token_sizes,
    )
    with torch.inference_mode():
        _warmup_layer_mhc(layer, token_sizes)
        if deepseek_model is not None:
            _warmup_hc_head(deepseek_model, token_sizes)
        torch.accelerator.synchronize()
    logger.info(
        "DeepSeek V4 mHC TileLang warmup finished in %.2f seconds.",
        time.perf_counter() - started,
    )
