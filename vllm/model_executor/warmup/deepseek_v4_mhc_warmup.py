# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Warm up DeepSeek V4 mHC TileLang kernels before serving requests.

Ported from lucifer1004/vllm-jasl with the two env-var knobs removed
(`VLLM_ENABLE_DEEPSEEK_V4_MHC_WARMUP`, `VLLM_DEEPSEEK_V4_MHC_WARMUP_TOKEN_SIZES`).
Gating is intrinsic: non-DSv4 models and layers without hc_* attributes
return early, so the warmup is a no-op except where it's needed.
"""

import time
from collections.abc import Iterable

import torch

from vllm.logger import init_logger
from vllm.tracing import instrument
from vllm.utils.math_utils import cdiv

logger = init_logger(__name__)


def _compute_mhc_pre_num_split(
    *,
    num_tokens: int,
    hidden_size: int,
    hc_mult: int,
    num_sms: int,
) -> int:
    block_k = 64
    block_m = 64
    k = hc_mult * hidden_size
    grid_size = cdiv(num_tokens, block_m)
    split_k = num_sms // grid_size
    num_block_k = cdiv(k, block_k)
    split_k = min(split_k, num_block_k // 4)
    return max(split_k, 1)


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

    # Warm up every size to avoid TileLang JIT during inference.
    candidates = list(range(1, max_tokens + 1))
    candidates.extend(cudagraph_capture_sizes)
    return _normalize_token_sizes(candidates, max_tokens=max_tokens)


def _find_first_mhc_layer(model: torch.nn.Module) -> torch.nn.Module | None:
    for module in model.modules():
        if module.__class__.__name__ != "DeepseekV4DecoderLayer":
            continue
        if all(
            hasattr(module, attr)
            for attr in (
                "hc_pre",
                "hc_post",
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

    # Use real RMSNorm weights so norm-fused TileLang kernels are warmed up
    # with the same tensors passed at runtime.
    norm_configs = (
        (
            layer.hc_attn_fn,
            layer.hc_attn_scale,
            layer.hc_attn_base,
            layer.attn_norm.weight.data,
            float(layer.attn_norm.variance_epsilon),
        ),
        (
            layer.hc_ffn_fn,
            layer.hc_ffn_scale,
            layer.hc_ffn_base,
            layer.ffn_norm.weight.data,
            float(layer.ffn_norm.variance_epsilon),
        ),
    )

    for i, size in enumerate(token_sizes):
        if i % 1000 == 0:
            logger.info(
                "mHC warmup progress: %d/%d (size=%d)",
                i,
                len(token_sizes),
                size,
            )
        residual_slice = residual[:size]
        # Dummy inputs for the fused post+pre variant.
        x_dummy = torch.zeros(
            size, hidden_size, dtype=torch.bfloat16, device=device
        )
        post_mix_dummy = torch.zeros(
            size, hc_mult, 1, dtype=torch.float32, device=device
        )
        comb_mix_dummy = torch.zeros(
            size, hc_mult, hc_mult, dtype=torch.float32, device=device
        )
        for fn, scale, base, norm_weight, norm_eps in norm_configs:
            layer_input, post_mix, comb_mix = layer.hc_pre(
                residual_slice,
                fn,
                scale,
                base,
                norm_weight=norm_weight,
                norm_eps=norm_eps,
            )
            layer.hc_post(layer_input, residual_slice, post_mix, comb_mix)

            # Warm up the fused post+pre variant used after the first layer.
            torch.ops.vllm.mhc_fused_post_pre_tilelang(
                x_dummy,
                residual_slice,
                post_mix_dummy,
                comb_mix_dummy,
                fn,
                scale,
                base,
                layer.rms_norm_eps,
                layer.hc_eps,
                layer.hc_eps,
                layer.hc_post_alpha,
                layer.hc_sinkhorn_iters,
                n_splits=1,
                tile_n=1,
                norm_weight=norm_weight,
                norm_eps=norm_eps,
            )


def _warmup_hc_head(
    model: torch.nn.Module,
    token_sizes: list[int],
) -> None:
    # Exercise the same HCHeadOp instance used during inference.
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


@instrument(span_name="DeepSeek V4 mHC warmup")
def deepseek_v4_mhc_warmup(
    model: torch.nn.Module,
    *,
    max_tokens: int,
    cudagraph_capture_sizes: list[int] | None = None,
) -> None:
    # Bail out early for non-DeepSeek-V4 models to avoid walking modules.
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
