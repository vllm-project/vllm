# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Warm up DeepSeek V4 mHC TileLang kernels before serving requests.

Ported from lucifer1004/vllm-jasl with the two env-var knobs removed
(`VLLM_ENABLE_DEEPSEEK_V4_MHC_WARMUP`, `VLLM_DEEPSEEK_V4_MHC_WARMUP_TOKEN_SIZES`).
Gating is intrinsic: non-DSv4 models and layers without hc_* attributes
return early, so the warmup is a no-op except where it's needed.

The warmup path matches each platform's inference code: NVIDIA fuses
RMSNorm into the TileLang kernels, while AMD/XPU apply RMSNorm outside
and use MHCPreOp / MHCFusedPostPreOp wrappers.
"""

from collections.abc import Iterable

import torch
from tqdm import tqdm

from vllm.distributed.parallel_state import is_global_first_rank
from vllm.tracing import instrument
from vllm.utils.math_utils import cdiv

# Auto-warmup token sizes. TileLang mHC kernels treat ``num_tokens`` as a
# dynamic dimension, but the underlying split-k / small-FMA / block-M paths
# have breakpoints at small powers of two. A sparse power-of-2 grid covers
# those distinct kernel configurations without warming up every integer up to
# ``max_num_batched_tokens``.
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

    # Warm up a sparse set of token sizes that covers the distinct kernel
    # configurations (small-FMA branches, split-k transitions, block-M
    # specializations) instead of every integer in [1, max_tokens]. Always
    # include ``max_tokens`` itself and any CUDA-graph capture sizes, since
    # those exact shapes are exercised at runtime.
    max_auto_tokens = min(max_tokens, _AUTO_WARMUP_MAX_TOKENS)
    candidates = [
        size
        for size in _DEFAULT_TOKEN_SIZE_CANDIDATES
        if size <= max_auto_tokens
    ]
    candidates.append(max_tokens)
    candidates.extend(cudagraph_capture_sizes)
    return _normalize_token_sizes(candidates, max_tokens=max_tokens)


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


def _warmup_layer_mhc(
    layer: torch.nn.Module,
    token_sizes: list[int],
    pbar: tqdm | None = None,
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

    # NVIDIA's decoder layer calls the TileLang ops directly and fuses
    # RMSNorm into mhc_pre / mhc_fused_post_pre. AMD/XPU layers wrap those
    # ops in MHCPreOp / MHCFusedPostPreOp and apply RMSNorm separately.
    use_fused_norm = not hasattr(layer, "mhc_pre")

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

    for size in token_sizes:
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
            if use_fused_norm:
                post_mix, comb_mix, layer_input = torch.ops.vllm.mhc_pre_tilelang(
                    residual_slice,
                    fn,
                    scale,
                    base,
                    layer.rms_norm_eps,
                    layer.hc_eps,
                    layer.hc_eps,
                    layer.hc_post_alpha,
                    layer.hc_sinkhorn_iters,
                    norm_weight=norm_weight,
                    norm_eps=norm_eps,
                )
                torch.ops.vllm.mhc_post_tilelang(
                    layer_input, residual_slice, post_mix, comb_mix
                )
            else:
                layer_input, post_mix, comb_mix = layer.hc_pre(
                    residual_slice, fn, scale, base
                )
                layer.hc_post(layer_input, residual_slice, post_mix, comb_mix)

            # Warm up the fused post+pre variant used after the first layer.
            if use_fused_norm:
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
            else:
                layer.mhc_fused_post_pre(
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
                )
        if pbar is not None:
            pbar.update(1)


def _warmup_hc_head(
    model: torch.nn.Module,
    token_sizes: list[int],
    pbar: tqdm | None = None,
) -> None:
    # Exercise the same HCHeadOp instance used during inference, or on
    # NVIDIA the direct TileLang kernel that is called from the model.
    hc_head_op = getattr(model, "hc_head_op", None)
    use_op = hc_head_op is None

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
        hs_slice = hidden_states[:size]
        if use_op:
            torch.ops.vllm.hc_head_fused_kernel_tilelang(
                hs_slice,
                model.hc_head_fn,
                model.hc_head_scale,
                model.hc_head_base,
                model.rms_norm_eps,
                model.hc_eps,
            )
        else:
            hc_head_op(
                hs_slice,
                model.hc_head_fn,
                model.hc_head_scale,
                model.hc_head_base,
                model.rms_norm_eps,
                model.hc_eps,
            )
        if pbar is not None:
            pbar.update(1)


@instrument(span_name="mHC warmup")
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

    total = len(token_sizes)
    if deepseek_model is not None:
        total += len(token_sizes)

    with torch.inference_mode():
        if is_global_first_rank():
            with tqdm(total=total, desc="mHC warmup") as pbar:
                _warmup_layer_mhc(layer, token_sizes, pbar)
                if deepseek_model is not None:
                    _warmup_hc_head(deepseek_model, token_sizes, pbar)
                torch.accelerator.synchronize()
        else:
            _warmup_layer_mhc(layer, token_sizes, None)
            if deepseek_model is not None:
                _warmup_hc_head(deepseek_model, token_sizes, None)
            torch.accelerator.synchronize()
