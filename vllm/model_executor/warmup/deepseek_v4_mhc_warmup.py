# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek V4 mHC TileLang warmup entry point."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from vllm.model_executor.kernels.mhc.tilelang_kernels import (
    HC_HEAD_FUSED_TILELANG_KERNEL,
    MHC_FUSED_TILELANG_KERNEL,
    MHC_POST_TILELANG_KERNEL,
    MHC_PRE_BIG_FUSE_TILELANG_KERNEL,
)
from vllm.tracing import instrument

if TYPE_CHECKING:
    from vllm.config import VllmConfig


def _find_first_mhc_layer(model: torch.nn.Module) -> torch.nn.Module | None:
    from vllm.models.deepseek_v4.nvidia.model import DeepseekV4DecoderLayer

    for module in model.modules():
        if isinstance(module, DeepseekV4DecoderLayer):
            return module
    return None


def _find_deepseek_v4_model(model: torch.nn.Module) -> torch.nn.Module | None:
    from vllm.models.deepseek_v4.nvidia.model import DeepseekV4Model

    for module in model.modules():
        if isinstance(module, DeepseekV4Model):
            return module
    return None


@instrument(span_name="DeepSeek V4 mHC warmup")
def deepseek_v4_mhc_warmup(
    model: torch.nn.Module,
    *,
    vllm_config: VllmConfig,
) -> None:
    config = getattr(model, "config", None)
    model_type = getattr(config, "model_type", None) if config is not None else None
    if model_type is not None and model_type != "deepseek_v4":
        return

    layer = _find_first_mhc_layer(model)
    if layer is None or layer.hc_attn_fn.device.type != "cuda":
        return

    hidden_size = int(layer.hidden_size)
    hc_mult = int(layer.hc_mult)
    has_broadcast = getattr(layer, "hc_attn_fn_broadcast", None) is not None
    attn_broadcast = (False, True) if has_broadcast else (False,)

    MHC_PRE_BIG_FUSE_TILELANG_KERNEL.warmup(
        vllm_config,
        hidden_size=hidden_size,
        hc_mult=hc_mult,
        use_norm_weight=True,
        is_broadcast=attn_broadcast,
        rms_eps=float(layer.rms_norm_eps),
        hc_pre_eps=float(layer.hc_eps),
        hc_sinkhorn_eps=float(layer.hc_eps),
        hc_post_mult_value=float(layer.hc_post_alpha),
        sinkhorn_repeat=int(layer.hc_sinkhorn_iters),
        norm_eps=float(layer.attn_norm.variance_epsilon),
    )
    MHC_PRE_BIG_FUSE_TILELANG_KERNEL.warmup(
        vllm_config,
        hidden_size=hidden_size,
        hc_mult=hc_mult,
        use_norm_weight=True,
        is_broadcast=(False,),
        rms_eps=float(layer.rms_norm_eps),
        hc_pre_eps=float(layer.hc_eps),
        hc_sinkhorn_eps=float(layer.hc_eps),
        hc_post_mult_value=float(layer.hc_post_alpha),
        sinkhorn_repeat=int(layer.hc_sinkhorn_iters),
        norm_eps=float(layer.ffn_norm.variance_epsilon),
    )
    MHC_POST_TILELANG_KERNEL.warmup(
        vllm_config,
        hidden_size=hidden_size,
        hc_mult=hc_mult,
    )
    MHC_FUSED_TILELANG_KERNEL.warmup(
        vllm_config,
        hidden_size=hidden_size,
        hc_mult=hc_mult,
    )

    deepseek_model = _find_deepseek_v4_model(model)
    if deepseek_model is not None:
        HC_HEAD_FUSED_TILELANG_KERNEL.warmup(
            vllm_config,
            hidden_size=int(deepseek_model.config.hidden_size),
            hc_mult=int(deepseek_model.hc_mult),
            rms_eps=float(deepseek_model.rms_norm_eps),
            hc_eps=float(deepseek_model.hc_eps),
        )
