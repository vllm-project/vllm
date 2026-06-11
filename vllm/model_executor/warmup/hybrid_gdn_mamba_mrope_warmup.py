# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Warm up hybrid GDN/Mamba and MRoPE kernels."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
    QwenGatedDeltaNetAttention,
)
from vllm.model_executor.layers.rotary_embedding.mrope import MRotaryEmbedding

logger = init_logger(__name__)


def _iter_qwen_gdn_layers(
    model: torch.nn.Module,
) -> Iterable[QwenGatedDeltaNetAttention]:
    for module in model.modules():
        if isinstance(module, QwenGatedDeltaNetAttention):
            yield module


def _iter_mrope_attention_modules(
    model: torch.nn.Module,
) -> Iterable[tuple[Any, MRotaryEmbedding]]:
    for module in model.modules():
        rotary_emb = getattr(module, "rotary_emb", None)
        if (
            isinstance(rotary_emb, MRotaryEmbedding)
            and rotary_emb.mrope_section
            and all(
                hasattr(module, attr)
                for attr in ("num_heads", "num_kv_heads", "head_dim")
            )
        ):
            yield module, rotary_emb


def has_hybrid_gdn_mamba_mrope(model: torch.nn.Module) -> bool:
    """Return whether the model has kernels covered by this warmup helper."""
    return any(_iter_qwen_gdn_layers(model)) or any(
        _iter_mrope_attention_modules(model)
    )


def _warmup_qwen_gdn_layer(
    layer: QwenGatedDeltaNetAttention,
    *,
    model_dtype: torch.dtype,
) -> None:
    qkv_dim = (layer.key_dim * 2 + layer.value_dim) // layer.tp_size
    device = layer.A_log.device
    qkv = torch.empty((1, qkv_dim), device=device, dtype=model_dtype)
    layer._warmup_prefill_kernels(qkv, 0)


def _warmup_mrope_module(
    module: Any,
    rotary_emb: MRotaryEmbedding,
    *,
    model_dtype: torch.dtype,
) -> None:
    device = rotary_emb.cos_sin_cache.device
    num_tokens = 1
    positions = torch.zeros((3, num_tokens), device=device, dtype=torch.long)
    q = torch.empty(
        (num_tokens, module.num_heads * module.head_dim),
        device=device,
        dtype=model_dtype,
    )
    k = torch.empty(
        (num_tokens, module.num_kv_heads * module.head_dim),
        device=device,
        dtype=model_dtype,
    )
    rotary_emb.forward_cuda(positions, q, k)


def hybrid_gdn_mamba_mrope_warmup(
    model: torch.nn.Module,
    *,
    model_dtype: torch.dtype,
) -> None:
    """Warm kernels missed by generic model dummy runs.

    Qwen3.5/Qwen3-Next style hybrid models use GDN/Mamba kernels for linear
    attention layers and MRoPE for some full-attention layers.  These kernels
    can be first compiled on the first real request if profile/dummy runs do not
    exercise their exact runtime variants.
    """
    with torch.inference_mode():
        warmed_gdn_keys: set[tuple[Any, ...]] = set()
        warmed_gdn_count = 0
        for layer in _iter_qwen_gdn_layers(model):
            gdn_key = (
                str(layer.A_log.device),
                model_dtype,
                layer.key_dim,
                layer.value_dim,
                layer.tp_size,
                layer.num_k_heads,
                layer.num_v_heads,
                layer.head_k_dim,
                layer.head_v_dim,
                layer.gqa_interleaved_layout,
            )
            if gdn_key in warmed_gdn_keys:
                continue
            warmed_gdn_keys.add(gdn_key)
            try:
                _warmup_qwen_gdn_layer(layer, model_dtype=model_dtype)
                warmed_gdn_count += 1
            except Exception:
                logger.warning(
                    "GDN/Mamba JIT warmup failed for layer %s. "
                    "First inference may JIT compile.",
                    layer.prefix,
                    exc_info=True,
                )

        warmed_mrope_keys: set[tuple[Any, ...]] = set()
        warmed_mrope_count = 0
        for module, rotary_emb in _iter_mrope_attention_modules(model):
            mrope_key = (
                str(rotary_emb.cos_sin_cache.device),
                model_dtype,
                module.num_heads,
                module.num_kv_heads,
                module.head_dim,
                rotary_emb.rotary_dim,
                tuple(rotary_emb.mrope_section or ()),
                rotary_emb.mrope_interleaved,
            )
            if mrope_key in warmed_mrope_keys:
                continue
            warmed_mrope_keys.add(mrope_key)
            try:
                _warmup_mrope_module(module, rotary_emb, model_dtype=model_dtype)
                warmed_mrope_count += 1
            except Exception:
                logger.warning(
                    "MRoPE JIT warmup failed. First inference may JIT compile.",
                    exc_info=True,
                )

    if warmed_gdn_count or warmed_mrope_count:
        logger.info(
            "Warmed up %d GDN/Mamba and %d MRoPE JIT kernel variant(s).",
            warmed_gdn_count,
            warmed_mrope_count,
        )
