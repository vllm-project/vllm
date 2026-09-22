# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.model_executor.layers.attention.attention import Attention
from vllm.model_executor.layers.attention.chunked_local_attention import (
    ChunkedLocalAttention,
)
from vllm.model_executor.layers.attention.cross_attention import CrossAttention
from vllm.model_executor.layers.attention.encoder_only_attention import (
    EncoderOnlyAttention,
)
from vllm.model_executor.layers.attention.mla_attention import MLAAttention
from vllm.model_executor.layers.attention.mm_encoder_attention import MMEncoderAttention
from vllm.model_executor.layers.attention.prefill_prefix_lm_attention import (
    PrefillPrefixLMAttention,
)
from vllm.model_executor.layers.attention.rswa_attention import RSWAAttention
from vllm.model_executor.layers.attention.static_sink_attention import (
    StaticSinkAttention,
)
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase


def is_deferred_attention_layer(layer: torch.nn.Module) -> bool:
    """Whether an attention-like layer requires deferred post-load processing."""
    return isinstance(layer, (AttentionLayerBase, MMEncoderAttention)) and callable(
        getattr(layer, "process_weights_after_loading", None)
    )


__all__ = [
    "Attention",
    "ChunkedLocalAttention",
    "CrossAttention",
    "EncoderOnlyAttention",
    "MLAAttention",
    "MMEncoderAttention",
    "PrefillPrefixLMAttention",
    "RSWAAttention",
    "StaticSinkAttention",
    "is_deferred_attention_layer",
]
