# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch
import torch.nn as nn

from vllm.config import VllmConfig, get_layers_from_vllm_config
from vllm.model_executor.layers.attention import Attention, CrossAttention
from vllm.v1.attention.backend import AttentionType
from vllm.v1.worker.cpu.encoder_cache import EncoderCache
from vllm.v1.worker.cpu.model_states.interface import ModelState


def init_model_state(
    vllm_config: VllmConfig,
    model: nn.Module,
    encoder_cache: EncoderCache | None,
    device: torch.device,
) -> ModelState:
    cls = resolve_model_state_cls(vllm_config, model)
    return cls(vllm_config, model, encoder_cache, device)


def resolve_model_state_cls(
    vllm_config: VllmConfig, model: nn.Module
) -> type[ModelState]:
    # Custom model states are GPU-owned until a CPU implementation is provided.
    if hasattr(model, "get_model_state_cls"):
        raise ValueError(
            f"Custom model state from {type(model).__name__} is not supported "
            "by CPU Model Runner V2 without Triton."
        )

    # Cross-attention encoder-decoder models (Whisper, CohereASR, ...).
    if any(isinstance(module, CrossAttention) for module in model.modules()):
        from vllm.v1.worker.cpu.model_states.encoder_decoder import (
            EncoderDecoderModelState,
        )

        return EncoderDecoderModelState

    # Encoder-only attention is non-causal and needs no KV cache.
    if any(
        layer.attn_type == AttentionType.ENCODER_ONLY
        for layer in get_layers_from_vllm_config(vllm_config, Attention).values()
    ):
        from vllm.v1.worker.cpu.model_states.encoder_only import EncoderOnlyModelState

        return EncoderOnlyModelState

    if vllm_config.model_config.is_hybrid or vllm_config.model_config.is_attention_free:
        from vllm.v1.worker.cpu.model_states.mamba_hybrid import MambaHybridModelState

        return MambaHybridModelState

    from vllm.v1.worker.cpu.model_states.default import DefaultModelState

    return DefaultModelState
