# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.model_executor.layers.attention import CrossAttention
from vllm.v1.worker.gpu.mm.encoder_cache import EncoderCache


def init_model_state(
    vllm_config: VllmConfig,
    model: nn.Module,
    encoder_cache: EncoderCache | None,
    device: torch.device,
):
    # Let the model provide its own ModelState if it defines one.
    if hasattr(model, "get_model_state_cls"):
        cls = model.get_model_state_cls()
        return cls(vllm_config, model, encoder_cache, device)

    # Cross-attention encoder-decoder models (Whisper, CohereASR, NemotronParse, ...)
    if any(isinstance(m, CrossAttention) for m in model.modules()):
        from vllm.v1.worker.gpu.model_states.encoder_decoder import (
            EncoderDecoderModelState,
        )

        return EncoderDecoderModelState(vllm_config, model, encoder_cache, device)

    if vllm_config.model_config.is_hybrid:
        from vllm.v1.worker.gpu.model_states.mamba_hybrid import MambaHybridModelState

        return MambaHybridModelState(vllm_config, model, encoder_cache, device)

    from vllm.v1.worker.gpu.model_states.default import DefaultModelState

    return DefaultModelState(vllm_config, model, encoder_cache, device)
