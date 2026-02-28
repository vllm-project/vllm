# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.v1.worker.gpu.mm.encoder_cache import EncoderCache
from vllm.v1.worker.gpu.model_states.default import ModelState
from vllm.v1.worker.gpu.model_states.interface import ModelStateInterface
from vllm.v1.worker.gpu.model_states.whisper import WhisperModelState


def create_model_state(
    vllm_config: VllmConfig,
    model: nn.Module,
    encoder_cache: EncoderCache | None,
    device: torch.device,
) -> ModelStateInterface:
    if "WhisperForConditionalGeneration" in vllm_config.model_config.architectures:
        return WhisperModelState(vllm_config, model, encoder_cache, device)
    return ModelState(vllm_config, model, encoder_cache, device)
