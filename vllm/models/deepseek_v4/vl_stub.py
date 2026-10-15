# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stub for platforms where the DeepSeek-V4 vision variant is unsupported."""

import torch
from torch import nn

from vllm.model_executor.models.interfaces import SupportsMultiModal
from vllm.models.deepseek_v4.common.mm_preprocess import (
    IMAGE_PLACEHOLDER,
    DeepseekV4VLDummyInputsBuilder,
    DeepseekV4VLMultiModalProcessor,
    DeepseekV4VLProcessingInfo,
)
from vllm.multimodal import MULTIMODAL_REGISTRY


@MULTIMODAL_REGISTRY.register_processor(
    DeepseekV4VLMultiModalProcessor,
    info=DeepseekV4VLProcessingInfo,
    dummy_inputs=DeepseekV4VLDummyInputsBuilder,
)
class DeepseekV4ForConditionalGeneration(nn.Module, SupportsMultiModal):
    """CPU/XPU placeholder.

    Registry inspection treats this as a multimodal generative model so
    processor-only routing metadata can be constructed on CPU. Instantiating
    the module for generation remains unsupported.
    """

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality == "image":
            return IMAGE_PLACEHOLDER
        raise ValueError(f"Unsupported modality: {modality!r}")

    def __init__(self, vllm_config, prefix: str = ""):
        super().__init__()
        raise NotImplementedError(
            "DeepSeek-V4 vision (DeepseekV4ForConditionalGeneration) is not "
            "supported on the current platform; use CUDA or ROCm."
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        raise NotImplementedError
