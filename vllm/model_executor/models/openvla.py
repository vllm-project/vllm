# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterable

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.model_executor.models.interfaces import (
    MultiModalEmbeddings,
    SupportsMultiModal,
    SupportsPP,
)
from vllm.sequence import IntermediateTensors


class OpenVLAForActionPrediction(nn.Module, SupportsMultiModal, SupportsPP):
    """Phase-1 registration skeleton for OpenVLA.

    The executable OpenVLA model is implemented in later phases.  Keeping this
    class importable lets config and architecture registration be verified
    independently before adding vision towers, projector, language model, and
    action decoding.
    """

    embed_input_ids = SupportsMultiModal.embed_input_ids

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("image"):
            return None
        raise ValueError("Only image modality is supported")

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        self.config = vllm_config.model_config.hf_config

    def get_language_model(self) -> nn.Module:
        raise NotImplementedError("OpenVLA execution is implemented in a later phase")

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        raise NotImplementedError("OpenVLA execution is implemented in a later phase")

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        *,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors:
        raise NotImplementedError("OpenVLA execution is implemented in a later phase")

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        raise NotImplementedError("OpenVLA execution is implemented in a later phase")

    def make_empty_intermediate_tensors(
        self,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> IntermediateTensors:
        raise NotImplementedError("OpenVLA execution is implemented in a later phase")

    def get_num_mm_encoder_tokens(self, num_image_tokens: int) -> int:
        raise NotImplementedError("OpenVLA execution is implemented in a later phase")

    def get_num_mm_connector_tokens(self, num_vision_tokens: int) -> int:
        raise NotImplementedError("OpenVLA execution is implemented in a later phase")

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        raise NotImplementedError("OpenVLA execution is implemented in a later phase")
