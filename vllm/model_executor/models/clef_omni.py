# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The Qwen3-Omni thinker with a Clef joint schema head."""

import math
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from transformers.feature_extraction_utils import BatchFeature

from vllm.config import VllmConfig
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
from vllm.multimodal import MULTIMODAL_REGISTRY

from .clef_joint_schema import JOINT_HEAD_WEIGHTS, JointSchemaHead, JointSchemaPooler
from .interfaces_base import default_pooling_type
from .qwen3_omni_moe_thinker import (
    Qwen3OmniMoeThinkerDummyInputsBuilder,
    Qwen3OmniMoeThinkerForConditionalGeneration,
    Qwen3OmniMoeThinkerMultiModalProcessor,
    Qwen3OmniMoeThinkerProcessingInfo,
)

MAX_FRAME_PIXELS = 256 * 32 * 32


def cap_frame_pixels(frames: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    """The reference reads each video frame at at most 256 * 32 * 32 pixels,
    in even sizes. Frames are NHWC uint8, numpy or tensor; tensors stay on
    their device (a CUDA tensor from torchcodec must not round-trip)."""
    height, width = frames.shape[1:3]
    if height * width <= MAX_FRAME_PIXELS:
        return frames
    scale = math.sqrt(MAX_FRAME_PIXELS / (width * height))
    new_width = max(2, round(width * scale / 2) * 2)
    new_height = max(2, round(height * scale / 2) * 2)
    is_tensor = isinstance(frames, torch.Tensor)
    tensor = frames if is_tensor else torch.from_numpy(frames)
    resized = F.interpolate(
        tensor.permute(0, 3, 1, 2).float(),
        size=(new_height, new_width),
        mode="bilinear",
    )
    if tensor.dtype == torch.uint8:
        resized = resized.round()
    resized = resized.to(tensor.dtype).permute(0, 2, 3, 1).contiguous()
    return resized if is_tensor else resized.numpy()


def _cap_video(item: Any) -> Any:
    if isinstance(item, tuple):
        frames, metadata = item
        return cap_frame_pixels(frames), metadata
    return cap_frame_pixels(item)


class ClefOmniMultiModalProcessor(Qwen3OmniMoeThinkerMultiModalProcessor):
    """Caps video frames at the reference's per-frame pixel count."""

    def _call_hf_processor(
        self,
        hf_data: Mapping[str, object],
        hf_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        if videos := hf_data.get("videos"):
            assert isinstance(videos, list)
            hf_data = {**hf_data, "videos": [_cap_video(v) for v in videos]}
        return super()._call_hf_processor(hf_data, hf_kwargs)


@default_pooling_type(tok_pooling_type="ALL")
@MULTIMODAL_REGISTRY.register_processor(
    ClefOmniMultiModalProcessor,
    info=Qwen3OmniMoeThinkerProcessingInfo,
    dummy_inputs=Qwen3OmniMoeThinkerDummyInputsBuilder,
)
class ClefOmniForJointSchema(Qwen3OmniMoeThinkerForConditionalGeneration):
    is_pooling_model = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        model_config = vllm_config.model_config
        head_config = model_config.hf_config.joint_head_config
        self.pooler = JointSchemaPooler(
            JointSchemaHead(**head_config), self._embed_lexical
        )
        # The head's weights ship as their own safetensors file.
        self.secondary_weights = [
            DefaultModelLoader.Source(
                model_or_path=model_config.model,
                revision=model_config.revision,
                prefix="pooler.head.",
                allow_patterns_overrides=[JOINT_HEAD_WEIGHTS],
            )
        ]

    def _embed_lexical(self, token_ids: torch.Tensor) -> torch.Tensor:
        # The head embeds options with the output embeddings. ParallelLMHead
        # forbids lookups, so use the vocab-parallel embedding's.
        return VocabParallelEmbedding.forward(self.language_model.lm_head, token_ids)
