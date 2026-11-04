# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only MonkeyOCRv2.

MonkeyOCRv2 pairs the dots.ocr vision encoder, with its axial RoPE
frequencies interleaved across height and width, with a Qwen3 language model.
"""

from collections.abc import Mapping

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.transformers_utils.configs.dotsocr import DotsVisionConfig

from .dots_ocr import (
    DotsOCRDummyInputsBuilder,
    DotsOCRForCausalLM,
    DotsOCRProcessingInfo,
    DotsVisionTransformer,
)
from .qwen2_vl import Qwen2VLMultiModalProcessor
from .utils import init_vllm_registered_model, maybe_prefix


class MonkeyOCRv2ProcessingInfo(DotsOCRProcessingInfo):
    def get_hf_config(self):
        config = self.ctx.get_hf_config()
        if isinstance(config.vision_config, dict):
            config.vision_config = DotsVisionConfig(**config.vision_config)
        return config

    def get_hf_processor(self, **kwargs: object):
        return self.ctx.get_hf_processor(**kwargs)


class MonkeyOCRv2DummyInputsBuilder(DotsOCRDummyInputsBuilder):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        image_token = self.info.get_hf_processor().image_token
        return image_token * mm_counts.get("image", 0)


class MonkeyOCRv2VisionTransformer(DotsVisionTransformer):
    def rot_pos_emb(self, grid_thw: list[list[int]]) -> torch.Tensor:
        pos_ids = torch.cat(self.get_pos_ids_by_grid(grid_thw), dim=0)
        max_grid_size = max(max(h, w) for _, h, w in grid_thw)
        rotary_pos_emb = self.rotary_pos_emb(max_grid_size)[pos_ids]
        # (h0, w0, h1, w1, ...) where dots.ocr has (h0, h1, ..., w0, w1, ...)
        return rotary_pos_emb.transpose(1, 2).flatten(1)


@MULTIMODAL_REGISTRY.register_processor(
    Qwen2VLMultiModalProcessor,
    info=MonkeyOCRv2ProcessingInfo,
    dummy_inputs=MonkeyOCRv2DummyInputsBuilder,
)
class MonkeyOCRv2ForCausalLM(DotsOCRForCausalLM):
    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("image"):
            return "<|vision_start|><|image_pad|><|vision_end|>"
        return None

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        nn.Module.__init__(self)

        self.config = vllm_config.model_config.hf_config
        self.quant_config = vllm_config.quant_config
        multimodal_config = vllm_config.model_config.multimodal_config
        assert multimodal_config is not None
        self.use_data_parallel = multimodal_config.mm_encoder_tp_mode == "data"
        if isinstance(self.config.vision_config, dict):
            self.config.vision_config = DotsVisionConfig(**self.config.vision_config)

        with self._mark_tower_model(vllm_config, "image"):
            self.vision_tower = MonkeyOCRv2VisionTransformer(
                self.config.vision_config,
                quant_config=self.quant_config,
                prefix=maybe_prefix(prefix, "vision_tower"),
            )

        with self._mark_language_model(vllm_config):
            self.language_model = init_vllm_registered_model(
                vllm_config=vllm_config,
                hf_config=self.config,
                prefix=maybe_prefix(prefix, "language_model"),
                architectures=["Qwen3ForCausalLM"],
            )

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )
