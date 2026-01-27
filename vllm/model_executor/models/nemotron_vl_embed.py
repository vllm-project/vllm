# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
LlamaNemotronVL Embedding Model for vLLM.

This model combines:
- SigLIP vision encoder
- Bidirectional LLaMA language model (non-causal attention)
- MLP projector for vision-to-language mapping
- Pooling for embedding output

Based on: nvidia/llama-nemotron-embed-vl-1b-v2

This model inherits from LlamaNemotronVLChatModel and specializes it for
embedding/pooling tasks by:
- Using SigLIP instead of C-RADIO for vision encoding
- Using bidirectional LLaMA instead of causal LLaMA
- Adding a pooler instead of generating logits
"""

from collections.abc import Iterable

import torch
import torch.nn as nn
import torchvision.transforms as T
from transformers import PretrainedConfig

from vllm.config import VllmConfig
from vllm.model_executor.layers.pooler import DispatchPooler
from vllm.model_executor.models.siglip import SiglipVisionModel
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.tokenizers import TokenizerLike

from .interfaces_base import VllmModelForPooling
from .internvl import (
    BaseInternVLDummyInputsBuilder,
    BaseInternVLMultiModalProcessor,
)
from .nemotron_vl import (
    LlamaNemotronVLChatModel,
    NemotronVLProcessingInfo,
    NemotronVLProcessor,
    build_transform,
)
from .utils import AutoWeightsLoader, WeightsMapper

# Special token for image context (different from chat model's "<image>")
IMG_CONTEXT = "<IMG_CONTEXT>"

# SigLIP normalization constants
SIGLIP_MEAN = (0.5, 0.5, 0.5)
SIGLIP_STD = (0.5, 0.5, 0.5)


def build_siglip_transform(input_size: int):
    """Build transform for SigLIP vision encoder with normalization.

    Extends the base transform from nemotron_vl with SigLIP-specific normalization.
    """
    base_transform = build_transform(input_size=input_size)
    return T.Compose(
        [
            base_transform,
            T.Normalize(mean=SIGLIP_MEAN, std=SIGLIP_STD),
        ]
    )


class LlamaNemotronVLEmbedProcessor(NemotronVLProcessor):
    """
    Processor for LlamaNemotronVL embedding model.

    Inherits from NemotronVLProcessor and specializes it for embedding tasks:
    - Uses SigLIP transform with normalization instead of base transform
    - Uses different image context token (<IMG_CONTEXT> vs <image>)
    """

    IMG_CONTEXT = "<IMG_CONTEXT>"

    def __init__(
        self,
        config: PretrainedConfig,
        tokenizer: TokenizerLike,
        *,
        min_dynamic_patch: int | None = None,
        max_dynamic_patch: int | None = None,
        dynamic_image_size: bool | None = None,
    ) -> None:
        super().__init__(
            config=config,
            tokenizer=tokenizer,
            image_processor=None,
            min_dynamic_patch=min_dynamic_patch,
            max_dynamic_patch=max_dynamic_patch,
            dynamic_image_size=dynamic_image_size,
        )

    def _get_transform(self) -> T.Compose:
        """Override to add SigLIP normalization."""
        return build_siglip_transform(input_size=self.image_size)

    def _replace_image_tokens(
        self,
        text: list[str],
        pixel_values_lst: list[torch.Tensor],
    ) -> list[str]:
        """Override with simpler token replacement for embedding model.

        No temporary placeholder needed because IMG_CONTEXT is <IMG_CONTEXT>,
        not <image>, so there's no collision risk.
        """
        for pixel_values in pixel_values_lst:
            num_patches = pixel_values.shape[0]
            feature_size = num_patches * self.num_image_token
            image_repl = self.get_image_repl(feature_size, num_patches)
            text = [t.replace("<image>", image_repl.full, 1) for t in text]
        return text


class LlamaNemotronVLEmbedProcessingInfo(NemotronVLProcessingInfo):
    """Processing info for LlamaNemotronVL embedding model."""

    def get_hf_processor(self, **kwargs: object) -> LlamaNemotronVLEmbedProcessor:
        """Override to create embedding-specific processor without image_processor."""
        return self.ctx.init_processor(
            LlamaNemotronVLEmbedProcessor,
            config=self.get_hf_config(),
            tokenizer=self.get_tokenizer(),
            **kwargs,
        )


@MULTIMODAL_REGISTRY.register_processor(
    BaseInternVLMultiModalProcessor[LlamaNemotronVLEmbedProcessingInfo],
    info=LlamaNemotronVLEmbedProcessingInfo,
    dummy_inputs=BaseInternVLDummyInputsBuilder[LlamaNemotronVLEmbedProcessingInfo],
)
class LlamaNemotronVLForEmbedding(LlamaNemotronVLChatModel, VllmModelForPooling):
    """
    LlamaNemotronVL model for embeddings.

    Inherits from LlamaNemotronVLChatModel and specializes it for embedding tasks:
    - Uses SigLIP vision encoder instead of C-RADIO
    - Uses bidirectional LLaMA (via llm_config) instead of causal LLaMA
    - Adds pooler for embedding output instead of generating logits
    """

    is_pooling_model = True

    # Weight mapping from checkpoint format to vLLM format
    # Different from parent class due to different vision model structure
    weight_mapper = WeightsMapper(
        orig_to_new_prefix={
            # Language model mapping
            "language_model.layers.": "language_model.model.layers.",
            "language_model.embed_tokens.": "language_model.model.embed_tokens.",
            "language_model.norm.": "language_model.model.norm.",
            # Vision model mapping (SiglipVisionModel has nested vision_model)
            "vision_model.encoder.": "vision_model.vision_model.encoder.",
            "vision_model.embeddings.": "vision_model.vision_model.embeddings.",
            "vision_model.pre_layrnorm.": "vision_model.vision_model.pre_layrnorm.",
            "vision_model.post_layernorm.": "vision_model.vision_model.post_layernorm.",
        }
    )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)

        config = vllm_config.model_config.hf_config

        # Override: get img_context_token_id from config (parent sets None)
        self.img_context_token_id = getattr(config, "img_context_token_id", None)

        # Initialize pooler for embedding output
        pooler_config = vllm_config.model_config.pooler_config
        assert pooler_config is not None
        self.pooler = DispatchPooler.for_embedding(pooler_config)

    def _init_vision_model(
        self,
        config: PretrainedConfig,
        quant_config,
        *,
        prefix: str,
    ) -> nn.Module:
        """Override to use SigLIP instead of C-RADIO."""
        return SiglipVisionModel(
            config.vision_config,
            quant_config=quant_config,
            prefix=prefix,
            use_head=False,
        )

    def _init_mlp1(self, config: PretrainedConfig) -> nn.Module:
        """Override to use different MLP structure for embedding model."""
        return super()._init_mlp1(
            config,
            vit_hidden_size=config.vision_config.hidden_size,
            vision_projection_hidden_size=config.get_text_config().hidden_size,
        )

    def _call_vision_model(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Override to handle SigLIP interface."""
        return self.vision_model(pixel_values)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Override to use different weight mapping for SigLIP."""
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=self.weight_mapper)
