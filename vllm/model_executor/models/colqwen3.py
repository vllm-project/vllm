# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
ColQwen3 late interaction model for multi-modal retrieval and reranking.

ColQwen3 extends Qwen3-VL with a ColBERT-style late interaction head,
producing per-token embeddings for both text and image inputs. It uses
MaxSim scoring for retrieval/reranking tasks.

This model supports the "token_embed" pooling task and is designed for
multi-vector retrieval of documents containing both text and images.

Reference: https://arxiv.org/abs/2407.01449 (ColPali)
Based on: Qwen3-VL backbone with custom text projection

Target models:
- TomoroAI/tomoro-colqwen3-embed-8b
- OpenSearch-AI/Ops-Colqwen3-4B
"""

from collections.abc import Iterable, Mapping
from typing import ClassVar, Literal

import torch
import torch.nn as nn
from transformers.models.qwen3_vl import Qwen3VLProcessor

from vllm.config import VllmConfig
from vllm.model_executor.layers.pooler.tokwise import pooler_for_token_embed
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.multimodal import MULTIMODAL_REGISTRY

from .interfaces_base import default_pooling_type
from .qwen2_vl import Qwen2VLMultiModalDataParser
from .qwen3_vl import (
    Qwen3VLDummyInputsBuilder,
    Qwen3VLForConditionalGeneration,
    Qwen3VLMultiModalProcessor,
    Qwen3VLProcessingInfo,
)
from .utils import AutoWeightsLoader, WeightsMapper


class ColQwen3ProcessingInfo(Qwen3VLProcessingInfo):
    """Processing info for ColQwen3 models.

    ColQwen3 models (TomoroAI, OpenSearch-AI, etc.) use custom HuggingFace
    configs (e.g. ColQwen3Config, OpsColQwen3Config) that are not instances
    of Qwen3VLConfig. We override get_hf_config() and get_hf_processor()
    to skip the strict type check, similar to OpenCUAProcessingInfo.
    """

    def get_hf_config(self):
        return self.ctx.get_hf_config()

    def get_hf_processor(self, **kwargs: object) -> Qwen3VLProcessor:
        # Force standard Qwen3VLProcessor even when trust_remote_code=True.
        # ColQwen3 custom processors (e.g. ColQwen3Processor) have
        # incompatible interfaces with vLLM's Qwen3VLMultiModalProcessor.
        # The standard Qwen3VLProcessor handles both text and image inputs
        # correctly for the Qwen3-VL backbone.
        return self.ctx.get_hf_processor(
            Qwen3VLProcessor,
            use_fast=kwargs.pop("use_fast", True),
            **kwargs,
        )

    @property
    def _supports_video(self) -> bool:
        """Check if the HF processor supports video inputs."""
        return hasattr(self.get_hf_processor(), "video_processor")

    def get_video_processor(self, **kwargs: object):
        if not self._supports_video:
            raise AttributeError(
                f"The processor for {self.ctx.model_config.model} does not "
                "support video inputs (no video_processor attribute)."
            )
        return self.get_hf_processor(**kwargs).video_processor  # type: ignore[attr-defined]

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        limits: dict[str, int | None] = {"image": None}
        if self._supports_video:
            limits["video"] = None
        return limits

    def get_mm_max_tokens_per_item(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> Mapping[str, int]:
        max_image_tokens = self.get_max_image_tokens()
        result: dict[str, int] = {"image": max_image_tokens}
        if self._supports_video:
            max_video_tokens = self.get_max_video_tokens(seq_len, mm_counts)
            result["video"] = max_video_tokens
        return result

    def get_data_parser(self):
        hf_config = self.get_hf_config()
        spatial_merge_size = hf_config.vision_config.spatial_merge_size
        return Qwen2VLMultiModalDataParser(
            spatial_merge_size,
            video_needs_metadata=self._supports_video,
            expected_hidden_size=self._get_expected_hidden_size(),
        )


@default_pooling_type(seq_pooling_type="CLS", tok_pooling_type="ALL")
@MULTIMODAL_REGISTRY.register_processor(
    Qwen3VLMultiModalProcessor,
    info=ColQwen3ProcessingInfo,
    dummy_inputs=Qwen3VLDummyInputsBuilder,
)
class ColQwen3Model(
    Qwen3VLForConditionalGeneration,
):
    """ColQwen3 late interaction model for multi-modal retrieval/reranking.

    This model extends Qwen3VLForConditionalGeneration with a ColBERT-style
    linear projection layer for per-token embeddings. It supports:
    - "token_embed" task: Per-token embeddings for late interaction scoring

    The model produces L2-normalized per-token embeddings by:
    1. Running the Qwen3-VL backbone (vision + language) to get hidden states
    2. Projecting hidden states through a linear layer (hidden_size -> embed_dim)
    3. L2-normalizing the projected embeddings

    ColBERT-style MaxSim scoring is computed externally, either client-side
    or via the late interaction scoring path in ServingScores.

    Attributes:
        custom_text_proj: Linear projection from hidden_size to embed_dim
        supports_late_interaction: Flag indicating this model uses late
            interaction scoring
    """

    # Mark this as a pooling model so vLLM routes to pooler path
    is_pooling_model = True

    # Mark this model as supporting late interaction scoring
    supports_late_interaction: ClassVar[Literal[True]] = True

    # Override hf_to_vllm_mapper to handle ColQwen3 weight naming.
    # NOTE: WeightsMapper applies ALL matching prefix rules sequentially
    # (no early exit), so more-specific prefixes must come first.
    #   TomoroAI:    "vlm.model.visual.", "vlm.model.language_model."
    #   ColPali:     "model.visual.", "model.language_model."
    #   OpenSearch:  "visual.", "language_model." (no outer prefix,
    #                re-prefixed to "model.*" in load_weights)
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            # TomoroAI naming convention (most specific first)
            "vlm.model.visual.": "visual.",
            "vlm.lm_head.": "language_model.lm_head.",
            "vlm.model.language_model.": "language_model.model.",
            # ColPali / nvidia naming convention
            "model.visual.": "visual.",
            "lm_head.": "language_model.lm_head.",
            # OpenSearch-AI: after re-prefix, "language_model.model.*"
            # becomes "model.language_model.model.*" — handle this before
            # the shorter "model.language_model." rule to avoid double map
            "model.language_model.model.": "language_model.model.",
            "model.language_model.": "language_model.model.",
        }
    )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "model"):
        super().__init__(vllm_config=vllm_config, prefix=prefix)

        config = vllm_config.model_config.hf_config
        head_dtype = vllm_config.model_config.head_dtype

        hidden_size = getattr(config, "hidden_size", None)
        if hidden_size is None and hasattr(config, "text_config"):
            hidden_size = config.text_config.hidden_size
        if hidden_size is None:
            raise ValueError(
                "Unable to determine text hidden size from config. "
                "Expected 'hidden_size' or 'text_config.hidden_size'."
            )
        self._proj_hidden_size = hidden_size

        # (TomoroAI: embed_dim, OpenSearch: dims, ColPali: dim)
        self.embed_dim: int | None = (
            getattr(config, "embed_dim", None)
            or getattr(config, "dims", None)
            or getattr(config, "dim", None)
            or getattr(config, "projection_dim", None)
            or getattr(config, "colbert_dim", None)
        )

        # Build the projection layer if embed_dim is known
        if self.embed_dim is not None:
            self.custom_text_proj = nn.Linear(
                hidden_size,
                self.embed_dim,
                bias=False,
                dtype=head_dtype,
            )
        else:
            # Will be created during load_weights when dim is inferred
            self.custom_text_proj = None

        pooler_config = vllm_config.model_config.pooler_config
        assert pooler_config is not None
        self.pooler = pooler_for_token_embed(
            pooler_config,
            projector=None,
        )

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors=None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor:
        """Run forward pass producing per-token embeddings."""
        hidden_states = super().forward(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )

        if not isinstance(hidden_states, torch.Tensor):
            return hidden_states  # type: ignore

        proj_dtype = self.custom_text_proj.weight.dtype  # type: ignore
        if hidden_states.dtype != proj_dtype:
            hidden_states = hidden_states.to(proj_dtype)

        # Project to embedding dimension and L2 normalize
        proj = self.custom_text_proj(hidden_states)  # type: ignore
        return torch.nn.functional.normalize(proj, p=2, dim=-1)

    # Names used for the projection layer across different ColQwen3 variants
    _PROJ_LAYER_NAMES = {
        "custom_text_proj",  # ColPali naming
        "embedding_proj_layer",  # TomoroAI naming
    }

    def _is_proj_weight(self, name: str) -> bool:
        """Check if a weight name belongs to the projection layer."""
        return any(proj_name in name for proj_name in self._PROJ_LAYER_NAMES)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load weights with special handling for ColQwen3 projection layer."""
        weights_list = list(weights)
        proj_weights: list[tuple[str, torch.Tensor]] = []
        model_weights: list[tuple[str, torch.Tensor]] = []

        # Scan all weight names to determine if re-prefixing is needed.
        # OpenSearch-AI models have unprefixed weights ("language_model.*",
        # "visual.*") that need "model." added so hf_to_vllm_mapper can
        # process them. Only re-prefix if ALL backbone weights are
        # unprefixed (no "vlm." or "model." prefix found).
        has_unprefixed = any(
            name.startswith("language_model.") or name.startswith("visual.")
            for name, _ in weights_list
        )
        has_prefixed = any(
            name.startswith("vlm.") or name.startswith("model.")
            for name, _ in weights_list
        )
        needs_reprefix = has_unprefixed and not has_prefixed

        for name, weight in weights_list:
            if self._is_proj_weight(name):
                proj_weights.append((name, weight))
            else:
                if needs_reprefix and not self._is_proj_weight(name):
                    name = "model." + name
                model_weights.append((name, weight))

        loader = AutoWeightsLoader(self)
        loaded = loader.load_weights(model_weights, mapper=self.hf_to_vllm_mapper)

        if proj_weights:
            model_dtype = next(self.language_model.parameters()).dtype
            model_device = next(self.language_model.parameters()).device

            for name, weight in proj_weights:
                if self.embed_dim is None and "weight" in name:
                    self.embed_dim = weight.shape[0]
                    has_bias = any("bias" in n for n, _ in proj_weights)
                    self.custom_text_proj = nn.Linear(
                        self._proj_hidden_size,
                        self.embed_dim,
                        bias=has_bias,
                        dtype=model_dtype,
                    )
                    self.custom_text_proj.to(model_device)

                if self.custom_text_proj is not None:
                    param_name = name.split(".")[-1]
                    param = getattr(self.custom_text_proj, param_name, None)
                    if param is not None:
                        weight = weight.to(device=param.device, dtype=param.dtype)
                        default_weight_loader(param, weight)
                        loaded.add(f"custom_text_proj.{param_name}")

        return loaded
