# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
ColQwen3 multimodal late interaction model for visual document retrieval.

ColQwen3 extends Qwen3-VL with ColBERT-style per-token embeddings for
multi-vector retrieval. Instead of generating text, it produces per-token
embeddings that support late interaction (MaxSim) scoring between queries
and document page images.

This module provides:

- :class:`ColQwen3ForRetrieval` — ColQwen3 with Qwen3-VL backbone, producing
  per-token embeddings for late interaction retrieval.

Supported models:

- ``TomoroAI/tomoro-colqwen3-embed-8b``
- ``TomoroAI/tomoro-colqwen3-embed-4b``

Reference: https://arxiv.org/abs/2407.01449
"""

from collections.abc import Iterable
from typing import ClassVar, Literal

import torch
from torch import nn

from vllm.config import PoolerConfig, VllmConfig
from vllm.model_executor.layers.pooler import Pooler
from vllm.model_executor.layers.pooler.tokwise import pooler_for_token_embed
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.utils import AutoWeightsLoader, WeightsMapper
from vllm.multimodal import MULTIMODAL_REGISTRY

from .interfaces_base import default_pooling_type
from .qwen3_vl import (
    Qwen3VLDummyInputsBuilder,
    Qwen3VLForConditionalGeneration,
    Qwen3VLMultiModalProcessor,
    Qwen3VLProcessingInfo,
)


@default_pooling_type(seq_pooling_type="LAST", tok_pooling_type="ALL")
@MULTIMODAL_REGISTRY.register_processor(
    Qwen3VLMultiModalProcessor,
    info=Qwen3VLProcessingInfo,
    dummy_inputs=Qwen3VLDummyInputsBuilder,
)
class ColQwen3ForRetrieval(Qwen3VLForConditionalGeneration):
    """ColQwen3 late interaction model with Qwen3-VL backbone.

    Produces per-token embeddings for ColBERT-style late interaction
    retrieval. Supports ``token_embed`` task for multi-vector document
    and query encoding.

    Architecture:
        Qwen3-VL vision encoder + LLM → hidden states → linear projection
        → L2 normalized per-token embeddings (default: 320-dim)
    """

    is_pooling_model: ClassVar[Literal[True]] = True

    supports_late_interaction: ClassVar[Literal[True]] = True

    # TomoroAI checkpoints use vlm.model.* prefix for backbone weights
    # and embedding_proj_layer.* at top level for the projection.
    # The colpali canonical checkpoints use model.* prefix.
    # We handle both via the weight mapper + load_weights override.
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            # TomoroAI checkpoint layout
            "vlm.model.visual.": "visual.",
            "vlm.lm_head.": "language_model.lm_head.",
            "vlm.model.language_model.": "language_model.model.",
            # colpali/standard checkpoint layout
            "model.visual.": "visual.",
            "lm_head.": "language_model.lm_head.",
            "model.language_model.": "language_model.model.",
        }
    )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "model"):
        super().__init__(vllm_config=vllm_config, prefix=prefix)

        config = vllm_config.model_config.hf_config

        # ColQwen3 embedding dimension (default 320)
        self.embed_dim: int = getattr(config, "embed_dim", None) or 320

        # Hidden size from the language model
        text_config = getattr(config, "text_config", config)
        hidden_size: int = getattr(text_config, "hidden_size", 4096)

        # Projection: hidden_size → embed_dim
        # Named to match checkpoint weight keys
        self.embedding_proj_layer = nn.Linear(
            hidden_size,
            self.embed_dim,
            bias=True,
        )

        # Pooler for token-level embeddings
        pooler_config = vllm_config.model_config.pooler_config
        assert pooler_config is not None
        self.pooler = self._build_pooler(pooler_config)

    def _build_pooler(self, pooler_config: PoolerConfig) -> Pooler:
        return pooler_for_token_embed(
            pooler_config,
            projector=None,  # Projection handled in forward()
        )

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors=None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor:
        # Get hidden states from the full Qwen3-VL pipeline
        hidden_states = super().forward(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )

        # Project to embedding dimension
        embeddings = self.embedding_proj_layer(hidden_states)

        # L2 normalize per-token
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=-1)

        return embeddings

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        return None  # Embedding model, no logits

    def load_weights(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
    ) -> set[str]:
        # Separate projection weights from backbone weights.
        # The projection layer is stored at top level in the checkpoint
        # (no vlm. or model. prefix).
        proj_names = {
            "embedding_proj_layer.weight",
            "embedding_proj_layer.bias",
            # colpali canonical naming
            "custom_text_proj.weight",
            "custom_text_proj.bias",
        }

        backbone_weights: list[tuple[str, torch.Tensor]] = []
        proj_weights: dict[str, torch.Tensor] = {}

        for name, weight in weights:
            if name in proj_names:
                # Normalize custom_text_proj → embedding_proj_layer
                key = name.replace("custom_text_proj", "embedding_proj_layer")
                proj_weights[key] = weight
            else:
                backbone_weights.append((name, weight))

        # Load backbone weights via parent (uses hf_to_vllm_mapper)
        loader = AutoWeightsLoader(self)
        loaded = loader.load_weights(backbone_weights, mapper=self.hf_to_vllm_mapper)

        # Load projection weights
        if "embedding_proj_layer.weight" in proj_weights:
            default_weight_loader(
                self.embedding_proj_layer.weight,
                proj_weights["embedding_proj_layer.weight"],
            )
            loaded.add("embedding_proj_layer.weight")

        if "embedding_proj_layer.bias" in proj_weights:
            default_weight_loader(
                self.embedding_proj_layer.bias,
                proj_weights["embedding_proj_layer.bias"],
            )
            loaded.add("embedding_proj_layer.bias")

        return loaded
