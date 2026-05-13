# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
ColPali late interaction model for multi-modal retrieval and reranking.

ColPali extends PaliGemma with a ColBERT-style late interaction head,
producing per-token embeddings for both text and image inputs. It uses
MaxSim scoring for retrieval/reranking tasks.

This model supports the "token_embed" pooling task and is designed for
multi-vector retrieval of documents containing both text and images.

Reference: https://arxiv.org/abs/2407.01449 (ColPali)
Based on: PaliGemma backbone (SigLIP + Gemma) with custom text projection

Target models:
- vidore/colpali-v1.3-hf
"""

from collections.abc import Iterable, Mapping

import torch
import torch.nn as nn
from transformers import BatchFeature, PaliGemmaProcessor

from vllm.config import VllmConfig
from vllm.model_executor.layers.pooler.tokwise import pooler_for_token_embed
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.multimodal import MULTIMODAL_REGISTRY

from .interfaces import SupportsLateInteraction
from .interfaces_base import default_pooling_type
from .paligemma import (
    PaliGemmaDummyInputsBuilder,
    PaliGemmaForConditionalGeneration,
    PaliGemmaMultiModalProcessor,
    PaliGemmaProcessingInfo,
)
from .utils import AutoWeightsLoader, WeightsMapper


class ColPaliProcessingInfo(PaliGemmaProcessingInfo):
    """Processing info for ColPali models.

    ColPali models use a custom HuggingFace config (ColPaliConfig) that is
    not an instance of PaliGemmaConfig. We override get_hf_config() and
    get_hf_processor() to skip the strict type check.
    """

    def get_hf_config(self):
        return self.ctx.get_hf_config()

    def get_hf_processor(self, **kwargs: object) -> PaliGemmaProcessor:
        # Force standard PaliGemmaProcessor even when trust_remote_code=True.
        return self.ctx.get_hf_processor(PaliGemmaProcessor, **kwargs)


class ColPaliMultiModalProcessor(PaliGemmaMultiModalProcessor):
    """Multimodal processor for ColPali."""

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        if mm_data:
            # The ColPali tokenizer_config.json ships with a small default
            # max_length (50) that truncates the 1024 image tokens inserted
            # by PaliGemmaProcessor, causing a token-count mismatch.
            # vLLM enforces its own max_model_len, so we disable HF
            # truncation to keep all image + text tokens intact.
            tok_kwargs = dict(tok_kwargs, truncation=False)
        return super()._call_hf_processor(
            prompt=prompt,
            mm_data=mm_data,
            mm_kwargs=mm_kwargs,
            tok_kwargs=tok_kwargs,
        )


@default_pooling_type(seq_pooling_type="CLS", tok_pooling_type="ALL")
@MULTIMODAL_REGISTRY.register_processor(
    ColPaliMultiModalProcessor,
    info=ColPaliProcessingInfo,
    dummy_inputs=PaliGemmaDummyInputsBuilder,
)
class ColPaliModel(
    PaliGemmaForConditionalGeneration,
    SupportsLateInteraction,
):
    """ColPali late interaction model for multi-modal retrieval/reranking.

    This model extends PaliGemmaForConditionalGeneration with a ColBERT-style
    linear projection layer for per-token embeddings. It supports:
    - "token_embed" task: Per-token embeddings for late interaction scoring

    The model produces L2-normalized per-token embeddings by:
    1. Running the PaliGemma backbone (vision + language) to get hidden states
    2. Projecting hidden states through a linear layer (hidden_size -> embed_dim)
    3. L2-normalizing the projected embeddings
    """

    # Mark this as a pooling model so vLLM routes to pooler path
    is_pooling_model = True

    # Override hf_to_vllm_mapper to handle ColPali weight naming.
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            # HF transformers checkpoint (vidore/colpali-v1.3-hf)
            # Weights: vlm.vision_tower.*, vlm.language_model.*,
            # vlm.multi_modal_projector.*
            "vlm.vision_tower.": "vision_tower.",
            "vlm.language_model.": "language_model.",
            "vlm.multi_modal_projector.": "multi_modal_projector.",
            # colpali-engine checkpoint naming
            "model.vision_tower.": "vision_tower.",
            "model.language_model.": "language_model.",
            "model.multi_modal_projector.": "multi_modal_projector.",
            "lm_head.": "language_model.lm_head.",
        }
    )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
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

        # ColPali uses embedding_dim=128, but also check other naming variants
        self.embed_dim: int | None = (
            getattr(config, "embedding_dim", None)
            or getattr(config, "embed_dim", None)
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
            projector=self.custom_text_proj,
        )

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors=None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor:
        return super().forward(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )

    # Names used for the projection layer across different ColPali variants
    _PROJ_LAYER_NAMES = {
        "custom_text_proj",  # vLLM internal naming
        "embedding_proj_layer",  # colpali-engine / HF naming
    }

    def _is_proj_weight(self, name: str) -> bool:
        """Check if a weight name belongs to the projection layer."""
        return any(proj_name in name for proj_name in self._PROJ_LAYER_NAMES)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load weights with special handling for ColPali projection layer."""
        weights_list = list(weights)
        proj_weights: list[tuple[str, torch.Tensor]] = []
        model_weights: list[tuple[str, torch.Tensor]] = []

        for name, weight in weights_list:
            if self._is_proj_weight(name):
                proj_weights.append((name, weight))
            else:
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

            # Update pooler projector for the lazy-creation path
            self.pooler.head.projector = self.custom_text_proj

        # Mark pooler projector params as loaded
        if hasattr(self, "pooler") and hasattr(self.pooler, "head"):
            head = self.pooler.head
            projector = getattr(head, "projector", None)
            if projector is not None and isinstance(projector, nn.Module):
                for pname, _ in projector.named_parameters():
                    loaded.add(f"pooler.head.projector.{pname}")

        return loaded
