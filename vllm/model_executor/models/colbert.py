# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
ColBERT late interaction model for retrieval and reranking.

ColBERT uses per-token embeddings and late interaction (MaxSim) scoring
instead of single-vector representations or cross-encoder concatenation.

Reference: https://arxiv.org/abs/2004.12832
"""

from collections.abc import Iterable
from typing import ClassVar, Literal

import torch
from torch import nn

from vllm.config import PoolerConfig, VllmConfig
from vllm.model_executor.layers.pooler import Pooler
from vllm.model_executor.layers.pooler.tokwise import pooler_for_token_embed

from .bert import BertEmbeddingModel, BertModel
from .interfaces_base import default_pooling_type


@default_pooling_type(seq_pooling_type="CLS", tok_pooling_type="ALL")
class ColBERTModel(BertEmbeddingModel):
    """ColBERT late interaction model for retrieval/reranking.

    This model extends BertEmbeddingModel with a ColBERT-style linear
    projection layer for per-token embeddings. It supports only:
    - "token_embed" task: Per-token embeddings for late interaction

    ColBERT is fundamentally a per-token embedding model - the linear
    projection is trained for per-token representations, not for CLS
    pooling. Use a dedicated dense embedding model if you need single-
    vector representations.

    The ColBERT scoring (MaxSim) is computed externally, either client-side
    or via the late interaction scoring path in ServingScores.

    Attributes:
        colbert_linear: Linear projection from hidden_size to colbert_dim
        supports_late_interaction: Flag indicating this model uses late
            interaction scoring
    """

    # Mark this model as supporting late interaction scoring
    supports_late_interaction: ClassVar[Literal[True]] = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        # Get config before calling super().__init__
        config = vllm_config.model_config.hf_config
        self.hidden_size = config.hidden_size
        self.head_dtype = vllm_config.model_config.head_dtype

        # ColBERT dimension - check various config field names used by different
        # ColBERT implementations. If not found in config, will be inferred
        # from loaded weights in load_weights()
        self.colbert_dim: int | None = (
            getattr(config, "colbert_dim", None)
            or getattr(config, "dim", None)
            or getattr(config, "projection_dim", None)
        )

        # Initialize parent (this will call _build_pooler)
        super().__init__(vllm_config=vllm_config, prefix=prefix)

    def _build_model(self, vllm_config: VllmConfig, prefix: str = "") -> BertModel:
        return BertModel(vllm_config=vllm_config, prefix=prefix)

    def _build_colbert_linear(self) -> nn.Linear:
        """Build the ColBERT linear projection layer."""
        if self.colbert_dim is None:
            raise ValueError("colbert_dim must be set before building the linear layer")
        return nn.Linear(
            self.hidden_size,
            self.colbert_dim,
            bias=False,
            dtype=self.head_dtype,
        )

    def _build_pooler(self, pooler_config: PoolerConfig) -> Pooler:
        # ColBERT linear projection: hidden_size -> colbert_dim
        # Original ColBERT uses bias=False
        # If colbert_dim is not set from config, it will be inferred during
        # load_weights and the linear layer will be created there
        if self.colbert_dim is not None:
            self.colbert_linear = self._build_colbert_linear()
        else:
            # Placeholder - will be created when weights are loaded
            self.colbert_linear = None

        # ColBERT only supports token_embed - it's fundamentally a per-token
        # embedding model.
        return pooler_for_token_embed(
            pooler_config,
            projector=self.colbert_linear,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        def _strip(name: str) -> str:
            for p in ("model.", "bert."):
                if name.startswith(p):
                    name = name[len(p) :]
            return name

        weights_list = list(weights)
        model_side: list[tuple[str, torch.Tensor]] = []
        colbert_side: list[tuple[str, torch.Tensor]] = []

        for name, weight in weights_list:
            stripped = _strip(name)
            # Handle different checkpoint naming conventions for ColBERT linear
            if stripped in ("linear.weight", "colbert_linear.weight"):
                colbert_side.append(("colbert_linear.weight", weight))
            elif stripped.startswith("linear.") or stripped.startswith(
                "colbert_linear."
            ):
                new_name = stripped.replace("linear.", "colbert_linear.")
                colbert_side.append((new_name, weight))
            else:
                model_side.append((stripped, weight))

        # Load base BERT weights using BertModel.load_weights which handles QKV fusion
        loaded: set[str] = set()
        loaded_model = self.model.load_weights(model_side)
        loaded.update({"model." + n for n in loaded_model})

        # Load ColBERT linear weights
        if colbert_side:
            for name, weight in colbert_side:
                if name == "colbert_linear.weight":
                    # Infer colbert_dim from weights if not set in config
                    if self.colbert_dim is None:
                        # Weight shape is [colbert_dim, hidden_size]
                        self.colbert_dim = weight.shape[0]
                        # Create the linear layer now that we know the dimension
                        self.colbert_linear = self._build_colbert_linear()
                        # Move to the same device as the model's existing parameters
                        device = next(self.model.parameters()).device
                        self.colbert_linear.to(device)
                        # Update the pooler's projector to use the new linear layer
                        self.pooler.head.projector = self.colbert_linear

                    # Load weights directly into the pooler's projector
                    weight = weight.to(self.pooler.head.projector.weight.device)
                    self.pooler.head.projector.weight.data.copy_(weight)
                    loaded.add("pooler.head.projector.weight")
                    break

        return loaded
