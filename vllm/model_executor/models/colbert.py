# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
ColBERT late interaction model for retrieval and reranking.

ColBERT uses per-token embeddings and late interaction (MaxSim) scoring
instead of single-vector representations or cross-encoder concatenation.

This module provides:

- :class:`ColBERTMixin` — mixin that adds ColBERT late-interaction support
  to any embedding model.
- :class:`ColBERTModel` — ColBERT with BERT backbone (original architecture).
- :class:`ColBERTModernBertModel` — ColBERT with ModernBERT backbone.
- :class:`ColBERTJinaRobertaModel` — ColBERT with Jina XLM-RoBERTa backbone.

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


class ColBERTMixin:
    """Mixin that adds ColBERT late interaction support to any embedding model.

    ColBERT (Contextualized Late Interaction over BERT) uses per-token
    embeddings with a linear projection layer.  This mixin provides:

    - ``supports_late_interaction`` class-var
    - ColBERT linear projection initialisation / lazy creation
    - Weight loading helpers for the projection layer
    - A builder for the token-embedding pooler

    **Integration:**

    1. Inherit from both ``ColBERTMixin`` and ``nn.Module``.
    2. In ``__init__``: call ``super().__init__()``, then
       :meth:`_init_colbert_components`, then create ``self.model``
       (the backbone) and ``self.pooler`` via :meth:`_build_colbert_pooler`.
    3. In ``load_weights``: use :meth:`_load_colbert_weights` to separate
       the ColBERT projection weight, then delegate the rest to the backbone.
    """

    supports_late_interaction: ClassVar[Literal[True]] = True

    # Set during _init_colbert_components
    colbert_dim: int | None
    colbert_linear: nn.Linear | None
    hidden_size: int
    head_dtype: torch.dtype

    # ------------------------------------------------------------------ init

    def _init_colbert_components(
        self,
        hidden_size: int,
        colbert_dim: int | None,
        head_dtype: torch.dtype,
    ) -> None:
        """Initialise ColBERT projection layer.

        Args:
            hidden_size: Hidden dimension of the encoder backbone.
            colbert_dim: Output dimension for ColBERT embeddings.  If
                ``None``, will be inferred from weights during loading (or
                auto-loaded from sentence-transformers Dense layers).
            head_dtype: Data type for the projection layer.
        """
        self.hidden_size = hidden_size
        self.colbert_dim = colbert_dim
        self.head_dtype = head_dtype

        if colbert_dim is not None:
            self.colbert_linear = self._build_colbert_linear()
        else:
            self.colbert_linear = None

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

    # ---------------------------------------------------------------- pooler

    def _build_colbert_pooler(self, pooler_config: PoolerConfig) -> Pooler:
        """Build pooler for ColBERT token embeddings.

        When ``colbert_linear`` is set, it is used as the projector.
        Otherwise ``pooler_for_token_embed`` falls back to auto-loading
        sentence-transformers Dense layers (``1_Dense/`` etc.).
        """
        return pooler_for_token_embed(
            pooler_config,
            projector=self.colbert_linear,
        )

    # --------------------------------------------------------- config helper

    @classmethod
    def get_colbert_dim_from_config(cls, hf_config) -> int | None:
        """Extract ColBERT dimension from a HuggingFace config.

        Checks ``colbert_dim``, ``dim`` and ``projection_dim`` in that order.
        """
        return (
            getattr(hf_config, "colbert_dim", None)
            or getattr(hf_config, "dim", None)
            or getattr(hf_config, "projection_dim", None)
        )

    # -------------------------------------------------------- weight loading

    def _load_colbert_weights(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
        colbert_weight_names: tuple[str, ...] = (
            "linear.weight",
            "colbert_linear.weight",
        ),
    ) -> tuple[list[tuple[str, torch.Tensor]], set[str]]:
        """Separate and load ColBERT projection weights.

        Scans *weights* for entries whose name ends with one of
        *colbert_weight_names*.  The matching weight is loaded into
        ``self.colbert_linear`` (creating it first if ``colbert_dim`` was
        not known at init time).

        Args:
            weights: Iterable of ``(name, tensor)`` weight pairs.
            colbert_weight_names: Suffixes that identify the ColBERT linear
                weight.

        Returns:
            ``(remaining_weights, loaded_names)`` — the weights that were
            **not** consumed and the set of names that were loaded.
        """
        weights_list = list(weights)
        other_weights: list[tuple[str, torch.Tensor]] = []
        colbert_weight: tuple[str, torch.Tensor] | None = None

        for name, weight in weights_list:
            if any(name.endswith(cw) for cw in colbert_weight_names):
                colbert_weight = (name, weight)
            else:
                other_weights.append((name, weight))

        loaded: set[str] = set()
        if colbert_weight is not None:
            _name, weight = colbert_weight
            if weight.dim() == 2:
                # Infer colbert_dim from weight shape if not set
                if self.colbert_dim is None:
                    self.colbert_dim = weight.shape[0]
                    self.colbert_linear = self._build_colbert_linear()
                    # Update the pooler's projector
                    if hasattr(self, "pooler") and hasattr(self.pooler, "head"):
                        self.pooler.head.projector = self.colbert_linear

                assert self.colbert_linear is not None
                # Move to same device as model
                if hasattr(self, "model"):
                    device = next(self.model.parameters()).device
                    self.colbert_linear.to(device)

                weight = weight.to(self.colbert_linear.weight.device)
                self.colbert_linear.weight.data.copy_(weight)
                loaded.add("pooler.head.projector.weight")

        return other_weights, loaded


# -----------------------------------------------------------------------
# Concrete model: ColBERT + BERT backbone  (original architecture)
# -----------------------------------------------------------------------


@default_pooling_type(seq_pooling_type="CLS", tok_pooling_type="ALL")
class ColBERTModel(ColBERTMixin, BertEmbeddingModel):
    """ColBERT late interaction model with BERT backbone.

    Supports the ``token_embed`` task (per-token embeddings for late
    interaction).  MaxSim scoring is computed externally.
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        config = vllm_config.model_config.hf_config

        # Must run before super().__init__ because _build_pooler reads these.
        colbert_dim = self.get_colbert_dim_from_config(config)
        self._init_colbert_components(
            hidden_size=config.hidden_size,
            colbert_dim=colbert_dim,
            head_dtype=vllm_config.model_config.head_dtype,
        )

        super().__init__(vllm_config=vllm_config, prefix=prefix)

    def _build_model(self, vllm_config: VllmConfig, prefix: str = "") -> BertModel:
        return BertModel(vllm_config=vllm_config, prefix=prefix)

    def _build_pooler(self, pooler_config: PoolerConfig) -> Pooler:
        return self._build_colbert_pooler(pooler_config)

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
            # Handle different checkpoint naming conventions
            if stripped in ("linear.weight", "colbert_linear.weight"):
                colbert_side.append(("colbert_linear.weight", weight))
            elif stripped.startswith("linear.") or stripped.startswith(
                "colbert_linear."
            ):
                new_name = stripped.replace("linear.", "colbert_linear.")
                colbert_side.append((new_name, weight))
            else:
                model_side.append((stripped, weight))

        loaded: set[str] = set()
        loaded_model = self.model.load_weights(model_side)
        loaded.update({"model." + n for n in loaded_model})

        if colbert_side:
            _, colbert_loaded = self._load_colbert_weights(colbert_side)
            loaded.update(colbert_loaded)

        return loaded


# -----------------------------------------------------------------------
# Concrete model: ColBERT + ModernBERT backbone
# -----------------------------------------------------------------------

from .modernbert import ModernBertModel  # noqa: E402


@default_pooling_type(seq_pooling_type="CLS", tok_pooling_type="ALL")
class ColBERTModernBertModel(ColBERTMixin, nn.Module):
    """ColBERT late interaction model with ModernBERT backbone.

    For ``lightonai/GTE-ModernColBERT-v1`` and similar models.
    The projection is auto-loaded from sentence-transformers ``1_Dense/``
    when not present in the main checkpoint.
    """

    is_pooling_model = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config

        colbert_dim = self.get_colbert_dim_from_config(config)
        self._init_colbert_components(
            hidden_size=config.hidden_size,
            colbert_dim=colbert_dim,
            head_dtype=vllm_config.model_config.head_dtype,
        )

        self.model = ModernBertModel(
            vllm_config=vllm_config,
            prefix=prefix,
        )

        pooler_config = vllm_config.model_config.pooler_config
        assert pooler_config is not None
        self.pooler = self._build_colbert_pooler(pooler_config)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors=None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.model(
            input_ids=input_ids,
            positions=positions,
            inputs_embeds=inputs_embeds,
            intermediate_tensors=intermediate_tensors,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        other_weights, colbert_loaded = self._load_colbert_weights(weights)

        # Strip "model." prefix added by the embedding adapter
        model_weights = [
            (n[len("model.") :] if n.startswith("model.") else n, w)
            for n, w in other_weights
        ]

        loaded_model = self.model.load_weights(model_weights)
        loaded = {"model." + n for n in loaded_model} | colbert_loaded

        # When the ST projector was auto-loaded during init
        # (not from the main checkpoint), mark its params as loaded
        # so the weight validator doesn't complain.
        if hasattr(self.pooler, "head"):
            head = self.pooler.head
            projector = getattr(head, "projector", None)
            if projector is not None and isinstance(projector, nn.Module):
                for name, _ in projector.named_parameters():
                    loaded.add(f"pooler.head.projector.{name}")

        return loaded


# -----------------------------------------------------------------------
# Concrete model: ColBERT + Jina XLM-RoBERTa backbone
# -----------------------------------------------------------------------

from .bert_with_rope import JinaRobertaModel  # noqa: E402


@default_pooling_type(seq_pooling_type="CLS", tok_pooling_type="ALL")
class ColBERTJinaRobertaModel(ColBERTMixin, nn.Module):
    """ColBERT late interaction model with Jina XLM-RoBERTa backbone.

    For ``jinaai/jina-colbert-v2`` and similar models.
    """

    is_pooling_model = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config

        colbert_dim = self.get_colbert_dim_from_config(config)
        self._init_colbert_components(
            hidden_size=config.hidden_size,
            colbert_dim=colbert_dim,
            head_dtype=vllm_config.model_config.head_dtype,
        )

        self.model = JinaRobertaModel(
            vllm_config=vllm_config,
            prefix=prefix,
        )

        pooler_config = vllm_config.model_config.pooler_config
        assert pooler_config is not None
        self.pooler = self._build_colbert_pooler(pooler_config)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors=None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.model(
            input_ids=input_ids,
            positions=positions,
            inputs_embeds=inputs_embeds,
            intermediate_tensors=intermediate_tensors,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        weights_list = list(weights)
        model_side: list[tuple[str, torch.Tensor]] = []
        colbert_side: list[tuple[str, torch.Tensor]] = []

        for name, weight in weights_list:
            stripped = name
            # Strip "model." prefix added by the embedding adapter
            if stripped.startswith("model."):
                stripped = stripped[len("model.") :]
            # Strip "roberta." prefix from checkpoint
            if stripped.startswith("roberta."):
                stripped = stripped[len("roberta.") :]

            if stripped in ("linear.weight", "colbert_linear.weight"):
                colbert_side.append(("colbert_linear.weight", weight))
            elif stripped.startswith("pooler."):
                # Skip HF pooler weights (not used in ColBERT)
                continue
            else:
                model_side.append((stripped, weight))

        loaded: set[str] = set()
        loaded_model = self.model.load_weights(model_side)
        loaded.update({"model." + n for n in loaded_model})

        if colbert_side:
            _, colbert_loaded = self._load_colbert_weights(colbert_side)
            loaded.update(colbert_loaded)

        return loaded
