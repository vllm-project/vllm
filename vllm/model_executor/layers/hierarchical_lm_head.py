# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Centroid-based vocabulary selection with compiled sparse scoring."""

from collections.abc import Iterable

import torch
from torch import nn
from transformers import PreTrainedConfig


@torch.compile(dynamic=True, fullgraph=True)
def selected_logits(
    lm_head_weights: torch.Tensor, sel: torch.Tensor, hidden: torch.Tensor
) -> torch.Tensor:
    """Keep the gather inside compilation so Inductor can fuse the reduction."""
    # Only token count varies within a model; keep kernel tiling model-specific.
    torch._dynamo.mark_static(lm_head_weights)
    torch._dynamo.mark_static(sel, 1)
    torch._dynamo.mark_static(hidden, 1)
    return (lm_head_weights[sel] * hidden[:, None, :]).sum(-1)


class HierarchicalLMHead(nn.Module):
    """Route to balanced vocabulary clusters and score selected token rows."""

    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        num_centroids: int,
        centroid_intermediate_top_k: int,
    ) -> None:
        super().__init__()
        if num_centroids <= 0 or vocab_size <= 0 or vocab_size % num_centroids:
            raise ValueError(
                "vocab_size must be positive and divisible by num_centroids"
            )
        if not 1 <= centroid_intermediate_top_k <= num_centroids:
            raise ValueError("top-k must be between 1 and num_centroids")
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_centroids = num_centroids
        self.centroid_intermediate_top_k = centroid_intermediate_top_k
        self.vocab_size_per_centroid = vocab_size // num_centroids
        self.num_selected = centroid_intermediate_top_k * self.vocab_size_per_centroid
        self.centroids = nn.Linear(hidden_size, num_centroids, bias=False)
        self.register_buffer(
            "token_ordering", torch.empty(vocab_size, dtype=torch.long)
        )

    @classmethod
    def from_config(
        cls,
        config: PreTrainedConfig,
        *,
        hidden_size: int,
        vocab_size: int,
    ) -> "HierarchicalLMHead | None":
        """Build the head when its draft checkpoint enables ordered embeddings."""
        if not getattr(config, "use_ordered_embeddings", False):
            return None
        return cls(
            hidden_size,
            vocab_size,
            getattr(config, "num_centroids", 2048),
            getattr(config, "centroid_intermediate_top_k", 32),
        )

    def _select_and_score(
        self, hidden_states: torch.Tensor, lm_head_weight: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if (
            lm_head_weight.ndim != 2
            or lm_head_weight.shape[0] < self.vocab_size
            or lm_head_weight.shape[1] != self.hidden_size
            or not lm_head_weight.is_floating_point()
            or lm_head_weight.dtype != hidden_states.dtype
        ):
            raise ValueError(
                "Expected unsharded, unpacked head weights matching hidden dtype"
            )
        top_k_indices = (
            self.centroids(hidden_states)
            .topk(self.centroid_intermediate_top_k, dim=-1)
            .indices
        )
        clusters = self.token_ordering.view(
            self.num_centroids, self.vocab_size_per_centroid
        )
        selected = clusters[top_k_indices].reshape(
            hidden_states.shape[0], self.num_selected
        )
        logits = selected_logits(lm_head_weight, selected, hidden_states)
        return logits, selected

    def forward(
        self, hidden_states: torch.Tensor, lm_head_weight: torch.Tensor
    ) -> torch.Tensor:
        """Return full-vocabulary logits with unselected tokens masked."""
        logits, indices = self._select_and_score(hidden_states, lm_head_weight)
        output = hidden_states.new_full(
            (hidden_states.shape[0], self.vocab_size),
            torch.finfo(hidden_states.dtype).min,
        )
        return output.scatter_(-1, indices, logits)

    def get_top_tokens(
        self, hidden_states: torch.Tensor, lm_head_weight: torch.Tensor
    ) -> torch.Tensor:
        """Return vocabulary argmax IDs without allocating full logits."""
        logits, indices = self._select_and_score(hidden_states, lm_head_weight)
        return indices.gather(-1, logits.argmax(-1, keepdim=True)).squeeze(-1)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load router tensors through the model's standard checkpoint loader."""
        tensors = dict(self.named_parameters()) | dict(self.named_buffers())
        loaded = set()
        for name, weight in weights:
            if name not in tensors:
                raise ValueError(f"Unexpected hierarchical LM-head tensor: {name}")
            if weight.shape != tensors[name].shape:
                raise ValueError(f"Incompatible hierarchical LM-head shape: {name}")
            if name == "token_ordering":
                ordering = weight.detach().to(device="cpu")
                expected = torch.arange(self.vocab_size, device="cpu")
                if ordering.dtype != torch.long or not torch.equal(
                    ordering.sort().values, expected
                ):
                    raise ValueError("token_ordering must be a vocabulary permutation")
            elif not weight.is_floating_point() or not weight.isfinite().all():
                raise ValueError(
                    "Centroid weights must be finite floating-point values"
                )
            with torch.no_grad():
                tensors[name].copy_(weight)
            loaded.add(name)
        return loaded
