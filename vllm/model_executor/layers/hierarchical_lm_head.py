# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Centroid-based vocabulary selection with compiled sparse scoring."""

from collections.abc import Iterable
from pathlib import Path

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


def load_router_checkpoint(path: str) -> dict[str, torch.Tensor]:
    """Read only the two router tensors from a local, weights-only sidecar."""
    checkpoint = Path(path).expanduser()
    if checkpoint.suffix == ".safetensors":
        from safetensors.torch import load_file

        state = load_file(checkpoint, device="cpu")
    else:
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    required = ("centroids.weight", "token_ordering")
    if not isinstance(state, dict) or any(
        not isinstance(state.get("masked_embedding." + key), torch.Tensor)
        for key in required
    ):
        raise ValueError(
            "Router sidecar must contain masked_embedding.centroids.weight "
            "and masked_embedding.token_ordering tensors"
        )
    return {key: state["masked_embedding." + key] for key in required}


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
        self.router_path: str | None = None
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
        router_path: str | None = None,
        top_k: int | None = None,
    ) -> "HierarchicalLMHead | None":
        """Resolve an embedded router or a sidecar without changing HF config."""
        if router_path is not None:
            if top_k is None:
                raise ValueError("Sidecar router requires an explicit top-k")
            state = load_router_checkpoint(router_path)
            centroids = state["centroids.weight"]
            if centroids.ndim != 2 or centroids.shape[1] != hidden_size:
                raise ValueError("Router centroid shape must match draft hidden size")
            num_centroids = centroids.shape[0]
        elif getattr(config, "use_ordered_embeddings", False):
            num_centroids = getattr(config, "num_centroids", 2048)
            if top_k is None:
                top_k = getattr(config, "centroid_intermediate_top_k", 32)
        else:
            if top_k is not None:
                raise ValueError("Top-k override requires a hierarchical router")
            return None
        head = cls(
            hidden_size,
            vocab_size,
            num_centroids,
            top_k,
        )
        head.router_path = router_path
        return head

    def load_sidecar_weights(self) -> set[str]:
        """Reload sidecar tensors through the same validation as embedded ones."""
        assert self.router_path is not None
        loaded = self.load_weights(load_router_checkpoint(self.router_path).items())
        return {"masked_embedding." + name for name in loaded}

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

    def validate_loaded_weights(self, loaded: set[str]) -> None:
        """Require the ordering buffer as well as the centroid parameters."""
        required = {"masked_embedding." + name for name in self.state_dict()}
        if missing := required - loaded:
            raise ValueError(f"Missing hierarchical LM-head weights: {sorted(missing)}")
