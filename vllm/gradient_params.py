# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from copy import deepcopy
from typing import Literal

import msgspec

from vllm.sampling_params import RequestOutputKind


class GradientParams(
    msgspec.Struct,
    omit_defaults=True,  # type: ignore[call-arg]
    array_like=True,
):  # type: ignore[call-arg]
    """API parameters for gradient computation.

    Given a prompt and target continuation tokens, computes gradients of
    a loss and/or per-token log-probabilities with respect to specified
    model components (e.g. input embeddings). This enables token-level
    attribution and interpretability analysis.

    Attributes:
        target_token_ids: The output token IDs to compute gradients for.
            These represent the continuation after the prompt.
        gradient_of: What quantity to differentiate:
            - "loss": scalar cross-entropy loss (single backward pass).
            - "token_log_probs": per-token log p(y_t | x, y_{<t})
              (one backward pass per selected target token).
            - "both": return both loss and per-token gradients.
        gradient_targets: What to differentiate with respect to:
            - "input_embeddings": gradients w.r.t. prompt token embeddings.
            - "output_embeddings": gradients w.r.t. target token embeddings.
        target_token_indices: If set, only compute per-token gradients for
            these target positions (0-indexed). None = all target tokens.
            Reduces compute when only a subset of tokens is of interest.
        aggregation: How to reduce the hidden_dim of gradient tensors:
            - "none": return full [num_tokens, hidden_dim] gradients.
            - "l2_norm": ||grad||_2 per token position (most common for
              attribution heatmaps).
            - "abs_sum": sum(|grad|) per token position.
        loss_function: Loss to use when gradient_of is "loss" or "both":
            - "cross_entropy": standard cross-entropy loss over targets.
            - "log_prob_sum": sum of log-probabilities of target tokens.
        return_log_probs: Whether to also return the log-probability values
            for each target token.
    """

    target_token_ids: list[int]

    gradient_of: Literal[
        "loss",
        "token_log_probs",
        "both",
    ] = "token_log_probs"

    gradient_targets: list[
        Literal[
            "input_embeddings",
            "output_embeddings",
        ]
    ] = msgspec.field(default_factory=lambda: ["input_embeddings"])

    target_token_indices: list[int] | None = None

    aggregation: Literal[
        "none",
        "l2_norm",
        "abs_sum",
    ] = "l2_norm"

    loss_function: Literal[
        "cross_entropy",
        "log_prob_sum",
    ] = "cross_entropy"

    return_log_probs: bool = True

    ## Internal use only
    output_kind: RequestOutputKind = RequestOutputKind.FINAL_ONLY

    def clone(self) -> "GradientParams":
        """Returns a deep copy of the GradientParams instance."""
        return deepcopy(self)

    def verify(self) -> None:
        """Validate gradient parameters."""
        if not self.target_token_ids:
            raise ValueError("target_token_ids must be non-empty")

        if self.target_token_indices is not None:
            num_targets = len(self.target_token_ids)
            for idx in self.target_token_indices:
                if idx < 0 or idx >= num_targets:
                    raise ValueError(
                        f"target_token_indices contains {idx}, but "
                        f"target_token_ids has only {num_targets} tokens "
                        f"(valid range: 0 to {num_targets - 1})"
                    )

        if not self.gradient_targets:
            raise ValueError("gradient_targets must contain at least one target")

        valid_targets = {"input_embeddings", "output_embeddings"}
        for target in self.gradient_targets:
            if target not in valid_targets:
                raise ValueError(
                    f"Unknown gradient target: {target!r}. "
                    f"Valid targets: {valid_targets}"
                )

    def __repr__(self) -> str:
        return (
            f"GradientParams("
            f"gradient_of={self.gradient_of!r}, "
            f"gradient_targets={self.gradient_targets}, "
            f"target_token_ids=[{len(self.target_token_ids)} tokens], "
            f"target_token_indices={self.target_token_indices}, "
            f"aggregation={self.aggregation!r}, "
            f"loss_function={self.loss_function!r}, "
            f"return_log_probs={self.return_log_probs})"
        )

    def __post_init__(self) -> None:
        assert self.output_kind == RequestOutputKind.FINAL_ONLY, (
            "For gradient computation output_kind has to be FINAL_ONLY"
        )
