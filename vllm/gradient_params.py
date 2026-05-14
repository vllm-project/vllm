# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Parameters for gradient computation requests.

Gradient computation enables token-level attribution and interpretability
by differentiating log p(y_t | x, y_{<t}) w.r.t. model embeddings.

Data flow:
  API request (GradientRequest)
    → ServingGradient tokenizes prompt/target
    → GradientParams created and sent through engine
    → EngineCore dispatches via collective_rpc (bypasses scheduler)
    → GradientRunner performs forward + backward passes
    → Results returned as GradientOutput
"""

from copy import deepcopy
from typing import Literal

import msgspec

from vllm.sampling_params import RequestOutputKind


class GradientParams(
    msgspec.Struct,
    omit_defaults=True,  # type: ignore[call-arg]
    array_like=True,
):  # type: ignore[call-arg]
    """Parameters controlling what and how to compute gradients.

    Attributes:
        target_token_ids: Output token IDs (the continuation after the prompt).
        gradient_of: What to differentiate:
            "loss" — scalar cross-entropy (one backward pass),
            "token_log_probs" — per-token log-probs (one backward per token),
            "both" — both of the above.
        gradient_targets: What to differentiate *with respect to*:
            "input_embeddings" and/or "output_embeddings".
        target_token_indices: Subset of target positions for per-token
            gradients (0-indexed). None means all targets.
        aggregation: Reduction over hidden_dim:
            "none" — full gradients, "l2_norm", or "abs_sum".
        loss_function: "cross_entropy" or "log_prob_sum".
        return_log_probs: Also return log p(y_t | x, y_{<t}) values.
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

    def __repr__(self) -> str:
        return (
            f"GradientParams("
            f"gradient_of={self.gradient_of!r}, "
            f"targets={self.gradient_targets}, "
            f"num_target_tokens={len(self.target_token_ids)}, "
            f"aggregation={self.aggregation!r})"
        )
