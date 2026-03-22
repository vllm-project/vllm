# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pluggable policies for dynamically deciding the number of draft tokens.

A DraftLengthPolicy decides, at each drafting step, whether to continue
generating more draft tokens or to stop early.  This abstraction lets
alternative early-exit heuristics be swapped in without touching the core
proposer logic.

Example usage (see SpecDecodeBaseProposer.__init__)::

    from vllm.v1.spec_decode.draft_length_policy import (
        DraftLengthPolicy,
        ConfidenceThresholdPolicy,
        AlwaysContinuePolicy,
    )

    threshold = speculative_config.draft_confidence_threshold
    policy: DraftLengthPolicy = (
        ConfidenceThresholdPolicy(threshold)
        if threshold > 0
        else AlwaysContinuePolicy()
    )
"""

from typing import Protocol, runtime_checkable

import torch


@runtime_checkable
class DraftLengthPolicy(Protocol):
    """Protocol for dynamic draft-length policies.

    A policy is called after each draft step to decide whether to generate
    one more token.  Returning ``False`` terminates the draft loop early.

    Args:
        step: Zero-based index of the token just generated (0 = first extra
              draft token after the seed token).
        token_probs: 1-D tensor of shape ``[batch_size]`` containing the
                     probability that the draft model assigned to the token
                     it just sampled for each sequence in the batch.

    Returns:
        ``True``  – continue drafting (generate token at ``step + 1``).
        ``False`` – stop drafting; the token at ``step`` is the last one.
    """

    def should_continue(self, step: int, token_probs: torch.Tensor) -> bool: ...


class ConfidenceThresholdPolicy:
    """Mean-confidence early-exit policy (DISCO-style).

    Stops drafting when the *mean* probability assigned to the sampled tokens
    across the batch falls below ``threshold``.  Using the mean rather than
    the minimum is more robust under mixed workloads: a single low-confidence
    request does not penalise high-confidence ones.

    Reference: arXiv:2405.04304 (DISCO).
    """

    def __init__(self, threshold: float) -> None:
        if not 0.0 < threshold <= 1.0:
            raise ValueError(
                f"ConfidenceThresholdPolicy threshold must be in (0, 1], "
                f"got {threshold}"
            )
        self.threshold = threshold

    def should_continue(self, step: int, token_probs: torch.Tensor) -> bool:
        return token_probs.mean().item() >= self.threshold


class AlwaysContinuePolicy:
    """No-op policy that never triggers early exit.

    Used when ``draft_confidence_threshold`` is 0 (DSL disabled).  Avoids
    per-step ``if dsl_enabled`` guards in the hot path while still
    satisfying the ``DraftLengthPolicy`` interface.
    """

    def should_continue(self, step: int, token_probs: torch.Tensor) -> bool:
        return True
