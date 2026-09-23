# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Core algorithms for speculative speculative decoding (SSD / Saguaro).

Implements the building blocks from https://arxiv.org/abs/2603.03251:

- Geometric fan-out allocation (Theorem 12): how many bonus-token guesses
  to prepare per accepted-prefix length under a total budget.
- Saguaro sampling (Definition 14): downweighting the top-F draft logits
  to concentrate residual mass on cached bonus-token guesses.
- Verification outcome prediction: top-F_k bonus-token guesses per
  position, excluding the drafted token (which cannot be the bonus token).
- SpeculationCache: maps predicted verification outcomes
  (num_accepted, bonus_token) to pre-computed speculations.
"""

import math
from dataclasses import dataclass, field

import torch

VerificationOutcome = tuple[int, int]
"""(num_accepted_draft_tokens, bonus_token_id)"""


def compute_geometric_fanout(
    budget: int,
    acceptance_rate: float,
    num_speculative_tokens: int,
    power_law_r: float = 1.0,
) -> list[int]:
    """Allocate bonus-token guesses per position (Theorem 12).

    Args:
        budget: Total number of verification outcomes to prepare for (B).
        acceptance_rate: Per-token draft acceptance rate (a_p).
        num_speculative_tokens: Speculative lookahead (K).
        power_law_r: Exponent of the power-law cache hit rate (r).

    Returns:
        Fan-out values ``[F_0, ..., F_K]`` (length K+1) summing to <= budget.

    """
    if budget < 1:
        raise ValueError(f"budget must be >= 1, got {budget}")
    if num_speculative_tokens < 1:
        raise ValueError(
            f"num_speculative_tokens must be >= 1, got {num_speculative_tokens}"
        )
    if power_law_r <= 0:
        raise ValueError(f"power_law_r must be > 0, got {power_law_r}")

    k = num_speculative_tokens
    a = min(max(acceptance_rate, 1e-4), 1.0 - 1e-4)
    inv = 1.0 / (1.0 + power_law_r)
    c = a**inv
    last_scale = (1.0 - a) ** -inv
    denom = c**k * last_scale + (1.0 - c**k) / (1.0 - c)
    f0 = budget / denom

    ideal = [f0 * c**i for i in range(k)]
    ideal.append(f0 * c**k * last_scale)

    if budget < k + 1:
        fanout = [0] * (k + 1)
        order = sorted(range(k + 1), key=lambda i: ideal[i], reverse=True)
        for i in order[:budget]:
            fanout[i] = 1
        return fanout

    fanout = [max(1, math.floor(f)) for f in ideal]
    remainders = sorted(range(k + 1), key=lambda i: ideal[i] - fanout[i], reverse=True)
    for i in remainders:
        if sum(fanout) >= budget:
            break
        fanout[i] += 1
    while sum(fanout) > budget:
        i = max(range(k + 1), key=lambda j: fanout[j])
        fanout[i] -= 1
    return fanout


def saguaro_adjust_logits(
    logits: torch.Tensor,
    fanout: int,
    downweight_c: float,
) -> torch.Tensor:
    """Apply the Saguaro sampling scheme (Definition 14) to draft logits.

    Downweights the probability of the top-``fanout`` tokens by a factor
    ``downweight_c``, pushing residual distribution mass onto them so the
    bonus token is more likely to land in the speculation cache.

    Args:
        logits: Draft logits of shape ``[..., vocab_size]``.
        fanout: Number of top tokens to downweight (F).
        downweight_c: Multiplicative factor in ``[0, 1]`` (C).

    Returns:
        Adjusted logits of the same shape.

    """
    if not 0.0 <= downweight_c <= 1.0:
        raise ValueError(f"downweight_c must be in [0, 1], got {downweight_c}")
    if fanout == 0 or downweight_c == 1.0:
        return logits
    top_indices = logits.topk(fanout, dim=-1).indices
    adjusted = logits.clone()
    if downweight_c == 0.0:
        adjusted.scatter_(-1, top_indices, float("-inf"))
    else:
        adjusted.scatter_add_(
            -1,
            top_indices,
            torch.full_like(top_indices, math.log(downweight_c), dtype=logits.dtype),
        )
    return adjusted


def predict_bonus_tokens(
    draft_logits: torch.Tensor,
    draft_token_ids: torch.Tensor,
    fanout: list[int],
) -> list[list[int]]:
    """Predict likely bonus tokens per accepted-prefix length.

    Takes the top-``fanout[k]`` tokens from the draft logits at position k,
    excluding the drafted token sent for verification at that position
    (which is guaranteed not to be the bonus token).

    Args:
        draft_logits: ``[num_positions, vocab_size]`` draft logits, one row
            per possible accepted-prefix length.
        draft_token_ids: ``[num_positions]`` drafted token at each position,
            or -1 if there is no token to exclude (e.g. the all-accepted
            position).
        fanout: Number of bonus-token guesses per position.

    Returns:
        Per-position lists of predicted bonus token ids.

    """
    num_positions = draft_logits.shape[0]
    if len(fanout) != num_positions or draft_token_ids.shape[0] != num_positions:
        raise ValueError(
            "draft_logits, draft_token_ids and fanout must have matching "
            f"lengths, got {num_positions}, {draft_token_ids.shape[0]}, "
            f"{len(fanout)}"
        )
    max_fanout = max(fanout)
    if max_fanout == 0:
        return [[] for _ in range(num_positions)]
    top_k = min(max_fanout + 1, draft_logits.shape[-1])
    top_indices = draft_logits.topk(top_k, dim=-1).indices.tolist()
    excluded = draft_token_ids.tolist()
    predictions: list[list[int]] = []
    for k in range(num_positions):
        guesses = [t for t in top_indices[k] if t != excluded[k]]
        predictions.append(guesses[: fanout[k]])
    return predictions


@dataclass
class SpeculationCache:
    """Pre-computed speculations keyed by predicted verification outcome.

    While the target model verifies round T, the drafter fills this cache
    with speculations for likely round-T outcomes. On a hit, the cached
    speculation is returned immediately, hiding all drafting latency.
    """

    cache: dict[VerificationOutcome, list[int]] = field(default_factory=dict)
    hits: int = 0
    misses: int = 0

    def put(
        self,
        num_accepted: int,
        bonus_token_id: int,
        speculation: list[int],
    ) -> None:
        self.cache[(num_accepted, bonus_token_id)] = speculation

    def get(self, num_accepted: int, bonus_token_id: int) -> list[int] | None:
        speculation = self.cache.get((num_accepted, bonus_token_id))
        if speculation is None:
            self.misses += 1
        else:
            self.hits += 1
        return speculation

    def clear(self) -> None:
        self.cache.clear()

    def __len__(self) -> int:
        return len(self.cache)

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0
