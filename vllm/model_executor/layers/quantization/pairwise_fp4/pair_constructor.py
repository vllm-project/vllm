# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import math

import torch


def construct_pairs(
    risk_scores: torch.Tensor,
    policy: str = "high_high",
    top_ratio: float = 0.05,
    secondary_risk_scores: torch.Tensor | None = None,
) -> torch.Tensor:
    _VALID = {
        "adjacent_sorted",
        "high_high",
        "high_low",
        "random_baseline",
        "joint_compatible",
    }
    if risk_scores.ndim != 1:
        raise ValueError(
            f"risk_scores must be 1-D, got {risk_scores.ndim}-D"
        )
    if policy not in _VALID:
        raise ValueError(
            f"Unknown policy '{policy}', must be one of {sorted(_VALID)}"
        )
    if policy == "joint_compatible" and secondary_risk_scores is None:
        raise ValueError(
            "secondary_risk_scores is required for joint_compatible policy"
        )
    if secondary_risk_scores is not None:
        if secondary_risk_scores.shape != risk_scores.shape:
            raise ValueError(
                f"Shape mismatch: risk_scores {risk_scores.shape} vs "
                f"secondary_risk_scores {secondary_risk_scores.shape}"
            )

    C = risk_scores.shape[0]
    top_ratio = max(0.0, min(1.0, top_ratio))
    K = math.ceil(C * top_ratio)
    if K % 2 != 0:
        K -= 1
    if K < 2:
        return torch.empty(0, 2, dtype=torch.int64)

    if policy == "adjacent_sorted":
        return _adjacent_sorted(risk_scores, K)
    if policy == "high_high":
        return _high_high(risk_scores, K)
    if policy == "high_low":
        return _high_low(risk_scores, K)
    if policy == "random_baseline":
        return _random_baseline(risk_scores, K)
    # joint_compatible
    return _joint_compatible(risk_scores, K, secondary_risk_scores)


def _top_k_sorted_indices(scores: torch.Tensor, K: int) -> torch.Tensor:
    _, idx = torch.sort(scores, descending=True)
    return idx[:K]


def _adjacent_sorted(risk_scores: torch.Tensor, K: int) -> torch.Tensor:
    idx = _top_k_sorted_indices(risk_scores, K)
    return idx.reshape(-1, 2).to(dtype=torch.int64, device="cpu")


def _high_high(risk_scores: torch.Tensor, K: int) -> torch.Tensor:
    idx = _top_k_sorted_indices(risk_scores, K)
    return idx.reshape(-1, 2).to(dtype=torch.int64, device="cpu")


def _high_low(risk_scores: torch.Tensor, K: int) -> torch.Tensor:
    idx = _top_k_sorted_indices(risk_scores, K)
    first_half = idx[: K // 2]
    second_half = idx[K // 2:].flip(0)
    pairs = torch.stack([first_half, second_half], dim=1)
    return pairs.to(dtype=torch.int64, device="cpu")


def _random_baseline(risk_scores: torch.Tensor, K: int) -> torch.Tensor:
    idx = _top_k_sorted_indices(risk_scores, K)
    gen = torch.Generator()
    gen.manual_seed(42)
    perm = torch.randperm(K, generator=gen)
    idx = idx[perm]
    return idx.reshape(-1, 2).to(dtype=torch.int64, device="cpu")


def _joint_compatible(
    risk_scores: torch.Tensor,
    K: int,
    secondary_risk_scores: torch.Tensor | None,
) -> torch.Tensor:
    combined = risk_scores + secondary_risk_scores  # type: ignore[operator]
    return _high_high(combined, K)
