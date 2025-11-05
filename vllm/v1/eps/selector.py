# ABOUTME: Union selection helpers for EPS pre-pass.
# ABOUTME: Scores groups using JL summaries and applies visit heuristics.

from __future__ import annotations

from collections.abc import Iterable
from typing import Dict, Sequence

from vllm.v1.eps.config import EpsRuntimeConfig


def _compute_threshold(scores: Sequence[float], top_pages: int | None) -> float:
    if not scores:
        return 0.0
    if top_pages is None or len(scores) < top_pages:
        return max(scores)
    sorted_scores = sorted(scores, reverse=True)
    return sorted_scores[top_pages - 1]


def select_union_groups(
    *,
    cfg: EpsRuntimeConfig,
    groups_by_recency: Sequence[int],
    energy_by_group: Dict[int, float],
) -> set[int]:
    if not groups_by_recency:
        return set()

    alpha = cfg.alpha * (1.1 if cfg.strict else 1.0)
    visit = set(groups_by_recency[: cfg.last_n])
    kept_scores: list[float] = [
        energy_by_group.get(g, 0.0)
        for g in visit
        if energy_by_group.get(g) not in (None, float("inf"))
    ]
    top_pages = cfg.top_pages

    for group_id in groups_by_recency:
        energy = energy_by_group.get(group_id)
        if group_id in visit:
            if energy not in (None, float("inf")):
                kept_scores.append(energy)
            continue

        if energy == float("inf"):
            visit.add(group_id)
            continue

        energy_value = 0.0 if energy is None else energy
        threshold = _compute_threshold(kept_scores, top_pages)
        limit = threshold / alpha if threshold > 0 else 0.0
        if energy_value >= limit:
            visit.add(group_id)
            kept_scores.append(energy_value)

    if top_pages is not None and len(visit) > top_pages:
        sorted_groups = sorted(visit, key=lambda g: energy_by_group.get(g, 0.0), reverse=True)
        keep = set(groups_by_recency[: cfg.last_n])
        for g in sorted_groups:
            if len(keep) >= top_pages:
                break
            keep.add(g)
        visit = keep

    return visit



def build_union_visit_set(
    total_groups: int,
    last_n: int,
    *,
    candidate_groups: Iterable[int] | None = None,
) -> set[int]:
    """Backward-compatible helper returning the newest last_n groups."""
    if total_groups <= 0:
        return set()
    groups = list(candidate_groups) if candidate_groups is not None else list(range(total_groups))
    if last_n <= 0:
        return set()
    return set(groups[: min(last_n, len(groups))])
