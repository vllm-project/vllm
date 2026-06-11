# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Semantic parameter selectors for weight offloading."""

from collections.abc import Iterable

import torch.nn as nn

from vllm.config.offload import PrefetchOffloadSelector

_ATTENTION_SEGMENTS = ("self_attn", "attn", "attention")
_DENSE_MLP_SEGMENTS = (
    "mlp.gate_up_proj",
    "mlp.down_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "feed_forward.w1",
    "feed_forward.w2",
    "feed_forward.w3",
)
_ROUTED_EXPERT_SEGMENTS = (
    "mlp.experts",
    "experts.w13_weight",
    "experts.w2_weight",
    "experts.gate_up_proj",
    "experts.down_proj",
)
_SHARED_EXPERT_SEGMENTS = ("mlp.shared_expert", "shared_expert")
_ROUTED_EXPERT_AUXILIARY_LEAF_SEGMENTS = ("scale",)


def _segment_match(name: str, pattern: str) -> bool:
    return f".{pattern}." in f".{name}."


def _matches_param_filter(name: str, patterns: Iterable[str]) -> bool:
    return any(_segment_match(name, pattern) for pattern in patterns)


def _matches_routed_expert_auxiliary(name: str) -> bool:
    leaf_name = name.rsplit(".", 1)[-1]
    return any(
        segment in leaf_name for segment in _ROUTED_EXPERT_AUXILIARY_LEAF_SEGMENTS
    )


def matches_selector(name: str, selector: PrefetchOffloadSelector) -> bool:
    """Return whether a parameter name matches a semantic selector."""
    if selector == "attention":
        return _matches_param_filter(name, _ATTENTION_SEGMENTS)
    if selector == "dense_mlp":
        return _matches_param_filter(name, _DENSE_MLP_SEGMENTS)
    if selector == "shared_experts":
        return _matches_param_filter(name, _SHARED_EXPERT_SEGMENTS)
    if selector == "routed_experts":
        return _matches_param_filter(name, _ROUTED_EXPERT_SEGMENTS) and not (
            matches_selector(name, "shared_experts")
            or _matches_param_filter(name, ("gate", "router"))
            or _matches_routed_expert_auxiliary(name)
        )
    raise ValueError(f"Unknown prefetch offload selector: {selector}")


def select_module_parameters(
    module: nn.Module,
    selectors: set[PrefetchOffloadSelector] | None = None,
    include_names: set[str] | None = None,
) -> list[str]:
    """Select module parameter names for offloading.

    Selector matches and include-name matches are unioned. If neither is
    provided, all module parameters are selected.
    """
    param_names = [name for name, _ in module.named_parameters()]
    if not selectors and not include_names:
        return param_names

    selected: set[str] = set()
    selectors = selectors or set()
    include_names = include_names or set()

    for name in param_names:
        if selectors and any(
            matches_selector(name, selector) for selector in selectors
        ):
            selected.add(name)
            continue
        if include_names and _matches_param_filter(name, include_names):
            selected.add(name)

    return [name for name in param_names if name in selected]
