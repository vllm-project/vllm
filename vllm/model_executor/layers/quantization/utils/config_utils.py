# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterable, Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING

import regex as re

if TYPE_CHECKING:
    from vllm.model_executor.layers.quantization.base_config import QuantizationConfig


def find_matching_patterns(
    layer_name: str,
    patterns: Iterable[str],
    fused_mapping: Mapping[str, list[str]] = MappingProxyType({}),
) -> list[set[str]]:
    """Return matching patterns for a layer or each shard of a fused layer.

    A pattern matching the fused layer directly takes precedence. Otherwise,
    return one set of matching patterns for every shard.
    """
    patterns = list(patterns)
    matches = [
        pattern for pattern in patterns if is_equal_or_regex_match(layer_name, pattern)
    ]
    if matches:
        return [set(matches)]

    proj_name = layer_name.split(".")[-1]
    if proj_name not in fused_mapping:
        return [set()]

    shard_names = [
        layer_name.replace(proj_name, shard_proj_name)
        for shard_proj_name in fused_mapping[proj_name]
    ]
    per_shard_matches = [
        {
            pattern
            for pattern in patterns
            if is_equal_or_regex_match(shard_name, pattern)
        }
        for shard_name in shard_names
    ]
    return per_shard_matches


def get_layer_name_after_index(layer_name: str) -> str:
    """Return the suffix following the final numeric component of a layer name."""
    parts = layer_name.split(".")
    for index in range(len(parts) - 1, -1, -1):
        if parts[index].isdigit():
            return ".".join(parts[index + 1 :])
    return layer_name


def is_equal_or_regex_match(
    value: str, target: str, check_contains: bool = False
) -> bool:
    """
    Checks whether a value is exactly equal or a regex match for target
    if target starts with 're:'. If check_contains is set to True,
    additionally checks if the target string is contained within the value.
    """

    if target.startswith("re:"):
        pattern = target[3:]
        if re.match(pattern, value):
            return True
    elif check_contains:
        if target.lower() in value.lower():
            return True
    elif target == value:
        return True
    return False


def is_shared_expert_quant_fse_compatible(
    quant_config: "QuantizationConfig | None",
    expert_prefix: str,
    shared_expert_prefix: str,
) -> tuple[bool, str | None]:
    """Check whether quantization permits fused shared-expert execution.

    Returns:
        A compatibility flag and, when incompatible, the reason.
    """
    if quant_config is None:
        return True, None

    from vllm.model_executor.layers.quantization.online.base import (
        OnlineQuantizationConfig,
    )
    from vllm.model_executor.layers.quantization.quark.quark import QuarkConfig

    if isinstance(quant_config, OnlineQuantizationConfig):
        targets = quant_config.args.targets
        if targets is None:
            is_compatible = quant_config.args.moe is not None and (
                quant_config.args.linear is None
                or quant_config.args.linear == quant_config.args.moe
            )
            if is_compatible:
                return True, None
            return (
                False,
                "online quantization must configure MoE quantization and use "
                "the same format for linear layers",
            )

        def get_target(prefix: str) -> str | None:
            matches = find_matching_patterns(prefix, targets)
            if any(len(match) > 1 for match in matches):
                raise ValueError(
                    f"Layer {prefix} matches multiple "
                    f"quantization_config.targets patterns: {matches}."
                )
            if any(not match for match in matches):
                return None
            selected = {targets[next(iter(match))] for match in matches}
            return selected.pop() if len(selected) == 1 else None

        expert_target = get_target(expert_prefix)
        if expert_target is None:
            return (
                False,
                f"routed experts at {expert_prefix} do not have a unique "
                "quantization target",
            )
        shared_targets = {
            target
            for projection in ("gate_up_proj", "down_proj")
            if (target := get_target(f"{shared_expert_prefix}.{projection}"))
            is not None
        }
        if not shared_targets or shared_targets == {expert_target}:
            return True, None
        return (
            False,
            f"shared expert projections at {shared_expert_prefix} use "
            f"{sorted(shared_targets)}, but routed experts at {expert_prefix} "
            f"use {expert_target}",
        )
    elif isinstance(quant_config, QuarkConfig):
        is_compatible = not any(
            "shared_expert" in str(entry)
            for entry in quant_config.quant_config.get("exclude", [])
        )
        if is_compatible:
            return True, None
        return False, f"Quark excludes shared experts at {shared_expert_prefix}"

    return (
        False,
        "shared-expert FSE quantization compatibility is not implemented for "
        f"{type(quant_config).__name__}",
    )
