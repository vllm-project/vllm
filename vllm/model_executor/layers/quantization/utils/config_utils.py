# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterable, Mapping
from types import MappingProxyType

import regex as re


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
