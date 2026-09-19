# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stateless comparison of rank-qualified weight checksums."""


def compare_weight_checksums(
    baseline: dict[str, str],
    current: dict[str, str],
) -> tuple[bool, list[str]]:
    """Return whether every tensor matches, and the keys that differ.

    The caller owns the baseline: a multi-API-process deployment routes
    requests to arbitrary processes, so no baseline can be kept server-side.
    Keys present in only one of the two maps count as mismatches.
    """
    mismatches = sorted(
        key
        for key in baseline.keys() | current.keys()
        if baseline.get(key) != current.get(key)
    )
    return not mismatches, mismatches
