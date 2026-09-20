# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stateless comparison of rank-qualified weight checksums."""


def compare_weight_checksums(
    baseline: dict[str, str],
    current: dict[str, str],
) -> tuple[bool, list[str]]:
    """Return whether every tensor matches, and the keys that differ.

    Keys present in only one of the two maps count as mismatches. The caller
    owns the baseline: with several API processes, any of them may serve any
    request, so no baseline can live server-side.
    """
    mismatches = sorted(
        key
        for key in baseline.keys() | current.keys()
        if baseline.get(key) != current.get(key)
    )
    return not mismatches, mismatches
