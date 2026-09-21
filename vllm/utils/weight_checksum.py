# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Rank-qualified weight checksum merging and comparison.

Pure mapping utilities: no torch, no config, no distributed state, so the
engine, the executor and the API process can all import them cheaply.
"""


def combine_weight_checksums(per_worker: list[dict[str, str]]) -> dict[str, str]:
    """Merge per-worker checksum maps into one rank-qualified map.

    Worker keys carry their parallel ranks, so the same logical weight appears
    once per shard. An overlapping key means a worker failed to qualify it.

    Raises:
        RuntimeError: If two workers report the same key.
    """
    combined: dict[str, str] = {}
    for worker_checksums in per_worker:
        duplicate_keys = combined.keys() & worker_checksums.keys()
        if duplicate_keys:
            duplicates = ", ".join(sorted(duplicate_keys))
            raise RuntimeError(f"Duplicate weight checksum keys: {duplicates}")
        combined.update(worker_checksums)
    return combined


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
