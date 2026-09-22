# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Rank-qualified weight checksum merging and comparison.

Pure mapping utilities: no torch, no config, no distributed state, so the
engine, the executor and the API process can all import them cheaply.

The keys produced by the workers are ``{rank prefix}{tensor name}``, where the
prefix is ``dp{dp}:pp{pp}:pcp{pcp}:tp{tp}:ep{ep}:``. The prefix is the only
record of which ranks a digest came from, so the functions here that need to
regroup digests parse it back out rather than take a second structure.
"""


def split_checksum_key(key: str) -> tuple[str, str]:
    """Split a checksum key into its rank prefix and tensor name.

    Splits on the first five colons, so a tensor name containing a colon still
    lands entirely in the name.

    Args:
        key: A rank-qualified checksum key from a worker or a baseline.

    Returns:
        The ``dp:pp:pcp:tp:ep:`` prefix and the tensor name.
    """
    parts = key.split(":", 5)
    if len(parts) != 6:
        return "", key
    return ":".join(parts[:5]) + ":", parts[5]


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


def compare_weight_checksum_reports(
    reports: list[dict[str, str]],
) -> tuple[bool, list[str], list[str]]:
    """Return whether every report agrees on every checksum it contains.

    Each report is one full ``checksum`` response, so an entry is a digest of
    one shard of one rank. The usual call is ``[baseline, current]``; passing
    more reports compares replicas against each other instead of against a
    single baseline, and passing one report only checks that it is readable.

    Only the same rank-qualified key is compared: the same tensor name on
    another rank holds a different shard, and its digest is expected to differ.

    A key that only some reports carry separates "the replicas disagree" from
    "these reports do not cover the same ranks", which is why the covered
    prefixes are returned alongside the verdict.

    Args:
        reports: One checksum mapping per report to compare.

    Returns:
        Whether every entry is identical in each report, the keys that are not,
        and the distinct rank prefixes the reports covered.
    """
    observed: dict[str, dict[int, str]] = {}
    for index, report in enumerate(reports):
        for key, digest in report.items():
            observed.setdefault(key, {})[index] = digest

    mismatches = sorted(
        key
        for key, digests in observed.items()
        if len(digests) != len(reports) or len(set(digests.values())) > 1
    )
    covered = sorted({split_checksum_key(key)[0] for key in observed})
    return not mismatches, mismatches, covered
