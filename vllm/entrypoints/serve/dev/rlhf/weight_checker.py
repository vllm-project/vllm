# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Rank-qualified checksum merging and one-shot comparison state."""


def _merge_weight_checksums(
    per_engine: list[dict[str, str]],
) -> dict[str, str]:
    """Merge engine results using the complete parallel-rank-qualified key."""
    merged: dict[str, str] = {}
    for engine_checksums in per_engine:
        duplicate_keys = merged.keys() & engine_checksums.keys()
        if duplicate_keys:
            duplicates = ", ".join(sorted(duplicate_keys))
            raise RuntimeError(f"Duplicate weight checksum keys: {duplicates}")
        merged.update(engine_checksums)
    return merged


class _WeightCheckerState:
    """Store the first checksum result in a verification cycle.

    Operations that mutate the baseline must be externally serialized.
    """

    def __init__(self):
        self.baseline: dict[str, str] | None = None

    def store_if_absent(self, checksums: dict[str, str]) -> bool:
        """Store checksums unless a comparison baseline already exists."""
        if self.baseline is not None:
            return False
        self.baseline = dict(checksums)
        return True

    def has_baseline(self) -> bool:
        """Return whether a comparison baseline is currently stored."""
        return self.baseline is not None

    def compare(self, current: dict[str, str]) -> tuple[bool, list[str]]:
        """Compare the current checksums with the stored baseline.

        Args:
            current: Complete rank-qualified keys mapped to SHA-256 digests.

        Returns:
            A tuple containing whether all tensors match and the names of changed,
            added, or missing tensors.

        Raises:
            RuntimeError: If no baseline has been stored.
        """
        if self.baseline is None:
            raise RuntimeError("No checksum baseline; call action='checksum' first")
        mismatches = sorted(
            key
            for key in self.baseline.keys() | current.keys()
            if self.baseline.get(key) != current.get(key)
        )
        # Compare is one-shot: clear the baseline so a second compare fails.
        self.baseline = None
        return not mismatches, mismatches
