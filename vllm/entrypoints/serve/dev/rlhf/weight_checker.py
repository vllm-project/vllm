# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""One-shot checksum baseline state for the Weight Checker endpoint."""


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
