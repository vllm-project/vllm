"""Exception types for the GLADIUS policy protocol.

These are used internally by PolicyLoader to classify a rejected snapshot;
PolicyLoader.poll() catches all of them and never lets them escape (see
policy.py), so request scheduling can never be taken down by a bad snapshot.
"""

from __future__ import annotations


class PolicySnapshotError(Exception):
    """Base class for all policy_snapshot.json validation failures."""


class PolicyCorruptError(PolicySnapshotError):
    """Snapshot is missing, malformed JSON, wrong types, or fails range/schema validation."""


class PolicyStaleError(PolicySnapshotError):
    """Snapshot's `generation` is <= the last-accepted generation."""


class PolicyEngineMismatchError(PolicySnapshotError):
    """Snapshot's `engine_id` or `model_id` does not match this scheduler's own."""
