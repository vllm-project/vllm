from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock
import pytest

from vllm.multimodal.inputs import MultiModalFeatureSpec


# =====================================================================
# Helpers
# =====================================================================


@dataclass
class _CacheTracker:
    """Tracks feature identifiers applied to the cache."""

    applied: list[str] = field(default_factory=list)


def _make_feature_spec(
    identifier: str = "feat-1",
    skip_mm_cache: Any = False,
    data: Any = None,
    modality: str = "image",
    mm_position: Any = None,
    mm_hash: str | None = "hash-1",
) -> MultiModalFeatureSpec:
    """Build a MultiModalFeatureSpec instance for testing."""
    return MultiModalFeatureSpec(
        data=data,
        modality=modality,
        identifier=identifier,
        mm_position=mm_position if mm_position is not None else MagicMock(),
        mm_hash=mm_hash,
        skip_mm_cache=skip_mm_cache,
    )


def _make_tracker(applied: list[str] | None = None) -> _CacheTracker:
    """Build a cache tracker instance."""
    return _CacheTracker(applied=list(applied) if applied is not None else [])


def _call_apply_features(
    tracker: _CacheTracker,
    features: list[MultiModalFeatureSpec],
) -> list[str]:
    """Apply non-skipped features to the tracker and return applied IDs."""
    for spec in features:
        # Bypass cache updates when per-feature skip_mm_cache is enabled.
        if spec.skip_mm_cache is True:
            continue
        tracker.applied.append(spec.identifier)
    return tracker.applied


# =====================================================================
# Core Contract
# =====================================================================


class TestSkipMmCacheContract:
    """Contract: skip_mm_cache defaults to False; per-item True bypasses cache."""

    def test_default_flag_is_false_and_applies_feature(self):
        """Omitting skip_mm_cache defaults to False and applies the feature."""
        spec = _make_feature_spec(identifier="id-default")
        tracker = _make_tracker()

        assert spec.skip_mm_cache is False

        applied = _call_apply_features(tracker, [spec])

        assert applied == ["id-default"]

    def test_mixed_batch_skips_only_true_items(self):
        """In a mixed batch, skip_mm_cache=True is skipped while False is applied."""
        tracker = _make_tracker()
        features = [
            _make_feature_spec(identifier="id-rendered", skip_mm_cache=True),
            _make_feature_spec(identifier="id-normal", skip_mm_cache=False),
        ]

        applied = _call_apply_features(tracker, features)

        assert applied == ["id-normal"]


# =====================================================================
# Key Boundaries
# =====================================================================


class TestSkipMmCacheBoundaries:
    """Contract: distinguish None vs False and handle empty feature lists."""

    @pytest.mark.parametrize(
        "flag_value,expected_applied",
        [
            (True, []),
            (False, ["id-boundary"]),
            (None, ["id-boundary"]),
        ],
    )
    def test_flag_boundaries_distinguish_true_from_false_and_none(
        self, flag_value: Any, expected_applied: list[str]
    ):
        """Only True bypasses cache handling; False and None apply normally."""
        tracker = _make_tracker()
        spec = _make_feature_spec(identifier="id-boundary", skip_mm_cache=flag_value)

        applied = _call_apply_features(tracker, [spec])

        assert applied == expected_applied

    def test_empty_features_applies_nothing(self):
        """An empty feature list leaves applied items unchanged."""
        tracker = _make_tracker(applied=["id-existing"])

        applied = _call_apply_features(tracker, [])

        assert applied == ["id-existing"]