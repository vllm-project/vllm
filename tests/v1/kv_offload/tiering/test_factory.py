# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Unit tests for SecondaryTierFactory.

These tests verify:
1. Pre-registration integrity — registered tier module paths can import
   and yield correct SecondaryTierManager subclasses (CI sentinel).
2. Multi-tier creation via factory with correct tier_type propagation.
3. Error paths — missing tier_type, unknown tier_type, duplicate registration.
"""

from unittest.mock import MagicMock

import pytest

from vllm.v1.kv_offload.tiering.base import SecondaryTierManager
from vllm.v1.kv_offload.tiering.example.manager import ExampleSecondaryTierManager
from vllm.v1.kv_offload.tiering.factory import SecondaryTierFactory

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def restore_registry():
    """Save and restore SecondaryTierFactory._registry between tests."""
    original = dict(SecondaryTierFactory._registry)
    yield
    SecondaryTierFactory._registry = original


def _make_mock_args():
    """Build common mock args for create_secondary_tier."""
    return MagicMock(), MagicMock()  # primary_kv_view, offloading_spec


# ---------------------------------------------------------------------------
# Pre-registration integrity (CI sentinel)
# ---------------------------------------------------------------------------


def test_pre_registered_tiers_can_be_imported():
    """CI sentinel: example/fs/obj paths must import and yield SecondaryTierManager."""
    for tier_type in SecondaryTierFactory._registry:
        cls = SecondaryTierFactory._registry[tier_type]()
        assert issubclass(cls, SecondaryTierManager)


def test_example_tier_registered():
    """Example tier is registered."""
    cls = SecondaryTierFactory._registry["example"]()
    assert cls is ExampleSecondaryTierManager


# ---------------------------------------------------------------------------
# Normal path — create_secondary_tier
# ---------------------------------------------------------------------------


def test_create_tier_from_registry():
    """Registered tier_type creates instance with correct tier_type."""
    primary_kv_view, offloading_spec = _make_mock_args()
    tier_config = {"type": "example"}

    tier = SecondaryTierFactory.create_secondary_tier(
        tier_config, primary_kv_view, offloading_spec
    )

    assert isinstance(tier, SecondaryTierManager)
    assert tier.tier_type == "example"


def test_create_multiple_tiers():
    """Multiple tier configs can be created with correct tier_types."""
    primary_kv_view, offloading_spec = _make_mock_args()
    configs = [
        {"type": "example", "custom_param": 1},
        {"type": "example", "custom_param": 2},
    ]

    tiers = [
        SecondaryTierFactory.create_secondary_tier(
            cfg.copy(), primary_kv_view, offloading_spec
        )
        for cfg in configs
    ]

    assert len(tiers) == 2
    assert all(tier.tier_type == "example" for tier in tiers)
    assert all(isinstance(tier, ExampleSecondaryTierManager) for tier in tiers)


def test_register_new_tier_type():
    """Verify that new tier types can be registered and created.

    This is how external projects add custom secondary tiers
    (e.g., llm-d FS backend was upstreamed as "fs" tier via this mechanism).
    """
    # Register a new tier type (reuse example manager for simplicity)
    SecondaryTierFactory.register_tier(
        "custom_tier",
        "vllm.v1.kv_offload.tiering.example.manager",
        "ExampleSecondaryTierManager",
    )

    primary_kv_view, offloading_spec = _make_mock_args()
    tier = SecondaryTierFactory.create_secondary_tier(
        {"type": "custom_tier", "custom_param": 99},
        primary_kv_view,
        offloading_spec,
    )

    assert tier.tier_type == "custom_tier"
    assert isinstance(tier, ExampleSecondaryTierManager)


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


def test_missing_tier_type_raises():
    """tier_config without 'type' → ValueError."""
    primary_kv_view, offloading_spec = _make_mock_args()
    tier_config: dict[str, str] = {}

    with pytest.raises(ValueError, match="must include 'type'"):
        SecondaryTierFactory.create_secondary_tier(
            tier_config, primary_kv_view, offloading_spec
        )


def test_unknown_tier_type_raises():
    """Unrecognized tier_type → ValueError with supported types list."""
    primary_kv_view, offloading_spec = _make_mock_args()
    tier_config = {"type": "nonexistent_tier"}

    with pytest.raises(
        ValueError,
        match=r"Unknown secondary tier type.*Supported types:",
    ):
        SecondaryTierFactory.create_secondary_tier(
            tier_config, primary_kv_view, offloading_spec
        )


def test_duplicate_registration_raises():
    """register_tier with existing type → ValueError."""
    with pytest.raises(ValueError, match="is already registered"):
        SecondaryTierFactory.register_tier("example", "some.module", "SomeClass")
