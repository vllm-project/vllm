# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Factory for creating secondary tier implementations.
"""

from vllm.v1.kv_offload.tiering.base import SecondaryTierManager


def create_secondary_tier(tier_config: dict) -> SecondaryTierManager:
    """
    Create a secondary tier from configuration.

    Args:
        tier_config: Dictionary with tier configuration containing:
            - type (required): Type of secondary tier (e.g., "dummy")
            - tier_name (required): Name for this tier
            - Additional tier-specific parameters are passed directly
              to the tier constructor

    Returns:
        SecondaryTierManager instance

    Raises:
        ValueError: If tier type is unknown or configuration is invalid
    """
    config = tier_config.copy()

    tier_type = config.pop("type", None)
    if not tier_type:
        raise ValueError("Secondary tier configuration must include 'type'")

    tier_name = config.pop("tier_name", None)
    if not tier_name:
        raise ValueError("Secondary tier configuration must include 'tier_name'")

    if tier_type == "dummy":
        from vllm.v1.kv_offload.tiering.dummy import DummySecondaryTier

        return DummySecondaryTier(tier_name=tier_name, **config)
    else:
        raise ValueError(
            f"Unknown secondary tier type: {tier_type}. Supported types: dummy"
        )
