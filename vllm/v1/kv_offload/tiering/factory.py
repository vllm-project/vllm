# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Factory for creating secondary tier implementations.
"""

from vllm.v1.kv_offload.tiering.base import SecondaryTierManager
from vllm.v1.kv_offload.tiering.dummy import DummySecondaryTier

SUPPORTED_TIERS: tuple[type[SecondaryTierManager], ...] = (DummySecondaryTier,)

_TIER_REGISTRY: dict[str, type[SecondaryTierManager]] = {
    cls.get_tier_type(): cls for cls in SUPPORTED_TIERS
}


def create_secondary_tier(tier_config: dict) -> SecondaryTierManager:
    """
    Create a secondary tier from configuration.

    Args:
        tier_config: Dictionary with tier configuration containing:
            - type (required): Type of secondary tier (e.g., "dummy")
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

    cls = _TIER_REGISTRY.get(tier_type)
    if cls is None:
        raise ValueError(
            f"Unknown secondary tier type: {tier_type!r}. "
            f"Supported types: {list(_TIER_REGISTRY)}"
        )
    return cls(**config)
