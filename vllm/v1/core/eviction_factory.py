"""
Factory for creating eviction policies from configuration.

This module provides a factory function to instantiate the appropriate eviction
policy based on CacheConfig settings.

Reference: PagedEviction paper (arXiv:2509.04377v1), Implementation Guide Example 4
"""

from vllm.config.cache import CacheConfig
from vllm.v1.core.eviction_policy import (
    EvictionPolicy,
    LRUEvictionPolicy,
    PagedEvictionPolicy,
)


def create_eviction_policy(cache_config: CacheConfig) -> EvictionPolicy:
    """
    Factory function to create eviction policy from cache configuration.

    This reads the eviction_policy field from CacheConfig and instantiates
    the appropriate policy with configured parameters.

    Args:
        cache_config: CacheConfig with eviction policy settings

    Returns:
        Instantiated eviction policy ready to use

    Raises:
        ValueError: If eviction_policy specifies an unknown policy type
    """

    if cache_config.eviction_policy == "lru":
        return LRUEvictionPolicy()

    elif cache_config.eviction_policy == "paged":
        return PagedEvictionPolicy(
            recency_weight=cache_config.paged_eviction_recency_weight,
            frequency_weight=cache_config.paged_eviction_frequency_weight,
            cache_weight=cache_config.paged_eviction_cache_weight,
            time_decay=cache_config.paged_eviction_time_decay,
            enable_prefill_eviction=cache_config.paged_eviction_enable_prefill,
            enable_decode_eviction=cache_config.paged_eviction_enable_decode,
        )

    else:
        raise ValueError(f"Unknown eviction policy: {cache_config.eviction_policy}")


def get_eviction_policy_info(policy: EvictionPolicy) -> dict[str, any]:  # type: ignore
    """
    Get information about an eviction policy for logging/debugging.

    Args:
        policy: The eviction policy to inspect

    Returns:
        Dictionary with policy type and configuration
    """

    if isinstance(policy, LRUEvictionPolicy):
        return {
            "type": "lru",
            "description": "Least Recently Used (baseline)",
        }

    elif isinstance(policy, PagedEvictionPolicy):
        return {
            "type": "paged",
            "description": "PagedEviction block-wise eviction",
            "recency_weight": policy.recency_weight,
            "frequency_weight": policy.frequency_weight,
            "cache_weight": policy.cache_weight,
            "time_decay": policy.time_decay,
            "prefill_enabled": policy.enable_prefill_eviction,
            "decode_enabled": policy.enable_decode_eviction,
        }

    else:
        return {
            "type": "unknown",
            "class": policy.__class__.__name__,
        }
