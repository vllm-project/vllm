# SPDX-License-Identifier: Apache-2.0

# First Party
from lmcache.v1.distributed.config import EvictionConfig
from lmcache.v1.distributed.eviction import EvictionPolicy
from lmcache.v1.distributed.eviction_policy.isolated_lru import (
    IsolatedLRUEvictionPolicy,
)
from lmcache.v1.distributed.eviction_policy.lru import LRUEvictionPolicy
from lmcache.v1.distributed.eviction_policy.noop import NoOpEvictionPolicy


def CreateEvictionPolicy(eviction_config: EvictionConfig) -> EvictionPolicy:
    """
    Factory method to create an eviction policy based on the provided configuration.

    Args:
        eviction_config (EvictionConfig): The configuration for the eviction policy.

    Returns:
        An instance of the specified eviction policy.
    """
    if eviction_config.eviction_policy == "LRU":
        return LRUEvictionPolicy()
    elif eviction_config.eviction_policy == "IsolatedLRU":
        return IsolatedLRUEvictionPolicy()
    elif eviction_config.eviction_policy == "noop":
        return NoOpEvictionPolicy()
    else:
        raise ValueError(
            f"Unsupported eviction policy: {eviction_config.eviction_policy}"
        )
