# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import TYPE_CHECKING, Optional, Union

# First Party
from lmcache.logging import init_logger
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.lookup_client.abstract_client import LookupClientInterface

if TYPE_CHECKING:
    # Third Party
    import torch

    # First Party
    from lmcache.v1.cache_engine import LMCacheEngine
    from lmcache.v1.metadata import LMCacheMetadata

logger = init_logger(__name__)


class LMCacheBypassLookupClient(LookupClientInterface):
    """
    Bypass lookup client that directly calls LMCacheEngine.lookup()
    instead of using ZMQ communication. This is particularly useful
    for MLA scenarios where only rank 0 needs to perform lookups.
    """

    def __init__(
        self,
        config: LMCacheEngineConfig,
        metadata: "LMCacheMetadata",
        lmcache_engine: "LMCacheEngine",
    ):
        """
        Initialize the bypass lookup client.

        Args:
            config: The LMCacheEngine configuration
            metadata: The LMCacheEngine metadata
            lmcache_engine: The LMCacheEngine instance to use for lookups
        """
        assert isinstance(config, LMCacheEngineConfig), (
            "LMCache v1 configuration should be passed."
        )

        self.lmcache_engine = lmcache_engine
        self.config = config

        # Use the token database from the provided LMCacheEngine
        self.token_database = self.lmcache_engine.token_database
        self.enable_blending = self.config.enable_blending

        logger.info("LMCacheBypassLookupClient initialized")

    def lookup(
        self,
        token_ids: Union["torch.Tensor", list[int]],
        lookup_id: str,
        request_configs: Optional[dict] = None,
    ) -> Optional[int]:
        try:
            if not self.enable_blending:
                # Process tokens to get hashes and offsets
                hashes = []
                offsets = []
                for start, end, key in self.token_database.process_tokens(
                    token_ids, make_key=False
                ):
                    hashes.append(key)
                    offsets.append(end - start)
                if not hashes:
                    return 0

                # Call LMCacheEngine lookup with hashes and offsets
                result = self.lmcache_engine.lookup(
                    hashes=hashes,
                    offsets=offsets,
                    lookup_id=lookup_id,
                    pin=True,
                    request_configs=request_configs,
                )
            else:
                # For blending mode, pass tokens directly
                result = self.lmcache_engine.lookup(
                    tokens=token_ids,
                    lookup_id=lookup_id,
                    pin=True,
                    request_configs=request_configs,
                )

            return result

        except Exception as e:
            logger.error("Error in bypass lookup: %s", e)
            return 0

    def supports_producer_reuse(self) -> bool:
        return True

    def close(self):
        # No resources to clean up for bypass client
        logger.info("LMCacheBypassLookupClient closed")
