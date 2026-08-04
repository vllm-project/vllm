# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import Optional, Union

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.lookup_client.abstract_client import LookupClientInterface
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.storage_backend.connector.mooncakestore_connector import (
    MooncakeStoreConfig,
    setup_mooncake_store,
)

logger = init_logger(__name__)


class MooncakeLookupClient(LookupClientInterface):
    def __init__(
        self,
        config: LMCacheEngineConfig,
        metadata: LMCacheMetadata,
        master_addr: str,
    ):
        # Third Party
        from mooncake.store import MooncakeDistributedStore

        self.store = MooncakeDistributedStore()
        # Deprecated
        # TODO(baoloongmao): use dict config instead in the future
        lookup_config = MooncakeStoreConfig(
            setup_config={
                "local_hostname": "localhost",
                "metadata_server": "P2PHANDSHAKE",
                "global_segment_size": "0",
                "local_buffer_size": str(16 * 1024 * 1024),
                "protocol": "tcp",
                "device_name": "",
                "master_server_address": master_addr,
            },
        )
        setup_mooncake_store(self.store, lookup_config)

        # Initialize token database for processing tokens
        assert isinstance(config, LMCacheEngineConfig), (
            "LMCache v1 configuration is should be passed."
        )

        # First Party
        from lmcache.v1.token_database import ChunkedTokenDatabase

        assert not config.enable_blending, (
            "LMCache v1 blending is not supported in MooncakeLookupClient yet."
        )
        self.token_database = ChunkedTokenDatabase(config, metadata)

    def lookup(
        self,
        token_ids: Union[torch.Tensor, list[int]],
        lookup_id: Optional[str] = None,
        request_configs: Optional[dict] = None,
    ) -> Optional[int]:
        # process token_ids to cacheengine keys
        keys = []
        ends = []
        for start, end, key in self.token_database.process_tokens(token_ids):
            assert isinstance(key, CacheEngineKey)
            keys.append(key.to_string())
            ends.append(end)

        # Use batch_is_exist to check all keys at once
        # rets is list of int: 1 = found, 0 = not found, -1 = error
        rets = self.store.batch_is_exist(keys)

        # Find the first key that doesn't exist (ret != 1)
        # This follows the same logic as cache engine's lookup method
        for i, ret in enumerate(rets):
            if ret != 1:  # Not found or error
                # Return the end position of the previous chunk
                # If i == 0, no chunks were found, return 0
                return ends[i - 1] if i > 0 else 0

        # All keys were found, return the last end position
        return ends[-1] if ends else 0

    def supports_producer_reuse(self) -> bool:
        """Return True as MooncakeLookupClient supports producer kvcache reuse"""
        return True

    def close(self):
        # nothing here
        pass
