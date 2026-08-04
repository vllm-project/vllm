# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import List, Optional
import abc

# First Party
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey
from lmcache.v1.protocol import ClientMetaMessage
from lmcache.v1.server.utils import LMSMemoryObj

logger = init_logger(__name__)


class LMSBackendInterface(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def put(
        self,
        client_meta: ClientMetaMessage,
        kv_chunk_bytes: bytearray,
    ) -> None:
        """
        Store the KV cache of the tokens into the cache server.

        Args:
            key: the key of the token chunk, in the format of CacheEngineKey
            client_meta: metadata sent by the client
            kv_chunk_bytes: the kv cache (bytearray) of the token chunk

        Returns:
            None
        """
        raise NotImplementedError

    @abc.abstractmethod
    def contains(
        self,
        key: CacheEngineKey,
    ) -> bool:
        """
        Query if a key is in the cache or not
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get(
        self,
        key: CacheEngineKey,
    ) -> Optional[LMSMemoryObj]:
        """
        Retrieve LMSMemoryObj by the given key

        Input:
            key: the CacheEngineKey

        Output:
            An LMSMemoryObj object that contains the KV cache bytearray
            with the some metadata
        """
        raise NotImplementedError

    @abc.abstractmethod
    def list_keys(
        self,
    ) -> List[CacheEngineKey]:
        """
        List all keys in the cache server

        Output:
            All keys in the cache server
        """
        raise NotImplementedError

    @abc.abstractmethod
    def close(self):
        """
        Do the cleanup things
        Children classes should override this method if necessary
        """
        pass
