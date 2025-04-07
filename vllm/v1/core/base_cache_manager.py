# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from typing import Optional

from vllm.logger import init_logger
from vllm.v1.core.kv_cache_utils import KVCacheBlock
from vllm.v1.metrics.stats import PrefixCacheStats
from vllm.v1.request import Request
from vllm.v1.core.kv_cache_utils import BlockHashType

logger = init_logger(__name__)


class BaseKVCacheManager(ABC):

    @property
    @abstractmethod
    def usage(self) -> float:
        raise NotImplementedError

    @abstractmethod
    def make_prefix_cache_stats(self) -> PrefixCacheStats:
        raise NotImplementedError

    @abstractmethod
    def get_computed_blocks(self, request: Request) -> tuple[list[KVCacheBlock],
                                         list[BlockHashType], int]:
        raise NotImplementedError

    @abstractmethod
    def allocate_slots(
        self,
        request: Request,
        num_tokens: int,
        new_computed_blocks: Optional[list[KVCacheBlock]] = None,
        new_computed_extended_blocks: Optional[list[BlockHashType]] = None,
        saved_blocks: Optional[set[int]] = None,
    ) -> Optional[list[KVCacheBlock]]:
        raise NotImplementedError

    @abstractmethod
    def free(self, request: Request) -> None:
        raise NotImplementedError

    @abstractmethod
    def reset_prefix_cache(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def get_num_common_prefix_blocks(
        self,
        request: Request,
        num_running_requests: int,
    ) -> int:
        raise NotImplementedError

    @abstractmethod
    def free_block_hashes(self, request: Request) -> None:
        raise NotImplementedError
    
    @abstractmethod
    def clear_swap_metadata(self) -> None:
        raise NotImplementedError