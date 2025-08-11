# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TypeVar, Union

import torch

from vllm.logger import init_logger
from vllm.utils import GiB_bytes, LRUCache
from vllm.utils.jsontree import json_map_leaves, json_reduce_leaves

from .inputs import MultiModalKwargs, MultiModalKwargsItem, NestedTensors

logger = init_logger(__name__)


@dataclass
class MultiModalCacheItemMetadata:
    size: int

    @classmethod
    def wraps(cls, value: "MultiModalCacheValue"):
        return cls(size=MultiModalCache.get_item_size(value))


MultiModalCacheValue = Union[
    MultiModalKwargs,
    MultiModalKwargsItem,
    Mapping[str, NestedTensors],
    MultiModalCacheItemMetadata,
]

_V = TypeVar("_V", bound=MultiModalCacheValue)


class MultiModalCache:

    @classmethod
    def get_leaf_size(
        cls,
        leaf: object,
        *,
        debug: bool = False,
    ) -> int:
        # MultiModalKwargs is not a subclass of dict
        if isinstance(leaf, MultiModalKwargs):
            return cls.get_item_size(leaf.data, debug=debug)

        # MultiModalKwargsItem is not a subclass of dict
        if isinstance(leaf, MultiModalKwargsItem):
            leaf_data = {k: v.data for k, v in leaf.items()}
            return cls.get_item_size(leaf_data, debug=debug)

        # sys.getsizeof doesn't work for tensors
        if isinstance(leaf, torch.Tensor):
            return leaf.nbytes

        if isinstance(leaf, MultiModalCacheItemMetadata):
            return leaf.size

        return sys.getsizeof(leaf)

    @classmethod
    def get_item_size(
        cls,
        value: MultiModalCacheValue,
        *,
        debug: bool = False,
    ) -> int:
        size = json_reduce_leaves(
            lambda a, b: a + b,
            json_map_leaves(lambda x: cls.get_leaf_size(x, debug=debug),
                            value),
        )

        if debug:
            logger.debug("Calculated size of %s to be %.2f GiB", type(value),
                         size / GiB_bytes)

        return size

    @classmethod
    def get_lru_cache(
        cls,
        capacity_gb: float,
        value_type: type[_V],
        *,
        debug: bool = False,
    ) -> LRUCache[str, _V]:
        return LRUCache(
            GiB_bytes * capacity_gb,
            getsizeof=lambda x: cls.get_item_size(x, debug=debug),
        )
