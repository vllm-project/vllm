from typing import List, Optional, Set, Iterable, Tuple, Dict
from abc import ABC, abstractmethod, abstractproperty

from vllm.utils import Device


from typing import Type, TypeVar, T

class RefCounter:
    BlockIndex = int
    RefCount = int

    def __init__(self, all_block_indices: Iterable[BlockIndex]):
        deduped = set(all_block_indices)
        self._refcounts: Dict[BlockIndex, RefCount] = {index: 0 for index in deduped}

    def incr(self, block_index: BlockIndex) -> RefCount:
        assert block_index in self._refcounts
        pre_incr_refcount = self._refcounts[block_index]

        assert pre_incr_refcount >= 0

        post_incr_refcount = pre_incr_refcount + 1
        self._refcounts[block_index] = post_incr_refcount
        return post_incr_refcount

    def decr(self, block_index: BlockIndex) -> RefCount:
        assert block_index in self._refcounts
        refcount = self._refcounts[block_index]

        assert refcount > 0
        refcount -= 1

        self._refcounts[block_index] = refcount

        return refcount
