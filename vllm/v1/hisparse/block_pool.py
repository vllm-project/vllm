# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.distributed.kv_events import KVCacheEvent
from vllm.v1.core.block_pool import BlockPool


class SharedEventQueueBlockPool(BlockPool):
    """A pool that publishes into another pool's live KV event queue.

    The owner rebinds its queue on every drain, so this reads it through the
    owner rather than holding a reference.
    """

    def __init__(self, *args, event_owner: BlockPool, **kwargs) -> None:
        self._event_owner = event_owner
        super().__init__(*args, **kwargs)

    @property
    def kv_event_queue(self) -> list[KVCacheEvent]:
        return self._event_owner.kv_event_queue

    @kv_event_queue.setter
    def kv_event_queue(self, events: list[KVCacheEvent]) -> None:
        # ``BlockPool.__init__`` seeds an empty queue and ``take_events``
        # swaps in a fresh one; both belong to the owner, which drains it.
        assert not events
