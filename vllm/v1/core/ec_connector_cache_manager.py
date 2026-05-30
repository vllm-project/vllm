# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections import OrderedDict
from typing import TYPE_CHECKING

from vllm.logger import init_logger
from vllm.utils.mem_constants import GiB_bytes, MiB_bytes
from vllm.v1.request import Request

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)


class ECConnectorCacheManager:
    """Bookkeeping for EC connector embedding slots (scheduler-side).

    Mirrors :class:`EncoderCacheManager` state shape so the scheduler can
    apply the same ``can_allocate`` / ``allocate`` / ``free_encoder_input``
    patterns. Producer paths release connector capacity when the consumer
    acknowledges transfer via ``mark_consumer_received``.

    Eviction from ``freeable`` is **lazy**: on each ``can_allocate``, entries are
    popped only until ``num_free_slots`` is large enough for **this** request's
    ``num_embeds``. Many 0-ref rows may remain in ``freeable`` until later
    allocations force more evictions; there is no "flush all on new request."
    """

    def __init__(self, cache_size: int, vllm_config: "VllmConfig"):
        self.cache_size = cache_size
        self.num_free_slots = cache_size
        self.num_freeable_slots = cache_size
        self.cached: dict[str, set[str]] = {}
        self.freeable: OrderedDict[str, int] = OrderedDict()
        self.freed: list[str] = []
        self._mm_hash_num_embeds: dict[str, int] = {}

        hidden_size = vllm_config.model_config.get_hidden_size()
        dtype_bytes = vllm_config.model_config.dtype.itemsize
        self._slot_bytes = hidden_size * dtype_bytes
        total_bytes = self._slot_bytes * cache_size
        logger.info(
            "ECConnectorCacheManager init: ec_connector_capacity_embeds=%d slots, "
            "per-slot=%d B (%.3f MB), total=%.2f MB (%.3f GB), "
            "free_slots=%d freeable_slots=%d",
            cache_size,
            self._slot_bytes,
            self._slot_bytes / MiB_bytes,
            total_bytes / MiB_bytes,
            total_bytes / GiB_bytes,
            self.num_free_slots,
            self.num_freeable_slots,
        )

    def check_and_update_cache(self, request: Request, input_id: int) -> bool:
        mm_hash = request.mm_features[input_id].identifier
        if mm_hash not in self.cached:
            return False
        if not self.cached[mm_hash]:
            num_encoder_embeds = self.freeable.pop(mm_hash)
            self.num_freeable_slots -= num_encoder_embeds
        self.cached[mm_hash].add(request.request_id)
        return True

    def can_allocate(
        self,
        request: Request,
        input_id: int,
        encoder_compute_budget: int,
        num_embeds_to_schedule: int,
    ) -> bool:
        num_embeds = request.get_num_encoder_embeds(input_id)
        if num_embeds > encoder_compute_budget:
            return False
        num_embeds += num_embeds_to_schedule
        if num_embeds <= self.num_free_slots:
            return True
        if num_embeds > self.num_freeable_slots:
            return False
        while num_embeds > self.num_free_slots:
            mm_hash, num_free_embeds = self.freeable.popitem(last=False)
            self.cached.pop(mm_hash, None)
            self.freed.append(mm_hash)
            self._mm_hash_num_embeds.pop(mm_hash, None)
            self.num_free_slots += num_free_embeds
        return True

    def allocate(self, request: Request, input_id: int) -> None:
        mm_hash = request.mm_features[input_id].identifier
        request_id = request.request_id
        if mm_hash not in self.cached:
            self.cached[mm_hash] = set()
        num_encoder_embeds = request.get_num_encoder_embeds(input_id)
        assert self.num_free_slots >= num_encoder_embeds
        assert self.num_freeable_slots >= num_encoder_embeds
        self.cached[mm_hash].add(request_id)
        self._mm_hash_num_embeds[mm_hash] = num_encoder_embeds
        self.num_free_slots -= num_encoder_embeds
        self.num_freeable_slots -= num_encoder_embeds

    def get_cached_input_ids(self, request: Request) -> set[int]:
        return {
            input_id
            for input_id in range(len(request.mm_features))
            if request.mm_features[input_id].identifier in self.cached
        }

    def free_encoder_input(self, request: Request, input_id: int) -> None:
        req_id = request.request_id
        mm_hash = request.mm_features[input_id].identifier
        if not self.cached.get(mm_hash, None):
            return
        self.cached[mm_hash].discard(req_id)
        if not self.cached[mm_hash]:
            num_encoder_embeds = request.get_num_encoder_embeds(input_id)
            self.freeable[mm_hash] = num_encoder_embeds
            self.num_freeable_slots += num_encoder_embeds

    def free(self, request: Request) -> None:
        input_ids = self.get_cached_input_ids(request).copy()
        for input_id in input_ids:
            self.free_encoder_input(request, input_id)

    def mark_consumer_received(self, mm_hash: str) -> None:
        """Producer: consumer ACKed the transfer. Consumer: load finished.

        Moves connector bookkeeping for ``mm_hash`` to ``freeable`` when refs
        were held in ``cached``.
        """
        if mm_hash not in self.cached:
            return
        self.cached.pop(mm_hash)
        num_encoder_embeds = self._mm_hash_num_embeds.pop(mm_hash, None)
        if num_encoder_embeds is None:
            return
        self.freeable[mm_hash] = num_encoder_embeds
        self.num_freeable_slots += num_encoder_embeds

    def get_freed_mm_hashes(self) -> list[str]:
        freed = self.freed
        self.freed = []
        return freed
