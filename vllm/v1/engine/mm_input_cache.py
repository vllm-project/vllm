# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Mapping
from typing import TYPE_CHECKING

from vllm.multimodal import MultiModalRegistry
from vllm.multimodal.cache import MultiModalCache, MultiModalCacheItemMetadata
from vllm.multimodal.inputs import MultiModalKwargsItem, NestedTensors

if TYPE_CHECKING:
    from vllm.config import ModelConfig

# The idea of multimodal input caching is based on having a client and
# a server, where the client executes in the frontend process (=P0) and the
# server in the core process (=P1).
#
# -- P0:
#  - BaseMultiModalProcessor calls MultiModalHasher to get the `mm_hash` of
#    each input multi-modal item (e.g. image),
#  - BaseMultiModalProcessor processes the input items into `mm_kwargs`,
#    which are MultiModalKwargsItem instances that each correspond to an
#    input multi-modal item.
#  - MultiModalInputCacheClient accepts the `mm_kwargs` and corresponding
#    `mm_hash` for each item. It stores the `mm_hash` as keys and the size
#    of `mm_kwargs`, but not the `mm_kwargs` themselves, to avoid taking
#    up additional memory in P0.
#  - The `mm_hash` is always sent to P1.
#  - The corresponding `mm_kwargs` are only sent to P1 if they are not cached
#    in MultiModalInputCacheServer.
#
# -- P1:
#  - If the `mm_hash` is cached (i.e. `mm_kwargs` are not sent from P0),
#    MultiModalInputCacheServer retrieves the corresponding `mm_kwargs`.
#  - If the `mm_hash` is not cached (i.e. `mm_kwargs` are sent from P0),
#    MultiModalInputCacheServer stores `mm_kwargs` under the key `mm_hash`.
#  - Either way, the `mm_hash` and corresponding `mm_kwargs` are sent to
#    the engine for model execution.
#
# Both Client and Server must perform cache update and eviction based on the
# same item size. This ensures that the keys of MultiModalInputCacheClient
# and MultiModalInputCacheServer are mirrored, allowing us to determine in P0
# whether a key is cached in MultiModalInputCacheServer by querying
# MultiModalInputCacheClient without having to communicate with P1.


class MultiModalInputCacheClient:
    """Used by P0 to check whether multi-modal kwargs are cached in P1."""

    def __init__(self, model_config: "ModelConfig",
                 mm_registry: MultiModalRegistry) -> None:
        super().__init__()

        self.enabled = mm_registry.enable_mm_input_cache(model_config)
        self.mm_cache = MultiModalCache.get_lru_cache(
            model_config.get_mm_input_cache_gb(),
            MultiModalCacheItemMetadata,
        )

    def get_and_update(
        self,
        mm_kwargs: list[MultiModalKwargsItem],
        mm_hashes: list[str],
    ) -> list[MultiModalKwargsItem]:
        if not self.enabled:
            return mm_kwargs

        assert len(mm_kwargs) == len(mm_hashes)

        out_mm_items = list[MultiModalKwargsItem]()
        for mm_item, mm_hash in zip(mm_kwargs, mm_hashes):
            if self.mm_cache.get(mm_hash) is not None:
                out_mm_items.append(mm_item.without_data())
            else:
                self.mm_cache[mm_hash] = \
                    MultiModalCacheItemMetadata.wraps(mm_item.require_data())
                out_mm_items.append(mm_item)

        return out_mm_items

    def reset(self) -> None:
        self.mm_cache.clear()


class MultiModalInputCacheServer:
    """Used by P1 to avoid requiring past multi-modal kwargs from P0."""

    def __init__(self, model_config: "ModelConfig",
                 mm_registry: MultiModalRegistry) -> None:
        super().__init__()

        self.enabled = mm_registry.enable_mm_input_cache(model_config)
        self.mm_cache = MultiModalCache.get_lru_cache(
            model_config.get_mm_input_cache_gb(),
            Mapping[str, NestedTensors],
        )

    def get_and_update(
        self,
        mm_kwargs: list[MultiModalKwargsItem],
        mm_hashes: list[str],
    ) -> list[MultiModalKwargsItem]:
        if not self.enabled:
            return mm_kwargs

        assert len(mm_kwargs) == len(mm_hashes)

        out_mm_items = list[MultiModalKwargsItem]()
        for mm_item, mm_hash in zip(mm_kwargs, mm_hashes):
            if (mm_data := mm_item.get_data()) is None:
                out_mm_items.append(mm_item.with_data(self.mm_cache[mm_hash]))
            else:
                self.mm_cache[mm_hash] = mm_data
                out_mm_items.append(mm_item)

        return out_mm_items

    def reset(self) -> None:
        self.mm_cache.clear()
