# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Sequence
from typing import TYPE_CHECKING, Optional

from vllm.multimodal import MultiModalKwargs, MultiModalRegistry
from vllm.multimodal.cache import MultiModalCache, MultiModalCacheItemMetadata
from vllm.utils import is_list_of

if TYPE_CHECKING:
    from vllm.config import ModelConfig

# The idea of multimodal input caching is based on having a client and
# a server, where the client executes in the frontend process (=P0) and the
# server in the core process (=P1).
#
# -- P0:
#  - BaseMultiModalProcessor calls MultiModalHasher to get the `mm_hash` of
#    each input multi-modal item (e.g. image),
#  - BaseMultiModalProcessor processes the input items into `mm_inputs`,
#    which are MultiModalKwargsItem instances that each correspond to an
#    input multi-modal item.
#  - MultiModalInputCacheClient accepts the `mm_inputs` and corresponding
#    `mm_hash` for each item. It stores the `mm_hash` as keys and the size
#    of `mm_inputs`, but not the `mm_inputs` themselves, to avoid taking
#    up additional memory in P0.
#  - The `mm_hash` is always sent to P1.
#  - The corresponding `mm_inputs` are only sent to P1 if they are not cached
#    in MultiModalInputCacheServer.
#
# -- P1:
#  - If the `mm_hash` is cached (i.e. `mm_inputs` are not sent from P0),
#    MultiModalInputCacheServer retrieves the corresponding `mm_inputs`.
#  - If the `mm_hash` is not cached (i.e. `mm_inputs` are sent from P0),
#    MultiModalInputCacheServer stores `mm_inputs` under the key `mm_hash`.
#  - Either way, the `mm_hash` and corresponding `mm_inputs` are sent to
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
        mm_inputs: Sequence[MultiModalKwargs],
        mm_hashes: list[str],
    ) -> Sequence[Optional[MultiModalKwargs]]:
        assert len(mm_inputs) == len(mm_hashes)

        if not self.enabled:
            assert is_list_of(mm_inputs, MultiModalKwargs)
            return mm_inputs

        full_mm_inputs = list[Optional[MultiModalKwargs]]()
        for mm_input, mm_hash in zip(mm_inputs, mm_hashes):
            if self.mm_cache.get(mm_hash) is not None:
                mm_input = None
            else:
                self.mm_cache[mm_hash] = \
                    MultiModalCacheItemMetadata.wraps(mm_input)

            full_mm_inputs.append(mm_input)

        return full_mm_inputs

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
            MultiModalKwargs,
        )

    def get_and_update(
        self,
        mm_inputs: Sequence[Optional[MultiModalKwargs]],
        mm_hashes: list[str],
    ) -> Sequence[MultiModalKwargs]:
        assert len(mm_inputs) == len(mm_hashes)

        if not self.enabled:
            assert is_list_of(mm_inputs, MultiModalKwargs)
            return mm_inputs

        full_mm_inputs = list[MultiModalKwargs]()
        for mm_input, mm_hash in zip(mm_inputs, mm_hashes):
            if mm_input is None:
                mm_input = self.mm_cache[mm_hash]
            else:
                self.mm_cache[mm_hash] = mm_input

            full_mm_inputs.append(mm_input)

        return full_mm_inputs

    def reset(self) -> None:
        self.mm_cache.clear()
