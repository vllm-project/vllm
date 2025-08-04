# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Sequence
from typing import TYPE_CHECKING, Optional

from vllm.multimodal import MultiModalKwargs
from vllm.multimodal.cache import MultiModalCache, MultiModalCacheItemMetadata
from vllm.utils import is_list_of

if TYPE_CHECKING:
    from vllm.config import ModelConfig

# The idea of multimodal IPC caching is based on having a client and
# a server, where the client executes in the frontend process (=P0) and the
# server in the core process (=P1).
#
# -- Client:
#  - BaseMultiModalProcessor processes MultiModalData into MultiModalKwargs
#    while outputting mm_hash as its identifier.
#  - MultiModalIPCCacheLookup keeps track of the cached entries and
#    determine whether to send the MultiModalKwargs to P1.
#
# -- Server:
#  - MultiModalIPCCache stores the MultiModalKwargs from P0.
#
# The keys of MultiModalIPCCacheLookup (in the client)
# and the keys of MultiModalIPCCacheLookup (in the server)
# are mirrored, and this allows us to avoid the serialization of `mm_inputs`
# (like pixel values) between client (=P0) and server (=P1) processes if the
# `mm_hash` is found in the client cache.

# Both Client and Server must use the same cache size (to remain mirrored).
# This cache size is set by the config variable `mm_ipc_cache_gb`.


class MultiModalIPCCacheLookup:
    """Used by P0 to check whether multi-modal kwargs are cached in P1."""

    def __init__(self, model_config: "ModelConfig") -> None:
        super().__init__()

        mm_config = model_config.multimodal_config

        self.enabled = (mm_config is not None
                        and mm_config.mm_ipc_cache_gb > 0)

        self.mm_cache = MultiModalCache.get_lru_cache(
            mm_config.mm_ipc_cache_gb if mm_config else 0,
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

    def reset(self) -> bool:
        self.mm_cache.clear()

        return True


class MultiModalIPCCache:
    """Used by P1 to avoid requiring past multi-modal kwargs from P0."""

    def __init__(self, model_config: "ModelConfig") -> None:
        super().__init__()

        mm_config = model_config.multimodal_config

        self.enabled = (mm_config is not None
                        and mm_config.mm_ipc_cache_gb > 0)

        self.mm_cache = MultiModalCache.get_lru_cache(
            mm_config.mm_ipc_cache_gb if mm_config else 0,
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

    def reset(self) -> bool:
        self.mm_cache.clear()

        return True
