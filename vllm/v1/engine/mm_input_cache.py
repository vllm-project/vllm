# SPDX-License-Identifier: Apache-2.0

from vllm.envs import VLLM_MM_INPUT_CACHE_GIB
from vllm.multimodal import MultiModalKwargs
from vllm.multimodal.processing import ProcessingCache

# The idea of multimodal preprocessing caching is based on having a client and
# a server, where the client executes in the frontend process (=P0) and the
# server in the core process (=P1).
#
# -- Client:
#  - BaseMultiModalProcessor to process MultiModalData into MultiModalKwargs
#    with built-in caching functionality, with mm_hash as its identifier.
#
# -- Server:
#  - MMInputCacheServer to perform caching of the received MultiModalKwargs.
#
# The caching for both client and server is mirrored, and this allows us
# to avoid the serialization of "mm_inputs" (like pixel values) between
# client (=P0) and server (=P1) processes if the mm_hash is found in the client
# cache.

# Both Client and Server must use the same cache size
# (to perform mirrored caching). This cache size is set by the environment
# variable VLLM_MM_INPUT_CACHE_GIB.


class MMInputCacheServer:

    def __init__(self, model_config):
        self.use_cache = not model_config.disable_mm_preprocessor_cache
        self.mm_cache = ProcessingCache.get_lru_cache(VLLM_MM_INPUT_CACHE_GIB,
                                                      MultiModalKwargs)

    def get_and_update(
        self,
        mm_inputs: list[MultiModalKwargs],
        mm_hashes: list[str],
    ) -> list[MultiModalKwargs]:
        assert len(mm_inputs) == len(mm_hashes)

        if not self.use_cache:
            return mm_inputs

        full_mm_inputs = []
        for mm_input, mm_hash in zip(mm_inputs, mm_hashes):
            assert mm_hash is not None
            if mm_input is None:
                mm_input = self.mm_cache[mm_hash]
            else:
                self.mm_cache[mm_hash] = mm_input

            full_mm_inputs.append(mm_input)

        return full_mm_inputs
