from typing import Any, Dict, List, Optional, Tuple

import PIL
from blake3 import blake3

from vllm.config import ModelConfig
from vllm.inputs import PromptType
from vllm.logger import init_logger
from vllm.multimodal import (MULTIMODAL_REGISTRY, MultiModalDataDict,
                             MultiModalKwargs, MultiModalRegistry)
from vllm.v1.utils import LRUDictCache

logger = init_logger(__name__)

# The idea of MM preprocessor caching is based on having a client and a server,
# where the client executes in the frontend process (=P0) and the server in the
# core process (=P1).
#
# -- Client: Executes the MM mapper and performs caching of the results.
# -- Server: Performs caching of the results
#
# The caching for both client and server is mirrored/similar, and this allows us
# to avoid the serialization of "mm_inputs" (like pixel values) between
# client (=P0) and server (=P1) processes.

# Both Client and Server must use the same cache size
# (to perform mirrored caching)
# TODO: Tune the MM cache size
MM_CACHE_SIZE = 256


class MMInputMapperClient:

    def __init__(
        self,
        model_config: ModelConfig,
        mm_registry: MultiModalRegistry = MULTIMODAL_REGISTRY,
    ):
        self.model_config = model_config
        self.mm_registry = mm_registry
        self.multi_modal_input_mapper = mm_registry.create_input_mapper(
            model_config)
        self.mm_registry.init_mm_limits_per_prompt(model_config)

        self.mm_cache = LRUDictCache[str, MultiModalKwargs](MM_CACHE_SIZE)

        # DEBUG: Set to None to disable
        self.mm_debug_cache_hit_ratio_steps = None
        self.mm_cache_hits = 0
        self.mm_cache_total = 0

    def cache_hit_ratio(self, steps):
        if self.mm_cache_total > 0 and self.mm_cache_total % steps == 0:
            logger.debug("MMInputMapper: cache_hit_ratio = %.2f ",
                         self.mm_cache_hits / self.mm_cache_total)

    def process_inputs(
        self,
        mm_data: MultiModalDataDict,
        mm_hashes: Optional[List[str]],
        mm_processor_kwargs: Optional[Dict[str, Any]],
        precomputed_mm_inputs: Optional[List[MultiModalKwargs]],
    ) -> Tuple[List[MultiModalKwargs], Optional[List[str]]]:
        if precomputed_mm_inputs is None:
            image_inputs = mm_data["image"]
            if not isinstance(image_inputs, list):
                image_inputs = [image_inputs]
            num_inputs = len(image_inputs)
        else:
            num_inputs = len(precomputed_mm_inputs)

        # Check if hash is enabled
        use_hash = mm_hashes is not None
        if use_hash:
            assert mm_hashes is not None
            assert num_inputs == len(
                mm_hashes), "num_inputs = {} len(mm_hashes) = {}".format(
                    num_inputs, len(mm_hashes))

        # Process each image input separately, so that later we can schedule
        # them in a fine-grained manner.
        # Apply caching (if enabled) and reuse precomputed inputs (if provided)
        ret_hashes: Optional[List[str]] = [] if use_hash else None
        ret_inputs: List[MultiModalKwargs] = []
        for input_id in range(num_inputs):
            if self.mm_debug_cache_hit_ratio_steps is not None:
                self.cache_hit_ratio(self.mm_debug_cache_hit_ratio_steps)

            mm_hash = None
            mm_input = None
            if use_hash:
                assert mm_hashes is not None
                mm_hash = mm_hashes[input_id]
                mm_input = self.mm_cache.get(mm_hash)

            self.mm_cache_total += 1
            if mm_input is None:
                if precomputed_mm_inputs is not None:
                    # Reuse precomputed input (for merged preprocessor)
                    mm_input = precomputed_mm_inputs[input_id]
                else:
                    # Apply MM mapper
                    mm_input = self.multi_modal_input_mapper(
                        {"image": [image_inputs[input_id]]},
                        mm_processor_kwargs=mm_processor_kwargs,
                    )

                if use_hash:
                    # Add to cache
                    assert mm_hash is not None
                    self.mm_cache.put(mm_hash, mm_input)
            else:
                self.mm_cache_hits += 1
                mm_input = None  # Avoids sending mm_input to Server

            if use_hash:
                assert mm_hash is not None
                assert ret_hashes is not None
                ret_hashes.append(mm_hash)
            ret_inputs.append(mm_input)

        return ret_inputs, ret_hashes


class MMInputMapperServer:

    def __init__(self, ):
        self.mm_cache = LRUDictCache[str, MultiModalKwargs](MM_CACHE_SIZE)

    def process_inputs(
        self,
        mm_inputs: List[Optional[MultiModalKwargs]],
        mm_hashes: List[str],
    ) -> List[MultiModalKwargs]:
        assert len(mm_inputs) == len(mm_hashes)

        full_mm_inputs = []
        for mm_input, mm_hash in zip(mm_inputs, mm_hashes):
            assert mm_hash is not None
            if mm_input is None:
                mm_input = self.mm_cache.get(mm_hash)
                assert mm_input is not None
            else:
                self.mm_cache.put(mm_hash, mm_input)

            full_mm_inputs.append(mm_input)

        return full_mm_inputs


class MMHasher:

    def __init__(self):
        pass

    def hash(self, prompt: PromptType) -> Optional[List[str]]:
        if "multi_modal_data" not in prompt:
            return None

        mm_data = prompt["multi_modal_data"]
        image_inputs = mm_data["image"]
        if not isinstance(image_inputs, list):
            image_inputs = [image_inputs]
        assert len(image_inputs) > 0

        ret = []
        for image in image_inputs:
            assert isinstance(image, PIL.Image.Image)

            # Convert image to bytes
            bytes = image.tobytes()

            # Hash image bytes
            hasher = blake3()
            hasher.update(bytes)
            ret.append(hasher.hexdigest())

        return ret
