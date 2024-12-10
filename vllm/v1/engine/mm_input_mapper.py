from typing import Any, Dict, List, Optional

import PIL
from blake3 import blake3

from vllm.config import ModelConfig
from vllm.multimodal import (MULTIMODAL_REGISTRY, MultiModalDataDict,
                             MultiModalKwargs, MultiModalRegistry)
from vllm.v1.utils import LRUDictCache

# Both Client and Server must use the same cache size
MM_CACHE_SIZE = 128


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

        self.mm_cache = LRUDictCache(MM_CACHE_SIZE)

        # Set to None to disable (TODO: Disable!)
        self.mm_debug_cache_hit_ratio_steps = 32
        self.mm_cache_hits = 0
        self.mm_cache_misses = 0

    def cache_hit_ratio(self, steps) -> float:
        total_steps = self.mm_cache_hits + self.mm_cache_misses

        if total_steps > 0 and total_steps % steps == 0:
            print("[debug] MMInputMapper: cache_hit_ratio = {}".format(
                self.mm_cache_hits / total_steps))

    def process_inputs(
        self,
        mm_data: MultiModalDataDict,
        mm_hashes: Optional[List[str]],
        mm_processor_kwargs: Optional[Dict[str, Any]],
    ) -> List[MultiModalKwargs]:
        image_inputs = mm_data["image"]
        if not isinstance(image_inputs, list):
            image_inputs = [image_inputs]

        use_hash = mm_hashes is not None
        if use_hash:
            assert len(image_inputs) == len(mm_hashes)  # Sanity

        # Process each image input separately so that later we can schedule
        # them in a fine-grained manner.
        # Utilize caching (if enabled)
        ret_hashes = [] if use_hash else None
        ret_inputs: List[MultiModalKwargs] = []
        for i in range(len(image_inputs)):
            if self.mm_debug_cache_hit_ratio_steps is not None:
                self.cache_hit_ratio(self.mm_debug_cache_hit_ratio_steps)

            if use_hash:
                mm_hash = mm_hashes[i]
                mm_input = self.mm_cache.get(mm_hash)
            else:
                mm_hash = None
                mm_input = None

            if mm_input is None:
                self.mm_cache_misses += 1
                mm_input = self.multi_modal_input_mapper(
                    {"image": [image_inputs[i]]},
                    mm_processor_kwargs=mm_processor_kwargs,
                )

                if use_hash:
                    self.mm_cache.put(mm_hash, mm_input)
            else:
                self.mm_cache_hits += 1
                mm_input = None  # Avoids sending mm_input to Server

            if use_hash:
                ret_hashes.append(mm_hash)
            ret_inputs.append(mm_input)

        return ret_inputs, ret_hashes


class MMInputMapperServer:

    def __init__(self, ):
        self.mm_cache = LRUDictCache(MM_CACHE_SIZE)

    def process_inputs(
        self,
        mm_inputs: List[Optional[MultiModalKwargs]],
        mm_hashes: List[Optional[str]],
    ) -> List[MultiModalKwargs]:
        assert len(mm_inputs) == len(mm_hashes)

        full_mm_inputs = []
        for mm_input, mm_hash in zip(mm_inputs, mm_hashes):
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

    def hash(self, mm_data: MultiModalDataDict) -> List[str]:
        image_inputs = mm_data["image"]
        if not isinstance(image_inputs, list):
            image_inputs = [image_inputs]

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
