import time
import PIL

from blake3 import blake3
from typing import Any, Dict, List, Optional

from vllm.config import ModelConfig
from vllm.multimodal import (MULTIMODAL_REGISTRY, MultiModalDataDict,
                             MultiModalKwargs, MultiModalRegistry)
from vllm.v1.utils import LRUDictCache


class MMInputMapper:

    def __init__(
        self,
        model_config: ModelConfig,
        mm_registry: MultiModalRegistry = MULTIMODAL_REGISTRY,
        mm_cache_size: int = 128,
    ):
        self.mm_registry = mm_registry
        self.multi_modal_input_mapper = mm_registry.create_input_mapper(
            model_config)
        self.mm_registry.init_mm_limits_per_prompt(model_config)

        self.mm_cache = LRUDictCache(size=mm_cache_size)
        self.mm_cache_hits = 0
        self.mm_cache_misses = 0

        # Set to None to disable (TODO: Disable!)
        self.mm_debug_cache_hit_ratio_steps = 32

    def cache_hit_ratio(self, steps) -> float:
        total_steps = self.mm_cache_hits + self.mm_cache_misses

        if total_steps > 0 and total_steps % steps == 0:
            print("[debug] MMInputMapper: cache_hit_ratio = {}".format(
                self.mm_cache_hits / total_steps))

    def process_inputs(
        self,
        mm_data: MultiModalDataDict,
        mm_hash: Optional[List[str]],
        mm_processor_kwargs: Optional[Dict[str, Any]],
    ) -> List[MultiModalKwargs]:
        image_inputs = mm_data["image"]
        if not isinstance(image_inputs, list):
            image_inputs = [image_inputs]

        use_hash = mm_hash is not None
        if use_hash:
            assert len(image_inputs) == len(mm_hash)  # Sanity

        # Process each image input separately so that later we can schedule
        # them in a fine-grained manner.
        # Utilize caching (if enabled)
        mm_inputs: List[MultiModalKwargs] = []
        for i in range(len(image_inputs)):
            if self.mm_debug_cache_hit_ratio_steps is not None:
                self.cache_hit_ratio(self.mm_debug_cache_hit_ratio_steps)

            mm_input = self.mm_cache.get(mm_hash[i]) if use_hash else None
            if mm_input is None:
                self.mm_cache_misses += 1
                mm_input = self.multi_modal_input_mapper(
                    {"image": [image_inputs[i]]},
                    mm_processor_kwargs=mm_processor_kwargs,
                )
                if use_hash:
                    self.mm_cache.put(mm_hash[i], mm_input)
            else:
                self.mm_cache_hits += 1

            mm_inputs.append(mm_input)

        return mm_inputs


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

            # FIXME(alexm): Remove debug

            # print("  type(data) = {}, data = {}".format(type(image), image))

            # Convert image to bytes
            start_time = time.time()
            bytes = image.tobytes()
            elapsed_time = time.time() - start_time
            # print("    tobytes time = {}".format(elapsed_time))

            # Hash image bytes
            start_time = time.time()
            hasher = blake3()
            hasher.update(bytes)
            ret.append(hasher.hexdigest())
            elapsed_time = time.time() - start_time
            # print("    hash time = {}".format(elapsed_time))
            # print("    hash val = {}".format(ret[-1]))

        return ret
