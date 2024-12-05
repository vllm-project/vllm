from typing import Any, Dict, List, Optional

from vllm.config import ModelConfig
from vllm.multimodal import (MULTIMODAL_REGISTRY, MultiModalDataDict,
                             MultiModalKwargs, MultiModalRegistry)


class MMInputMapper:

    def __init__(
        self,
        model_config: ModelConfig,
        mm_registry: MultiModalRegistry = MULTIMODAL_REGISTRY,
    ):
        self.mm_registry = mm_registry
        self.multi_modal_input_mapper = mm_registry.create_input_mapper(
            model_config)
        self.mm_registry.init_mm_limits_per_prompt(model_config)

    def process_inputs(
        self,
        mm_data: MultiModalDataDict,
        mm_processor_kwargs: Optional[Dict[str, Any]],
    ) -> List[MultiModalKwargs]:
        image_inputs = mm_data["image"]
        if not isinstance(image_inputs, list):
            image_inputs = [image_inputs]

        # Process each image input separately so that later we can schedule
        # them in a fine-grained manner.
        mm_inputs: List[MultiModalKwargs] = []
        num_images = len(image_inputs)
        for i in range(num_images):
            mm_input = self.multi_modal_input_mapper(
                {"image": [image_inputs[i]]},
                mm_processor_kwargs=mm_processor_kwargs,
            )
            mm_inputs.append(mm_input)
        return mm_inputs
