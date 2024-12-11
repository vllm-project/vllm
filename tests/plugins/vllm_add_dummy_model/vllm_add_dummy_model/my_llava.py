from typing import Optional

import torch

from vllm.inputs import INPUT_REGISTRY
from vllm.model_executor.models.llava import (LlavaForConditionalGeneration,
                                              dummy_data_for_llava,
                                              get_max_llava_image_tokens,
                                              input_processor_for_llava)
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.multimodal import MULTIMODAL_REGISTRY


@MULTIMODAL_REGISTRY.register_image_input_mapper()
@MULTIMODAL_REGISTRY.register_max_image_tokens(get_max_llava_image_tokens)
@INPUT_REGISTRY.register_dummy_data(dummy_data_for_llava)
@INPUT_REGISTRY.register_input_processor(input_processor_for_llava)
class MyLlava(LlavaForConditionalGeneration):

    def compute_logits(
            self, hidden_states: torch.Tensor,
            sampling_metadata: SamplingMetadata) -> Optional[torch.Tensor]:
        # this dummy model always predicts the first token
        logits = super().compute_logits(hidden_states, sampling_metadata)
        if logits is not None:
            logits.zero_()
            logits[:, 0] += 1.0
        return logits
