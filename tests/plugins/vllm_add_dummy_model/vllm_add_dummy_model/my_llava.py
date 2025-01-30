from typing import Optional

import torch

from vllm.model_executor.models.llava import (LlavaDummyInputsBuilder,
                                              LlavaForConditionalGeneration,
                                              LlavaMultiModalProcessor,
                                              LlavaProcessingInfo)
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.multimodal import MULTIMODAL_REGISTRY


@MULTIMODAL_REGISTRY.register_processor(LlavaMultiModalProcessor,
                                        info=LlavaProcessingInfo,
                                        dummy_inputs=LlavaDummyInputsBuilder)
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
