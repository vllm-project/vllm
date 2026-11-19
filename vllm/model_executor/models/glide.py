# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLiDE (fastino/GLiDE-27B): a Qwen3.5 decision model with recurrent depth.

The checkpoint is a `Qwen3_5ForConditionalGeneration` without an LM head, plus
`readout.safetensors`, one row per answer code, and `decision_config.json`.
It is read like Decision 3.0: the readout scores the codes at the last prompt
token and is served as the `score` head of a classification model.
`GlideForSequenceClassificationConfig` applies the checkpoint's attention mode,
label space and recurrent depth (`midloop`), which runs blocks of middle layers
more than once inside the forward pass.
"""

from transformers.models.qwen3_vl import Qwen3VLProcessor

from vllm.multimodal import MULTIMODAL_REGISTRY

from .d3 import D3ForSequenceClassification
from .interfaces_base import default_pooling_type
from .qwen3_5 import Qwen3_5ForConditionalGeneration, Qwen3_5ProcessingInfo
from .qwen3_vl import Qwen3VLDummyInputsBuilder, Qwen3VLMultiModalProcessor

# The image budget of the checkpoint's runtime (`glide/images.py`), which
# differs from its processor config: 256 x 256 to 1024 x 1024 pixels.
IMAGE_SIZE = {"shortest_edge": 65_536, "longest_edge": 1_048_576}


class GlideProcessingInfo(Qwen3_5ProcessingInfo):
    def get_hf_processor(self, **kwargs: object) -> Qwen3VLProcessor:
        processor = super().get_hf_processor(**kwargs)
        processor.image_processor.size = dict(IMAGE_SIZE)
        return processor


@default_pooling_type(seq_pooling_type="LAST")
@MULTIMODAL_REGISTRY.register_processor(
    Qwen3VLMultiModalProcessor,
    info=GlideProcessingInfo,
    dummy_inputs=Qwen3VLDummyInputsBuilder,
)
class GlideForSequenceClassification(D3ForSequenceClassification):
    # The checkpoint keeps the names of a Qwen3.5 generation checkpoint.
    hf_to_vllm_mapper = Qwen3_5ForConditionalGeneration.hf_to_vllm_mapper
