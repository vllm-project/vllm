# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# QianfanOCR is built on InternVL with a Qwen3 language backbone.
# The model architecture and weights are fully compatible with InternVLChatModel,
# only the config model_type / architectures strings differ.

from vllm.multimodal import MULTIMODAL_REGISTRY

from .internvl import (
    InternVLChatModel,
    InternVLDummyInputsBuilder,
    InternVLMultiModalProcessor,
    InternVLProcessingInfo,
)


@MULTIMODAL_REGISTRY.register_processor(
    InternVLMultiModalProcessor,
    info=InternVLProcessingInfo,
    dummy_inputs=InternVLDummyInputsBuilder,
)
class QianfanOCRForConditionalGeneration(InternVLChatModel):
    """QianfanOCR multimodal model.

    Identical in structure to InternVLChatModel (InternViT vision encoder +
    pixel-shuffle MLP connector + Qwen3 language model).  This class exists
    solely to register the ``QianfanOCRForConditionalGeneration`` architecture
    name that appears in the model's config.json.
    """
