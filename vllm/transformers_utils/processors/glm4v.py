# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Adapted from
# https://github.com/zai-org/CogAgent
from transformers import PreTrainedTokenizer
from transformers.image_processing_utils_fast import BaseImageProcessorFast
from transformers.image_utils import PILImageResampling
from transformers.processing_utils import ProcessorMixin


class GLM4VImageProcessorFast(BaseImageProcessorFast):
    """
    Port of https://huggingface.co/zai-org/glm-4v-9b/blob/main/tokenization_chatglm.py#L177
    to HF Transformers.
    """

    resample = PILImageResampling.BICUBIC
    image_mean = [0.48145466, 0.4578275, 0.40821073]
    image_std = [0.26862954, 0.26130258, 0.27577711]
    size = {"height": 1120, "width": 1120}
    do_resize = True
    do_rescale = True
    do_normalize = True


class GLM4VProcessor(ProcessorMixin):
    attributes = ["image_processor", "tokenizer"]

    def __init__(
        self,
        image_processor: GLM4VImageProcessorFast,
        tokenizer: PreTrainedTokenizer,
    ) -> None:
        self.image_processor = image_processor
        self.tokenizer = tokenizer
