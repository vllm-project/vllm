# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright 2025 Bytedance Ltd. and/or its affiliates.
"""BAGEL processor for image and text inputs."""

from transformers import AutoProcessor
from transformers.feature_extraction_utils import BatchFeature
from transformers.image_utils import ImageInput
from transformers.processing_utils import ProcessingKwargs, ProcessorMixin, Unpack
from transformers.tokenization_utils_base import PreTokenizedInput, TextInput


class BagelProcessorKwargs(ProcessingKwargs, total=False):  # type: ignore[call-arg]
    _defaults = {
        "images_kwargs": {
            "return_tensors": "pt",
        },
    }


class BagelProcessor(ProcessorMixin):
    """
    Constructs a BAGEL processor which wraps a
    SigLIP image processor and a Qwen2 tokenizer.
    """

    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "SiglipImageProcessor"
    tokenizer_class = "AutoTokenizer"

    def __call__(
        self,
        text: TextInput
        | PreTokenizedInput
        | list[TextInput]
        | list[PreTokenizedInput] = None,
        images: ImageInput = None,
        **kwargs: Unpack[BagelProcessorKwargs],
    ):
        """
        Main method to prepare for the model one or several sequences(s) and image(s).
        """
        output_kwargs = self._merge_kwargs(
            BagelProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        if images is not None:
            # Process images with the image processor
            pixel_values = self.image_processor(
                images, **output_kwargs["images_kwargs"]
            )
        else:
            pixel_values = {}

        text_inputs = (
            self.tokenizer(text, **output_kwargs["text_kwargs"])
            if text is not None
            else {}
        )

        return BatchFeature(**pixel_values, **text_inputs)

    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to Qwen2TokenizerFast's batch_decode.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to Qwen2TokenizerFast's decode.
        """
        return self.tokenizer.decode(*args, **kwargs)

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))


AutoProcessor.register("BagelProcessor", BagelProcessor)
