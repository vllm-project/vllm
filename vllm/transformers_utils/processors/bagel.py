# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright 2025 Bytedance Ltd. and/or its affiliates.
"""BAGEL processor for image and text inputs."""

from transformers import AutoProcessor
from transformers.feature_extraction_utils import BatchFeature
from transformers.image_utils import ImageInput
from transformers.processing_utils import ProcessorMixin
from transformers.tokenization_utils_base import PreTokenizedInput, TextInput


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
        **kwargs,
    ):
        """
        Main method to prepare for the model one or several sequences(s) and image(s).
        """
        if images is not None:
            # Process images with the image processor
            # Ensure return_tensors is set to "pt" for PyTorch tensors
            image_kwargs = {**kwargs}
            if "return_tensors" not in image_kwargs:
                image_kwargs["return_tensors"] = "pt"
            pixel_values = self.image_processor(images, **image_kwargs)
        else:
            pixel_values = None

        text_inputs = self.tokenizer(text, **kwargs) if text is not None else None

        if pixel_values is not None and text_inputs is not None:
            # Combine text and image inputs into BatchFeature
            combined = dict(text_inputs)
            combined["pixel_values"] = pixel_values["pixel_values"]
            return BatchFeature(combined)
        elif pixel_values is not None:
            return pixel_values
        elif text_inputs is not None:
            return BatchFeature(dict(text_inputs))
        else:
            return BatchFeature({})

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
