# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cheers (UMM) processor for image and text inputs."""

from transformers import AutoProcessor
from transformers.feature_extraction_utils import BatchFeature
from transformers.image_utils import ImageInput
from transformers.processing_utils import ProcessingKwargs, ProcessorMixin, Unpack
from transformers.tokenization_utils_base import PreTokenizedInput, TextInput


class CheersProcessorKwargs(ProcessingKwargs, total=False):  # type: ignore[call-arg]
    _defaults = {
        "images_kwargs": {
            "return_tensors": "pt",
        },
    }


class CheersProcessor(ProcessorMixin):
    """
    Constructs a Cheers processor which wraps a
    SigLIP image processor and a Qwen2 tokenizer.
    """

    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "AutoImageProcessor"
    tokenizer_class = "AutoTokenizer"

    def __call__(
        self,
        text: TextInput
        | PreTokenizedInput
        | list[TextInput]
        | list[PreTokenizedInput] = None,
        images: ImageInput = None,
        **kwargs: Unpack[CheersProcessorKwargs],
    ):
        output_kwargs = self._merge_kwargs(
            CheersProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        if images is not None:
            import torch
            if isinstance(images, (list, tuple)):
                all_pv = []
                all_ghw = []
                for img in images:
                    result = self.image_processor(
                        img, **output_kwargs["images_kwargs"]
                    )
                    all_pv.append(result["pixel_values"])
                    if "grid_hws" in result:
                        all_ghw.append(result["grid_hws"])
                pixel_values = {
                    "pixel_values": torch.cat(all_pv, dim=0),
                }
                if all_ghw:
                    pixel_values["grid_hws"] = torch.cat(all_ghw, dim=0)
            else:
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

        return BatchFeature(data={**pixel_values, **text_inputs})

    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))


AutoProcessor.register("CheersProcessor", CheersProcessor)
