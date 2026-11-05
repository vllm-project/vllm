# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from transformers import AutoTokenizer
from transformers.models.qwen2_vl import Qwen2VLImageProcessor
from transformers.models.qwen3_vl import Qwen3VLProcessor, Qwen3VLVideoProcessor
from transformers.processing_utils import ProcessingKwargs, ProcessorMixin


class Lfm3VlProcessorKwargs(ProcessingKwargs, total=False):  # type: ignore[call-arg]
    _defaults = {
        "text_kwargs": {
            "padding": False,
            "return_token_type_ids": False,
            "return_mm_token_type_ids": False,
        },
        "videos_kwargs": {"return_metadata": True},
    }


class Lfm3VlProcessor(Qwen3VLProcessor):
    valid_processor_kwargs = Lfm3VlProcessorKwargs

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        # The hub checkpoints point the sub-processors at remote code that wraps
        # these stock classes; load them directly and reuse vLLM's tokenizer.
        tokenizer = kwargs.pop("tokenizer", None)
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(
                pretrained_model_name_or_path, **kwargs
            )
        return cls(
            image_processor=Qwen2VLImageProcessor.from_pretrained(
                pretrained_model_name_or_path, **kwargs
            ),
            tokenizer=tokenizer,
            video_processor=Qwen3VLVideoProcessor.from_pretrained(
                pretrained_model_name_or_path, **kwargs
            ),
        )

    # Qwen3-VL's adds `mm_token_type_ids`, which LFM3-VL does not return
    model_input_names = ProcessorMixin.model_input_names

    def _process_videos(self, videos, **kwargs):
        # The checkpoints set `return_metadata: false`, but the frame timestamps
        # of the placeholder need the metadata.
        kwargs["return_metadata"] = True
        return super()._process_videos(videos, **kwargs)
