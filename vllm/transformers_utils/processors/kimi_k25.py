# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch
from transformers import BatchFeature
from transformers.processing_utils import ProcessorMixin

from vllm.multimodal.inputs import VisionChunk


class KimiK25Processor(ProcessorMixin):
    attributes = ["tokenizer"]
    tokenizer_class = "AutoTokenizer"

    def __init__(
        self, media_processor=None, tokenizer=None, media_token_id: int | None = None
    ):
        super().__init__(tokenizer)
        self.media_processor = media_processor
        self.media_token_id = media_token_id
        assert self.media_token_id is not None

    def __call__(
        self,
        vision_chunks: list[VisionChunk] | None = None,
        *,
        text: list[int] | str,
        **kwargs,
    ) -> BatchFeature:
        """
        Args:
            vision_chunks: List of VisionChunk items to be processed.
                For image: VisionChunkImage with type='image', image=PIL.Image
                For video_chunk: VisionChunkVideo with type='video_chunk',
                  video_chunk=list[PIL.Image]
            text: The token ids to be fed to a model (required).
        Returns:
            [`BatchFeature`]: A [`BatchFeature`] with the following fields:

            - **input_ids** -- list of token ids to be fed to a model.
            - **pixel_values** -- Pixel values to be fed to a model.
              Returned when `vision_chunks` is not `None`.
            - **grid_thws** -- list of image 3D grid in LLM.
              Returned when `vision_chunks` is not `None`.
        """
        mm_inputs = {}
        input_ids = self.tokenizer.encode(text) if isinstance(text, str) else text
        if vision_chunks is not None:
            assert isinstance(vision_chunks, list)
            mm_inputs = self.media_processor.preprocess(vision_chunks)

            num_tokens_per_chunk = [
                self.media_processor.media_tokens_calculator(chunk)
                for chunk in vision_chunks
            ]

            new_input_ids = []
            for token in input_ids:
                if token == self.media_token_id:
                    new_input_ids.extend(
                        [self.media_token_id] * num_tokens_per_chunk.pop(0)
                    )
                else:
                    new_input_ids.append(token)
            input_ids = new_input_ids

        # XXX: _apply_hf_processor_text_mm will call tolist() on input_ids
        return BatchFeature(
            data={
                "input_ids": torch.tensor([input_ids]),
                **mm_inputs,
            }
        )
