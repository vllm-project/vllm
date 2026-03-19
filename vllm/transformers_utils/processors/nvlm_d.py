# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# adapted from https://huggingface.co/nvidia/NVLM-D-72B/blob/main/modeling_nvlm_d.py
# --------------------------------------------------------
# NVLM-D
# Copyright (c) 2024 NVIDIA
# Licensed under Apache 2.0 License [see LICENSE for details]
# --------------------------------------------------------
from vllm.multimodal.processing import PromptUpdateDetails
from vllm.tokenizers.hf import HfTokenizer

from .internvl import InternVLImageProcessor, InternVLProcessor


class NVLMProcessor(InternVLProcessor):
    def __init__(
        self,
        image_processor: InternVLImageProcessor,
        tokenizer: HfTokenizer,
        *,
        image_seq_length: int,
        start_image_token: str = "<Image>",
        end_image_token: str = "</Image>",
        ctx_image_token: str = "<|vision_pad|>",
    ) -> None:
        super().__init__(
            image_processor=image_processor,
            tokenizer=tokenizer,
            image_seq_length=image_seq_length,
            start_image_token=start_image_token,
            end_image_token=end_image_token,
            ctx_image_token=ctx_image_token,
        )

    def get_image_repl(
        self,
        num_patches: int | None,
        num_features: int | None = None,
    ) -> PromptUpdateDetails[str]:
        if num_patches is None:
            raise NotImplementedError("Embedding inputs are not supported")

        num_features = num_patches * self.image_seq_length

        tile_pos_identifiers = [f"<tile_{i}>" for i in range(1, num_patches)]
        if self.image_processor.use_thumbnail:
            tile_pos_identifiers += ["<tile_global_thumbnail>"]

        context_size = num_features // num_patches
        features = "".join(
            (identifier + self.ctx_image_token * context_size)
            for identifier in tile_pos_identifiers
        )

        # We include the start and end as well because "<Image><tile" is
        # tokenized as ["<Image", "><", "tile"], resulting in assertion error
        # when trying to find "<tile" as a subsequence of "<Image><tile"
        repl = self.start_image_token + features + self.end_image_token

        return PromptUpdateDetails.select_text(repl, self.ctx_image_token)
