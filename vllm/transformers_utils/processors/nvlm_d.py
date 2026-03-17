# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# adapted from https://huggingface.co/nvidia/NVLM-D-72B/blob/main/modeling_nvlm_d.py
# --------------------------------------------------------
# NVLM-D
# Copyright (c) 2024 NVIDIA
# Licensed under Apache 2.0 License [see LICENSE for details]
# --------------------------------------------------------
from vllm.multimodal.processing import PromptUpdateDetails

from .internvl import BaseInternVLProcessor

IMG_PAD = "<|vision_pad|>"


class NVLMProcessor(BaseInternVLProcessor):
    @property
    def image_token_id(self) -> int:
        return self.tokenizer.get_vocab()[IMG_PAD]

    def get_image_repl(
        self,
        feature_size: int,
        num_patches: int | None,
    ) -> PromptUpdateDetails[str]:
        if num_patches is None:
            raise NotImplementedError("Embedding inputs are not supported")

        tile_pos_identifiers = [f"<tile_{i}>" for i in range(1, num_patches)]
        if self.use_thumbnail:
            tile_pos_identifiers += ["<tile_global_thumbnail>"]

        context_size = feature_size // num_patches
        features = "".join(
            identifier + IMG_PAD * context_size for identifier in tile_pos_identifiers
        )

        # We include the start and end as well because "<Image><tile" is
        # tokenized as ["<Image", "><", "tile"], resulting in assertion error
        # when trying to find "<tile" as a subsequence of "<Image><tile"
        repl = "<Image>" + features + "</Image>"

        return PromptUpdateDetails.select_text(repl, IMG_PAD)
