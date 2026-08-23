# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from fractions import Fraction

from transformers import LlavaNextProcessor
from transformers.image_processing_utils import select_best_resolution


class Granite4VisionProcessor(LlavaNextProcessor):
    """Processor for Granite 4 Vision.

    Extends LlavaNextProcessor to account for the Window Q-Former
    downsampling when computing the number of image features.

    This processor is needed because the granite4_vision processor type
    is not yet in the transformers version pinned by vLLM.
    """

    model_type = "granite4_vision"

    def __init__(
        self,
        image_processor=None,
        tokenizer=None,
        patch_size=None,
        vision_feature_select_strategy=None,
        chat_template=None,
        image_token="<image>",
        num_additional_image_tokens=0,
        downsample_rate=None,
        **kwargs,
    ):
        super().__init__(
            image_processor=image_processor,
            tokenizer=tokenizer,
            patch_size=patch_size,
            vision_feature_select_strategy=vision_feature_select_strategy,
            chat_template=chat_template,
            image_token=image_token,
            num_additional_image_tokens=num_additional_image_tokens,
        )
        self.downsample_rate = downsample_rate

    def _get_number_of_features(
        self,
        orig_height: int,
        orig_width: int,
        height: int,
        width: int,
    ) -> int:
        image_grid_pinpoints = self.image_processor.image_grid_pinpoints

        height_best_resolution, width_best_resolution = select_best_resolution(
            [orig_height, orig_width], image_grid_pinpoints
        )
        scale_height = height_best_resolution // height
        scale_width = width_best_resolution // width

        patches_height = height // self.patch_size
        patches_width = width // self.patch_size
        if self.downsample_rate is not None:
            ds_rate = Fraction(self.downsample_rate)
            patches_height = int(patches_height * ds_rate)
            patches_width = int(patches_width * ds_rate)

        unpadded_features, newline_features = self._get_unpadded_features(
            orig_height,
            orig_width,
            patches_height,
            patches_width,
            scale_height,
            scale_width,
        )
        base_features = (
            patches_height * patches_width + self.num_additional_image_tokens
        )
        return unpadded_features + newline_features + base_features
