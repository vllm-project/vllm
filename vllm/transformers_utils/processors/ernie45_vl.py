# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# yapf: disable
# ruff: noqa: E501
# coding=utf-8
# Copyright (c) 2025 Baidu.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""Processor class for Ernie_4_5_VL."""

import math
from typing import Any, Dict, List, Union

import numpy as np
import torch
from PIL import Image
from collections import defaultdict


from transformers.utils import logging
from transformers.processing_utils import ProcessorMixin
from transformers.image_processing_utils import BatchFeature
from transformers.image_utils import ChannelDimension


logger = logging.get_logger(__name__)


def round_by_factor(number: int, factor: int) -> int:
    """Returns the closest integer to 'number' that is divisible by 'factor'."""
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
    return math.floor(number / factor) * factor


def smart_resize(
    height: int,
    width: int,
    factor: int = 28,
    min_pixels: int = 4 * 28 * 28,
    max_pixels: int = 16384 * 28 * 28,
):
    """
    Rescales the image so that the following conditions are met:

    1. Both dimensions (height and width) are divisible by 'factor'.

    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].

    3. The aspect ratio of the image is maintained as closely as possible.
    """
    MAX_RATIO = 200
    if max(height, width) / min(height, width) > MAX_RATIO:
        if height > width:
            new_width = max(factor, round_by_factor(width, factor))
            new_height = floor_by_factor(new_width * MAX_RATIO, factor)
        else:
            new_height = max(factor, round_by_factor(height, factor))
            new_width = floor_by_factor(new_height * MAX_RATIO, factor)

        logger.info(
            f"absolute aspect ratio must be smaller than {MAX_RATIO}, got {max(height, width) / min(height, width)},\
              resize to {max(new_height, new_width) / min(new_height, new_width)}"
        )

        height = new_height
        width = new_width

    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)

    if min_pixels > h_bar * w_bar or h_bar * w_bar > max_pixels:
        raise ValueError(f"encounter invalid h_bar: {h_bar}, w_bar: {w_bar}")

    return h_bar, w_bar


IDS_TYPE_FLAG = {"text": 0, "image": 1, "video": 2, "audio": 3}


class Ernie4_5_VLProcessor(ProcessorMixin):
    """
    Processes multimodal chat messages into model-ready inputs,
    handling text, images, and videos with 3D positional embeddings.
    """

    attributes = ["image_processor", "tokenizer"]
    valid_kwargs = [
        "chat_template",
        "spatial_conv_size",
        "temporal_conv_size",
        "image_min_pixels",
        "image_max_pixels",
        "video_min_pixels",
        "video_max_pixels",
        "video_target_frames",
        "video_frames_sample",
        "video_max_frames",
        "video_min_frames",
        "video_fps",
    ]
    image_processor_class = "AutoImageProcessor"
    tokenizer_class = "AutoTokenizer"

    CLS_TOKEN = "<|begin_of_sentence|>"
    SEP_TOKEN = "<|end_of_sentence|>"
    IMG_START = "<|IMAGE_START|>"
    IMG_END = "<|IMAGE_END|>"
    VID_START = "<|VIDEO_START|>"
    VID_END = "<|VIDEO_END|>"

    def __init__(
        self,
        image_processor=None,
        tokenizer=None,
        chat_template=None,
        spatial_conv_size: int = 2,
        temporal_conv_size: int = 2,
        image_min_pixels: int = 4 * 28 * 28,
        image_max_pixels: int = 6177 * 28 * 28,
        video_min_pixels: int = 299 * 28 * 28,
        video_max_pixels: int = 1196 * 28 * 28,
        video_target_frames: int = -1,
        video_frames_sample: str = "leading",
        video_max_frames: int = 180,
        video_min_frames: int = 16,
        video_fps: int = 2,
        **kwargs,
    ):
        super().__init__(image_processor, tokenizer)
        self.tokenizer.ignored_index = -100

        # Convolution sizes for patch aggregation
        self.spatial_conv_size = spatial_conv_size
        self.temporal_conv_size = temporal_conv_size

        # Pixel constraints
        self.image_min_pixels = image_min_pixels
        self.image_max_pixels = image_max_pixels
        self.video_min_pixels = video_min_pixels
        self.video_max_pixels = video_max_pixels

        # Video sampling parameters
        self.target_frames = video_target_frames
        self.frames_sample = video_frames_sample
        self.max_frames = video_max_frames
        self.min_frames = video_min_frames
        self.fps = video_fps

        # Special tokens and IDs
        self.cls_token = self.CLS_TOKEN
        self.sep_token = self.SEP_TOKEN
        self.image_start = self.IMG_START
        self.image_end = self.IMG_END
        self.video_start = self.VID_START
        self.video_end = self.VID_END
        self.image_patch_id = self.tokenizer.convert_tokens_to_ids(
            "<|IMAGE_PLACEHOLDER|>"
        )

        self.token_type_mapping = self._build_token_type_mapping()
        self.is_training = True
        self.role_prefixes = {"system": "", "user": "User: ", "bot": "Assistant: "}

    def _build_token_type_mapping(self) -> Dict[Any, int]:
        mapping = defaultdict(lambda: IDS_TYPE_FLAG["text"])
        for token in (self.IMG_START, self.IMG_END, self.VID_START, self.VID_END):
            mapping[token] = IDS_TYPE_FLAG["image"]
        mapping[self.image_patch_id] = IDS_TYPE_FLAG["image"]
        return mapping

    def __call__(
        self,
        text: Union[str, List[str]],
        images: List[Image.Image] = [],
        videos: List[List[Image.Image]] = [],
        **kwargs,
    ) -> BatchFeature:
        """
        Convert chat messages into model inputs.
        Returns a dict with input_ids, token_type_ids, position_ids, images, grid_thw, image_type_ids, labels.
        """
        outputs = {
            "input_ids": [],
            "token_type_ids": [],
            "position_ids": [],
            "images": [],
            "grid_thw": [],
            "image_type_ids": [],
            "cur_position": 0,
            "pic_cnt": 0,
            "video_cnt": 0,
        }
        if not isinstance(text, list):
            text = [text]
        
        if len(text) == 0:
            raise ValueError("Processor no text is provided")
        
        # only support single element
        texts = text[0]

        new_video_seg = True
        for text_with_image in texts.split(self.VID_START + "<|video@placeholder|>" + self.VID_END):
            new_text_seg = True
            if not new_video_seg:
                self._add_video(videos[outputs["video_cnt"]], outputs)
            for text in text_with_image.split(self.IMG_START + "<|image@placeholder|>" + self.IMG_END):
                if not new_text_seg:
                    self._add_image(images[outputs["pic_cnt"]], outputs)
                self._add_text(text, outputs)
                new_text_seg = False
            new_video_seg = False

        for key in ["cur_position", "pic_cnt", "video_cnt"]:
            outputs.pop(key, None)

        outputs = self._pack_outputs(outputs)
        for key in outputs.keys():
            if isinstance(outputs[key], np.ndarray):
                if key in ["images", "grid_thw"]:
                    outputs[key] = torch.tensor(np.array(outputs[key]))
                else:
                    outputs[key] = torch.tensor(np.array([outputs[key]]))

        return BatchFeature(data=outputs)

    def _add_special_token(self, token: Union[str, int], outputs: Dict) -> None:
        """add special token to outputs"""
        token_id = (
            token
            if isinstance(token, int)
            else self.tokenizer.convert_tokens_to_ids(token)
        )
        outputs["input_ids"].append(token_id)
        outputs["token_type_ids"].append(self.token_type_mapping[token])
        pos = outputs["cur_position"]
        outputs["position_ids"].append([pos] * 3)
        outputs["cur_position"] += 1
    
    def _add_text(self, text: str, outputs: Dict) -> None:
        """add text to outputs"""
        tokens = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(text))
        outputs["input_ids"].extend(tokens)
        outputs["token_type_ids"].extend([IDS_TYPE_FLAG["text"]] * len(tokens))

        start = outputs["cur_position"]
        for i in range(len(tokens)):
            outputs["position_ids"].append([start + i] * 3)
        outputs["cur_position"] += len(tokens)

    def _add_image(self, img: Image.Image, outputs: Dict) -> None:
        """add image to outputs"""
        outputs["pic_cnt"] += 1
        self._add_special_token(self.IMG_START, outputs)

        patches_h, patches_w = self.image_processor.get_smarted_resize(
            img.height,
            img.width,
            min_pixels=self.image_min_pixels,
            max_pixels=self.image_max_pixels,
        )[1]
        num_tokens = (patches_h * patches_w) // (self.spatial_conv_size**2)

        outputs["input_ids"].extend([self.image_patch_id] * num_tokens)
        outputs["token_type_ids"].extend([IDS_TYPE_FLAG["image"]] * num_tokens)

        pos_ids = self._compute_3d_positions(
            1, patches_h, patches_w, outputs["cur_position"]
        )
        outputs["position_ids"].extend(pos_ids)
        outputs["cur_position"] = np.max(pos_ids) + 1

        # Preprocess pixels
        ret = self.image_processor.preprocess(
            images=[img.convert("RGB")],
            do_normalize=False,
            do_rescale=False,
            predetermined_grid_thw=np.array([[patches_h, patches_w]]),
            do_convert_rgb=True,
            input_data_format=ChannelDimension.LAST,
        )
        outputs["images"].append(ret["pixel_values"])
        outputs["grid_thw"].append(ret["image_grid_thw"])
        outputs["image_type_ids"].append(0)

        self._add_special_token(self.IMG_END, outputs)

    def _add_video(
        self, pixel_stack: List[np.ndarray], outputs: Dict
    ) -> None:
        outputs["video_cnt"] += 1
        self._add_special_token(self.VID_START, outputs)

        patches_h, patches_w = self.image_processor.get_smarted_resize(
            pixel_stack.shape[1],
            pixel_stack.shape[2],
            min_pixels=self.video_min_pixels,
            max_pixels=self.video_max_pixels,
        )[1]
        num_frames = pixel_stack.shape[0]
        num_tokens = (num_frames * patches_h * patches_w) // (
            self.spatial_conv_size**2 * self.temporal_conv_size
        )

        ret = self.image_processor.preprocess(
            images=None,
            videos=pixel_stack,
            do_normalize=False,
            do_rescale=False,
            predetermined_grid_thw=np.array([[patches_h, patches_w]] * num_frames),
            do_convert_rgb=True,
            input_data_format=ChannelDimension.LAST,
        )
        outputs["images"].append(ret["pixel_values_videos"])
        outputs["grid_thw"].append(ret["video_grid_thw"])
        outputs["image_type_ids"].extend([1] * num_frames)

        outputs["input_ids"].extend([self.image_patch_id] * num_tokens)
        outputs["token_type_ids"].extend([IDS_TYPE_FLAG["video"]] * num_tokens)

        pos_ids = self._compute_3d_positions(
            num_frames, patches_h, patches_w, outputs["cur_position"]
        )
        outputs["position_ids"].extend(pos_ids)
        outputs["cur_position"] = np.max(pos_ids) + 1

        self._add_special_token(self.VID_END, outputs)

    def _compute_3d_positions(
        self, t: int, h: int, w: int, start_idx: int
    ) -> List[List[int]]:
        # Downsample time if needed
        t_eff = t // self.temporal_conv_size if t != 1 else 1
        gh, gw = h // self.spatial_conv_size, w // self.spatial_conv_size
        time_idx = np.repeat(np.arange(t_eff), gh * gw)
        h_idx = np.tile(np.repeat(np.arange(gh), gw), t_eff)
        w_idx = np.tile(np.arange(gw), t_eff * gh)

        coords = list(zip(time_idx, h_idx, w_idx))
        return [
            [start_idx + ti, start_idx + hi, start_idx + wi] for ti, hi, wi in coords
        ]

    def _pack_outputs(self, outs: Dict) -> Dict[str, Any]:
        # Stack or nullify image-related fields
        if not outs["images"]:
            outs["images"] = None
            outs["grid_thw"] = None
            outs["image_type_ids"] = None
        else:
            outs["images"] = np.vstack(outs["images"])
            outs["grid_thw"] = np.vstack(outs["grid_thw"])
            outs["image_type_ids"] = np.array(outs["image_type_ids"])

        # Convert lists to arrays
        outs["input_ids"] = np.array(outs["input_ids"], dtype=np.int64)
        outs["token_type_ids"] = np.array(outs["token_type_ids"], dtype=np.int64)
        outs["position_ids"] = np.array(outs["position_ids"], dtype=np.int64)
        return outs

    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to Ernie4_5_VLTokenizer's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to Ernie4_5_VLTokenizer's [`~PreTrainedTokenizer.decode`].
        Please refer to the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)

    @property
    def model_input_names(self):
        """get model input names"""
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        return list(tokenizer_input_names) + list(image_processor_input_names)


__all__ = ["Ernie4_5_VLProcessor"]