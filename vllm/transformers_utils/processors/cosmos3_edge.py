# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import math
from typing import Optional, Union

import torch
from transformers import AutoTokenizer
from transformers.feature_extraction_utils import BatchFeature
from transformers.image_utils import ChannelDimension, PILImageResampling, SizeDict
from transformers.models.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor
from transformers.models.qwen3_vl.video_processing_qwen3_vl import (
    Qwen3VLVideoProcessor,
    Qwen3VLVideoProcessorInitKwargs,
    get_image_size,
    smart_resize,
)
from transformers.models.siglip2.image_processing_siglip2 import (
    Siglip2ImageProcessor,
    convert_image_to_patches,
)
from transformers.processing_utils import Unpack
from transformers.utils import TensorType
from transformers.video_utils import group_videos_by_shape, reorder_videos
from torchvision.transforms import InterpolationMode


def round_by_factor(number: int, factor: int) -> int:
    """Returns the closest integer to 'number' that is divisible by 'factor'."""
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
    return math.floor(number / factor) * factor


class Cosmos3EdgeImagesKwargs(Siglip2ImageProcessor.valid_kwargs, total=False):
    # global setting for all images, can be overridden by per-image kwargs
    max_pixels: Optional[int]
    min_pixels: Optional[int]

    # per-image overrides, e.g. [{"min_pixels": 256*28*28, "max_pixels": 1280*28*28}, None, ...]
    per_image_kwargs: Optional[list[dict]]


class Cosmos3EdgeImageProcessor(Siglip2ImageProcessor):
    resample = PILImageResampling.BICUBIC
    valid_kwargs = Cosmos3EdgeImagesKwargs

    def __init__(self, **kwargs: Unpack[Cosmos3EdgeImagesKwargs]):
        super().__init__(**kwargs)

    def _resize_image(
        self,
        image: torch.Tensor,
        max_ratio=200,
        resample: Optional[Union[PILImageResampling, InterpolationMode, int]] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Rescales the image so that the following conditions are met:

        1. Both dimensions (height and width) are divisible by 'factor'.
        2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].
        3. The aspect ratio of the image is maintained as closely as possible.
        """
        image_min_pixels = kwargs.get("min_pixels", None) or self.size.get("shortest_edge", None)
        image_max_pixels = kwargs.get("max_pixels", None) or self.size.get("longest_edge", None)
        assert image_min_pixels is not None and image_max_pixels is not None, "When do_resize is True, min_pixels and max_pixels must be provided."
        assert image_max_pixels >= image_min_pixels, "The max_pixels of image must be greater than or equal to min_pixels."

        _, height, width = image.shape
        if max(height, width) / min(height, width) > max_ratio:
            raise ValueError(
                f"absolute aspect ratio must be smaller than {max_ratio}, got {max(height, width) / min(height, width)}"
            )
        factor = self.merge_size * self.patch_size
        h_bar = max(factor, round_by_factor(height, factor))
        w_bar = max(factor, round_by_factor(width, factor))
        if h_bar * w_bar > image_max_pixels:
            beta = math.sqrt((height * width) / image_max_pixels)
            h_bar = floor_by_factor(height / beta, factor)
            w_bar = floor_by_factor(width / beta, factor)
        elif h_bar * w_bar < image_min_pixels:
            beta = math.sqrt(image_min_pixels / (height * width))
            h_bar = ceil_by_factor(height * beta, factor)
            w_bar = ceil_by_factor(width * beta, factor)

        image = self.resize(
            image,
            size=SizeDict(height=h_bar, width=w_bar),
            resample=resample,
        )
        return image

    def _preprocess(
        self,
        images: list["torch.Tensor"],
        do_resize: bool,
        patch_size: int,
        max_num_patches: int,
        resample: Optional[Union[PILImageResampling, InterpolationMode, int]],
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: Optional[Union[float, list[float]]],
        image_std: Optional[Union[float, list[float]]],
        return_tensors: Optional[Union[str, TensorType]],
        **kwargs,
    ) -> BatchFeature:
        per_image_kwargs = kwargs.pop("per_image_kwargs", None)
        pixel_values = []
        spatial_shapes = []
        for idx, image in enumerate(images):
            # merge global kwargs with per-image overrides (if provided)
            image_kwargs = kwargs.copy()
            if per_image_kwargs is not None and idx < len(per_image_kwargs) and per_image_kwargs[idx]:
                image_kwargs.update(per_image_kwargs[idx])
            if do_resize:
                image = self._resize_image(
                    image,
                    max_ratio=200,
                    resample=resample,
                    **image_kwargs,
                )

            image = self.rescale_and_normalize(image, do_rescale, rescale_factor, do_normalize, image_mean, image_std)

            # (num_channels, height, width) -> (num_patches, patch_size * patch_size * num_channels)
            patches = convert_image_to_patches(image, patch_size)

            num_patches_height = image.shape[1] // patch_size
            num_patches_width = image.shape[2] // patch_size

            spatial_shapes.append((num_patches_height, num_patches_width))
            pixel_values.append(patches)

        pixel_values = torch.cat(pixel_values, dim=0)

        spatial_shapes = torch.tensor(spatial_shapes)
        t_dim = torch.ones((spatial_shapes.shape[0], 1),
                           dtype=spatial_shapes.dtype,
                            device=spatial_shapes.device)
        image_grid_thw = torch.cat([t_dim, spatial_shapes], dim=1)

        batch_feature = BatchFeature(
            data={
                "pixel_values": pixel_values,
                "image_grid_thw": image_grid_thw,
            },
            tensor_type=return_tensors,
        )
        return batch_feature


class Cosmos3EdgeVideoProcessor(Qwen3VLVideoProcessor):
    def __init__(self, **kwargs: Unpack[Qwen3VLVideoProcessorInitKwargs]):
        super().__init__(**kwargs)

    def _preprocess(
        self,
        videos: list[torch.Tensor],
        do_convert_rgb: bool = True,
        do_resize: bool = True,
        size: Optional[SizeDict] = None,
        resample: Optional[
            Union[PILImageResampling, InterpolationMode, int]
        ] = PILImageResampling.BICUBIC,
        do_rescale: bool = True,
        rescale_factor: float = 1 / 255.0,
        do_normalize: bool = True,
        image_mean: Optional[Union[float, list[float]]] = None,
        image_std: Optional[Union[float, list[float]]] = None,
        patch_size: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        **kwargs,
    ):
        merge_size = self.merge_size
        grouped_videos, grouped_videos_index = group_videos_by_shape(videos)
        resized_videos_grouped = {}

        for shape, stacked_videos in grouped_videos.items():
            B, T, C, H, W = stacked_videos.shape
            num_frames, height, width = T, H, W
            if do_resize:
                # 1. Both dimensions (height and width) are divisible by 'factor'.
                # 2. The total number of pixels (t * h * w) is within the range ['min_pixels', 'max_pixels'].
                resized_height, resized_width = smart_resize(
                    num_frames=num_frames,
                    height=height,
                    width=width,
                    temporal_factor=1,
                    factor=patch_size * merge_size,
                    min_pixels=size.shortest_edge,
                    max_pixels=size.longest_edge,
                )
                stacked_videos = stacked_videos.view(B * T, C, H, W)
                stacked_videos = self.resize(
                    stacked_videos,
                    size=SizeDict(height=resized_height, width=resized_width),
                    resample=resample,
                )
                stacked_videos = stacked_videos.view(B, T, C, resized_height, resized_width)
            resized_videos_grouped[shape] = stacked_videos
        resized_videos = reorder_videos(resized_videos_grouped, grouped_videos_index)

        # Group videos by size for further processing
        # Needed in case do_resize is False, or resize returns videos with different sizes
        grouped_videos, grouped_videos_index = group_videos_by_shape(resized_videos)
        processed_videos_grouped = {}
        processed_grids = {}
        for shape, stacked_videos in grouped_videos.items():
            resized_height, resized_width = get_image_size(stacked_videos[0], channel_dim=ChannelDimension.FIRST)
            # Fused rescale and normalize
            stacked_videos = self.rescale_and_normalize(
                stacked_videos, do_rescale, rescale_factor, do_normalize, image_mean, image_std
            )
            patches = stacked_videos

            batch_size, grid_t, channel = patches.shape[:3]
            grid_h, grid_w = resized_height // patch_size, resized_width // patch_size

            # [1, 16, 3, 320, 608] -> [1, 16, 3, 20, 16, 38, 16]
            patches = patches.view(
                batch_size, # 0
                grid_t, # 1
                channel, # 2
                grid_h, # 3
                patch_size, # 4
                grid_w, # 5
                patch_size, # 6
            )
            # [1, 16, 3, 20, 16, 38, 16] -> [1, 16, 20, 38, 16, 16, 3]
            # After permute: [batch_size, grid_t, grid_h, grid_w, patch_size, patch_size, channel]
            patches = patches.permute(0, 1, 3, 5, 4, 6, 2)

            # [1, 16, 20, 38, 16, 16, 3] -> [1, 5120, 768]
            flatten_patches = patches.reshape(
                batch_size,
                grid_t * grid_h * grid_w,
                patch_size * patch_size * channel,
            )

            processed_videos_grouped[shape] = flatten_patches
            processed_grids[shape] = [[grid_t, grid_h, grid_w]] * batch_size

        processed_videos = reorder_videos(processed_videos_grouped, grouped_videos_index)
        processed_grids = reorder_videos(processed_grids, grouped_videos_index)
        pixel_values_videos = torch.cat(processed_videos, dim=0)
        video_grid_thw = torch.tensor(processed_grids)
        data = {
            "pixel_values_videos": pixel_values_videos,
            "video_grid_thw": video_grid_thw,
        }

        return BatchFeature(data=data, tensor_type=return_tensors)

class Cosmos3EdgeProcessor(Qwen3VLProcessor):
    r"""
    Constructs a Qwen3VL processor which wraps a Qwen3VL image processor and a Qwen2 tokenizer into a single processor.
    [`Qwen3VLProcessor`] offers all the functionalities of [`Qwen2VLImageProcessor`] and [`Qwen2TokenizerFast`]. See the
    [`~Qwen3VLProcessor.__call__`] and [`~Qwen3VLProcessor.decode`] for more information.
    Args:
        image_processor ([`Qwen2VLImageProcessor`], *optional*):
            The image processor is a required input.
        tokenizer ([`Qwen2TokenizerFast`], *optional*):
            The tokenizer is a required input.
        video_processor ([`Qwen3VLVideoProcessor`], *optional*):
            The video processor is a required input.
        chat_template (`str`, *optional*): A Jinja template which will be used to convert lists of messages
            in a chat into a tokenizable string.
    """
    attributes = ["image_processor", "tokenizer", "video_processor"]

    # Transformers' auto loaders cannot resolve these vLLM-native image and
    # video processors without remote code or global auto-class registration.
    @classmethod
    def _get_arguments_from_pretrained(
        cls,
        pretrained_model_name_or_path,
        processor_dict=None,
        **kwargs,
    ):
        component_kwargs = dict(kwargs)
        component_kwargs.pop("trust_remote_code", None)

        image_processor = Cosmos3EdgeImageProcessor.from_pretrained(
            pretrained_model_name_or_path,
            **component_kwargs,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path,
            trust_remote_code=False,
            **component_kwargs,
        )
        video_processor = Cosmos3EdgeVideoProcessor.from_pretrained(
            pretrained_model_name_or_path,
            **component_kwargs,
        )

        return [
            image_processor,
            tokenizer,
            video_processor,
        ]


__all__ = [
    "Cosmos3EdgeProcessor",
]