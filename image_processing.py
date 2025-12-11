# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved. Except portions as noted which are Copyright (c) 2023 OpenGVLab and licensed under the MIT license found in LICENSE.
from abc import ABC, abstractmethod
from dataclasses import dataclass
import math
from typing import Callable, Optional
import numpy as np
import random
from PIL import Image
import albumentations as A

import einops
import torch
from torchvision import transforms as T
from torchvision.transforms import Compose
from torchvision.transforms.functional import InterpolationMode

from data_loading.conversation_sample import (
    ImageMedia,
    VideoFrameMedia,
)

IMAGENET_PIXEL_MEAN = [0.485, 0.456, 0.406]
IMAGENET_PIXEL_STD = [0.229, 0.224, 0.225]
SIGLIP_PIXEL_MEAN = [0.5, 0.5, 0.5]
SIGLIP_PIXEL_STD = [0.5, 0.5, 0.5]
CLIP_PIXEL_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_PIXEL_STD = [0.26862954, 0.26130258, 0.27577711]
RADIO_G_PIXEL_MEAN = [0.4850, 0.4560, 0.4060]
RADIO_G_PIXEL_STD = [0.2230, 0.2240, 0.2250]


pixel_statistics = {
    "clip": (CLIP_PIXEL_MEAN, CLIP_PIXEL_STD),
    "siglip": (SIGLIP_PIXEL_MEAN, SIGLIP_PIXEL_STD),
    "internvit": (IMAGENET_PIXEL_MEAN, IMAGENET_PIXEL_STD),
    "radio": (CLIP_PIXEL_MEAN, CLIP_PIXEL_STD),
    "radio-g": (RADIO_G_PIXEL_MEAN, RADIO_G_PIXEL_STD),
    "huggingface": (SIGLIP_PIXEL_MEAN, SIGLIP_PIXEL_STD),
    "radio_siglip_move": (CLIP_PIXEL_MEAN, CLIP_PIXEL_STD),
    "cradio-v1": (CLIP_PIXEL_MEAN, CLIP_PIXEL_STD),
    "cradio-g": (CLIP_PIXEL_MEAN, CLIP_PIXEL_STD),
}


# From https://github.com/OpenGVLab/InternVL/blob/c62fa4f7c850165d7386bdc48ac6bc5a6fab0864/internvl_chat/internvl/train/dataset.py#L685
# Copyright (c) 2023 OpenGVLab.
def find_closest_aspect_ratio(
    aspect_ratio: float,
    target_ratios: list[tuple[int, int]],
    width: int,
    height: int,
    image_size: int,
) -> tuple[int, int]:
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def find_closest_area_weighted_aspect_ratio(
    aspect_ratio: float,
    target_ratios: list[tuple[int, int]],
    width: int,
    height: int,
    image_size: int,
):
    """
    Find the best number of tiles based on the aspect ratio and the area covered by the tiles.
    """
    best_factor = float("-inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        factor_based_on_area_n_ratio = min(
            (ratio[0] * ratio[1] * image_size * image_size) / area, 0.6
        ) * min(target_aspect_ratio / aspect_ratio, aspect_ratio / target_aspect_ratio)
        if factor_based_on_area_n_ratio > best_factor:
            best_factor = factor_based_on_area_n_ratio
            best_ratio = ratio
    return best_ratio


# Mike's optimized ToTensor.
def _fast_to_tensor(pic) -> torch.Tensor:
    np_img = np.array(pic, copy=False)
    img = torch.from_numpy(np_img)
    img = img.permute(2, 0, 1)  # HWC to CHW
    fp_img = img.to(dtype=torch.float32, memory_format=torch.contiguous_format)
    fp_img.div_(255)
    return fp_img


@dataclass
class ImageTilingParams:
    media: ImageMedia | VideoFrameMedia
    num_tiles: int
    num_embeddings: int


class ImageTilingStrategy(ABC):
    """
    Base class for image tiling strategies.
    A tiling strategy is a function that takes a list of media and returns a list of image tiling parameters.
    These can then be used to apply the tiling to the media.

    Subclasses must implement the `compute_params` and `apply_params` methods.

    The `transform` method is a convenience method that computes the transformation parameters and applies the transformation to the media.

    """

    def transform(
        self,
        media_list: list[ImageMedia | VideoFrameMedia],
        num_tokens_available: int | None = None,
    ) -> list[torch.Tensor]:
        """
        Transform the media and compute the transformation parameters.
        """
        transform_media_list = self.compute_params(media_list, num_tokens_available)
        return [
            self.apply_params(transform_media, **kwargs)
            for transform_media in transform_media_list
        ]

    @abstractmethod
    def compute_params(
        self, media_list: list[ImageMedia | VideoFrameMedia], num_tokens_available: int, max_num_tiles: int | None = None, **kwargs
    ) -> list[ImageTilingParams]:
        """
        Compute the transformation parameters and the number of tokens to use for the media.

        Args:
            media_list: List of media to transform
            num_tokens_available: Number of tokens available for all media
            max_num_tiles: Maximum number of tiles allowed (optional, defaults to instance's max_num_tiles if not provided)

        Returns:
            list of transformation parameters with the media
        """
        ...

    @abstractmethod
    def apply_params(self, transform_media: ImageTilingParams, **kwargs) -> list[torch.Tensor]:
        """
        Apply the transformation parameters to the media.

        Args:
            transform_media: The media to apply the transformation to

        Returns:
            list of transformed media tensors
        """
        ...

    @abstractmethod
    def stack(
        self, images: list[torch.Tensor]
    ) -> tuple[torch.Tensor, list[tuple[int, int]], list[int], list[int]]:
        """
        Stack the images into a single tensor.

        Args:
            media_list: List of images to stack

        Returns:
            tuple of (stacked media, image sizes, vision cu lengths, vision max lengths)
        """
        ...


class _FixedSizeStrategy(ImageTilingStrategy):
    """
    Base class for fixed size image tiling strategies.
    """

    def __init__(
        self,
        vision_model_type: str,
        target_width: int,
        target_height: int,
        embeddings_per_image: int,
    ):
        self._vision_model_type = vision_model_type
        self._target_width = target_width
        self._target_height = target_height
        self._embeddings_per_image = embeddings_per_image
        self._transform = self._build_transform(
            (target_width, target_height), vision_model_type
        )

    # Based on https://github.com/openai/CLIP/blob/dcba3cb2e2827b402d2701e7e1c7d9fed8a20ef1/clip/clip.py#L79
    # and https://github.com/OpenGVLab/InternVL/blob/aa521e6eb1df4cf153aa4118fcf13e673c055d46/internvl_chat/internvl/train/dataset.py#L276
    @staticmethod
    def _build_transform(target_size: tuple[int, int], vision_model_type: str):
        """
        Build a transform for a given vision model type and target size.
        """
        if vision_model_type in ("siglip", "internvit", "radio", "radio-g", "cradio-g"):
            pixel_mean, pixel_std = pixel_statistics[vision_model_type]

            transform = T.Compose(
                [
                    T.Lambda(
                        lambda img: img.convert("RGB") if img.mode != "RGB" else img
                    ),
                    T.Resize(
                        (target_size[1], target_size[0]),
                        interpolation=InterpolationMode.BICUBIC,
                    ),
                    T.ToTensor(), #T.Lambda(lambda img: _fast_to_tensor(img)),
                    T.Normalize(mean=pixel_mean, std=pixel_std),
                ]
            )
        # From the official CLIP repo.
        elif vision_model_type == "clip":
            pixel_mean, pixel_std = pixel_statistics[vision_model_type]

            transform = Compose(
                [
                    T.Resize(
                        (target_size[1], target_size[0]),
                        interpolation=InterpolationMode.BICUBIC,
                    ),
                    T.Lambda(
                        lambda img: img.convert("RGB") if img.mode != "RGB" else img
                    ),
                    T.ToTensor(), #T.Lambda(lambda img: _fast_to_tensor(img)),
                    T.Normalize(mean=pixel_mean, std=pixel_std),
                ]
            )
        elif vision_model_type.startswith("hf://"):
            from megatron.core.models.huggingface.module import get_hf_model_type

            model_type = get_hf_model_type(vision_model_type)
            if "siglip" in model_type:
                from transformers.models.siglip.image_processing_siglip import (
                    SiglipImageProcessor,
                )

                processor = SiglipImageProcessor(
                    size={"height": target_size[1], "width": target_size[0]}
                )

                def transform(x):
                    x = x.convert("RGB") if x.mode != "RGB" else x
                    x = processor(x, return_tensors="pt")
                    return x["pixel_values"][0]
            else:
                raise NotImplementedError(
                    f"image processing not defined for huggingface model {vision_model_type}"
                )
        else:
            raise NotImplementedError(
                f"image processing not defined for vision model {vision_model_type}"
            )

        return transform

    def stack(
        self, images: list[torch.Tensor]
    ) -> tuple[torch.Tensor, list[tuple[int, int]], list[int] | None, list[int] | None]:
        return (
            torch.stack(images) if len(images) > 0 else None,
            torch.tensor(
                [(img.shape[1], img.shape[2]) for img in images], dtype=torch.int32
            ) if len(images) > 0 else None,
            None,
            None,
        )


class NoTilingStrategy(_FixedSizeStrategy):
    """
    A simple image transformation that resizes the image to the target width and height.
    """

    def __init__(
        self,
        vision_model_type: str,
        target_width: int,
        target_height: int,
        embeddings_per_image: int,
    ):
        super().__init__(
            vision_model_type=vision_model_type,
            target_width=target_width,
            target_height=target_height,
            embeddings_per_image=embeddings_per_image,
        )

    def apply_params(self, transform_media: ImageTilingParams, **kwargs) -> list[torch.Tensor]:
        return [self._transform(transform_media.media.value)]

    def compute_params(
        self,
        media_list: list[ImageMedia | VideoFrameMedia],
        num_tokens_available: Optional[int] = None,
        max_num_tiles: int | None = None,
        **kwargs,
    ) -> list[ImageTilingParams]:
        return [
            ImageTilingParams(
                media=media, num_tiles=1, num_embeddings=self._embeddings_per_image
            )
            for media in media_list
        ]

    def __str__(self):
        return f"SimpleImageTransform(vision_model_type={self._vision_model_type}, num_tokens_per_image={self._embeddings_per_image})"


@dataclass
class ImageTilingParamsV1(ImageTilingParams):
    tiling: tuple[int, int]


class ImageTilingStrategyV1(_FixedSizeStrategy):
    """Tiling image transformation.

    This transformation splits the image into a grid of tiles and applies the transformation to each tile.
    """

    # Based on https://github.com/openai/CLIP/blob/dcba3cb2e2827b402d2701e7e1c7d9fed8a20ef1/clip/clip.py#L79
    # and https://github.com/OpenGVLab/InternVL/blob/aa521e6eb1df4cf153aa4118fcf13e673c055d46/internvl_chat/internvl/train/dataset.py#L276

    def __init__(
        self,
        vision_model_type: str,
        tile_size: int,
        use_thumbnail: bool,
        min_num_tiles: int,
        max_num_tiles: int,
        embeddings_per_tile: int,
        find_closest_aspect_ratio_fn=find_closest_aspect_ratio,
    ):
        super().__init__(
            vision_model_type=vision_model_type,
            target_width=tile_size,
            target_height=tile_size,
            embeddings_per_image=embeddings_per_tile,
        )

        # print(f"Transformation params: {vision_model_type=}, {use_tiling=}, {tile_size=}, {use_thumbnail=}, {augment=}, {min_num_tiles=}, {max_num_tiles=}, {find_closest_aspect_ratio_fn=}")
        self._tile_size = tile_size
        self._use_thumbnail = use_thumbnail
        self._min_num_tiles = min_num_tiles
        self._max_num_tiles = max_num_tiles
        self._find_closest_aspect_ratio_fn = find_closest_aspect_ratio_fn

        # Calculate all possible aspect ratios for each max_num_tiles.
        self.target_ratios = {
            max_num_tiles: sorted(
                set(
                    (x, y)
                    for n in range(self._min_num_tiles, max_num_tiles + 1)
                    for x in range(1, n + 1)
                    for y in range(1, n + 1)
                    if x * y <= max_num_tiles and x * y >= self._min_num_tiles
                ),
                key=lambda x: x[0] * x[1],
            )
            for max_num_tiles in range(self._min_num_tiles, self._max_num_tiles + 1)
        }

        self.transform = A.Compose([
            A.OneOf([
                A.GaussNoise(var_limit=(5.0, 30.0)),
                A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5)),
            ], p=0.3),
            A.OneOf([
                A.MedianBlur(blur_limit=5),
                A.GaussianBlur(blur_limit=5),
            ], p=0.2),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.5),
            A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=15, val_shift_limit=15, p=0.3),
            A.ImageCompression(quality_lower=70, quality_upper=100, p=0.3),
        ])

    def apply_params(self, transform_media: ImageTilingParams, data_augment: bool = False, **kwargs) -> list[torch.Tensor]:
        assert isinstance(transform_media, ImageTilingParamsV1)
        image = transform_media.media.value

        if data_augment:
            image = self.transform(image=np.asarray(image))["image"]
            image = Image.fromarray(image)

        # calculate the target width and height
        target_width = self._tile_size * transform_media.tiling[0]
        target_height = self._tile_size * transform_media.tiling[1]
        blocks = transform_media.tiling[0] * transform_media.tiling[1]

        # resize the image
        resized_img = image.resize((target_width, target_height))
        processed_images = []
        for i in range(blocks):
            box = (
                (i % (target_width // self._tile_size)) * self._tile_size,
                (i // (target_width // self._tile_size)) * self._tile_size,
                ((i % (target_width // self._tile_size)) + 1) * self._tile_size,
                ((i // (target_width // self._tile_size)) + 1) * self._tile_size,
            )
            # split the image
            split_img = resized_img.crop(box)
            processed_images.append(split_img)
        assert len(processed_images) == blocks
        if self._use_thumbnail and len(processed_images) != 1:
            thumbnail_img = image.resize((self._tile_size, self._tile_size))
            processed_images.append(thumbnail_img)

        return [self._transform(img) for img in processed_images]

    def compute_params(
        self,
        media_list: list[ImageMedia | VideoFrameMedia],
        num_tokens_available: Optional[int] = None,
        max_num_tiles: int | None = None,
        data_augment: bool = False,
        tiling_augment_prob: float = 0.4,
        **kwargs,
    ) -> list[ImageTilingParamsV1]:
        # Use provided max_num_tiles or fall back to instance's max_num_tiles
        # Clamp to self._max_num_tiles since target_ratios are only pre-computed up to that value
        effective_max_num_tiles = max_num_tiles if max_num_tiles is not None else self._max_num_tiles
        effective_max_num_tiles = min(effective_max_num_tiles, self._max_num_tiles)
        
        max_num_tiles_to_use = min(
            num_tokens_available // self._embeddings_per_image, effective_max_num_tiles
        )

        # calculate the existing image aspect ratio
        target_ratios = self.target_ratios[max_num_tiles_to_use]

        params = []
        for media in media_list:
            if isinstance(media, ImageMedia):
                img_size = (media.width, media.height)
            elif isinstance(media, VideoFrameMedia):
                img_size = (media.video_width, media.video_height)
            else:
                raise ValueError(f"Unsupported media type: {type(media)}")

            aspect_ratio = img_size[0] / img_size[1]

            # find the closest aspect ratio to the target
            tiling = self._find_closest_aspect_ratio_fn(
                aspect_ratio, target_ratios, img_size[0], img_size[1], self._tile_size
            )
            if data_augment and isinstance(media, ImageMedia) and random.random() < tiling_augment_prob:
                tiling = self.augment_tiling(tiling)
            num_tiles = tiling[0] * tiling[1]
            if self._use_thumbnail and num_tiles != 1:
                num_tiles += 1

            params.append(
                ImageTilingParamsV1(
                    media=media,
                    num_tiles=num_tiles,
                    num_embeddings=num_tiles * self._embeddings_per_image,
                    tiling=tiling,
                )
            )

        return params

    def augment_tiling(self, tiling: tuple[int, int]) -> tuple[int, int]:
        def num_tiles(tiling: tuple[int, int]) -> int:
            return tiling[0] * tiling[1]

        def plus_minus_one(tiling: tuple[int, int], minus_prob: float = 0.65) -> tuple[int, int]:
            if random.random() < minus_prob:
                # Minus one
                if tiling[0] == 1 and tiling[1] == 1:
                    return tiling
                elif tiling[0] == 1:
                    return (tiling[0], tiling[1] - 1)
                elif tiling[1] == 1:
                    return (tiling[0] - 1, tiling[1])
                else:
                    if random.random() < 0.5:
                        return (tiling[0] - 1, tiling[1])
                    else:
                        return (tiling[0], tiling[1] - 1)
            else:
                # Plus one
                if num_tiles(tiling) < self._max_num_tiles:
                    tiling0 = (tiling[0] + 1, tiling[1])
                    tiling1 = (tiling[0], tiling[1] + 1)
                    if num_tiles(tiling0) > self._max_num_tiles and num_tiles(tiling1) > self._max_num_tiles:
                        return tiling
                    elif num_tiles(tiling0) > self._max_num_tiles:
                        return tiling1
                    elif num_tiles(tiling1) > self._max_num_tiles:
                        return tiling0
                    else:
                        if random.random() < 0.5:
                            return tiling0
                        else:
                            return tiling1
                return tiling

        new_tiling = plus_minus_one(tiling)
        return new_tiling

    def __str__(self):
        return f"TilingImageTransform(vision_model_type={self._vision_model_type}, tile_size={self._tile_size}, use_thumbnail={self._use_thumbnail}, min_num_tiles={self._min_num_tiles}, max_num_tiles={self._max_num_tiles}, embeddings_per_tile={self._embeddings_per_image}, find_closest_aspect_ratio_fn={self._find_closest_aspect_ratio_fn})"


class TileDegradationStrategy(ImageTilingStrategy):
    """Strategy for tiling images and video frames, each with their own tiling strategy, while trying to match the
    number of tokens left in the sample by reducing the number of tiles if needed.
    """

    # Based on https://github.com/openai/CLIP/blob/dcba3cb2e2827b402d2701e7e1c7d9fed8a20ef1/clip/clip.py#L79
    # and https://github.com/OpenGVLab/InternVL/blob/aa521e6eb1df4cf153aa4118fcf13e673c055d46/internvl_chat/internvl/train/dataset.py#L276

    def __init__(
        self,
        image_strategy: ImageTilingStrategy,
        video_frame_strategy: ImageTilingStrategy,
        embeddings_per_tile: int,
        max_num_tiles: int,
        tile_degradation_map: dict[int, int] = {12: 8, 8: 6, 6: 4, 4: 2, 2: 1},
    ):
        self._image_strategy = image_strategy
        self._video_frame_strategy = video_frame_strategy
        self._embeddings_per_tile = embeddings_per_tile
        self._max_num_tiles = max_num_tiles
        self._tile_degradation_map = tile_degradation_map

    def apply_params(self, transform_media: ImageTilingParams, **kwargs) -> list[torch.Tensor]:
        if isinstance(transform_media.media, ImageMedia):
            return self._image_strategy.apply_params(transform_media, **kwargs)
        elif isinstance(transform_media.media, VideoFrameMedia):
            return self._video_frame_strategy.apply_params(transform_media, **kwargs)
        else:
            raise ValueError(f"Unsupported media type: {type(transform_media.media)}")

    def compute_params(
        self,
        media_list: list[ImageMedia | VideoFrameMedia],
        num_tokens_available: int | None = None,
        max_num_tiles: int | None = None,
        **kwargs,
    ) -> list[ImageTilingParams]:
        # Use provided max_num_tiles or fall back to instance's max_num_tiles
        effective_max_num_tiles = max_num_tiles if max_num_tiles is not None else self._max_num_tiles
        max_num_tiles_to_use = effective_max_num_tiles
        degradation_map = self._tile_degradation_map

        while True:
            params = []
            img_num_tiles = []
            for media in media_list:
                if isinstance(media, ImageMedia):
                    media_params = self._image_strategy.compute_params(
                        [media], max_num_tiles_to_use * self._embeddings_per_tile, max_num_tiles_to_use, **kwargs
                    )[0]
                elif isinstance(media, VideoFrameMedia):
                    max_num_tiles_to_use = 1
                    media_params = self._video_frame_strategy.compute_params(
                        [media], max_num_tiles_to_use * self._embeddings_per_tile, max_num_tiles_to_use, **kwargs
                    )[0]
                else:
                    raise ValueError(f"Unsupported media type: {type(media)}")
                img_num_tiles.append(media_params.num_tiles)
                params.append(media_params)
            if max_num_tiles_to_use == 1 or num_tokens_available is None:
                break
            if sum(img_num_tiles) * self._embeddings_per_tile > num_tokens_available:
                if max_num_tiles_to_use in degradation_map:
                    max_num_tiles_to_use = degradation_map[max_num_tiles_to_use]
                else:
                    # End of degradation
                    break
            else:
                break
        return params

    def stack(
        self, images: list[torch.Tensor]
    ) -> tuple[torch.Tensor, list[tuple[int, int]], list[int] | None, list[int] | None]:
        return self._image_strategy.stack(images)

    def __str__(self):
        return f"TileDegradationImageTransform(max_num_tiles={self._max_num_tiles}, image_transform={self._image_strategy}, video_frame_transform={self._video_frame_strategy})"


@dataclass
class DynamicResolutionParams(ImageTilingParams):
    patch_size: tuple[int, int]


class DynamicResolutionImageTilingStrategy(ImageTilingStrategy):
    """Preprocess an image with dynamic resolution for vision transformers.

    This function resizes an image to optimize the number of patches while respecting
    constraints on minimum/maximum patches, minimum side length, and compatibility
    with pixel shuffle or convolution merging operations.

    The algorithm works by:
    1. Computing the initial patch grid size based on the image dimensions and res_step
    2. Scaling the patch grid to fit within the max_patches constraint
    3. Ensuring the result has at least min_patches
    4. Optionally enforcing a minimum side length constraint
    5. Rounding patch dimensions to even numbers for pixel_shuffle/conv_merging compatibility
    6. Resizing the image to the computed target dimensions

    Note:
        The function preserves aspect ratio as much as possible while satisfying all constraints.
        When constraints conflict (e.g., min_side vs max_patches), the function prioritizes
        staying within max_patches while maximizing the image size.

    Example:
        >>> from PIL import Image
        >>> img = Image.open("example.jpg")  # 800x600 image
        >>> strategy = DynamicResolutionImageTilingStrategy(vision_model_type="radio", min_patches=4, max_patches=64, res_step=14, get_num_embeddings=lambda x, y: x * y * 2)
        >>> params = strategy.compute_params([img])
        >>> img_tensor = strategy.apply_params(params[0])
        >>> # Returns image resized to maintain aspect ratio with 4-64 patches of size 14x14
    """

    def __init__(
        self,
        vision_model_type: str,
        min_num_patches: int,
        patch_size: int,
        get_num_embeddings: Callable[[int, int], int],
        factor_max: float = 1.0,
        pixel_shuffle: bool = False,
        min_side: int | None = None,
        conv_merging: bool = False,
        use_thumbnail: bool = False,
        thumbnail_size: int = 448,
        thumbnail_area_threshold: float = 0.8,
        max_num_patches: int = 0,
        apply_data_augment: bool = False,
    ):
        """
        Args:
            vision_model_type: Vision model type.
            min_num_patches: Minimum number of patches required. Defaults to 1.
            max_num_patches: Maximum number of patches allowed. Defaults to 0 (no maximum).
            patch_size: Resolution step size (patch dimension). Defaults to 16.
            get_num_embeddings: Function to get the number of embeddings from the patch size (width, height).
            factor_max: Maximum scaling factor to apply. Defaults to 1.0.
            pixel_shuffle: Whether to ensure compatibility with pixel shuffle operations by rounding to even patch
                dimensions. Defaults to False.
            min_side: Minimum side length in pixels. If specified, ensures at least one side meets this constraint.
                Defaults to None.
            conv_merging: Whether to ensure compatibility with convolution merging by rounding to even patch dimensions.
                Defaults to False.
            use_thumbnail: Whether to add a thumbnail image when processing. Defaults to False.
            thumbnail_size: Size of the thumbnail image (width and height). Defaults to 448.
            thumbnail_area_threshold: Maximum area percentage (0.0-1.0) of the resized image relative to thumbnail area
                for which to add a thumbnail. If the resized image area is larger than this threshold of the thumbnail
                area, no thumbnail will be added. Defaults to 0.8 (80%).
            apply_data_augment: Whether to apply data augmentation to the image. Defaults to False.
        """
        assert "radio" in vision_model_type, (
            "Dynamic resolution is only supported for radio models"
        )
        self._vision_model_type = vision_model_type
        self._min_num_patches = min_num_patches
        self._max_num_patches = max_num_patches if max_num_patches > 0 else float("inf")
        self._patch_size = patch_size
        self._get_num_embeddings = get_num_embeddings
        self._factor_max = factor_max
        self._pixel_shuffle = pixel_shuffle
        self._min_side = min_side
        self._conv_merging = conv_merging
        self._use_thumbnail = use_thumbnail
        self._thumbnail_size = thumbnail_size
        self._thumbnail_area_threshold = thumbnail_area_threshold
        pixel_mean, pixel_std = pixel_statistics[self._vision_model_type]
        self._transform = T.Compose(
            [
                T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
                T.ToTensor(), #T.Lambda(lambda img: _fast_to_tensor(img)),
                T.Normalize(mean=pixel_mean, std=pixel_std),
            ]
        )
        self._apply_data_augment = apply_data_augment

    def apply_params(self, params: DynamicResolutionParams, **kwargs) -> list[torch.Tensor]:
        # resize the image
        resized_img = params.media.value.resize(
            (
                params.patch_size[0] * self._patch_size,
                params.patch_size[1] * self._patch_size,
            )
        )
        processed_images = [resized_img]
        
        # Add thumbnail if enabled and image area is below threshold
        if self._use_thumbnail:
            # Calculate areas
            resized_area = resized_img.size[0] * resized_img.size[1]
            thumbnail_area = self._thumbnail_size * self._thumbnail_size
            area_ratio = resized_area / thumbnail_area
            
            # Only add thumbnail if resized image area is less than threshold % of thumbnail area
            if area_ratio < self._thumbnail_area_threshold:
                thumbnail_img = params.media.value.resize((self._thumbnail_size, self._thumbnail_size))
                processed_images.append(thumbnail_img)
        
        return [self._transform(img) for img in processed_images]

    def process_media(
        self,
        media: ImageMedia | VideoFrameMedia,
        num_tokens_available: int,
        data_augment: bool = False,
        tiling_augment_prob: float = 0.4,
    ) -> DynamicResolutionParams:
        """Process a single media item and return its parameters.
        
        Args:
            media: The media item to process
            num_tokens_available: Number of tokens available for this media
            data_augment: Whether to apply data augmentation to the image. Defaults to False.
        Returns:
            DynamicResolutionParams for the media
        """
        current_num_tokens_available = num_tokens_available
        if isinstance(media, ImageMedia):
            orig_width, orig_height = media.width, media.height
        elif isinstance(media, VideoFrameMedia):
            orig_width, orig_height = media.video_width, media.video_height
            # current_num_tokens_available = 1024 #TEMP: hack for video
        else:
            raise ValueError(f"Unsupported media type: {type(media)}")

        closest_patch_height = round(orig_height / self._patch_size + 0.5)
        closest_patch_width = round(orig_width / self._patch_size + 0.5)
        patches = closest_patch_height * closest_patch_width

        factor = min(math.sqrt(current_num_tokens_available / patches), self._factor_max)
        target_patch_height = math.floor(factor * closest_patch_height)
        target_patch_width = math.floor(factor * closest_patch_width)

        # We only consider self._min_num_patches if it is greater than current_num_tokens_available.
        if current_num_tokens_available > self._min_num_patches and target_patch_height * target_patch_width < self._min_num_patches:
            up_factor = math.sqrt(
                self._min_num_patches / (target_patch_height * target_patch_width)
            )
            target_patch_height = math.ceil(up_factor * target_patch_height)
            target_patch_width = math.ceil(up_factor * target_patch_width)

        if (
            self._min_side is not None
            and min(target_patch_width, target_patch_height) * self._patch_size
            < self._min_side
        ):
            if target_patch_width <= target_patch_height:
                up_factor = self._min_side / (target_patch_width * self._patch_size)
                new_patch_height = math.ceil(up_factor * target_patch_height)
                new_patch_width = math.ceil(up_factor * target_patch_width)

                if new_patch_height * new_patch_width > current_num_tokens_available:
                    # If only one side can be min_side, make as big as possible at native aspect ratio while staying below max_patches
                    if (
                        max(current_num_tokens_available // new_patch_width, 1)
                        * self._patch_size
                        < self._min_side
                    ):
                        up_factor = math.sqrt(
                            current_num_tokens_available
                            / (target_patch_height * target_patch_width)
                        )
                        target_patch_height = math.floor(
                            up_factor * target_patch_height
                        )
                        target_patch_width = math.floor(
                            up_factor * target_patch_width
                        )
                    target_patch_width = new_patch_width
                    target_patch_height = max(
                        current_num_tokens_available // new_patch_width, 1
                    )
                else:
                    target_patch_height = new_patch_height
                    target_patch_width = new_patch_width
            else:
                up_factor = self._min_side / (
                    target_patch_height * self._patch_size
                )
                new_patch_height = math.ceil(up_factor * target_patch_height)
                new_patch_width = math.ceil(up_factor * target_patch_width)

                if new_patch_height * new_patch_width > current_num_tokens_available:
                    # If only one side can be min_side, make as big as possible at native aspect ratio while staying below max_patches
                    if (
                        max(current_num_tokens_available // new_patch_height, 1)
                        * self._patch_size
                        < self._min_side
                    ):
                        up_factor = math.sqrt(
                            current_num_tokens_available
                            / (target_patch_height * target_patch_width)
                        )
                        target_patch_height = math.floor(
                            up_factor * target_patch_height
                        )
                        target_patch_width = math.floor(
                            up_factor * target_patch_width
                        )
                    else:
                        target_patch_height = new_patch_height
                        target_patch_width = max(
                            current_num_tokens_available // new_patch_height, 1
                        )
                else:
                    target_patch_height = new_patch_height
                    target_patch_width = new_patch_width

        # Round patch grid to be divisible by 2 (pixel-shuffle OR conv-merging)
        # or by 4 when BOTH are enabled (two successive 2x reductions)
        if self._pixel_shuffle or self._conv_merging:
            required_divisor = 4 if (self._pixel_shuffle and self._conv_merging) else 2

            rem_h = target_patch_height % required_divisor
            if rem_h != 0:
                inc_h = required_divisor - rem_h
                if (target_patch_height + inc_h) * target_patch_width <= current_num_tokens_available:
                    target_patch_height += inc_h
                else:
                    target_patch_height = max(required_divisor, target_patch_height - rem_h)

            rem_w = target_patch_width % required_divisor
            if rem_w != 0:
                inc_w = required_divisor - rem_w
                if target_patch_height * (target_patch_width + inc_w) <= current_num_tokens_available:
                    target_patch_width += inc_w
                else:
                    target_patch_width = max(required_divisor, target_patch_width - rem_w)

        if data_augment and self._apply_data_augment and random.random() < tiling_augment_prob:
            target_patch_width, target_patch_height = self.augment_resolution(target_patch_width, target_patch_height, current_num_tokens_available)

        #TEMP: hack for video
        if isinstance(media, VideoFrameMedia):
            target_patch_width = 32
            target_patch_height = 32

        # Calculate embeddings for the main dynamic resolution image
        num_embeddings = self._get_num_embeddings(
            target_patch_width * self._patch_size,
            target_patch_height * self._patch_size,
        )

        token_count = target_patch_width * target_patch_height
        
        # Add thumbnail embeddings if enabled and image area is below threshold
        num_tiles = 1  # Base dynamic resolution image
        if self._use_thumbnail:
            # Calculate areas
            resized_area = (target_patch_width * self._patch_size) * (target_patch_height * self._patch_size)
            thumbnail_area = self._thumbnail_size * self._thumbnail_size
            area_ratio = resized_area / thumbnail_area
            
            # Only add thumbnail if resized image area is less than threshold % of thumbnail area
            if area_ratio < self._thumbnail_area_threshold:
                num_tiles += 1  # Add 1 for thumbnail
                # Add embeddings for thumbnail (thumbnail_size x thumbnail_size)
                num_embeddings += self._get_num_embeddings(self._thumbnail_size, self._thumbnail_size)
                token_count += self._thumbnail_size // self._patch_size * self._thumbnail_size // self._patch_size

        return DynamicResolutionParams(
            media=media,
            num_tiles=num_tiles,
            num_embeddings=num_embeddings,
            patch_size=(target_patch_width, target_patch_height),
        ), token_count
    
    def augment_resolution(self, target_patch_width: int, target_patch_height: int, current_num_tokens_available: int) -> tuple[int, int]:

        min_num_patch_one_side = 32

        if random.random() < 0.5:
            # Minus one
            if target_patch_width <= min_num_patch_one_side and target_patch_height <= min_num_patch_one_side:
                return target_patch_width, target_patch_height
            elif target_patch_width <= min_num_patch_one_side:
                return target_patch_width, target_patch_height - min_num_patch_one_side
            elif target_patch_height <= min_num_patch_one_side:
                return target_patch_width - min_num_patch_one_side, target_patch_height
            else:
                if random.random() < 0.5:
                    return target_patch_width - min_num_patch_one_side, target_patch_height
                else:
                    return target_patch_width, target_patch_height - min_num_patch_one_side
        else:
            # Plus one
            if target_patch_width * target_patch_height < current_num_tokens_available:
                if random.random() < 0.5:
                    return target_patch_width + min_num_patch_one_side, target_patch_height
                else:
                    return target_patch_width, target_patch_height + min_num_patch_one_side
            return target_patch_width, target_patch_height

    def compute_params(
        self,
        media_list: list[ImageMedia | VideoFrameMedia],
        num_tokens_available: int | None = None,
        max_num_tiles: int | None = None,
        data_augment: bool = False,
        **kwargs,
    ) -> list[ImageTilingParams]:
        """Compute parameters for all media with iterative token budgeting.
        
        Args:
            media_list: List of media items to process
            num_tokens_available: Total number of tokens available across all media
            max_num_tiles: Maximum number of tiles (unused in this implementation)
            data_augment: Whether to apply data augmentation to the image. Defaults to False.
        Returns:
            List of ImageTilingParams for each media item
        """
        num_tokens_available = num_tokens_available * (4 if self._pixel_shuffle else 1) * (4 if self._conv_merging else 1)
        # When the number of available token is too small, allow self._min_num_patches per media and
        # let the sample be truncated.
        num_tokens_available = max(num_tokens_available, self._min_num_patches * len(media_list))

        # Clip the number of tokens available per media to be between min and max patches.
        num_tokens_available_per_media = [
            max(min(num_tokens_available, self._max_num_patches), self._min_num_patches)
            for _ in range(len(media_list))]

        # In theory this could be a while True loop, but in case the process_media method slightly
        # changes, I want to make sure we don't get stuck in an infinite loop.
        for _ in range(10):
            # Step 1: Process each media with current token budget
            params = []
            token_counts = []
            
            for media, tokens_for_media in zip(media_list, num_tokens_available_per_media):
                param, token_count = self.process_media(media, tokens_for_media, data_augment=data_augment)
                params.append(param)
                token_counts.append(token_count)
            
            # Step 2: Check if total tokens is within budget
            total_tokens = sum(token_counts)
            
            if total_tokens <= num_tokens_available:
                # We're within budget, return the params
                return params
            
            # Step 3: We're over budget, need to scale down
            # Calculate scaling factor to get under budget
            scaling_factor = num_tokens_available / total_tokens
            
            # Recalculate token budgets for each media based on scaling
            # Each media gets a proportional share of the total budget
            scaled_down_num_tokens_available_per_media = [
                max(self._min_num_patches, int(token_count * scaling_factor))
                for token_count in token_counts
            ]
            scaled_down = any([
                scaled_down_num_tokens_available_per_media[i] < num_tokens_available_per_media[i]
                for i in range(len(num_tokens_available_per_media))])
            # If there was not scaling down, we're stuck just use min_num_patches per media, else
            # try with the scaled down num_tokens_available_per_media.
            if not scaled_down:
                num_tokens_available_per_media = [self._min_num_patches] * len(media_list)
            else:
                num_tokens_available_per_media = scaled_down_num_tokens_available_per_media
        return params

    def stack(
        self, images: list[torch.Tensor]
    ) -> tuple[torch.Tensor, list[tuple[int, int]], list[int] | None, list[int] | None]:
        imgs_sizes = torch.tensor(
            [[img.shape[1], img.shape[2]] for img in images], dtype=torch.int32
        )

        def rearrange_img(x):
            py = x.shape[-2] // self._patch_size
            px = x.shape[-1] // self._patch_size
            x = einops.rearrange(
                x,
                "c (py yy) (px xx) -> (py px) (c yy xx)",
                py=py,
                yy=self._patch_size,
                px=px,
                xx=self._patch_size,
            )
            return x

        if len(images) > 0:
            imgs = [rearrange_img(img) for img in images]

            current_length = 0
            max_length = 0
            vision_cu_lengths = [0]
            for img in imgs:
                if max_length < img.shape[0]:
                    max_length = img.shape[0]
                current_length += img.shape[0]
                vision_cu_lengths.append(current_length)

            vision_cu_lengths = torch.tensor(vision_cu_lengths, dtype=torch.int32)
            vision_max_lengths = torch.tensor(max_length, dtype=torch.int32)

            return (
                torch.cat(imgs, dim=0).unsqueeze(0),
                imgs_sizes,
                vision_cu_lengths,
                vision_max_lengths,
            )
        else:
            return (
                torch.tensor([[0]], dtype=torch.float32),
                torch.tensor([[0,0]], dtype=torch.int32),
                None,
                None,
            )

    def __str__(self):
        return f"DynamicResolutionImageTransform(vision_model_type={self._vision_model_type}, min_num_patches={self._min_num_patches}, patch_size={self._patch_size}, pixel_shuffle={self._pixel_shuffle}, conv_merging={self._conv_merging}, use_thumbnail={self._use_thumbnail}, thumbnail_size={self._thumbnail_size}, thumbnail_area_threshold={self._thumbnail_area_threshold})"


@dataclass
class MatchTilingDynamicResolutionParams(ImageTilingParams):
    tiling: tuple[int, int]


class MatchTilingDynamicResolutionStrategy(ImageTilingStrategy):
    """
    Strategy that uses tiling logic to determine optimal image dimensions but processes
    the image as a single dynamic resolution image instead of splitting into tiles.
    
    This combines the aspect ratio optimization from ImageTilingStrategyV1 with the
    dynamic resolution processing from DynamicResolutionImageTilingStrategy.
    
    Also includes tile degradation logic similar to TileDegradationStrategy.
    """
    
    def __init__(
        self,
        vision_model_type: str,
        tile_size: int,
        use_thumbnail: bool,
        min_num_tiles: int,
        max_num_tiles: int,
        embeddings_per_tile: int,
        patch_size: int,
        get_num_embeddings: Callable[[int, int], int],
        find_closest_aspect_ratio_fn=find_closest_aspect_ratio,
        pixel_shuffle: bool = False,
        conv_merging: bool = False,
        tile_degradation_map: dict[int, int] = None,
        video_frame_strategy: ImageTilingStrategy = None,
        enable_tile_degradation: bool = True,
    ):
        """
        Args:
            vision_model_type: Vision model type (should support dynamic resolution)
            tile_size: Size of each tile for tiling calculation
            use_thumbnail: Whether tiling logic should include thumbnail
            min_num_tiles: Minimum number of tiles for tiling calculation
            max_num_tiles: Maximum number of tiles for tiling calculation
            embeddings_per_tile: Embeddings per tile for tiling calculation
            patch_size: Patch size for dynamic resolution processing
            get_num_embeddings: Function to get number of embeddings from dimensions
            find_closest_aspect_ratio_fn: Function to find closest aspect ratio
            pixel_shuffle: Whether to ensure compatibility with pixel shuffle
            conv_merging: Whether to ensure compatibility with convolution merging
            tile_degradation_map: Map for degrading tiles when tokens are insufficient
            video_frame_strategy: Strategy for processing video frames
            enable_tile_degradation: Whether to enable tile degradation (default: True)
        """
        assert "radio" in vision_model_type, (
            "MatchTilingDynamicResolution is only supported for radio models"
        )
        
        self._vision_model_type = vision_model_type
        self._tile_size = tile_size
        self._use_thumbnail = use_thumbnail
        self._min_num_tiles = min_num_tiles
        self._max_num_tiles = max_num_tiles
        self._embeddings_per_tile = embeddings_per_tile
        self._patch_size = patch_size
        self._get_num_embeddings = get_num_embeddings
        self._find_closest_aspect_ratio_fn = find_closest_aspect_ratio_fn
        self._pixel_shuffle = pixel_shuffle
        self._conv_merging = conv_merging
        self._enable_tile_degradation = enable_tile_degradation
        
        # Tile degradation logic (similar to TileDegradationStrategy)
        if tile_degradation_map is None:
            self._tile_degradation_map = {12: 8, 8: 6, 6: 4, 4: 2, 2: 1}
        else:
            self._tile_degradation_map = tile_degradation_map
        
        # Video frame strategy (similar to TileDegradationStrategy)
        if video_frame_strategy is None:
            self._video_frame_strategy = NoTilingStrategy(
                vision_model_type=vision_model_type,
                target_width=tile_size,
                target_height=tile_size,
                embeddings_per_image=embeddings_per_tile,
            )
        else:
            self._video_frame_strategy = video_frame_strategy
        
        # Calculate all possible aspect ratios for each max_num_tiles (borrowed from ImageTilingStrategyV1)
        self.target_ratios = {
            max_num_tiles: sorted(
                set(
                    (x, y)
                    for n in range(self._min_num_tiles, max_num_tiles + 1)
                    for x in range(1, n + 1)
                    for y in range(1, n + 1)
                    if x * y <= max_num_tiles and x * y >= self._min_num_tiles
                ),
                key=lambda x: x[0] * x[1],
            )
            for max_num_tiles in range(self._min_num_tiles, self._max_num_tiles + 1)
        }
        
        # Set up transform for dynamic resolution processing
        pixel_mean, pixel_std = pixel_statistics[self._vision_model_type]
        self._transform = T.Compose(
            [
                T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
                T.ToTensor(),
                T.Normalize(mean=pixel_mean, std=pixel_std),
            ]
        )

    def apply_params(self, params: MatchTilingDynamicResolutionParams, **kwargs) -> list[torch.Tensor]:
        # Handle video frames using the video frame strategy
        if isinstance(params.media, VideoFrameMedia):
            return self._video_frame_strategy.apply_params(params, **kwargs)
        
        # Handle images with dynamic resolution processing
        image = params.media.value
        # Calculate the target width and height (same logic as ImageTilingStrategyV1)
        target_width = self._tile_size * params.tiling[0]
        target_height = self._tile_size * params.tiling[1]
        
        # Resize the image to the target dimensions (same as ImageTilingStrategyV1)
        resized_img = image.resize((target_width, target_height))
        
        # Process as single dynamic resolution image
        processed_images = [resized_img]
        
        # Add thumbnail if use_thumbnail=True and there's more than 1 tile (same as ImageTilingStrategyV1)
        blocks = params.tiling[0] * params.tiling[1]
        if self._use_thumbnail and blocks != 1:
            thumbnail_img = image.resize((self._tile_size, self._tile_size))
            processed_images.append(thumbnail_img)
        
        return [self._transform(img) for img in processed_images]

    def compute_params(
        self,
        media_list: list[ImageMedia | VideoFrameMedia],
        num_tokens_available: int | None = None,
        max_num_tiles: int | None = None,
        **kwargs,
    ) -> list[MatchTilingDynamicResolutionParams]:
        # Implement tile degradation logic similar to TileDegradationStrategy
        # Use provided max_num_tiles or fall back to instance's max_num_tiles
        # Clamp to self._max_num_tiles since target_ratios are only pre-computed up to that value
        effective_max_num_tiles = max_num_tiles if max_num_tiles is not None else self._max_num_tiles
        effective_max_num_tiles = min(effective_max_num_tiles, self._max_num_tiles)
        max_num_tiles_to_use = effective_max_num_tiles
        degradation_map = self._tile_degradation_map
        
        while True:
            params = []
            total_embeddings_needed = 0
            
            for media in media_list:
                if isinstance(media, ImageMedia):
                    # Use tiling logic for images
                    img_size = (media.width, media.height)
                    aspect_ratio = img_size[0] / img_size[1]
                    
                    # Find the closest aspect ratio to the target
                    target_ratios = self.target_ratios[max_num_tiles_to_use]
                    tiling = self._find_closest_aspect_ratio_fn(
                        aspect_ratio, target_ratios, img_size[0], img_size[1], self._tile_size
                    )
                    
                    # Calculate target dimensions for dynamic resolution processing
                    target_width = self._tile_size * tiling[0]
                    target_height = self._tile_size * tiling[1]
                    num_embeddings = self._get_num_embeddings(target_width, target_height)
                    
                    # Account for thumbnail (same logic as ImageTilingStrategyV1)
                    num_tiles = 1  # Base dynamic resolution image
                    blocks = tiling[0] * tiling[1]
                    if self._use_thumbnail and blocks != 1:
                        num_tiles += 1  # Add 1 for thumbnail
                        # Add embeddings for thumbnail (tile_size x tile_size)
                        num_embeddings += self._get_num_embeddings(self._tile_size, self._tile_size)
                    
                    media_params = MatchTilingDynamicResolutionParams(
                        media=media,
                        num_tiles=num_tiles,
                        num_embeddings=num_embeddings,
                        tiling=tiling,
                    )
                elif isinstance(media, VideoFrameMedia):
                    # Use video frame strategy for video frames (always 1 tile)
                    video_params = self._video_frame_strategy.compute_params(
                        [media], 1 * self._embeddings_per_tile
                    )[0]
                    media_params = MatchTilingDynamicResolutionParams(
                        media=media,
                        num_tiles=video_params.num_tiles,
                        num_embeddings=video_params.num_embeddings,
                        tiling=(1, 1),  # Video frames always use 1x1 tiling
                    )
                else:
                    raise ValueError(f"Unsupported media type: {type(media)}")
                
                params.append(media_params)
                total_embeddings_needed += media_params.num_embeddings
            
            # Check if we need to degrade (only if degradation is enabled)
            if not self._enable_tile_degradation:
                break
            if max_num_tiles_to_use == 1 or num_tokens_available is None:
                break
            if total_embeddings_needed > num_tokens_available:
                if max_num_tiles_to_use in degradation_map:
                    max_num_tiles_to_use = degradation_map[max_num_tiles_to_use]
                    # Recalculate target ratios for the new max_num_tiles_to_use
                    if max_num_tiles_to_use not in self.target_ratios:
                        self.target_ratios[max_num_tiles_to_use] = sorted(
                            set(
                                (x, y)
                                for n in range(self._min_num_tiles, max_num_tiles_to_use + 1)
                                for x in range(1, n + 1)
                                for y in range(1, n + 1)
                                if x * y <= max_num_tiles_to_use and x * y >= self._min_num_tiles
                            ),
                            key=lambda x: x[0] * x[1],
                        )
                else:
                    # End of degradation
                    break
            else:
                break
        
        return params

    def stack(
        self, images: list[torch.Tensor]
    ) -> tuple[torch.Tensor, list[tuple[int, int]], list[int], list[int]]:
        """Stack images using dynamic resolution approach with sequence packing"""
        imgs_sizes = torch.tensor(
            [[img.shape[1], img.shape[2]] for img in images], dtype=torch.int32
        )

        def rearrange_img(x):
            py = x.shape[-2] // self._patch_size
            px = x.shape[-1] // self._patch_size
            x = einops.rearrange(
                x,
                "c (py yy) (px xx) -> (py px) (c yy xx)",
                py=py,
                yy=self._patch_size,
                px=px,
                xx=self._patch_size,
            )
            return x

        if len(images) > 0:
            imgs = [rearrange_img(img) for img in images]

            current_length = 0
            max_length = 0
            vision_cu_lengths = [0]
            for img in imgs:
                if max_length < img.shape[0]:
                    max_length = img.shape[0]
                current_length += img.shape[0]
                vision_cu_lengths.append(current_length)

            vision_cu_lengths = torch.tensor(vision_cu_lengths, dtype=torch.int32)
            vision_max_lengths = torch.tensor(max_length, dtype=torch.int32)

            return (
                torch.cat(imgs, dim=0).unsqueeze(0),
                imgs_sizes,
                vision_cu_lengths,
                vision_max_lengths,
            )
        else:
            return (
                torch.tensor([[0]], dtype=torch.float32),
                torch.tensor([[0,0]], dtype=torch.int32),
                None,
                None,
            )

    def __str__(self):
        return f"MatchTilingDynamicResolutionStrategy(vision_model_type={self._vision_model_type}, tile_size={self._tile_size}, use_thumbnail={self._use_thumbnail}, min_num_tiles={self._min_num_tiles}, max_num_tiles={self._max_num_tiles}, patch_size={self._patch_size}, pixel_shuffle={self._pixel_shuffle}, conv_merging={self._conv_merging}, enable_tile_degradation={self._enable_tile_degradation}, video_frame_strategy={self._video_frame_strategy})"


@dataclass
class MaskedTilingDynamicResolutionParams(ImageTilingParams):
    tiling: tuple[int, int]


class MaskedTilingDynamicResolutionStrategy(ImageTilingStrategy):
    """
    Like MatchTilingDynamicResolutionStrategy, but ensures tiles are isolated in the
    vision encoder by emitting per-tile packed samples (block-diagonal attention across tiles).
    """

    def __init__(
        self,
        vision_model_type: str,
        tile_size: int,
        use_thumbnail: bool,
        min_num_tiles: int,
        max_num_tiles: int,
        embeddings_per_tile: int,
        patch_size: int,
        get_num_embeddings: Callable[[int, int], int],
        find_closest_aspect_ratio_fn=find_closest_aspect_ratio,
        pixel_shuffle: bool = False,
        conv_merging: bool = False,
        tile_degradation_map: dict[int, int] = None,
        video_frame_strategy: ImageTilingStrategy = None,
        enable_tile_degradation: bool = True,
    ):
        assert "radio" in vision_model_type, (
            "MaskedTilingDynamicResolution is only supported for radio models"
        )

        self._vision_model_type = vision_model_type
        self._tile_size = tile_size
        self._use_thumbnail = use_thumbnail
        self._min_num_tiles = min_num_tiles
        self._max_num_tiles = max_num_tiles
        self._embeddings_per_tile = embeddings_per_tile
        self._patch_size = patch_size
        self._get_num_embeddings = get_num_embeddings
        self._find_closest_aspect_ratio_fn = find_closest_aspect_ratio_fn
        self._pixel_shuffle = pixel_shuffle
        self._conv_merging = conv_merging
        self._enable_tile_degradation = enable_tile_degradation

        if tile_degradation_map is None:
            self._tile_degradation_map = {12: 8, 8: 6, 6: 4, 4: 2, 2: 1}
        else:
            self._tile_degradation_map = tile_degradation_map

        if video_frame_strategy is None:
            self._video_frame_strategy = NoTilingStrategy(
                vision_model_type=vision_model_type,
                target_width=tile_size,
                target_height=tile_size,
                embeddings_per_image=embeddings_per_tile,
            )
        else:
            self._video_frame_strategy = video_frame_strategy

        self.target_ratios = {
            max_num_tiles: sorted(
                set(
                    (x, y)
                    for n in range(self._min_num_tiles, max_num_tiles + 1)
                    for x in range(1, n + 1)
                    for y in range(1, n + 1)
                    if x * y <= max_num_tiles and x * y >= self._min_num_tiles
                ),
                key=lambda x: x[0] * x[1],
            )
            for max_num_tiles in range(self._min_num_tiles, self._max_num_tiles + 1)
        }

        pixel_mean, pixel_std = pixel_statistics[self._vision_model_type]
        self._transform = T.Compose(
            [
                T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
                T.ToTensor(),
                T.Normalize(mean=pixel_mean, std=pixel_std),
            ]
        )

    def apply_params(self, params: MaskedTilingDynamicResolutionParams, **kwargs) -> list[torch.Tensor]:
        # Handle video frames using the video frame strategy
        if isinstance(params.media, VideoFrameMedia):
            return self._video_frame_strategy.apply_params(params, **kwargs)

        image = params.media.value
        nx, ny = params.tiling
        target_width = self._tile_size * nx
        target_height = self._tile_size * ny

        resized_img = image.resize((target_width, target_height))

        processed_images = []
        # Emit per-tile images (each becomes an isolated packed sample later)
        for j in range(ny):
            for i in range(nx):
                box = (
                    i * self._tile_size,
                    j * self._tile_size,
                    (i + 1) * self._tile_size,
                    (j + 1) * self._tile_size,
                )
                tile_img = resized_img.crop(box)
                processed_images.append(tile_img)

        if self._use_thumbnail and (nx * ny) != 1:
            thumbnail_img = image.resize((self._tile_size, self._tile_size))
            processed_images.append(thumbnail_img)

        return [self._transform(img) for img in processed_images]

    def compute_params(
        self,
        media_list: list[ImageMedia | VideoFrameMedia],
        num_tokens_available: int | None = None,
        max_num_tiles: int | None = None,
        data_augment: bool = False,
        tiling_augment_prob: float = 0.4,
        **kwargs,
    ) -> list[MaskedTilingDynamicResolutionParams]:
        effective_max_num_tiles = max_num_tiles if max_num_tiles is not None else self._max_num_tiles
        effective_max_num_tiles = min(effective_max_num_tiles, self._max_num_tiles)
        max_num_tiles_to_use = effective_max_num_tiles
        degradation_map = self._tile_degradation_map

        while True:
            params = []
            total_embeddings_needed = 0

            for media in media_list:
                if isinstance(media, ImageMedia):
                    img_size = (media.width, media.height)
                    aspect_ratio = img_size[0] / img_size[1]

                    target_ratios = self.target_ratios[max_num_tiles_to_use]
                    tiling = self._find_closest_aspect_ratio_fn(
                        aspect_ratio, target_ratios, img_size[0], img_size[1], self._tile_size
                    )
                    
                    # Apply tiling augmentation if enabled
                    if data_augment and isinstance(media, ImageMedia) and random.random() < tiling_augment_prob:
                        tiling = self.augment_tiling(tiling)

                    blocks = tiling[0] * tiling[1]
                    # Each tile is tile_size x tile_size
                    per_tile_emb = self._get_num_embeddings(self._tile_size, self._tile_size)
                    num_embeddings = blocks * per_tile_emb

                    num_tiles = blocks
                    if self._use_thumbnail and blocks != 1:
                        num_tiles += 1
                        num_embeddings += self._get_num_embeddings(self._tile_size, self._tile_size)

                    media_params = MaskedTilingDynamicResolutionParams(
                        media=media,
                        num_tiles=num_tiles,
                        num_embeddings=num_embeddings,
                        tiling=tiling,
                    )
                elif isinstance(media, VideoFrameMedia):
                    video_params = self._video_frame_strategy.compute_params(
                        [media], 1 * self._embeddings_per_tile
                    )[0]
                    media_params = MaskedTilingDynamicResolutionParams(
                        media=media,
                        num_tiles=video_params.num_tiles,
                        num_embeddings=video_params.num_embeddings,
                        tiling=(1, 1),
                    )
                else:
                    raise ValueError(f"Unsupported media type: {type(media)}")

                params.append(media_params)
                total_embeddings_needed += media_params.num_embeddings

            if not self._enable_tile_degradation:
                break
            if max_num_tiles_to_use == 1 or num_tokens_available is None:
                break
            if total_embeddings_needed > num_tokens_available:
                if max_num_tiles_to_use in degradation_map:
                    max_num_tiles_to_use = degradation_map[max_num_tiles_to_use]
                    if max_num_tiles_to_use not in self.target_ratios:
                        self.target_ratios[max_num_tiles_to_use] = sorted(
                            set(
                                (x, y)
                                for n in range(self._min_num_tiles, max_num_tiles_to_use + 1)
                                for x in range(1, n + 1)
                                for y in range(1, n + 1)
                                if x * y <= max_num_tiles_to_use and x * y >= self._min_num_tiles
                            ),
                            key=lambda x: x[0] * x[1],
                        )
                else:
                    break
            else:
                break

        return params

    def augment_tiling(self, tiling: tuple[int, int]) -> tuple[int, int]:
        def num_tiles(tiling: tuple[int, int]) -> int:
            return tiling[0] * tiling[1]

        def plus_minus_one(tiling: tuple[int, int], minus_prob: float = 0.65) -> tuple[int, int]:
            if random.random() < minus_prob:
                # Minus one
                if tiling[0] == 1 and tiling[1] == 1:
                    return tiling
                elif tiling[0] == 1:
                    return (tiling[0], tiling[1] - 1)
                elif tiling[1] == 1:
                    return (tiling[0] - 1, tiling[1])
                else:
                    if random.random() < 0.5:
                        return (tiling[0] - 1, tiling[1])
                    else:
                        return (tiling[0], tiling[1] - 1)
            else:
                # Plus one
                if num_tiles(tiling) < self._max_num_tiles:
                    tiling0 = (tiling[0] + 1, tiling[1])
                    tiling1 = (tiling[0], tiling[1] + 1)
                    if num_tiles(tiling0) > self._max_num_tiles and num_tiles(tiling1) > self._max_num_tiles:
                        return tiling
                    elif num_tiles(tiling0) > self._max_num_tiles:
                        return tiling1
                    elif num_tiles(tiling1) > self._max_num_tiles:
                        return tiling0
                    else:
                        if random.random() < 0.5:
                            return tiling0
                        else:
                            return tiling1
                return tiling

        new_tiling = plus_minus_one(tiling)
        return new_tiling

    def stack(
        self, images: list[torch.Tensor]
    ) -> tuple[torch.Tensor, list[tuple[int, int]], list[int], list[int]]:
        # Identical to dynamic resolution packing; each tile is already an independent image sample
        imgs_sizes = torch.tensor(
            [[img.shape[1], img.shape[2]] for img in images], dtype=torch.int32
        )

        def rearrange_img(x):
            py = x.shape[-2] // self._patch_size
            px = x.shape[-1] // self._patch_size
            x = einops.rearrange(
                x,
                "c (py yy) (px xx) -> (py px) (c yy xx)",
                py=py,
                yy=self._patch_size,
                px=px,
                xx=self._patch_size,
            )
            return x

        if len(images) > 0:
            imgs = [rearrange_img(img) for img in images]

            current_length = 0
            max_length = 0
            vision_cu_lengths = [0]
            for img in imgs:
                if max_length < img.shape[0]:
                    max_length = img.shape[0]
                current_length += img.shape[0]
                vision_cu_lengths.append(current_length)

            vision_cu_lengths = torch.tensor(vision_cu_lengths, dtype=torch.int32)
            vision_max_lengths = torch.tensor(max_length, dtype=torch.int32)

            return (
                torch.cat(imgs, dim=0).unsqueeze(0),
                imgs_sizes,
                vision_cu_lengths,
                vision_max_lengths,
            )
        else:
            return (
                torch.tensor([[0]], dtype=torch.float32),
                torch.tensor([[0,0]], dtype=torch.int32),
                None,
                None,
            )

    def __str__(self):
        return f"MaskedTilingDynamicResolutionStrategy(vision_model_type={self._vision_model_type}, tile_size={self._tile_size}, use_thumbnail={self._use_thumbnail}, min_num_tiles={self._min_num_tiles}, max_num_tiles={self._max_num_tiles}, patch_size={self._patch_size}, pixel_shuffle={self._pixel_shuffle}, conv_merging={self._conv_merging}, enable_tile_degradation={self._enable_tile_degradation}, video_frame_strategy={self._video_frame_strategy})"

def create_image_tiling_strategy(args):
    """
    Create an image tiling strategy based on the provided arguments.
    
    This function encapsulates the logic for creating the appropriate image tiling strategy
    based on the training/evaluation configuration. It can be used by both training (task_encoder)
    and evaluation code outside of data_loading/.
    
    Args:
        args: Arguments object with the following relevant attributes:
            - img_h, img_w: Image height and width
            - patch_dim: Patch dimension
            - vision_model_type: Vision model type (e.g., 'radio', 'clip', 'siglip')
            - disable_vision_class_token: Whether to disable vision class token
            - pixel_shuffle: Whether to use pixel shuffle
            - use_tile_tags: Whether to use tile tags
            - max_num_tiles: Maximum number of tiles
            - tokenizer_prompt_format: Tokenizer prompt format
            - image_break_token: Image break token (optional)
            - conv_merging: Whether to use convolution merging
            - dynamic_resolution: Whether to use dynamic resolution
            - match_tiling_dynamic_resolution: Whether to match tiling with dynamic resolution
            - use_area_weighted_aspect_ratio: Whether to use area-weighted aspect ratio
            - use_thumbnail: Whether to use thumbnail
            - dynamic_resolution_min_patches: Minimum number of patches for dynamic resolution
            - dynamic_resolution_min_side: Minimum side length for dynamic resolution (optional)
            - thumbnail_area_threshold: Thumbnail area threshold (optional)
            - use_tiling: Whether to use tiling
    
    Returns:
        ImageTilingStrategy: The created image tiling strategy
    """
    from megatron.core.models.vision.clip_vit_model import get_num_image_embeddings
    
    assert args.img_h == args.img_w, "img_h and img_w must be the same"
    
    match_tiling_dynamic_resolution = args.match_tiling_dynamic_resolution
    masked_tiling_dynamic_resolution = getattr(args, "masked_tiling_dynamic_resolution", False)
    dynamic_resolution = args.dynamic_resolution
    use_tiling = args.use_tiling
    use_area_weighted_aspect_ratio = args.use_area_weighted_aspect_ratio
    
    if match_tiling_dynamic_resolution:
        assert dynamic_resolution, "must enable --dynamic-resolution if using --match-tiling-dynamic-resolution"
        assert not use_tiling, "cannot use --use-tiling and --match-tiling-dynamic-resolution together"
    if masked_tiling_dynamic_resolution:
        assert dynamic_resolution, "must enable --dynamic-resolution if using --masked-tiling-dynamic-resolution"
        assert not use_tiling, "cannot use --use-tiling and --masked-tiling-dynamic-resolution together"
        assert not match_tiling_dynamic_resolution, "cannot combine --masked-tiling-dynamic-resolution with --match-tiling-dynamic-resolution"
    
    if dynamic_resolution:
        if masked_tiling_dynamic_resolution:
            num_image_embeddings_per_tile = get_num_image_embeddings(
                img_h=args.img_h,
                img_w=args.img_w,
                patch_dim=args.patch_dim,
                vision_model_type=args.vision_model_type,
                disable_vision_class_token=args.disable_vision_class_token,
                class_token_len=1,
                pixel_shuffle=args.pixel_shuffle,
                use_tile_tags=args.use_tile_tags,
                max_num_tiles=args.max_num_tiles,
                tokenizer_type=args.tokenizer_prompt_format,
                use_image_break_token=args.image_break_token is not None,
                conv_merging=args.conv_merging,
            )
            image_tiling_strategy = MaskedTilingDynamicResolutionStrategy(
                vision_model_type=args.vision_model_type,
                tile_size=args.img_h,
                use_thumbnail=args.use_thumbnail,
                min_num_tiles=1,
                max_num_tiles=args.max_num_tiles,
                embeddings_per_tile=num_image_embeddings_per_tile,
                patch_size=args.patch_dim,
                get_num_embeddings=lambda width, height: get_num_image_embeddings(
                    img_h=height,
                    img_w=width,
                    patch_dim=args.patch_dim,
                    vision_model_type=args.vision_model_type,
                    disable_vision_class_token=args.disable_vision_class_token,
                    class_token_len=1,
                    pixel_shuffle=args.pixel_shuffle,
                    use_tile_tags=args.use_tile_tags,
                    max_num_tiles=args.max_num_tiles,
                    tokenizer_type=args.tokenizer_prompt_format,
                    use_image_break_token=args.image_break_token is not None,
                    conv_merging=args.conv_merging,
                ),
                find_closest_aspect_ratio_fn=(
                    find_closest_area_weighted_aspect_ratio
                    if use_area_weighted_aspect_ratio
                    else find_closest_aspect_ratio
                ),
                pixel_shuffle=args.pixel_shuffle,
                conv_merging=args.conv_merging,
            )
        elif match_tiling_dynamic_resolution:
            num_image_embeddings_per_tile = get_num_image_embeddings(
                img_h=args.img_h,
                img_w=args.img_w,
                patch_dim=args.patch_dim,
                vision_model_type=args.vision_model_type,
                disable_vision_class_token=args.disable_vision_class_token,
                class_token_len=1,
                pixel_shuffle=args.pixel_shuffle,
                use_tile_tags=args.use_tile_tags,
                max_num_tiles=args.max_num_tiles,
                tokenizer_type=args.tokenizer_prompt_format,
                use_image_break_token=args.image_break_token is not None,
                conv_merging=args.conv_merging,
            )
            image_tiling_strategy = MatchTilingDynamicResolutionStrategy(
                vision_model_type=args.vision_model_type,
                tile_size=args.img_h,
                use_thumbnail=args.use_thumbnail,
                min_num_tiles=1,
                max_num_tiles=args.max_num_tiles,
                embeddings_per_tile=num_image_embeddings_per_tile,
                patch_size=args.patch_dim,
                get_num_embeddings=lambda width, height: get_num_image_embeddings(
                    img_h=height,
                    img_w=width,
                    patch_dim=args.patch_dim,
                    vision_model_type=args.vision_model_type,
                    disable_vision_class_token=args.disable_vision_class_token,
                    class_token_len=1,
                    pixel_shuffle=args.pixel_shuffle,
                    use_tile_tags=args.use_tile_tags,
                    max_num_tiles=args.max_num_tiles,
                    tokenizer_type=args.tokenizer_prompt_format,
                    use_image_break_token=args.image_break_token is not None,
                    conv_merging=args.conv_merging,
                ),
                find_closest_aspect_ratio_fn=(
                    find_closest_area_weighted_aspect_ratio
                    if use_area_weighted_aspect_ratio
                    else find_closest_aspect_ratio
                ),
                pixel_shuffle=args.pixel_shuffle,
                conv_merging=args.conv_merging,
            )
        else:
            image_tiling_strategy = DynamicResolutionImageTilingStrategy(
                vision_model_type=args.vision_model_type,
                min_num_patches=args.dynamic_resolution_min_patches,
                patch_size=args.patch_dim,
                get_num_embeddings=lambda width, height: get_num_image_embeddings(
                    img_h=height,
                    img_w=width,
                    patch_dim=args.patch_dim,
                    vision_model_type=args.vision_model_type,
                    disable_vision_class_token=args.disable_vision_class_token,
                    class_token_len=1,
                    pixel_shuffle=args.pixel_shuffle,
                    use_tile_tags=args.use_tile_tags,
                    max_num_tiles=args.max_num_tiles,
                    tokenizer_type=args.tokenizer_prompt_format,
                    use_image_break_token=args.image_break_token is not None,
                    conv_merging=args.conv_merging,
                ),
                pixel_shuffle=args.pixel_shuffle,
                min_side=args.dynamic_resolution_min_side,
                conv_merging=args.conv_merging,
                use_thumbnail=args.use_thumbnail,
                thumbnail_size=args.img_h,
                thumbnail_area_threshold=args.thumbnail_area_threshold,
                max_num_patches=args.dynamic_resolution_max_patches,
                apply_data_augment=args.apply_data_augment,
            )
    else:
        num_image_embeddings_per_tile = get_num_image_embeddings(
            img_h=args.img_h,
            img_w=args.img_w,
            patch_dim=args.patch_dim,
            vision_model_type=args.vision_model_type,
            disable_vision_class_token=args.disable_vision_class_token,
            class_token_len=1,
            pixel_shuffle=args.pixel_shuffle,
            use_tile_tags=args.use_tile_tags,
            max_num_tiles=args.max_num_tiles,
            tokenizer_type=args.tokenizer_prompt_format,
            use_image_break_token=args.image_break_token is not None,
            conv_merging=args.conv_merging,
        )
        if use_tiling:
            image_strategy = ImageTilingStrategyV1(
                vision_model_type=args.vision_model_type,
                tile_size=args.img_h,
                use_thumbnail=args.use_thumbnail,
                min_num_tiles=1,
                max_num_tiles=args.max_num_tiles,
                embeddings_per_tile=num_image_embeddings_per_tile,
                find_closest_aspect_ratio_fn=(
                    find_closest_area_weighted_aspect_ratio
                    if use_area_weighted_aspect_ratio
                    else find_closest_aspect_ratio
                ),
            )
        else:
            image_strategy = NoTilingStrategy(
                vision_model_type=args.vision_model_type,
                embeddings_per_image=num_image_embeddings_per_tile,
                target_width=args.img_w,
                target_height=args.img_h,
            )
        image_tiling_strategy = TileDegradationStrategy(
            image_strategy=image_strategy,
            video_frame_strategy=NoTilingStrategy(
                vision_model_type=args.vision_model_type,
                embeddings_per_image=num_image_embeddings_per_tile,
                target_width=args.img_w,
                target_height=args.img_h,
            ),
            embeddings_per_tile=num_image_embeddings_per_tile,
            max_num_tiles=args.max_num_tiles,
        )
    
    return image_tiling_strategy
