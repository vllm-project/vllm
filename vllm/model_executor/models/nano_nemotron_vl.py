# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# --------------------------------------------------------
# Adapted from
# https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/internvl.py
# under Apache-2.0 License
#     LICENSE is in root directory.
# --------------------------------------------------------

import copy
import math
import random
import typing
from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Annotated, Any, Literal, TypeAlias, TypeVar

import einops
import numpy.typing as npt
import regex as re
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
from transformers import BatchFeature, PretrainedConfig, TensorType
from typing_extensions import assert_never

from vllm.config import VllmConfig
from vllm.config.multimodal import BaseDummyOptions, VideoDummyOptions
from vllm.model_executor.layers.activation import ReLUSquaredActivation
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.interfaces import (
    HasInnerState,
    IsHybrid,
    MultiModalEmbeddings,
    SupportsMultiModal,
    SupportsMultiModalPruning,
)
from vllm.model_executor.models.internvl import (
    calculate_internvl_targets,
    get_internvl_target_ratios,
)
from vllm.model_executor.models.module_mapping import MultiModelKeys
from vllm.model_executor.models.nemotron_h import NemotronHForCausalLM
from vllm.model_executor.models.radio import RadioModel
from vllm.model_executor.models.utils import (
    init_vllm_registered_model,
    maybe_prefix,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.evs import (
    compute_retained_tokens_count,
    compute_retention_mask,
)
from vllm.multimodal.inputs import (
    MultiModalDataDict,
    MultiModalFieldConfig,
    MultiModalKwargsItems,
    VideoItem,
)
from vllm.multimodal.parse import (
    ImageEmbeddingItems,
    ImageProcessorItems,
    MultiModalDataItems,
    MultiModalDataParser,
)
from vllm.multimodal.processing import (
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    PromptReplacement,
    PromptUpdate,
    PromptUpdateDetails,
    _seq2tokens,
)
from vllm.multimodal.profiling import BaseDummyInputsBuilder
from vllm.sequence import IntermediateTensors
from vllm.tokenizers import TokenizerLike, cached_tokenizer_from_config
from vllm.transformers_utils.configs.radio import RadioConfig
from vllm.utils.tensor_schema import TensorSchema, TensorShape

from .utils import _merge_multimodal_embeddings

# Configure PIL to handle large images without warnings
# This prevents DecompressionBombWarning for legitimate large images
Image.MAX_IMAGE_PIXELS = None  # Disable the limit entirely
# Alternative: Set a specific higher limit
# Image.MAX_IMAGE_PIXELS = 300000000  # ~300M pixels
# Image.LOAD_TRUNCATED_IMAGES = True


# TODO(nhaber): does use_thumbnail=True work?
# TODO(nhaber): mixing images and videos will mess up the "text_prompt_length" calculation.


IMG_START = "<img>"
IMG_END = "</img>"
IMG_CONTEXT = "<image>"

# Profiling
# MAX_FRAMES = 16
DEFAULT_NUM_TILES = 12


def num_image_token_per_tile(
    *, width: int, height: int, patch_size: int, downsample_ratio: int
) -> int:
    num_patches = (width // patch_size) * (height // patch_size)
    num_tokens = num_patches // (downsample_ratio**2)
    return num_tokens


def width_and_height_for_max_num_tokens_available(
    *,
    target_num_tokens_post_shuffle: int,
    patch_size: int,
    downsample_ratio: int,
) -> tuple[int, int]:
    """
    TODO(nhaber): optimize this so it squeezes closer to target number of tokens.
    Calculate image dimensions that produce approximately `target` tokens after
    pixel_shuffle.

    With pixel_shuffle enabled, each 2x2 patch grid becomes 1 token, so we
    need 4*B patches to get B tokens.

    Examples:
    >>> width, height = width_and_height_for_max_num_tokens_available(
    ...     target_num_tokens_post_shuffle=8192,
    ...     patch_size=16,
    ...     downsample_ratio=2,
    ... )
    >>> assert width, height == (2880, 2880)
    >>> assert (width // 16) * (height // 16) // 2**2 == 8100  # tokens post-shuffle
    >>> assert (
    ...     num_image_token_per_tile(
    ...         width=width, height=height, patch_size=16, downsample_ratio=2
    ...     )
    ...     == 8100
    ... )
    """
    side_pixels = (
        math.isqrt(target_num_tokens_post_shuffle) * downsample_ratio * patch_size
    )
    assert isinstance(side_pixels, int) and side_pixels % patch_size == 0
    return side_pixels, side_pixels


@dataclass
class DynamicResolutionParams:
    media: Image.Image
    num_tiles: int
    num_embeddings: int
    patch_size: tuple[int, int]


class NanoNemotronVLImagePixelInputs(TensorSchema):
    """
    Dimensions:
        - bn: Batch size * number of images
        - bnp: Batch size * number of images * (1 + num_patches)
        - c: Number of channels (3)
        - h: Height of each image patch
        - w: Width of each image patch
    """

    type: Literal["pixel_values"]
    pixel_values_flat: torch.Tensor
    imgs_sizes: torch.Tensor
    image_feature_sizes: torch.Tensor


class NanoNemotronVLImageEmbeddingInputs(TensorSchema):
    """
    Dimensions:
        - n: Number of images
        - f: Total image feature size
        - h: Hidden size (must match the hidden size of language model backbone)
    """

    type: Literal["image_embeds"]
    data: Annotated[torch.Tensor | list[torch.Tensor], TensorShape("n", "f", "h")]


NanoNemotronVLImageInputs: TypeAlias = (
    NanoNemotronVLImagePixelInputs | NanoNemotronVLImageEmbeddingInputs
)


class NanoNemotronVLVideoPixelInputs(TensorSchema):
    """
    Dimensions:
        - bvf: Batch size * number of videos * num_frames
        - bn: Batch size * number of videos
        - f: Number of frames
        - c: Number of channels (3)
        - h: Height of each video frame
        - w: Width of each video frame
    """

    type: Literal["pixel_values_videos"]
    pixel_values_flat: Annotated[torch.Tensor, TensorShape("bvf", 3, "h", "w")]
    num_patches: Annotated[torch.Tensor, TensorShape("bn")]
    frames_indices: Annotated[torch.Tensor, TensorShape("bvf")]
    frame_duration_ms: Annotated[torch.Tensor, TensorShape("bn")]


class NanoNemotronVLVideoEmbeddingInputs(TensorSchema):
    """
    Dimensions:
        - n: Number of videos
        - f: Total video feature size
        - h: Hidden size (must match the hidden size of language model backbone)
    """

    type: Literal["video_embeds"]
    data: Annotated[torch.Tensor | list[torch.Tensor], TensorShape("n", "f", "h")]


NanoNemotronVLVideoInputs: TypeAlias = (
    NanoNemotronVLVideoPixelInputs | NanoNemotronVLVideoEmbeddingInputs
)


def dynamic_preprocess(
    image, *, image_size=512, max_num_tiles=12, use_thumbnail=True, idx=0
):
    orig_width, orig_height = image.size

    target_ratios = get_internvl_target_ratios(1, max_num_tiles)

    blocks, target_width, target_height = calculate_internvl_targets(
        orig_width=orig_width,
        orig_height=orig_height,
        target_ratios=target_ratios,
        image_size=image_size,
        use_thumbnail=False,
    )
    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)

    processed_images = [
        img.convert("RGB") if img.mode != "RGB" else img for img in processed_images
    ]
    processed_images = [
        T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BICUBIC)(
            img
        )
        for img in processed_images
    ]
    processed_images = [T.ToTensor()(img) for img in processed_images]
    return processed_images


def image_to_pixel_values(
    image: Image.Image,
    *,
    input_size: int,
    max_num: int,
    use_thumbnail: bool,
    idx: int,
) -> torch.Tensor:
    images = dynamic_preprocess(
        image,
        image_size=input_size,
        max_num_tiles=max_num,
        use_thumbnail=use_thumbnail,
        idx=idx,
    )

    pixel_values = torch.stack(images)
    return pixel_values


def video_to_pixel_values(
    video: npt.NDArray,
    *,
    input_size: int,
    max_num_tiles: int = 1,
    use_thumbnail: bool,
) -> torch.Tensor:
    assert max_num_tiles == 1, "Video modality always uses one tile"

    # Convert each frame to a single resized tile tensor consistent
    # with image path
    frames_tensors: list[torch.Tensor] = []
    for frame in video:
        pil_frame = dynamic_preprocess(
            Image.fromarray(frame, mode="RGB"),
            image_size=input_size,
            max_num_tiles=max_num_tiles,
            use_thumbnail=use_thumbnail,
            idx=0,
        )
        # dynamic_preprocess returns tensors already; take the single tile
        assert len(pil_frame) >= 1
        frames_tensors.append(pil_frame[-1])

    return torch.stack(frames_tensors)


def input_conditioner(
    x: torch.Tensor, norm_mean: torch.Tensor, norm_std: torch.Tensor
) -> torch.Tensor:
    assert isinstance(x, torch.Tensor), "x must be a tensor"
    assert isinstance(norm_mean, torch.Tensor), "norm_mean must be a tensor"
    assert isinstance(norm_std, torch.Tensor), "norm_std must be a tensor"
    return (x - norm_mean) / norm_std


def calculate_timestamps(
    indices: list[int] | torch.Tensor,
    frame_duration_ms: int,
):
    if not isinstance(indices, list):
        indices = indices.tolist()

    timestamps = [int(i) * frame_duration_ms / 1000.0 for i in indices]
    return timestamps


class BaseNanoNemotronVLProcessor(ABC):
    """
    This model doesn't define its own HF processor,
    so we implement our own one here.

    The code to insert image tokens is based on:
    https://huggingface.co/OpenGVLab/InternVL2-1B/blob/main/modeling_internvl_chat.py#L252
    """

    def __init__(
        self,
        config: PretrainedConfig,
        tokenizer: TokenizerLike,
        *args,
        max_num_tiles: int | None = None,
        **kwargs,
    ) -> None:
        super().__init__()

        self.config = config
        self.tokenizer = tokenizer

        self.max_num_tiles = max_num_tiles or DEFAULT_NUM_TILES
        image_size: int = config.force_image_size
        self.patch_size: int = getattr(config, "patch_size", 16)
        # self.downsample_ratio: float = self.config.downsample_ratio

        self.image_size = image_size
        self.use_thumbnail: bool = config.use_thumbnail
        self.norm_mean = torch.tensor(config.norm_mean).reshape(1, 3, 1, 1)
        self.norm_std = torch.tensor(config.norm_std).reshape(1, 3, 1, 1)

    @property
    @abstractmethod
    def image_token_id(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def get_image_repl(
        self,
        feature_size: int,
        num_patches: int | None,
    ) -> PromptUpdateDetails[str]:
        raise NotImplementedError

    def get_num_image_tokens(
        self,
        *,
        image_width: int,
        image_height: int,
        max_num_tiles: int,
    ) -> int:
        target_ratios = get_internvl_target_ratios(1, max_num_tiles)

        num_patches, _, _ = calculate_internvl_targets(
            orig_width=image_width,
            orig_height=image_height,
            target_ratios=target_ratios,
            image_size=self.image_size,
            use_thumbnail=self.use_thumbnail,
        )

        return num_patches * num_image_token_per_tile(
            width=image_width,
            height=image_height,
            patch_size=self.patch_size,
            downsample_ratio=self.downsample_ratio,
        )

    def _images_to_pixel_values_lst(
        self,
        text_prompt_length: int,
        images: list[Image.Image],
        max_num_tiles: int,
    ) -> tuple[list[torch.Tensor], list[int]]:
        return [
            image_to_pixel_values(
                image,
                input_size=self.image_size,
                max_num=max_num_tiles,
                use_thumbnail=self.use_thumbnail,
                idx=idx,
            )
            for idx, image in enumerate(images)
        ]

    def _preprocess_image(
        self,
        text: list[str],
        images: list[Image.Image],
        max_num_tiles: int,
    ) -> tuple[list[str], dict[str, torch.Tensor]]:
        if len(images) == 0:
            image_inputs = {}
        else:
            assert len(text) == 1, (
                "hf_processor is called on the output of get_dummy_text, "
                "which should be a single string"
            )
            text_prompt_length = len(
                self.tokenizer(
                    text[0].replace("<image>", ""), add_special_tokens=False
                )["input_ids"]
            )
            pixel_values_lst, token_counts = self._images_to_pixel_values_lst(
                text_prompt_length=text_prompt_length,
                images=images,
                max_num_tiles=max_num_tiles,
            )
            image_inputs = {
                "pixel_values_flat": input_conditioner(
                    torch.cat(pixel_values_lst), self.norm_mean, self.norm_std
                ),
                "image_num_patches": torch.tensor(
                    [len(item) for item in pixel_values_lst]
                ),
                "image_feature_sizes": token_counts,
            }

            assert len(text) == 1, (
                "hf_processor is called on the output of get_dummy_text, "
                "which should be a single string"
            )
            parts = [x for x in re.split(r"(<image>)", text[0]) if x]
            assert parts.count("<image>") == len(pixel_values_lst), (
                "the number of <image> tokens in the text should be the "
                "same as the number of images"
            )

            for i, (pixel_values, feature_size) in enumerate(
                zip(pixel_values_lst, token_counts, strict=True)
            ):
                num_patches = pixel_values.shape[0]
                image_repl = self.get_image_repl(feature_size, num_patches)
                parts[i] = parts[i].replace("<image>", image_repl.full)
            text = ["".join(parts)]
        return text, image_inputs

    def _make_batch_input(self, input_item: Any | list[Any] | None = None):
        if input_item is None:
            input_item = []
        if not isinstance(input_item, list):
            input_item = [input_item]
        return input_item

    @abstractmethod
    def __call__(
        self,
        text: str | list[str] | None = None,
        images: Image.Image | list[Image.Image] | None = None,
        return_tensors: str | TensorType | None = None,
        max_num_tiles: int | None = None,
    ) -> BatchFeature:
        raise NotImplementedError


class DynamicResolutionImageTiler(BaseNanoNemotronVLProcessor):
    CLIP_PIXEL_MEAN = [0.48145466, 0.4578275, 0.40821073]
    CLIP_PIXEL_STD = [0.26862954, 0.26130258, 0.27577711]

    def __init__(
        self,
        config: PretrainedConfig,
        tokenizer: TokenizerLike,
        *args,
        max_model_len: int,
        max_num_tiles: int | None = None,
        min_num_patches: int,
        max_num_patches: int,
        factor_max: float = 1.0,
        pixel_shuffle: bool = True,
        min_side: int | None = None,
        conv_merging: bool = False,
        use_thumbnail: bool = False,
        thumbnail_size: int = 448,
        thumbnail_area_threshold: float = 0.8,
        apply_data_augment: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(
            config=config, tokenizer=tokenizer, max_num_tiles=max_num_tiles, **kwargs
        )

        self._patch_size: int = getattr(config, "patch_size", 16)
        self.max_model_len = max_model_len
        self._min_num_patches = min_num_patches
        self._max_num_patches = max_num_patches if max_num_patches > 0 else float("inf")
        self._factor_max = factor_max
        self._pixel_shuffle = pixel_shuffle
        self._min_side = min_side
        self._conv_merging = conv_merging
        self._use_thumbnail = use_thumbnail
        self._thumbnail_size = thumbnail_size
        self._thumbnail_area_threshold = thumbnail_area_threshold
        self.norm_mean = torch.tensor(self.CLIP_PIXEL_MEAN).reshape(1, 3, 1, 1)
        self.norm_std = torch.tensor(self.CLIP_PIXEL_STD).reshape(1, 3, 1, 1)
        self._transform = T.Compose(
            [
                T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
                T.ToTensor(),  # T.Lambda(lambda img: _fast_to_tensor(img)),
                # T.Normalize(mean=pixel_mean, std=pixel_std), - This is done down below with input_conditioner
            ]
        )
        self._apply_data_augment = apply_data_augment
        reduction_factor = 1 / self.config.downsample_ratio
        assert reduction_factor == 2.0, (
            "I don't understand what's going on if this isn't 4"
        )
        self.downsample_ratio = int(reduction_factor) ** (pixel_shuffle + conv_merging)
        assert self.downsample_ratio == 2, (
            f"I don't understand what's going on if {self.downsample_ratio=} isn't 2"
        )

    def _get_num_embeddings(self, width: int, height: int) -> int:
        return num_image_token_per_tile(
            width=width,
            height=height,
            patch_size=self._patch_size,
            downsample_ratio=self.downsample_ratio,
        )

    def max_num_tokens_available(self, text_prompt_length: int) -> int:
        return self.max_model_len - text_prompt_length - 4

    def _images_to_pixel_values_lst(
        self,
        text_prompt_length: int,
        images: list[Image.Image],
        max_num_tiles: int,
    ) -> tuple[list[torch.Tensor], list[int]]:
        num_tokens_available = self.max_num_tokens_available(text_prompt_length)
        params_per_image = self.compute_params(images, num_tokens_available)

        feature_sizes = []
        images = []
        for param in params_per_image:
            for t in self.apply_params(param):
                if t.ndim == 3:
                    t = t.unsqueeze(0)
                images.append(t)
                feature_sizes.append(param.num_embeddings)
        print(f"{feature_sizes=}")
        print(f"{params_per_image=}")
        return images, feature_sizes

    feature_size_cache: dict[
        Image.Image, int
    ] = {}  # TODO(nhaber): Find a less silly way of doing this... Why can't this be an instance variable?

    def get_cached_feature_size(self, image: Image.Image) -> int:
        feature_size = self.feature_size_cache[id(image)]
        del self.feature_size_cache[id(image)]
        return feature_size

    def apply_params(self, params: DynamicResolutionParams) -> list[torch.Tensor]:
        resized_img = params.media.resize(
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
                thumbnail_img = params.media.resize(
                    (self._thumbnail_size, self._thumbnail_size)
                )
                processed_images.append(thumbnail_img)

        return [self._transform(img) for img in processed_images]

    def process_media(
        self,
        media: Image.Image,
        num_tokens_available: int,
        data_augment: bool = False,
        tiling_augment_prob: float = 0.4,
    ) -> tuple[DynamicResolutionParams, int]:
        """Process a single media item and return its parameters.

        Args:
            media: The media item to process
            num_tokens_available: Number of tokens available for this media
            data_augment: Whether to apply data augmentation to the image. Defaults to False.
        Returns:
            DynamicResolutionParams for the media
        """
        current_num_tokens_available = num_tokens_available
        assert isinstance(media, Image.Image), (
            "Dynamic resolution is only supported for image media"
        )
        orig_width, orig_height = media.width, media.height
        # TODO(nhaber): Ask Tyler - the round + 0.5 code is dangerous [banker's rounding], no?
        closest_patch_height = round(orig_height / self._patch_size + 0.5)
        closest_patch_width = round(orig_width / self._patch_size + 0.5)
        patches = closest_patch_height * closest_patch_width

        factor = min(
            math.sqrt(current_num_tokens_available / patches), self._factor_max
        )
        target_patch_height = math.floor(factor * closest_patch_height)
        target_patch_width = math.floor(factor * closest_patch_width)

        # We only consider self._min_num_patches if it is greater than current_num_tokens_available.
        if (
            current_num_tokens_available > self._min_num_patches
            and target_patch_height * target_patch_width < self._min_num_patches
        ):
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
                        target_patch_width = math.floor(up_factor * target_patch_width)
                    target_patch_width = new_patch_width
                    target_patch_height = max(
                        current_num_tokens_available // new_patch_width, 1
                    )
                else:
                    target_patch_height = new_patch_height
                    target_patch_width = new_patch_width
            else:
                up_factor = self._min_side / (target_patch_height * self._patch_size)
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
                        target_patch_width = math.floor(up_factor * target_patch_width)
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
                if (
                    target_patch_height + inc_h
                ) * target_patch_width <= current_num_tokens_available:
                    target_patch_height += inc_h
                else:
                    target_patch_height = max(
                        required_divisor, target_patch_height - rem_h
                    )

            rem_w = target_patch_width % required_divisor
            if rem_w != 0:
                inc_w = required_divisor - rem_w
                if (
                    target_patch_height * (target_patch_width + inc_w)
                    <= current_num_tokens_available
                ):
                    target_patch_width += inc_w
                else:
                    target_patch_width = max(
                        required_divisor, target_patch_width - rem_w
                    )

        if (
            data_augment
            and self._apply_data_augment
            and random.random() < tiling_augment_prob
        ):
            target_patch_width, target_patch_height = self.augment_resolution(
                target_patch_width, target_patch_height, current_num_tokens_available
            )

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
            resized_area = (target_patch_width * self._patch_size) * (
                target_patch_height * self._patch_size
            )
            thumbnail_area = self._thumbnail_size * self._thumbnail_size
            area_ratio = resized_area / thumbnail_area

            # Only add thumbnail if resized image area is less than threshold % of thumbnail area
            if area_ratio < self._thumbnail_area_threshold:
                num_tiles += 1  # Add 1 for thumbnail
                # Add embeddings for thumbnail (thumbnail_size x thumbnail_size)
                num_embeddings += self._get_num_embeddings(
                    self._thumbnail_size, self._thumbnail_size
                )
                token_count += (
                    self._thumbnail_size
                    // self._patch_size
                    * self._thumbnail_size
                    // self._patch_size
                )

        return DynamicResolutionParams(
            media=media,
            num_tiles=num_tiles,
            num_embeddings=num_embeddings,
            patch_size=(target_patch_width, target_patch_height),
        ), token_count

    def augment_resolution(
        self,
        target_patch_width: int,
        target_patch_height: int,
        current_num_tokens_available: int,
    ) -> tuple[int, int]:
        min_num_patch_one_side = 32

        if random.random() < 0.5:
            # Minus one
            if (
                target_patch_width <= min_num_patch_one_side
                and target_patch_height <= min_num_patch_one_side
            ):
                return target_patch_width, target_patch_height
            elif target_patch_width <= min_num_patch_one_side:
                return target_patch_width, target_patch_height - min_num_patch_one_side
            elif target_patch_height <= min_num_patch_one_side:
                return target_patch_width - min_num_patch_one_side, target_patch_height
            else:
                if random.random() < 0.5:
                    return (
                        target_patch_width - min_num_patch_one_side,
                        target_patch_height,
                    )
                else:
                    return (
                        target_patch_width,
                        target_patch_height - min_num_patch_one_side,
                    )
        else:
            # Plus one
            if target_patch_width * target_patch_height < current_num_tokens_available:
                if random.random() < 0.5:
                    return (
                        target_patch_width + min_num_patch_one_side,
                        target_patch_height,
                    )
                else:
                    return (
                        target_patch_width,
                        target_patch_height + min_num_patch_one_side,
                    )
            return target_patch_width, target_patch_height

    def compute_params(
        self,
        media_list: list[Image.Image],
        num_tokens_available: int | None = None,
        data_augment: bool = False,
    ) -> list[DynamicResolutionParams]:
        """Compute parameters for all media with iterative token budgeting.

        Args:
            media_list: List of media items to process
            num_tokens_available: Total number of tokens available across all media
            data_augment: Whether to apply data augmentation to the image. Defaults to
            False.
        Returns:
            List of ImageTilingParams for each media item
        """
        num_tokens_available = (
            num_tokens_available
            * (4 if self._pixel_shuffle else 1)
            * (4 if self._conv_merging else 1)
        )
        # When the number of available token is too small, allow self._min_num_patches per media and
        # let the sample be truncated.
        num_tokens_available = max(
            num_tokens_available, self._min_num_patches * len(media_list)
        )

        # Clip the number of tokens available per media to be between min and max patches.
        num_tokens_available_per_media = [
            max(min(num_tokens_available, self._max_num_patches), self._min_num_patches)
            for _ in range(len(media_list))
        ]

        # In theory this could be a while True loop, but in case the process_media method slightly
        # changes, I want to make sure we don't get stuck in an infinite loop.
        for _ in range(10):
            # Step 1: Process each media with current token budget
            params = []
            token_counts = []

            for media, tokens_for_media in zip(
                media_list, num_tokens_available_per_media
            ):
                param, token_count = self.process_media(
                    media, tokens_for_media, data_augment=data_augment
                )
                params.append(param)
                token_counts.append(token_count)
                self.feature_size_cache[id(param.media)] = param.num_embeddings

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
            scaled_down = any(
                [
                    scaled_down_num_tokens_available_per_media[i]
                    < num_tokens_available_per_media[i]
                    for i in range(len(num_tokens_available_per_media))
                ]
            )
            # If there was not scaling down, we're stuck just use min_num_patches per media, else
            # try with the scaled down num_tokens_available_per_media.
            if not scaled_down:
                num_tokens_available_per_media = [self._min_num_patches] * len(
                    media_list
                )
            else:
                num_tokens_available_per_media = (
                    scaled_down_num_tokens_available_per_media
                )
        assert_never(num_tokens_available_per_media)

    @staticmethod
    def stack(images: list[torch.Tensor], patch_size: int) -> torch.Tensor:
        assert len(images) > 0, "No images to stack"

        def rearrange_img(x):
            py = x.shape[-2] // patch_size
            px = x.shape[-1] // patch_size
            x = einops.rearrange(
                x,
                "c (py yy) (px xx) -> (py px) (c yy xx)",
                py=py,
                yy=patch_size,
                px=px,
                xx=patch_size,
            )
            return x

        imgs = [rearrange_img(img) for img in images]
        pixel_values_flat = torch.cat(imgs, dim=0).unsqueeze(0)
        return pixel_values_flat


class NanoNemotronVLProcessor(DynamicResolutionImageTiler):
    """
    HF Processor  with extended video processing logic.
    Code for video processing is adapted from video example:
    https://huggingface.co/OpenGVLab/InternVL3-1B#inference-with-transformers
    """

    def __init__(
        self,
        config: PretrainedConfig,
        tokenizer: TokenizerLike,
        *,
        max_model_len: int,
        min_num_patches: int,
        max_num_patches: int,
        max_num_tiles: int | None = None,
        dynamic_image_size: bool | None = None,
        video_token: str | None = None,
        video_pruning_rate: float | None = None,
    ) -> None:
        super().__init__(
            config=config,
            tokenizer=tokenizer,
            max_model_len=max_model_len,
            max_num_tiles=max_num_tiles,
            min_num_patches=min_num_patches,
            max_num_patches=max_num_patches,
            dynamic_image_size=dynamic_image_size,
        )
        # add extra video token for video processing
        self.video_token = video_token
        self.video_pruning_rate = video_pruning_rate

        # Pre-tokenize special tokens for video processing
        # to avoid repeated tokenization
        self._img_start_token_ids = tokenizer.encode(
            IMG_START, add_special_tokens=False
        )
        self._img_end_token_ids = tokenizer.encode(IMG_END, add_special_tokens=False)
        self._img_context_token_ids = tokenizer.encode(
            IMG_CONTEXT, add_special_tokens=False
        )

    @property
    def supports_video(self) -> bool:
        return self.video_token_id is not None

    @property
    def video_token_id(self) -> int | None:
        if self.video_token is None:
            return None
        return self.tokenizer.get_vocab().get(self.video_token, None)

    @property
    def image_token_id(self) -> int:
        return self.tokenizer.convert_tokens_to_ids(IMG_CONTEXT)

    def _videos_to_pixel_values_lst(
        self,
        videos: list[npt.NDArray],
        max_num_tiles: int,
        dynamic_image_size: bool | None = None,
    ) -> list[torch.Tensor]:
        return [
            video_to_pixel_values(
                video,
                input_size=self.image_size,
                max_num_tiles=max_num_tiles,
                use_thumbnail=self.use_thumbnail,
            )
            for video in videos
        ]

    def _preprocess_video(
        self,
        text: list[str],
        videos: list[tuple[npt.NDArray, dict[str, Any]]],
        max_num_tiles: int,
        dynamic_image_size: bool | None = None,
    ):
        if len(videos) == 0 or not self.supports_video:
            video_inputs = {}
        else:
            videos_lst = [v[0] for v in videos]
            video_metadata_lst = [v[1] for v in videos]
            pixel_values_lst_video = self._videos_to_pixel_values_lst(
                videos_lst,
                max_num_tiles=max_num_tiles,
                dynamic_image_size=dynamic_image_size,
            )

            # We use frame duration in milliseconds (as integer) to ensure
            # we have consistent timestamps calculation. At preprocessing
            # fps parameter is given in fp32, while at inference it is bf16
            # which leads to inaccurate timestamp calculation and causes
            # timestamp values to differ.In rare cases this causes
            # mismatching number of output tokens for tokenized  frame prefixes
            frame_duration_ms_lst = [
                int(1000.0 / metadata["fps"]) for metadata in video_metadata_lst
            ]
            frames_indices_lst = [
                metadata["frames_indices"] for metadata in video_metadata_lst
            ]

            video_inputs = {
                "pixel_values_flat_video": input_conditioner(
                    torch.cat(pixel_values_lst_video), self.norm_mean, self.norm_std
                ),
                "video_num_patches": torch.tensor(
                    [len(item) for item in pixel_values_lst_video]
                ),
                "frames_indices": frames_indices_lst,
                "frame_duration_ms": torch.tensor(frame_duration_ms_lst),
            }

            image_size: int = self.config.force_image_size
            patch_size: int = self.config.patch_size
            downsample_ratio = self.config.downsample_ratio
            tokens_in_single_frame = int(
                (image_size * image_size // patch_size**2) * (downsample_ratio**2)
            )

            for pixel_values, video_metadata, frames_indices, frame_duration_ms in zip(
                pixel_values_lst_video,
                video_metadata_lst,
                frames_indices_lst,
                frame_duration_ms_lst,
            ):
                num_frames = pixel_values.shape[0]

                if (
                    self.video_pruning_rate is not None
                    and self.video_pruning_rate > 0.0
                ):
                    # Start of EVS-specific code
                    num_tokens = compute_retained_tokens_count(
                        tokens_per_frame=tokens_in_single_frame,
                        num_frames=num_frames,
                        q=self.video_pruning_rate,
                    )

                    # Here we just need placeholders that won't actually be replaced -
                    # we just need to make sure the total number of tokens is correct
                    # assign all tokens to the first frame
                    tokens_per_frame = [num_tokens] + [0] * (num_frames - 1)

                    # End of EVS-specific code
                else:
                    tokens_per_frame = [tokens_in_single_frame] * num_frames

                video_repl = self.get_video_repl(
                    tokens_per_frame=tokens_per_frame,
                    frames_indices=frames_indices,
                    frame_duration_ms=frame_duration_ms,
                    tokenizer=self.tokenizer,
                    img_start_token_ids=self._img_start_token_ids,
                    img_end_token_ids=self._img_end_token_ids,
                    img_context_token_ids=self._img_context_token_ids,
                )

                # video_repl.full is a list of token IDs
                # Convert token IDs back to text for the HF processor flow
                video_repl_text = self.tokenizer.decode(
                    video_repl.full, skip_special_tokens=False
                )
                text = [t.replace("<video>", video_repl_text, 1) for t in text]
        return text, video_inputs

    def __call__(
        self,
        text: str | list[str] | None = None,
        images: Image.Image | list[Image.Image] | None = None,
        videos: list[tuple[npt.NDArray, dict[str, Any]]] | None = None,
        return_tensors: str | TensorType | None = None,
        max_num_tiles: int | None = None,
        dynamic_image_size: bool | None = None,
    ) -> BatchFeature:
        # Use default if not provided
        if max_num_tiles is None:
            max_num_tiles = self.max_num_tiles

        text, images, videos = [
            self._make_batch_input(x) for x in (text, images, videos)
        ]

        text, image_inputs = self._preprocess_image(
            text=text,
            images=images,
            max_num_tiles=max_num_tiles,
        )

        text, video_inputs = self._preprocess_video(
            text=text,
            videos=videos,
            max_num_tiles=1,
            dynamic_image_size=dynamic_image_size,
        )

        text_inputs = self.tokenizer(text, add_special_tokens=False)

        combined_outputs = {**text_inputs, **image_inputs, **video_inputs}

        return BatchFeature(combined_outputs, tensor_type=return_tensors)

    def get_image_repl(
        self,
        feature_size: int,
        num_patches: int | None,
    ) -> PromptUpdateDetails[str]:
        repl_features = IMG_CONTEXT * feature_size
        repl_full = IMG_START + repl_features + IMG_END

        return PromptUpdateDetails.select_text(repl_full, IMG_CONTEXT)

    @classmethod
    def get_video_repl(
        cls,
        *,
        tokens_per_frame: list[int],
        frames_indices: list[int],
        frame_duration_ms: int,
        tokenizer: TokenizerLike,
        img_start_token_ids: list[int],
        img_end_token_ids: list[int],
        img_context_token_ids: list[int],
    ) -> PromptUpdateDetails[list[int]]:
        """
        Build prompt replacement for a video.
        The replacement returned is not actually used to replace the placeholder
        tokens - it's just used to make sure we allocate the correct number
        of tokens.
        Actual replacement is done in embed_multimodal of
        NemotronH_Nano_VL_V2
        (specifically in _process_video_input -> _create_final_video_embeddings).
        There, we create the final embeddings with text embeddings for indicator tokens
        and video embeddings for video tokens.
        This is a single function that handles all cases - non EVS, EVS dummy, EVS real.
        The differentiation is done via tokens_per_frame parameter.
        - non EVS case - constant value same value across all frames
        - EVS dummy - Doesn't matter how tokens are distributed between frames - just
                        make sure the total number of tokens is correct.
        - EVS real (called from get_real_video_repl_for_evs) - different value per frame
        Args:
            tokens_per_frame (list[int]): number of tokens per frame
            frames_indices (list[int]): frame indices
            frame_duration_ms (int): duration of each frame in milliseconds
            tokenizer (TokenizerLike): tokenizer to use for tokenizing frame separators
            img_start_token_ids (list[int]): pre-tokenized IMG_START tokens
            img_end_token_ids (list[int]): pre-tokenized IMG_END tokens
            img_context_token_ids (list[int]): pre-tokenized IMG_CONTEXT tokens
        """
        # TODO: Add support of frame_duration_ms to be None
        # At preprocessing step we should allow absent / metadata without
        # frames_indices field.
        timestamps_enabled = frame_duration_ms is not None

        if timestamps_enabled:
            timestamps = calculate_timestamps(frames_indices, frame_duration_ms)

            assert len(timestamps) == len(tokens_per_frame), (
                "timestamps and tokens_per_frame must have the same length"
            )
            frame_separators = [
                f"Frame {i + 1} sampled at {timestamp:.2f} seconds: "
                for i, timestamp in enumerate(timestamps)
            ]
        else:
            frame_separators = [
                f"Frame {i + 1}: " for i, _ in enumerate(tokens_per_frame)
            ]

        # Tokenize frame separator independently
        frame_separators_tokenized = [
            _seq2tokens(tokenizer, sep) for sep in frame_separators
        ]

        # Tokenize each component independently to avoid tokenizer merging tokens
        # across boundaries. This ensures consistent tokenization regardless of
        # num_tokens_per_frame values.
        all_token_ids = []
        for i, num_tokens in enumerate(tokens_per_frame):
            frame_sep_token_ids = frame_separators_tokenized[i]
            all_token_ids.extend(frame_sep_token_ids)

            # Add pre-tokenized special tokens
            all_token_ids.extend(img_start_token_ids)
            all_token_ids.extend(img_context_token_ids * num_tokens)
            all_token_ids.extend(img_end_token_ids)

        return PromptUpdateDetails.from_seq(all_token_ids)


class BaseNanoNemotronVLProcessingInfo(BaseProcessingInfo):
    """Basic image-only ProcessingInfo for InternVL-style models."""

    @abstractmethod
    def get_hf_processor(
        self,
        **kwargs: object,
    ) -> DynamicResolutionImageTiler:
        raise NotImplementedError

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"image": None}

    def get_num_image_tokens(
        self,
        *,
        image_width: int,
        image_height: int,
        max_num_tiles: int,
        processor: BaseNanoNemotronVLProcessor | None,
    ) -> int:
        if processor is None:
            processor = self.get_hf_processor()

        return processor.get_num_image_tokens(
            image_width=image_width,
            image_height=image_height,
            max_num_tiles=max_num_tiles,
        )

    def get_max_image_tokens(self) -> int:
        processor = self.get_hf_processor()
        # Use default max_num_tiles for max tokens calculation
        max_num_tiles = processor.max_num_tiles
        target_width, target_height = self.get_image_size_with_most_features(
            max_num_tiles
        )

        return self.get_num_image_tokens(
            image_width=target_width,
            image_height=target_height,
            max_num_tiles=max_num_tiles,
            processor=processor,
        )


_I = TypeVar("_I", bound=BaseNanoNemotronVLProcessingInfo)


ArgType = TypeVar("ArgType")


def get_vision_config_arg(
    hf_config: PretrainedConfig,
    arg_name: str,
    *,
    default_value: ArgType,
) -> ArgType:
    vision_config = hf_config.vision_config
    if not hasattr(vision_config, "args"):
        return default_value
    value = vision_config.args.get(arg_name, default_value)
    assert isinstance(value, type(default_value)), (
        f"Value for {arg_name} is not of type {type(default_value)}"
    )
    return value


class NanoNemotronVLProcessingInfo(BaseNanoNemotronVLProcessingInfo):
    """ProcessingInfo extended for video processing"""

    @property
    def supports_video(self):
        return False  # TODO(nhaber): add video support

    def get_supported_mm_limits(self):
        video_limit = {"video": None} if self.supports_video else {}
        return {**super().get_supported_mm_limits(), **video_limit}

    def get_video_token(self) -> str | None:
        return IMG_CONTEXT

    def get_video_pruning_rate(self) -> float | None:
        return self.ctx.get_mm_config().video_pruning_rate

    def get_num_frames_with_most_features(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> int:
        max_images = mm_counts.get("image", 0)
        max_videos = mm_counts.get("video", 0)

        processor = self.get_hf_processor()  # we get the CustomProcessor here

        max_image_tokens = self.get_max_image_tokens() * max_images
        max_total_frames = (seq_len - max_image_tokens) // num_image_token_per_tile(
            width=256,
            height=256,
            patch_size=processor._patch_size,
            downsample_ratio=processor.downsample_ratio,
        )  # TODO(nhaber): get 256 dynamically
        max_frames_per_video = max_total_frames // max(max_videos, 1)
        return max(max_frames_per_video, 1)

    def get_hf_processor(self, **kwargs: object) -> NanoNemotronVLProcessor:
        return self.ctx.init_processor(
            NanoNemotronVLProcessor,
            config=self.get_hf_config(),
            tokenizer=self.get_tokenizer(),
            video_token=self.get_video_token(),
            video_pruning_rate=self.get_video_pruning_rate(),
            max_model_len=self.ctx.model_config.max_model_len,
            min_num_patches=get_vision_config_arg(
                self.get_hf_config(), "min_num_patches", default_value=4
            ),
            max_num_patches=get_vision_config_arg(
                self.get_hf_config(), "max_num_patches", default_value=0
            ),
            **kwargs,
        )


class NanoNemotronBaseVLMultiModalProcessor(BaseMultiModalProcessor[_I]):
    """Basic image-only MultiModalProcessor for InternVL-style models."""

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        image_num_patches = hf_inputs.get("image_num_patches", torch.empty(0))

        return dict(
            pixel_values_flat=MultiModalFieldConfig.flat_from_sizes(
                "image", image_num_patches
            ),
            image_num_patches=MultiModalFieldConfig.batched("image"),
            image_feature_sizes=MultiModalFieldConfig.batched("image"),
            image_embeds=MultiModalFieldConfig.batched("image"),
        )

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        hf_processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)

        out_mm_data = out_mm_kwargs.get_data()
        if "image_num_patches" in out_mm_data:
            image_num_patches = out_mm_data["image_num_patches"]
            assert isinstance(image_num_patches, torch.Tensor)
            image_num_patches = image_num_patches.tolist()
        elif "image_embeds" in out_mm_data:
            # to compute num_patches (similar to Qwen2-VL)
            image_num_patches = [None] * len(out_mm_data["image_embeds"])
        else:
            image_num_patches = []

        def get_replacement_custom(item_idx: int):
            images = mm_items.get_items(
                "image", (ImageEmbeddingItems, ImageProcessorItems)
            )

            if isinstance(images, ImageEmbeddingItems):
                feature_size = images.get_feature_size(item_idx)
            else:
                image = images.get(item_idx)
                feature_size = hf_processor.get_cached_feature_size(image)

            num_patches = None
            local_image_num_patches = image_num_patches
            if isinstance(local_image_num_patches, torch.Tensor):
                local_image_num_patches = local_image_num_patches.tolist()
            if isinstance(local_image_num_patches, (list, tuple)) and item_idx < len(
                local_image_num_patches
            ):
                num_patches = int(local_image_num_patches[item_idx])

            return hf_processor.get_image_repl(feature_size, num_patches)

        return [
            PromptReplacement(
                modality="image",
                target="<image>",
                replacement=get_replacement_custom,
            )
        ]


class NanoNemotronVLMultiModalProcessor(
    NanoNemotronBaseVLMultiModalProcessor[NanoNemotronVLProcessingInfo]
):
    """MultiModalProcessor extended for video support"""

    def _get_data_parser(self) -> MultiModalDataParser:
        return MultiModalDataParser(video_needs_metadata=True)

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        image_fields = super()._get_mm_fields_config(hf_inputs, hf_processor_mm_kwargs)
        if self.info.supports_video:
            video_num_patches = hf_inputs.get("video_num_patches", torch.empty(0))

            video_fields = dict(
                pixel_values_flat_video=MultiModalFieldConfig.flat_from_sizes(
                    "video", video_num_patches
                ),
                video_num_patches=MultiModalFieldConfig.batched("video"),
                frames_indices=MultiModalFieldConfig.batched("video"),
                frame_duration_ms=MultiModalFieldConfig.batched("video"),
            )
        else:
            video_fields = {}

        return image_fields | video_fields

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        prompt_repl = super()._get_prompt_updates(
            mm_items=mm_items,
            hf_processor_mm_kwargs=hf_processor_mm_kwargs,
            out_mm_kwargs=out_mm_kwargs,
        )

        hf_processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)

        out_mm_data = out_mm_kwargs.get_data()
        if "video_num_patches" in out_mm_data:
            video_num_patches = out_mm_data["video_num_patches"]
            assert isinstance(video_num_patches, torch.Tensor)
            video_num_patches = video_num_patches.tolist()
        else:
            video_num_patches = []

        def get_video_replacement_internvl(item_idx: int):
            feature_size = num_image_token_per_tile(
                width=256,
                height=256,
                patch_size=hf_processor._patch_size,
                downsample_ratio=hf_processor.downsample_ratio,
            )  # TODO(nhaber): get 256 dynamically
            video, metadata = mm_items["video"][item_idx]
            num_patches = video_num_patches[item_idx]
            if num_patches is not None:
                assert isinstance(num_patches, int)

            video_pruning_rate = self.info.ctx.get_mm_config().video_pruning_rate
            if video_pruning_rate is not None and video_pruning_rate > 0.0:
                # Start of EVS-specific code
                num_tokens = compute_retained_tokens_count(
                    tokens_per_frame=feature_size,
                    num_frames=num_patches,
                    q=video_pruning_rate,
                )
                # Here we just need placeholders that won't actually be replaced -
                # we just need to make sure the total number of tokens is correct
                # assign all tokens to the first frame
                tokens_per_frame = [num_tokens] + [0] * (num_patches - 1)

                # End of EVS-specific code
            else:
                tokens_per_frame = [feature_size] * num_patches

            frame_duration_ms = int(1000 / metadata["fps"])
            return hf_processor.get_video_repl(
                tokens_per_frame=tokens_per_frame,
                frames_indices=metadata["frames_indices"],
                frame_duration_ms=frame_duration_ms,
                tokenizer=hf_processor.tokenizer,
                img_start_token_ids=hf_processor._img_start_token_ids,
                img_end_token_ids=hf_processor._img_end_token_ids,
                img_context_token_ids=hf_processor._img_context_token_ids,
            )

        if self.info.supports_video:
            prompt_repl = [
                *prompt_repl,
                PromptReplacement(
                    modality="video",
                    target="<video>",
                    replacement=get_video_replacement_internvl,
                ),
            ]

        return prompt_repl


class NanoNemotronVLDummyInputsBuilder(BaseDummyInputsBuilder[_I]):
    """Basic image-only DummyInputsBuilder for InternVL-style models."""

    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        num_images = mm_counts.get("image", 0)

        return "<image>" * num_images

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions] | None = None,
    ) -> MultiModalDataDict:
        num_images = mm_counts.get("image", 0)
        processor = self.info.get_hf_processor()
        B = processor.max_num_tokens_available(text_prompt_length=num_images)
        target_width, target_height = width_and_height_for_max_num_tokens_available(
            target_num_tokens_post_shuffle=B,
            patch_size=processor._patch_size,
            downsample_ratio=processor.downsample_ratio,
        )

        image_overrides = mm_options.get("image") if mm_options else None

        return {
            "image": self._get_dummy_images(
                width=target_width,
                height=target_height,
                num_images=num_images,
                overrides=image_overrides,
            )
        }


class NanoNemotronVLDummyInputsBuilder(
    NanoNemotronVLDummyInputsBuilder[NanoNemotronVLProcessingInfo]
):
    """DummyInputsBuilder extended for video support"""

    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        num_videos = mm_counts.get("video", 0)

        return super().get_dummy_text(mm_counts) + "<video>" * num_videos

    def _get_dummy_videos(
        self,
        *,
        width: int,
        height: int,
        num_frames: int,
        num_videos: int,
        overrides: VideoDummyOptions | None = None,
    ) -> list[VideoItem]:
        video = super()._get_dummy_videos(
            width=width,
            height=height,
            num_frames=num_frames,
            num_videos=1,
            overrides=overrides,
        )[0]
        video_items = []
        for _ in range(num_videos):
            video_metadata = {
                "total_num_frames": num_frames,
                "fps": 2,
                "duration": num_frames / 2.0,
                "video_backend": "opencv_dynamic",
                "frames_indices": [i for i in range(num_frames)],
                "do_sample_frames": False,
            }
            video_item = (video.copy(), video_metadata)
            video_items.append(video_item)

        return video_items

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions] | None = None,
    ) -> MultiModalDataDict:
        dummy_image = super().get_dummy_mm_data(
            seq_len=seq_len, mm_counts=mm_counts, mm_options=mm_options
        )
        if self.info.supports_video:
            config = self.info.get_hf_config()
            image_size: int = config.force_image_size
            target_num_frames = self.info.get_num_frames_with_most_features(
                seq_len, mm_counts
            )
            num_videos = mm_counts.get("video", 0)
            video_overrides = mm_options.get("video") if mm_options else None
            dummy_video = {
                "video": self._get_dummy_videos(
                    width=image_size,
                    height=image_size,
                    num_frames=target_num_frames,
                    num_videos=num_videos,
                    overrides=video_overrides,
                )
            }
        else:
            dummy_video = {}
        return {**dummy_image, **dummy_video}


@MULTIMODAL_REGISTRY.register_processor(
    NanoNemotronVLMultiModalProcessor,
    info=NanoNemotronVLProcessingInfo,
    dummy_inputs=NanoNemotronVLDummyInputsBuilder,
)
class NemotronH_Nano_VL_V2(
    nn.Module, HasInnerState, IsHybrid, SupportsMultiModal, SupportsMultiModalPruning
):
    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("image"):
            return "<image>"
        if modality.startswith("video"):
            return "<video>"
        return None

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        multimodal_config = vllm_config.model_config.multimodal_config
        image_size = config.force_image_size
        patch_size = config.patch_size
        self.patch_size = patch_size
        self.template = config.template
        self.downsample_ratio = config.downsample_ratio
        self.ps_version = config.ps_version
        self.image_tag_type = config.image_tag_type
        self.video_pruning_rate = multimodal_config.video_pruning_rate
        self.language_model = init_vllm_registered_model(
            vllm_config=vllm_config,
            hf_config=config.text_config,
            prefix=maybe_prefix(prefix, "language_model"),
        )
        self.vision_model = self.get_vit_model_from_radio_config(config).to(
            self.language_model.config.dtype
        )

        # Construct the vision projection.
        vit_hidden_size = config.vit_hidden_size
        vision_projection_hidden_size = config.projector_hidden_size
        llm_hidden_size = config.text_config.hidden_size

        self.mlp1 = nn.Sequential(
            RMSNorm(
                hidden_size=vit_hidden_size * int(1 / self.downsample_ratio) ** 2,
                eps=1e-5,
            ),
            nn.Linear(
                vit_hidden_size * int(1 / self.downsample_ratio) ** 2,
                vision_projection_hidden_size,
                bias=False,
            ),
            ReLUSquaredActivation(),
            nn.Linear(vision_projection_hidden_size, llm_hidden_size, bias=False),
        )
        self.mlp1 = self.mlp1.to(self.language_model.config.dtype)

        self.config = config
        self.model_config = vllm_config.model_config

        # Pre-tokenize special tokens for video processing
        # to avoid repeated tokenization
        tokenizer = cached_tokenizer_from_config(vllm_config.model_config)
        self._img_start_token_ids = tokenizer.encode(
            IMG_START, add_special_tokens=False
        )
        self._img_end_token_ids = tokenizer.encode(IMG_END, add_special_tokens=False)
        self._img_context_token_ids = tokenizer.encode(
            IMG_CONTEXT, add_special_tokens=False
        )

    def pixel_shuffle_dynamic_res(self, x, *, imgs_sizes):
        scale_factor = self.downsample_ratio
        patch_dim = self.patch_size
        seq_lens = torch.prod(imgs_sizes // patch_dim, dim=-1)
        splits = torch.split(x, seq_lens.tolist(), dim=-2)
        out = []
        for i, sv in enumerate(splits):
            h = imgs_sizes[i][0] // patch_dim
            w = imgs_sizes[i][1] // patch_dim
            sv = sv.reshape(sv.shape[0], h, w, -1)

            n, h, w, c = sv.size()

            sv = sv.view(n, h, int(w * scale_factor), int(c / scale_factor))
            sv = sv.permute(0, 2, 1, 3).contiguous()
            sv = sv.view(
                n,
                int(w * scale_factor),
                int(h * scale_factor),
                int(c / (scale_factor * scale_factor)),
            )

            if self.ps_version == "v2":
                sv = sv.permute(0, 2, 1, 3).contiguous()

            sv = sv.reshape(sv.shape[0], -1, sv.shape[-1])
            out.append(sv)

        x = torch.cat(out, dim=-2)

        return x

    def extract_feature(self, pixel_values: torch.Tensor, imgs_sizes: torch.Tensor):
        vit_embeds = self.vision_model(pixel_values, imgs_sizes=imgs_sizes)
        vit_embeds = vit_embeds.to(dtype=torch.bfloat16)
        vit_embeds = self.pixel_shuffle_dynamic_res(vit_embeds, imgs_sizes=imgs_sizes)
        vit_embeds = self.mlp1(vit_embeds)
        return vit_embeds

    def _parse_and_validate_image_input(
        self, **kwargs: object
    ) -> NanoNemotronVLImageInputs | None:
        pixel_values_flat = kwargs.pop("pixel_values_flat", None)
        image_feature_sizes = kwargs.pop("image_feature_sizes", None)
        image_num_patches = kwargs.pop("image_num_patches", None)
        image_embeds = kwargs.pop("image_embeds", None)

        if pixel_values_flat is None and image_embeds is None:
            return None

        if image_embeds is not None:
            return NanoNemotronVLImageEmbeddingInputs(
                type="image_embeds",
                data=image_embeds,
            )

        if pixel_values_flat is not None:
            imgs_sizes = torch.tensor(
                [[pv.shape[1], pv.shape[2]] for pv in pixel_values_flat],
                dtype=torch.int32,
            )
            pixel_values_flat = DynamicResolutionImageTiler.stack(
                pixel_values_flat, self.patch_size
            )
            return NanoNemotronVLImagePixelInputs(
                type="pixel_values",
                pixel_values_flat=pixel_values_flat,
                imgs_sizes=imgs_sizes,
                image_feature_sizes=image_feature_sizes,
                validate=False,
            )

        raise AssertionError("This line should be unreachable.")

    def _process_image_input(
        self, image_input: NanoNemotronVLImageInputs
    ) -> tuple[torch.Tensor, ...]:
        if image_input["type"] == "image_embeds":
            return image_input["data"]

        assert self.vision_model is not None

        image_embeds = self.extract_feature(
            image_input.pixel_values_flat, image_input.imgs_sizes
        )
        image_feature_sizes = image_input.image_feature_sizes.tolist()
        print(f"{image_feature_sizes=}")

        if len(image_feature_sizes) == 1:
            return (image_embeds.view(-1, self.config.text_config.hidden_size),)

        image_embeds = image_embeds.view(-1, self.config.text_config.hidden_size)
        return image_embeds.split(image_feature_sizes)

    def _process_video_input(
        self, video_input: NanoNemotronVLVideoPixelInputs
    ) -> tuple[torch.Tensor, ...]:
        """Process video input and create final embeddings with video content
        and indicator tokens."""
        # Get video embeddings using the same processing as images
        assert typing.assert_never(video_input), (
            "TODO(nhaber): Video input is not supported for this hacky RL enablement branch for vision"
        )
        video_embeddings = self._process_image_input(video_input)

        final_video_embeddings: tuple[torch.Tensor, ...] = ()

        image_rows = image_cols = self.config.force_image_size
        downsample_ratio = self.config.downsample_ratio
        patch_size = self.config.patch_size
        rows = int(image_rows * downsample_ratio // patch_size)
        cols = int(image_cols * downsample_ratio // patch_size)
        video_pruning_rate = self.video_pruning_rate
        video_num_frames = video_input["num_patches"].tolist()
        video_frames_indices = video_input["frames_indices"].split(video_num_frames)
        # Calculate video feature dimensions (number of frames and
        # their feature size (AKA tokens per frame))
        # TODO: Maybe this can be optimized to avoid the loop?
        for i, single_video_embeddings in enumerate(video_embeddings):
            num_frames = video_num_frames[i]
            frames_indices = video_frames_indices[i].tolist()
            frame_duration_ms = video_input["frame_duration_ms"][i].item()
            assert single_video_embeddings.shape[0] % num_frames == 0

            if video_pruning_rate is not None and video_pruning_rate > 0.0:
                # Start of EVS-specific code
                retention_mask = compute_retention_mask(
                    single_video_embeddings,
                    video_size_thw=(num_frames, rows, cols),
                    spatial_merge_size=1,
                    q=video_pruning_rate,
                )

                # apply retention mask
                single_video_embeddings = single_video_embeddings[retention_mask]

                # calculate the actual number of retained tokens per frame
                retention_mask_thw = retention_mask.reshape(num_frames, rows, cols)
                num_tokens_per_frame = (
                    retention_mask_thw.sum(dim=(1, 2)).long().tolist()
                )
                # End of EVS-specific code
            else:
                feature_size = single_video_embeddings.shape[0] // num_frames
                num_tokens_per_frame = [feature_size] * num_frames

            final_video_embeddings += (
                self._create_final_video_embeddings(
                    single_video_embeddings,
                    num_tokens_per_frame,
                    frames_indices,
                    frame_duration_ms,
                ),
            )

        return final_video_embeddings

    def _create_final_video_embeddings(
        self,
        video_embeddings: torch.Tensor,
        num_tokens_per_frame: list[int],
        frames_indices: list[int],
        frame_duration_ms: int,
    ) -> torch.Tensor:
        """Create final embeddings that combine video embeddings with
        text embeddings of indicator tokens.

        These final embeddings contain:
        - Actual video embeddings in positions corresponding to video content
        - Text embeddings for indicator tokens (<img>, </img>, and
          frame separation text) in their respective positions

        These embeddings will replace the placeholder embeddings to create
        input_embeds for the LLM.
        """
        device = video_embeddings.device
        tokenizer = cached_tokenizer_from_config(self.model_config)

        # Generate video replacement token IDs using get_video_repl
        # This tokenizes each frame separator independently, then uses pre-tokenized
        # special tokens to ensure consistent tokenization regardless of
        # num_tokens_per_frame values.
        video_repl = NanoNemotronVLProcessor.get_video_repl(
            tokens_per_frame=num_tokens_per_frame,
            frames_indices=frames_indices,
            frame_duration_ms=frame_duration_ms,
            tokenizer=tokenizer,
            img_start_token_ids=self._img_start_token_ids,
            img_end_token_ids=self._img_end_token_ids,
            img_context_token_ids=self._img_context_token_ids,
        )

        # video_repl.full is a list of token IDs
        repl_token_ids = torch.tensor(video_repl.full, device=device)

        # Get embedding token IDs for image context (use pre-tokenized version)
        embed_token_ids = torch.tensor(self._img_context_token_ids, device=device)

        # Create mask for video embedding positions
        is_video_embed = torch.isin(repl_token_ids, embed_token_ids)

        # Create final video embeddings, merging text embeddings for indicator
        # tokens with video embeddings
        text_embeddings = self.get_language_model().embed_input_ids(repl_token_ids)
        final_video_embeddings = _merge_multimodal_embeddings(
            inputs_embeds=text_embeddings,
            multimodal_embeddings=video_embeddings,
            is_multimodal=is_video_embed,
        )

        return final_video_embeddings

    def _parse_and_validate_video_input(
        self, **kwargs: object
    ) -> NanoNemotronVLVideoPixelInputs | None:
        pixel_values_flat_video = kwargs.pop("pixel_values_flat_video", None)
        video_num_patches = kwargs.pop("video_num_patches", None)
        video_embeds = kwargs.pop("video_embeds", None)
        frames_indices = kwargs.pop("frames_indices", None)
        frame_duration_ms = kwargs.pop("frame_duration_ms", None)

        if pixel_values_flat_video is None and video_embeds is None:
            return None

        if video_embeds is not None:
            return NanoNemotronVLVideoEmbeddingInputs(
                type="video_embeds",
                data=video_embeds,
            )

        if pixel_values_flat_video is not None:
            if torch.is_tensor(frames_indices):
                frames_indices = frames_indices.flatten()
            else:
                frames_indices = torch.cat([f.flatten() for f in frames_indices], dim=0)

            frame_duration_ms = frame_duration_ms.flatten()
            expected_h = expected_w = self.config.force_image_size
            num_frames = video_num_patches[0].item()
            resolve_bindings = {"h": expected_h, "w": expected_w, "f": num_frames}

            return NanoNemotronVLVideoPixelInputs(
                type="pixel_values_videos",
                pixel_values_flat=pixel_values_flat_video,
                num_patches=video_num_patches,
                frames_indices=frames_indices,
                frame_duration_ms=frame_duration_ms,
                resolve_bindings=resolve_bindings,
            )

        raise AssertionError("This line should be unreachable.")

    def _parse_and_validate_multimodal_inputs(self, **kwargs: object) -> dict:
        modalities = {}
        # Preserve the order of modalities if there are multiple of them
        # from the order of kwargs.
        for input_key in kwargs:
            if (
                input_key in ("pixel_values_flat", "image_embeds")
                and "images" not in modalities
            ):
                modalities["images"] = self._parse_and_validate_image_input(**kwargs)
            if input_key in ("pixel_values_flat_video",) and "videos" not in modalities:
                modalities["videos"] = self._parse_and_validate_video_input(**kwargs)

        return modalities

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        # Validate the multimodal input keyword arguments
        modalities = self._parse_and_validate_multimodal_inputs(**kwargs)
        if modalities is None:
            return []

        # # The result multimodal_embeddings is tuple of tensors, with each
        # tensor corresponding to a multimodal data item (image or video).
        multimodal_embeddings: tuple[torch.Tensor, ...] = ()

        # NOTE: It is important to iterate over the keys in this dictionary
        # to preserve the order of the modalities.
        for modality in modalities:
            if modality == "images":
                image_input = modalities["images"]
                image_embeddings = self._process_image_input(image_input)
                multimodal_embeddings += tuple(image_embeddings)
            if modality == "videos":
                video_input = modalities["videos"]
                video_embeddings = self._process_video_input(video_input)
                multimodal_embeddings += tuple(video_embeddings)

        return multimodal_embeddings

    def get_language_model(self) -> torch.nn.Module:
        return self.language_model

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors:
        if intermediate_tensors is not None:
            input_ids = None
            inputs_embeds = None

        hidden_states = self.language_model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )

        return hidden_states

    def get_mm_mapping(self) -> MultiModelKeys:
        """
        Get the module prefix in multimodal models
        """
        return MultiModelKeys.from_string_field(
            language_model="language_model",
            connector="mlp1",
            tower_model="vision_model",
        )

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        return self.language_model.compute_logits(hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        adapter_dict = dict(self.mlp1.named_parameters())

        def is_llm(name: str) -> bool:
            return name.startswith("language_model")

        def is_adapter_weights(weight: tuple[str, torch.Tensor]):
            return weight[0].startswith("mlp1")

        def is_vision_weights(name: str) -> bool:
            return name.startswith("vision_model.radio_model.")

        # Separate weights by component
        llm_weights = []
        vision_weights = []

        for name, w in weights:
            if is_llm(name):
                # Strip 'language_model.' prefix for LLM weights
                llm_weights.append((".".join(name.split(".")[1:]), w))
            elif is_adapter_weights((name, w)):
                # Load vision-language adapter weights directly
                trimmed_name = ".".join(name.split(".")[1:])
                param = adapter_dict[trimmed_name]
                with torch.no_grad():
                    default_weight_loader(param, w)
            elif is_vision_weights(name):
                # Convert: vision_model.radio_model.* → radio_model.*
                hf_key = name[len("vision_model.") :]  # Remove "vision_model." prefix
                vision_weights.append((hf_key, w))

        self.language_model.load_weights(llm_weights)
        self.vision_model.load_weights(vision_weights)

    def print_architecture(self, detailed: bool = True, save_to_file: str = None):
        """
        Print model architecture with parameter names, shapes, and sizes.

        Args:
            detailed: If True, show detailed parameter breakdown
            save_to_file: If provided, save output to this file path
        """
        import sys
        from io import StringIO

        # Capture output if saving to file
        original_stdout = sys.stdout
        if save_to_file:
            sys.stdout = StringIO()

        try:
            print("=" * 100)
            print("NemotronH_Nano_VL_V2 Model Architecture")
            print("=" * 100)

            total_params = 0
            param_groups = {
                "language_model": [],
                "vision_model": [],
                "mlp1": [],
                "other": [],
            }

            for name, param in self.named_parameters():
                param_size = param.numel()
                total_params += param_size

                # Group parameters by main component
                if name.startswith("language_model"):
                    param_groups["language_model"].append(
                        (name, param.shape, param_size, param.dtype)
                    )
                elif name.startswith("vision_model"):
                    param_groups["vision_model"].append(
                        (name, param.shape, param_size, param.dtype)
                    )
                elif name.startswith("mlp1"):
                    param_groups["mlp1"].append(
                        (name, param.shape, param_size, param.dtype)
                    )
                else:
                    param_groups["other"].append(
                        (name, param.shape, param_size, param.dtype)
                    )

                if detailed:
                    print(
                        f"{name:<70} | Shape: {str(param.shape):<25} | "
                        f"Size: {param_size:>12,} | Dtype: {param.dtype}"
                    )

            print("=" * 100)
            print("Summary by Component:")
            print("-" * 60)

            for component, params in param_groups.items():
                if params:  # Only show components that have parameters
                    component_total = sum(size for _, _, size, _ in params)
                    percentage = (
                        (component_total / total_params) * 100
                        if total_params > 0
                        else 0
                    )
                    print(
                        f"{component:<20} | Parameters: {len(params):>4} | "
                        f"Total Size: {component_total:>15,} | "
                        f"{percentage:>6.2f}%"
                    )

            print("-" * 60)
            print(f"{'Total Parameters':<20} | {total_params:>15,}")

            # Estimate memory usage (assuming bfloat16 = 2 bytes per parameter)
            memory_mb = total_params * 2 / (1024**2)
            memory_gb = memory_mb / 1024
            print(f"{'Est. Memory (MB)':<20} | {memory_mb:>15.2f}")
            print(f"{'Est. Memory (GB)':<20} | {memory_gb:>15.2f}")
            print("=" * 100)

            # Save to file if requested
            if save_to_file:
                output = sys.stdout.getvalue()
                sys.stdout = original_stdout
                with open(save_to_file, "w") as f:
                    f.write(output)
                print(f"Architecture saved to: {save_to_file}")
                print(output)  # Also print to console

        finally:
            if save_to_file and sys.stdout != original_stdout:
                sys.stdout = original_stdout

    def get_vit_model_from_radio_config(self, hf_config):
        hf_config_vision = hf_config.vision_config
        model_name = hf_config_vision.args.get("model")
        if model_name is None:
            raise ValueError(f"Unsupported vit model type: {model_name}")

        preferred_resolution = getattr(hf_config_vision, "preferred_resolution", None)
        image_size = preferred_resolution[0] if preferred_resolution else 224
        patch_size = getattr(hf_config_vision, "patch_size", 16)

        radio_config = RadioConfig(
            model_name=model_name,
            image_size=image_size,
            patch_size=patch_size,
            norm_mean=hf_config.norm_mean,
            norm_std=hf_config.norm_std,
            reg_tokens=get_vision_config_arg(
                hf_config, "register_multiple", default_value=10
            ),
        )

        return RadioModel(config=radio_config)

    def copy_inputs_before_cuda_graphs(self, input_buffers, **kwargs):
        return self.language_model.mamba_cache.copy_inputs_before_cuda_graphs(
            input_buffers, **kwargs
        )

    def get_seqlen_agnostic_capture_inputs(self, batch_size: int):
        return self.language_model.mamba_cache.get_seqlen_agnostic_capture_inputs(
            batch_size
        )

    @classmethod
    def get_mamba_state_shape_from_config(cls, vllm_config: "VllmConfig"):
        text_config = vllm_config.model_config.hf_config.text_config
        temp_vllm_config = copy.deepcopy(vllm_config)
        temp_vllm_config.model_config.hf_config = text_config
        return NemotronHForCausalLM.get_mamba_state_shape_from_config(temp_vllm_config)

    @classmethod
    def get_mamba_state_dtype_from_config(cls, vllm_config: "VllmConfig"):
        text_config = vllm_config.model_config.hf_config.text_config
        temp_vllm_config = copy.deepcopy(vllm_config)
        temp_vllm_config.model_config.hf_config = text_config
        return NemotronHForCausalLM.get_mamba_state_dtype_from_config(temp_vllm_config)
