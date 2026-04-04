# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# --------------------------------------------------------
# Adapted from
# https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/internvl.py
# under Apache-2.0 License
#     LICENSE is in root directory.
# --------------------------------------------------------

import math
import warnings
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cached_property
from typing import Any, TypeVar

import einops
import numpy as np
import numpy.typing as npt
import regex as re
import torch
from PIL import Image
from transformers import BatchFeature, PretrainedConfig, TensorType

from vllm.model_executor.models.parakeet import ParakeetExtractor
from vllm.multimodal.evs import compute_retained_tokens_count
from vllm.multimodal.inputs import AudioItem
from vllm.multimodal.processing.processor import PromptUpdateDetails, _seq2tokens
from vllm.tokenizers.hf import HfTokenizer

from .internvl import calculate_internvl_targets, get_internvl_target_ratios

_T = TypeVar("_T")


IMG_START = "<img>"
IMG_END = "</img>"
IMG_CONTEXT = "<image>"
AUDIO_START = "<so_start>"
AUDIO_END = "<so_end>"
AUDIO_CONTEXT = "<so_embedding>"

# Profiling
# MAX_FRAMES = 16
DEFAULT_NUM_TILES = 12

# Configure PIL to handle large images without warnings
# This prevents DecompressionBombWarning for legitimate large images
Image.MAX_IMAGE_PIXELS = None  # Disable the limit entirely
# Alternative: Set a specific higher limit
# Image.MAX_IMAGE_PIXELS = 300000000  # ~300M pixels


def calculate_timestamps(
    indices: list[int] | torch.Tensor,
    frame_duration_ms: int,
):
    if not isinstance(indices, list):
        indices = indices.tolist()

    timestamps = [int(i) * frame_duration_ms / 1000.0 for i in indices]
    return timestamps


def input_conditioner(x: torch.Tensor, norm_mean: torch.Tensor, norm_std: torch.Tensor):
    return (x - norm_mean) / norm_std


def _bicubic_from_ndarray(
    array: npt.NDArray[Any], *, size: tuple[int, int]
) -> torch.Tensor:
    """
    Convert a 4D NHWC ndarray to NCHW and interpolate with bicubic.
    Suppresses PyTorch's non-writable NumPy warning because interpolate copies,
    and torch.from_numpy(array) is discarded at the end of function scope.
    """

    with warnings.catch_warnings():
        msg = "The given NumPy array is not writ.*"
        # Apparently, different versions of PyTorch use writable or writeable.
        warnings.filterwarnings("ignore", message=msg, category=UserWarning)
        tensor = torch.from_numpy(array)
    assert tensor.ndim == 4, f"{tensor.ndim=}"
    tensor = tensor.permute(0, 3, 1, 2)
    return (
        torch.nn.functional.interpolate(
            tensor, size=size, mode="bicubic", align_corners=False, antialias=True
        )
        / 255.0
    )


def dynamic_preprocess(
    image,
    *,
    image_size=512,
    max_num_tiles=12,
    use_thumbnail=True,
    idx=0,
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

    image = np.asarray(
        image.convert("RGB") if image.mode != "RGB" else image, dtype=np.uint8
    )

    image = np.expand_dims(image, axis=0)

    resized_img = _bicubic_from_ndarray(image, size=(target_height, target_width))
    B, C, H, W = resized_img.shape
    hp, wp = H // image_size, W // image_size
    patches = (
        resized_img.reshape(B, C, hp, image_size, wp, image_size)
        .permute(0, 2, 4, 1, 3, 5)
        .reshape(B * hp * wp, C, image_size, image_size)
    )

    if use_thumbnail and patches.shape[0] > 1:
        thumb = _bicubic_from_ndarray(image, size=(image_size, image_size))
        patches = torch.cat([patches, thumb], dim=0)

    return list(patches)


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


def _compute_aspect_preserving_size(
    orig_w: int,
    orig_h: int,
    target_num_patches: int,
    patch_size: int,
    downsample_ratio: float,
) -> tuple[int, int]:
    """Compute target pixel dimensions that preserve aspect ratio.

    Mirrors Megatron-LM image_processing.py video frame resizing:
    target area in patch-grid space is *target_num_patches*, distributed
    according to the source aspect ratio, then snapped to a multiple of
    the required divisor (2 for pixel-shuffle).
    """
    aspect_wh = orig_w / max(orig_h, 1)
    ph = round(math.sqrt(target_num_patches / aspect_wh))
    pw = round(math.sqrt(target_num_patches * aspect_wh))
    ph = max(ph, 1)
    pw = max(pw, 1)

    reduction_factor = int(round(1 / downsample_ratio))
    required_divisor = reduction_factor  # 2 for pixel-shuffle
    if required_divisor > 1:
        rem_h = ph % required_divisor
        rem_w = pw % required_divisor
        ph_up = ph + (required_divisor - rem_h if rem_h else 0)
        ph_down = ph - rem_h
        pw_up = pw + (required_divisor - rem_w if rem_w else 0)
        pw_down = pw - rem_w
        if ph_up * pw_up <= target_num_patches:
            ph, pw = ph_up, pw_up
        else:
            ph = max(required_divisor, ph_down)
            pw = max(required_divisor, pw_down)

    return pw * patch_size, ph * patch_size  # (width, height) in pixels


def get_video_target_size_and_feature_size(
    orig_w: int,
    orig_h: int,
    target_patches: int,
    maintain_aspect_ratio: bool,
    patch_size: int,
    downsample_ratio: float,
) -> tuple[int, int, int]:
    """Compute target (width, height) and feature_size for video resize and token count.

    Used by video_to_pixel_values (resize) and get_video_replacement_internvl
    (seq length calc) so both use the same dimensions.
    """
    if maintain_aspect_ratio:
        target_w, target_h = _compute_aspect_preserving_size(
            orig_w=orig_w,
            orig_h=orig_h,
            target_num_patches=target_patches,
            patch_size=patch_size,
            downsample_ratio=downsample_ratio,
        )
    else:
        reduction_factor = int(round(1 / downsample_ratio))
        side = int(math.sqrt(target_patches))
        side = max(reduction_factor, (side // reduction_factor) * reduction_factor)
        target_w = side * patch_size
        target_h = side * patch_size

    feature_size = int((target_h // patch_size) * downsample_ratio) * int(
        (target_w // patch_size) * downsample_ratio
    )
    return target_w, target_h, feature_size


def video_to_pixel_values(
    video: npt.NDArray,
    *,
    input_size: int,
    video_target_num_patches: int | None = None,
    video_maintain_aspect_ratio: bool = False,
    patch_size: int = 16,
    downsample_ratio: float = 0.5,
) -> torch.Tensor:
    # (num_frames, H, W, C) -> (num_frames, C, H, W)
    video_tensor = torch.from_numpy(video).permute(0, 3, 1, 2)

    if video_target_num_patches is not None:
        # Resize to target patch count (aspect-preserving or square).
        orig_h, orig_w = video_tensor.shape[2], video_tensor.shape[3]
        target_w, target_h, _ = get_video_target_size_and_feature_size(
            orig_w=orig_w,
            orig_h=orig_h,
            target_patches=video_target_num_patches,
            maintain_aspect_ratio=video_maintain_aspect_ratio,
            patch_size=patch_size,
            downsample_ratio=downsample_ratio,
        )
        if video_tensor.shape[2] != target_h or video_tensor.shape[3] != target_w:
            return _bicubic_from_ndarray(video, size=(target_h, target_w))
    elif video_tensor.shape[2] != input_size or video_tensor.shape[3] != input_size:
        return _bicubic_from_ndarray(video, size=(input_size, input_size))

    video_tensor = video_tensor / 255.0

    return video_tensor


class DynamicResolutionImageTiler:
    CONV_MERGING = False
    PIXEL_SHUFFLE = True
    USE_THUMBNAIL = False

    def __init__(
        self,
        *,
        max_model_len: int,
        patch_size: int,
        min_num_patches: int,
        max_num_patches: int,
        downsample_ratio: int,
        norm_mean: Sequence[float],
        norm_std: Sequence[float],
        factor_max: float = 1.0,
        use_thumbnail: bool = False,
    ) -> None:
        assert use_thumbnail is False, "use_thumbnail is not supported"
        self._patch_size: int = patch_size
        self._max_model_len = max_model_len
        self._min_num_patches = min_num_patches
        self._max_num_patches = max_num_patches if max_num_patches > 0 else float("inf")
        self._factor_max = factor_max
        self.norm_mean = torch.tensor(norm_mean).reshape(3, 1, 1)
        self.norm_std = torch.tensor(norm_std).reshape(3, 1, 1)
        assert downsample_ratio < 1
        reduction_factor = 1 / downsample_ratio
        assert reduction_factor == 2.0
        self._downsample_ratio = int(reduction_factor) ** (
            self.PIXEL_SHUFFLE + self.CONV_MERGING
        )
        assert self._downsample_ratio == 2

    def _get_num_embeddings(self, width: int, height: int) -> int:
        num_patches = (width // self._patch_size) * (height // self._patch_size)
        num_tokens = num_patches // (self._downsample_ratio**2)
        return num_tokens

    def width_and_height_for_max_num_tokens_available(
        self,
        target_num_tokens_post_shuffle: int,
    ) -> tuple[int, int]:
        """
        TODO: optimize this so it squeezes closer to target number of tokens.
        Calculate image dimensions that produce approximately `target` tokens after
        pixel_shuffle.

        With pixel_shuffle enabled, each 2x2 patch grid becomes 1 token, so we
        need 4*B patches to get B tokens.

        Examples:
        >>> PATCH_SIZE = 16
        >>> DOWNSAMPLE_RATIO = 0.5
        >>> tiler = DynamicResolutionImageTiler(
        ...     max_model_len=16384,
        ...     patch_size=PATCH_SIZE,
        ...     downsample_ratio=DOWNSAMPLE_RATIO,
        ...     min_num_patches=4,
        ...     max_num_patches=0,
        ... )
        >>> width, height = tiler.width_and_height_for_max_num_tokens_available(
        ...     target_num_tokens_post_shuffle=8192,
        ... )
        >>> assert width, height == (2880, 2880)
        >>> assert (width // PATCH_SIZE) * (
        ...     height // PATCH_SIZE
        ... ) // 2**2 == 8100  # tokens post-shuffle
        >>> assert tiler._get_num_embeddings(width=width, height=height) == 8100
        """
        side_pixels = (
            math.isqrt(target_num_tokens_post_shuffle)
            * self._downsample_ratio
            * self._patch_size
        )
        assert isinstance(side_pixels, int) and side_pixels % self._patch_size == 0
        return side_pixels, side_pixels

    def max_num_tokens_available(self, text_prompt_length: int) -> int:
        return self._max_model_len - text_prompt_length - 4

    def _images_to_pixel_values_lst(
        self,
        text_prompt_length: int,
        images: list[Image.Image],
    ) -> tuple[list[torch.Tensor], list[int]]:
        num_tokens_available = self.max_num_tokens_available(text_prompt_length)
        params_per_image = self.compute_params(images, num_tokens_available)

        feature_sizes = []
        images = []
        for param in params_per_image:
            for t in self.apply_params(param):
                assert t.ndim == 3, f"{t.ndim=}: expected 3 dim tensor"
                images.append(t)
                feature_sizes.append(param.num_embeddings)
        return images, feature_sizes

    @dataclass
    class DynamicResolutionParams:
        media: Image.Image
        num_tiles: int
        num_embeddings: int
        patch_size: tuple[int, int]

    def apply_params(self, params: DynamicResolutionParams) -> list[torch.Tensor]:
        target_size = (
            params.patch_size[1] * self._patch_size,
            params.patch_size[0] * self._patch_size,
        )
        image = np.asarray(
            params.media.convert("RGB") if params.media.mode != "RGB" else params.media,
            dtype=np.uint8,
        )
        image = np.expand_dims(image, axis=0)
        resized_img = _bicubic_from_ndarray(image, size=target_size)
        return list(resized_img)

    def process_media(
        self,
        media: Image.Image,
        num_tokens_available: int,
    ) -> tuple[DynamicResolutionParams, int]:
        """Process a single media item and return its parameters.

        Args:
            media: The media item to process
            num_tokens_available: Number of tokens available for this media
        Returns:
            DynamicResolutionParams for the media
        """
        current_num_tokens_available = num_tokens_available
        assert isinstance(media, Image.Image), (
            "Dynamic resolution is only supported for image media"
        )
        orig_width, orig_height = media.width, media.height
        closest_patch_height = round(orig_height / self._patch_size + 0.5)
        closest_patch_width = round(orig_width / self._patch_size + 0.5)
        patches = closest_patch_height * closest_patch_width

        factor = min(
            math.sqrt(current_num_tokens_available / patches), self._factor_max
        )
        target_patch_height = math.floor(factor * closest_patch_height)
        target_patch_width = math.floor(factor * closest_patch_width)

        # Consider self._min_num_patches if > current_num_tokens_available.
        if (
            current_num_tokens_available > self._min_num_patches
            and target_patch_height * target_patch_width < self._min_num_patches
        ):
            up_factor = math.sqrt(
                self._min_num_patches / (target_patch_height * target_patch_width)
            )
            target_patch_height = math.ceil(up_factor * target_patch_height)
            target_patch_width = math.ceil(up_factor * target_patch_width)

        # Round patch grid to be divisible by 2 (pixel-shuffle OR conv-merging)
        # or by 4 when BOTH are enabled (two successive 2x reductions)
        if self.PIXEL_SHUFFLE or self.CONV_MERGING:
            required_divisor = 4 if (self.PIXEL_SHUFFLE and self.CONV_MERGING) else 2

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

        # Calculate embeddings for the main dynamic resolution image
        num_embeddings = self._get_num_embeddings(
            target_patch_width * self._patch_size,
            target_patch_height * self._patch_size,
        )

        token_count = target_patch_width * target_patch_height

        # Add thumbnail embeddings if enabled and image area is below threshold
        num_tiles = 1  # Base dynamic resolution image

        return self.DynamicResolutionParams(
            media=media,
            num_tiles=num_tiles,
            num_embeddings=num_embeddings,
            patch_size=(target_patch_width, target_patch_height),
        ), token_count

    def compute_params(
        self,
        media_list: list[Image.Image],
        num_tokens_available: int,
    ) -> list[DynamicResolutionParams]:
        """Compute parameters for all media with iterative token budgeting.

        Args:
            media_list: List of media items to process
            num_tokens_available: Total number of tokens available across all media
        Returns:
            List of ImageTilingParams for each media item
        """
        num_tokens_available = (
            num_tokens_available
            * (4 if self.PIXEL_SHUFFLE else 1)
            * (4 if self.CONV_MERGING else 1)
        )
        # When the number of available token is too small,
        # allow self._min_num_patches per media and let the sample be truncated.
        num_tokens_available = max(
            num_tokens_available, self._min_num_patches * len(media_list)
        )

        # Clip the number of tokens available per media to >min and <max patches.
        num_tokens_available_per_media = [
            int(
                max(
                    min(num_tokens_available, self._max_num_patches),
                    self._min_num_patches,
                )
            )
            for _ in range(len(media_list))
        ]

        # prevent infinite loop in any case
        for _ in range(10):
            # Step 1: Process each media with current token budget
            params = []
            token_counts = []

            for media, tokens_for_media in zip(
                media_list, num_tokens_available_per_media
            ):
                param, token_count = self.process_media(media, tokens_for_media)
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
            scaled_down = any(
                [
                    scaled_down_num_tokens_available_per_media[i]
                    < num_tokens_available_per_media[i]
                    for i in range(len(num_tokens_available_per_media))
                ]
            )
            # If there wasn't scaling down, we're stuck with min_num_patches per media,
            # else try with the scaled down num_tokens_available_per_media.
            if not scaled_down:
                num_tokens_available_per_media = [self._min_num_patches] * len(
                    media_list
                )
            else:
                num_tokens_available_per_media = (
                    scaled_down_num_tokens_available_per_media
                )
        ctx = f"{params=} {total_tokens=} {num_tokens_available=}"
        raise ValueError(
            f"Should be unreachable - `return params` above must be reached: {ctx}"
        )

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
        tokenizer: HfTokenizer,
        *args,
        max_model_len: int,
        max_num_tiles: int | None = None,
        **kwargs,
    ) -> None:
        super().__init__()

        self.config = config
        self.tokenizer = tokenizer

        self.max_num_tiles = max_num_tiles or DEFAULT_NUM_TILES
        image_size: int = config.force_image_size
        patch_size: int = config.patch_size
        downsample_ratio: int = config.downsample_ratio

        self.num_image_token = int(
            (image_size // patch_size) ** 2 * (downsample_ratio**2)
        )
        self.image_size = image_size
        self.use_thumbnail: bool = config.use_thumbnail
        self.norm_mean = torch.Tensor(config.norm_mean).reshape(1, 3, 1, 1)
        self.norm_std = torch.Tensor(config.norm_std).reshape(1, 3, 1, 1)

        self.dynamic_tiler: DynamicResolutionImageTiler | None = None
        if self.use_dynamic_resolution(config):
            self.dynamic_tiler = DynamicResolutionImageTiler(
                max_model_len=max_model_len,
                patch_size=patch_size,
                downsample_ratio=downsample_ratio,
                min_num_patches=config.vision_config.args["min_num_patches"],
                max_num_patches=config.vision_config.args["max_num_patches"],
                norm_mean=config.norm_mean,
                norm_std=config.norm_std,
            )

    @staticmethod
    def use_dynamic_resolution(config: PretrainedConfig) -> bool:
        return "min_num_patches" in config.vision_config.args

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

        return num_patches * self.num_image_token

    def _images_to_pixel_values_lst(
        self,
        images: list[Image.Image],
        max_num_tiles: int,
    ) -> list[torch.Tensor]:
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
    ) -> tuple[list[str], dict[str, Any]]:
        if len(images) == 0:
            return text, {}

        image_inputs: dict[str, Any]
        if tiler := self.dynamic_tiler:
            sans_images = text[0].replace("<image>", "")
            text_prompt_length = len(
                self.tokenizer(sans_images, add_special_tokens=False).input_ids
            )
            pixel_values_lst, num_tokens_per_image = tiler._images_to_pixel_values_lst(
                text_prompt_length=text_prompt_length,
                images=images,
            )
            imgs_sizes = [(pv.shape[-2], pv.shape[-1]) for pv in pixel_values_lst]
            normalized = [
                input_conditioner(img, tiler.norm_mean, tiler.norm_std)
                for img in pixel_values_lst
            ]
            image_num_patches = torch.tensor([1] * len(num_tokens_per_image))
            image_inputs = {
                "pixel_values_flat": normalized,
                "imgs_sizes": imgs_sizes,
                "num_tokens_per_image": num_tokens_per_image,
            }
        else:
            pixel_values_lst = self._images_to_pixel_values_lst(images, max_num_tiles)
            image_num_patches = torch.tensor([len(item) for item in pixel_values_lst])
            pixel_values_flat = input_conditioner(
                torch.cat(pixel_values_lst), self.norm_mean, self.norm_std
            )
            image_inputs = {
                "pixel_values_flat": pixel_values_flat,
                "image_num_patches": image_num_patches,
            }
            num_tokens_per_image = [
                self.num_image_token * len(item) for item in pixel_values_lst
            ]

        assert len(text) == 1, (
            "hf_processor is called on the output of get_dummy_text, "
            "which should be a single string"
        )
        parts = [x for x in re.split(r"(<image>)", text[0]) if x]
        assert parts.count("<image>") == len(num_tokens_per_image), (
            f"Expected {len(num_tokens_per_image)} <image> tokens in text "
            f"but found {parts.count('<image>')}"
        )

        for i, (feature_size, num_patches) in enumerate(
            zip(num_tokens_per_image, image_num_patches, strict=True)
        ):
            image_repl = self.get_image_repl(feature_size, num_patches)
            parts[i] = parts[i].replace("<image>", image_repl.full)
        text = ["".join(parts)]

        return text, image_inputs

    def _make_batch_input(self, input_item: _T | list[_T] | None = None) -> list[_T]:
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
        *,
        return_tensors: str | TensorType | None = None,
        max_num_tiles: int | None = None,
        **kwargs,
    ) -> BatchFeature:
        raise NotImplementedError


class NanoNemotronVLProcessor(BaseNanoNemotronVLProcessor):
    """
    HF Processor with extended video processing logic.
    Code for video processing is adapted from video example:
    https://huggingface.co/OpenGVLab/InternVL3-1B#inference-with-transformers
    """

    def __init__(
        self,
        config: PretrainedConfig,
        tokenizer: HfTokenizer,
        *,
        max_model_len: int,
        max_num_tiles: int | None = None,
        video_token: str | None = None,
        video_pruning_rate: float | None = None,
    ) -> None:
        super().__init__(
            config=config,
            tokenizer=tokenizer,
            max_model_len=max_model_len,
            max_num_tiles=max_num_tiles,
        )
        # add extra video token for video processing
        self.video_token = video_token
        self.video_pruning_rate = video_pruning_rate

        # Video params live exclusively in vision_config
        vision_config = getattr(config, "vision_config", config)
        self.video_temporal_patch_size: int = getattr(
            vision_config, "video_temporal_patch_size", 1
        )
        self.video_maintain_aspect_ratio: bool = getattr(
            vision_config, "video_maintain_aspect_ratio", False
        )

        # Resolve video frame target size: exactly one of video_target_num_patches
        # or video_target_img_size may be set (mirrors Megatron's
        # DynamicResolutionImageTilingStrategy validation).
        target_num_patches = getattr(vision_config, "video_target_num_patches", None)
        target_img_size = getattr(vision_config, "video_target_img_size", None)
        if target_num_patches is not None and target_img_size is not None:
            raise ValueError(
                "Exactly one of video_target_num_patches or "
                "video_target_img_size must be set, got both"
            )
        if target_num_patches is not None:
            self.video_target_num_patches: int | None = target_num_patches
        elif target_img_size is not None:
            base_patches = math.ceil(target_img_size / config.patch_size)
            self.video_target_num_patches = base_patches * base_patches
        else:
            self.video_target_num_patches = None

        self.audio_extractor: ParakeetExtractor | None = None
        raw_sound_config = getattr(config, "sound_config", None)
        if raw_sound_config is not None:
            self.audio_extractor = ParakeetExtractor(raw_sound_config)

        # Pre-tokenize special tokens for video processing
        # to avoid repeated tokenization
        self._img_start_token_ids = tokenizer.encode(
            IMG_START, add_special_tokens=False
        )
        self._img_end_token_ids = tokenizer.encode(IMG_END, add_special_tokens=False)
        self._img_context_token_ids = tokenizer.encode(
            IMG_CONTEXT, add_special_tokens=False
        )

    @cached_property
    def num_video_token(self) -> int:
        """Token count per video frame, accounting for video_target_num_patches.

        When video_target_num_patches is set the per-frame feature count
        differs from the image-based num_image_token.  We use a square
        dummy (1:1) to compute the feature_size because the dummy video is
        square and the user confirmed that is acceptable.
        """
        if self.video_target_num_patches is not None:
            _, _, feature_size = get_video_target_size_and_feature_size(
                orig_w=self.image_size,
                orig_h=self.image_size,
                target_patches=self.video_target_num_patches,
                maintain_aspect_ratio=self.video_maintain_aspect_ratio,
                patch_size=self.config.patch_size,
                downsample_ratio=self.config.downsample_ratio,
            )
            return feature_size
        return self.num_image_token

    @property
    def supports_video(self) -> bool:
        return True

    @property
    def video_token_id(self) -> int:
        assert self.video_token is not None
        return self.tokenizer.get_vocab()[self.video_token]

    @property
    def image_token_id(self) -> int:
        return self.tokenizer.convert_tokens_to_ids(IMG_CONTEXT)

    def _videos_to_pixel_values_lst(
        self,
        videos: list[npt.NDArray],
    ) -> list[torch.Tensor]:
        return [
            video_to_pixel_values(
                video,
                input_size=self.image_size,
                video_target_num_patches=self.video_target_num_patches,
                video_maintain_aspect_ratio=self.video_maintain_aspect_ratio,
                patch_size=self.config.patch_size,
                downsample_ratio=self.config.downsample_ratio,
            )
            for video in videos
        ]

    def _preprocess_video(
        self,
        text: list[str],
        videos: list[tuple[npt.NDArray, dict[str, Any]]],
    ) -> tuple[list[str], dict[str, Any]]:
        if len(videos) == 0 or not self.supports_video:
            return text, {}

        videos_lst = [v[0] for v in videos]
        video_metadata_lst = [v[1] for v in videos]
        pixel_values_lst_video = self._videos_to_pixel_values_lst(
            videos_lst,
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
        video_num_patches = torch.tensor([len(item) for item in pixel_values_lst_video])
        video_inputs = {
            "pixel_values_flat_video": input_conditioner(
                torch.cat(pixel_values_lst_video), self.norm_mean, self.norm_std
            ),
            "video_num_patches": video_num_patches,
            "frames_indices": frames_indices_lst,
            "frame_duration_ms": torch.tensor(frame_duration_ms_lst),
        }

        patch_size: int = self.config.patch_size
        downsample_ratio = self.config.downsample_ratio

        T = self.video_temporal_patch_size

        for pixel_values, video_metadata, frames_indices, frame_duration_ms in zip(
            pixel_values_lst_video,
            video_metadata_lst,
            frames_indices_lst,
            frame_duration_ms_lst,
        ):
            num_frames = pixel_values.shape[0]
            frame_h, frame_w = pixel_values.shape[-2], pixel_values.shape[-1]
            tokens_in_single_frame = int(
                (frame_h * frame_w // patch_size**2) * (downsample_ratio**2)
            )
            num_tubelets = math.ceil(num_frames / T) if T > 1 else num_frames

            if self.video_pruning_rate is not None and self.video_pruning_rate > 0.0:
                # Start of EVS-specific code
                num_tokens = compute_retained_tokens_count(
                    tokens_per_frame=tokens_in_single_frame,
                    num_frames=num_tubelets,
                    q=self.video_pruning_rate,
                )

                # Here we just need placeholders that won't actually be replaced -
                # we just need to make sure the total number of tokens is correct
                # assign all tokens to the first frame
                tokens_per_frame = [num_tokens] + [0] * (num_tubelets - 1)

                # End of EVS-specific code
            else:
                tokens_per_frame = [tokens_in_single_frame] * num_tubelets

            video_repl = self.get_video_repl(
                tokens_per_frame=tokens_per_frame,
                frames_indices=frames_indices,
                frame_duration_ms=frame_duration_ms,
                tokenizer=self.tokenizer,
                img_start_token_ids=self._img_start_token_ids,
                img_end_token_ids=self._img_end_token_ids,
                img_context_token_ids=self._img_context_token_ids,
                video_temporal_patch_size=T,
            )

            # video_repl.full is a list of token IDs
            # Convert token IDs back to text for the HF processor flow
            video_repl_text = self.tokenizer.decode(
                video_repl.full, skip_special_tokens=False
            )
            text = [t.replace("<video>", video_repl_text, 1) for t in text]

        return text, video_inputs

    def _preprocess_audio(
        self,
        text: list[str],
        audios: list[npt.NDArray],
    ) -> tuple[list[str], dict[str, Any]]:
        if len(audios) == 0:
            return text, {"audio_num_clips": []}

        assert self.audio_extractor is not None
        extractor = self.audio_extractor

        parts = [x for x in re.split(f"({re.escape(AUDIO_CONTEXT)})", text[0]) if x]
        token_count = parts.count(AUDIO_CONTEXT)
        if token_count != len(audios):
            raise ValueError(
                "Number of audio tokens in text does not match the number "
                f"of audios (tokens={token_count}, audios={len(audios)})."
            )
        audio_index = 0
        for idx, part in enumerate(parts):
            if part == AUDIO_CONTEXT:
                audio_repl = self.get_audio_repl(audios[audio_index])
                parts[idx] = audio_repl.full
                audio_index += 1
        text = ["".join(parts)]
        audio_inputs = extractor(
            audios,
            sampling_rate=extractor.sampling_rate,
            return_tensors="pt",
        )
        audio_inputs = {
            "input_audio_features": audio_inputs.input_features,
            "feature_attention_mask": audio_inputs.attention_mask,
            "audio_num_clips": audio_inputs.audio_num_clips,
        }

        return text, audio_inputs

    def __call__(
        self,
        text: str | list[str] | None = None,
        images: Image.Image | list[Image.Image] | None = None,
        videos: tuple[npt.NDArray, dict[str, Any]]
        | list[tuple[npt.NDArray, dict[str, Any]]]
        | None = None,
        audios: AudioItem | list[AudioItem] | None = None,
        *,
        return_tensors: str | TensorType | None = None,
        max_num_tiles: int | None = None,
        **kwargs,
    ) -> BatchFeature:
        # Use default if not provided
        if max_num_tiles is None:
            max_num_tiles = self.max_num_tiles

        text = self._make_batch_input(text)
        images = self._make_batch_input(images)
        videos = self._make_batch_input(videos)
        audios = self._make_batch_input(audios)

        text, image_inputs = self._preprocess_image(
            text=text,
            images=images,
            max_num_tiles=max_num_tiles,
        )

        text, video_inputs = self._preprocess_video(
            text=text,
            videos=videos,
        )

        text, audio_inputs = self._preprocess_audio(
            text=text,
            audios=audios,
        )

        text_inputs = self.tokenizer(text, add_special_tokens=False)

        combined_inputs = {**text_inputs, **video_inputs, **audio_inputs}
        frames_indices = combined_inputs.get("frames_indices")
        ragged_frames_indices = (
            isinstance(frames_indices, list)
            and len({len(frame_indices) for frame_indices in frames_indices}) > 1
        )
        if ragged_frames_indices:
            combined_inputs.pop("frames_indices")

        if self.dynamic_tiler is None:
            batch = BatchFeature(
                {**combined_inputs, **image_inputs},
                tensor_type=return_tensors,
            )
        else:
            batch = BatchFeature(combined_inputs, tensor_type=return_tensors)
            # allow images to be exempt from the BatchFeature validation:
            # We will .stack() them in _parse_and_validate_image_input
            batch.update(image_inputs)
        if ragged_frames_indices:
            assert isinstance(frames_indices, list)
            batch["frames_indices"] = [
                torch.as_tensor(frame_indices, dtype=torch.int64)
                for frame_indices in frames_indices
            ]
        return batch

    def get_image_repl(
        self,
        feature_size: int,
        num_patches: int | None,
    ) -> PromptUpdateDetails[str]:
        repl_features = IMG_CONTEXT * feature_size
        repl_full = IMG_START + repl_features + IMG_END

        return PromptUpdateDetails.select_text(repl_full, IMG_CONTEXT)

    def get_audio_repl(
        self,
        audio: npt.NDArray,
    ) -> PromptUpdateDetails[str]:
        assert self.audio_extractor is not None
        num_tokens = self.audio_extractor.audio_token_count(len(audio))
        repl_full = f"{AUDIO_START}{AUDIO_CONTEXT * num_tokens}{AUDIO_END}"
        return PromptUpdateDetails.select_text(repl_full, AUDIO_CONTEXT)

    @classmethod
    def get_video_repl(
        cls,
        *,
        tokens_per_frame: list[int],
        frames_indices: list[int],
        frame_duration_ms: int,
        tokenizer: HfTokenizer,
        img_start_token_ids: list[int],
        img_end_token_ids: list[int],
        img_context_token_ids: list[int],
        video_temporal_patch_size: int = 1,
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
                (one per tubelet when T > 1)
            frames_indices (list[int]): orig. frame indices
                (one per frame, before tubelet subsampling)
            frame_duration_ms (int): duration of each frame in milliseconds
            tokenizer (TokenizerLike): tokenizer to use for tokenizing frame separators
            img_start_token_ids (list[int]): pre-tokenized IMG_START tokens
            img_end_token_ids (list[int]): pre-tokenized IMG_END tokens
            img_context_token_ids (list[int]): pre-tokenized IMG_CONTEXT tokens
            video_temporal_patch_size (int): temporal patch size for videos
        """
        # TODO: Add support of frame_duration_ms to be None
        # At preprocessing step we should allow absent / metadata without
        # frames_indices field.
        timestamps_enabled = frame_duration_ms is not None
        T = video_temporal_patch_size
        num_frames = len(frames_indices)

        if T > 1 and timestamps_enabled:
            all_timestamps = calculate_timestamps(frames_indices, frame_duration_ms)

            frame_separators = []
            for group_idx, i in enumerate(range(0, num_frames, T)):
                group_frames = []
                for j in range(T):  # Every frame in the group
                    frame_idx = i + j
                    if frame_idx < num_frames:
                        # Valid idx (haven't padded to mult. of T yet)
                        ts = all_timestamps[frame_idx]
                        frame_str = "Frame" if j == 0 else "frame"
                        group_frames.append(
                            f"{frame_str} {frame_idx + 1} sampled at {ts:.2f} seconds"
                        )
                if group_frames:
                    # Join by `and` if there are >1 frame, otherwise no `and`
                    # Prepend \n to match training format (except first group)
                    sep = " and ".join(group_frames) + ": "
                    if group_idx > 0:
                        sep = "\n" + sep
                    frame_separators.append(sep)
        elif timestamps_enabled:
            timestamps = calculate_timestamps(frames_indices, frame_duration_ms)

            assert len(timestamps) == len(tokens_per_frame), (
                "timestamps and tokens_per_frame must have the same length"
            )
            frame_separators = [
                ("\n" if i > 0 else "")
                + f"Frame {i + 1} sampled at {timestamp:.2f} seconds: "
                for i, timestamp in enumerate(timestamps)
            ]
        else:
            frame_separators = [
                ("\n" if i > 0 else "") + f"Frame {i + 1}: "
                for i, _ in enumerate(tokens_per_frame)
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
