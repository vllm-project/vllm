# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

import math
from typing import Any

import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F
from transformers import BatchFeature, ProcessorMixin, TensorType
from typing_extensions import TypedDict, Unpack

MAX_PIXELS = 60_000_000  # 60-megapixel ceiling ≈ 8200 × 7300 px

# Vision preprocessing constants
VISION_MEAN = (0.5, 0.5, 0.5)
VISION_STD = (0.5, 0.5, 0.5)
VISION_SCALE = 1 / 255


def _make_writeable(arr: np.ndarray) -> np.ndarray:
    """Return *arr* itself if it is already writeable, otherwise try to flip the
    write flag in-place and finally fall back to `arr.copy()`.
    This guarantees the buffer handed to `torch.from_numpy()` is always
    writeable, silencing the PyTorch warning about undefined behaviour.
    """
    if arr.flags.writeable:
        return arr

    # First, try the cheap path — in-place flag toggle (works for mmap'd arrays
    # and some shared memory buffers):
    try:
        arr.setflags(write=True)
        return arr  # success: no data copy
    except ValueError:
        # Buffer is inherently read-only (e.g. backed by PyAV / PIL): make copy
        return arr.copy()


def extract_image_pil(image: PIL.Image.Image) -> torch.Tensor | None:
    if image.width * image.height > MAX_PIXELS:
        raise ValueError(
            f"Image (w={image.width}, h={image.height}) > MAX=`{MAX_PIXELS}`"
        )
    img = image if image.mode == "RGB" else image.convert("RGB")
    arr = np.asarray(img)
    arr = _make_writeable(arr)
    return torch.from_numpy(arr)


def get_image_size_for_max_num_patches(
    image_height: int,
    image_width: int,
    patch_size: int,
    max_num_patches: int,
    min_num_patches: int | None = None,
    eps: float = 1e-5,
    pixel_shuffle_scale: int = 1,
) -> tuple[int, int]:
    r"""Compute a target resolution whose patch grid satisfies patching parametrization.

    Args:
        image_height (`int`):
            Height in pixels of the source image prior to any resizing.
        image_width (`int`):
            Width in pixels of the source image prior to any resizing.
        patch_size (`int`):
            Size of the square patch used by the vision encoder.
        max_num_patches (`int`):
            Upper bound on `(height / patch_size) * (width / patch_size)` after
            resizing.
        min_num_patches (`int`, *optional*):
            Lower bound on the number of patches. When provided the image will
            be scaled up if necessary.
        eps (`float`, *optional*, defaults to 1e-5):
            Convergence tolerance for the internal binary search to determine
            the target dimensions.
        pixel_shuffle_scale (`int`, *optional*, defaults to 1):
            Additional stride multiplier applied when pixel shuffle later
            reduces spatial resolution.

    Returns:
        `tuple[int, int]`: Height and width (in pixels) that are multiples of
        `patch_size * pixel_shuffle_scale` and respect both the maximum and
        optional minimum patch-count constraints.
    """

    def get_scaled_image_size(scale, original_size, patch_size, pixel_shuffle_scale):
        scaled_size = scale * original_size
        divisor = patch_size * pixel_shuffle_scale
        scaled_size = math.ceil(scaled_size / divisor) * divisor
        scaled_size = max(divisor, scaled_size)
        return int(scaled_size)

    # Ensure divisibility
    divisor = patch_size * pixel_shuffle_scale
    adjusted_height = math.ceil(image_height / divisor) * divisor
    adjusted_height = max(divisor, adjusted_height)
    adjusted_width = math.ceil(image_width / divisor) * divisor
    adjusted_width = max(divisor, adjusted_width)

    num_patches = (adjusted_height / patch_size) * (adjusted_width / patch_size)

    if min_num_patches is not None and num_patches < min_num_patches:
        # Scale up
        scale_min, scale_max = 1.0, 100.0
        while (scale_max - scale_min) >= eps:
            scale = (scale_min + scale_max) / 2
            target_height = get_scaled_image_size(
                scale, image_height, patch_size, pixel_shuffle_scale
            )
            target_width = get_scaled_image_size(
                scale, image_width, patch_size, pixel_shuffle_scale
            )
            num_patches = (target_height / patch_size) * (target_width / patch_size)
            if num_patches >= min_num_patches:
                scale_max = scale
            else:
                scale_min = scale
        scale = scale_max
        target_height = get_scaled_image_size(
            scale, image_height, patch_size, pixel_shuffle_scale
        )
        target_width = get_scaled_image_size(
            scale, image_width, patch_size, pixel_shuffle_scale
        )
        return target_height, target_width
    elif num_patches <= max_num_patches:
        return adjusted_height, adjusted_width
    else:
        # Scale down
        scale_min, scale_max = eps / 10, 1.0
        while (scale_max - scale_min) >= eps:
            scale = (scale_min + scale_max) / 2
            target_height = get_scaled_image_size(
                scale, image_height, patch_size, pixel_shuffle_scale
            )
            target_width = get_scaled_image_size(
                scale, image_width, patch_size, pixel_shuffle_scale
            )
            num_patches = (target_height / patch_size) * (target_width / patch_size)
            if num_patches <= max_num_patches:
                scale_min = scale
            else:
                scale_max = scale
        scale = scale_min
        target_height = get_scaled_image_size(
            scale, image_height, patch_size, pixel_shuffle_scale
        )
        target_width = get_scaled_image_size(
            scale, image_width, patch_size, pixel_shuffle_scale
        )
        return target_height, target_width


_MEAN_TENSOR = torch.tensor(VISION_MEAN, dtype=torch.float32).view(1, 1, 1, -1)
_STD_TENSOR = torch.tensor(VISION_STD, dtype=torch.float32).view(1, 1, 1, -1)


def prepare_image_tensor(
    image: torch.Tensor,
    scale: float = VISION_SCALE,
) -> torch.Tensor:
    r"""Standardize RGB images prior to patch extraction via rescaling and whitening.

    Args:
        image (`torch.Tensor`):
            Tensor with shape `(..., height, width, 3)` containing RGB values.
            The tensor is converted to floating point if needed.
        scale (`float`, *optional*, defaults to `VISION_SCALE`):
            Scalar multiplier applied before normalization.
    Returns:
        `torch.Tensor`: Normalized tensor with the same shape as the input and
        dtype `torch.float32`.
    """
    if not torch.is_floating_point(image):
        image = image.float()
    rescaled = image * scale

    # Use precomputed tensors and move to the correct device if needed
    mean_tensor = _MEAN_TENSOR.to(image.device)
    std_tensor = _STD_TENSOR.to(image.device)

    normalized = (rescaled - mean_tensor) / std_tensor
    return normalized


def patchify_vision(image: torch.Tensor, patch_size: int) -> torch.Tensor:
    r"""Convert normalized images into flattened ViT-style patches.

    Args:
        image (`torch.Tensor`):
            Tensor of shape `(num_images, height, width, channels)`.
        patch_size (`int`):
            Edge length of the square patches

    Returns:
        `torch.Tensor`:
            Patch tensor where each position stores the flattened pixels
            belonging to that patch.

    Raises:
        ValueError: If `height` or `width` is not divisible by `patch_size`.
    """
    num_images, height, width, channels = image.shape
    if height % patch_size or width % patch_size:
        raise ValueError(
            "Dimensions of images "
            f"{image.shape} are not divisible by patch_size={patch_size}."
        )
    patches = image.reshape(
        num_images,
        height // patch_size,
        patch_size,
        width // patch_size,
        patch_size,
        channels,
    )
    patches = patches.permute(0, 1, 3, 2, 4, 5)
    patches = patches.reshape(
        num_images,
        height // patch_size,
        width // patch_size,
        channels * patch_size * patch_size,
    )
    return patches


def process_vision_for_patches(
    images: torch.Tensor,
    patch_size: int,
    max_num_patches: int,
    min_num_patches: int | None = None,
    pixel_shuffle_scale: int = 1,
) -> tuple[torch.Tensor, list[int]]:
    r"""Resize, normalize, and patchify RGB images for the vision encoder.

    Args:
        images (`torch.Tensor`):
            Either `(height, width, channels)` for a single image or
            `(num_images, height, width, channels)` for a batch. Channels are
            expected to be RGB.
        patch_size (`int`):
            Edge length of square patches; implicitly controls resize grid granularity.
        max_num_patches (`int`):
            Maximum number of patches allowed after resizing.
        min_num_patches (`int`, *optional*):
            Minimum number of patches. If provided, the routine upsamples images
            as needed to satisfy the lower bound.
        pixel_shuffle_scale (`int`, *optional*, defaults to 1):
            Pixel shuffle scale factor; influences the target grid that the
            function produces.

    Returns:
        `tuple[torch.Tensor, list[int]]`: A pair `(patches, dims_virtual)`
        where `patches` has shape `(num_images, target_h / patch_size, target_w
        / patch_size, channels * patch_size**2)` and `dims_virtual` encodes
        effective `(images, height, width)` dimensions after optional pixel
        shuffling.
    """
    # Add batch dim if single image
    if images.dim() == 3:
        images = images.unsqueeze(0)

    # Permute to channel first for resize
    images = images.permute(0, 3, 1, 2)

    # Get target dimensions
    _, _, orig_height, orig_width = images.shape
    target_height, target_width = get_image_size_for_max_num_patches(
        orig_height,
        orig_width,
        patch_size,
        max_num_patches,
        min_num_patches=min_num_patches,
        pixel_shuffle_scale=pixel_shuffle_scale,
    )

    # Resize
    images = F.interpolate(
        images,
        size=(target_height, target_width),
        mode="bilinear",
        align_corners=False,
    )

    # Back to channel last
    images = images.permute(0, 2, 3, 1)

    # Normalize
    images = prepare_image_tensor(images)

    # Patchify
    patches = patchify_vision(images, patch_size=patch_size)

    # Calculate dimensions for the patches
    n_images, h_patches, w_patches, _ = patches.shape
    dims_virtual = (
        [1, h_patches, w_patches]
        if pixel_shuffle_scale == 1
        else [1, h_patches // pixel_shuffle_scale, w_patches // pixel_shuffle_scale]
    )

    return patches, dims_virtual


class IsaacImageProcessorKwargs(TypedDict, total=False):
    patch_size: int
    max_num_patches: int
    min_num_patches: int
    pixel_shuffle_scale: int


class IsaacImageProcessor:
    patch_size = 16
    max_num_patches = 6144
    min_num_patches = 256
    pixel_shuffle_scale = 2

    valid_kwargs = IsaacImageProcessorKwargs
    model_input_names = ["pixel_values", "image_grid_thw"]

    def __init__(self, kwargs):
        self.patch_size = kwargs.pop("patch_size", self.patch_size)
        self.vision_max_num_patches = kwargs.pop(
            "vision_max_num_patches", self.max_num_patches
        )
        self.vision_min_num_patches = kwargs.pop(
            "vision_min_num_patches", self.min_num_patches
        )
        self.pixel_shuffle_scale = kwargs.pop("pixel_shuffle_scale", 2)

    def preprocess(
        self,
        images: list[torch.Tensor],
        return_tensors: str | TensorType | None,
        **kwargs: Unpack[IsaacImageProcessorKwargs],
    ) -> BatchFeature:
        """Preprocess images into format compatible with vLLM input processing."""

        all_pixel_values: list[torch.Tensor] = []
        all_image_grids: list[torch.Tensor] = []

        for image in images:
            image_tensor = extract_image_pil(image)

            patches, dims_virtual = process_vision_for_patches(
                image_tensor,
                patch_size=self.patch_size,
                max_num_patches=self.vision_max_num_patches,
                min_num_patches=self.vision_min_num_patches,
                pixel_shuffle_scale=self.pixel_shuffle_scale,
            )

            # Isaac packs a dummy temporal dim for images
            patches = patches.unsqueeze(1)  # [N, T=1, Hp, Wp, D]

            hp, wp, dim = patches.shape[-3], patches.shape[-2], patches.shape[-1]
            current_num_patches = hp * wp
            pixel_values = patches.reshape(current_num_patches, dim)  # [N_tokens, D]

            # Use real patch dimensions for image_grid_thw, not virtual dimensions
            # This ensures the vision model receives correct grid info for pixel shuffle
            dims_real = [1, hp, wp]  # Real patch dimensions
            image_grid_thw = torch.tensor(dims_real).unsqueeze(0)

            all_pixel_values.append(pixel_values)
            all_image_grids.append(image_grid_thw)

        if all_pixel_values:
            final_pixel_values = torch.cat(all_pixel_values, dim=0)
            final_image_grids = torch.cat(all_image_grids, dim=0)
        else:
            final_pixel_values = torch.empty(0, 0)
            final_image_grids = torch.empty(0, 3)

        return BatchFeature(
            data={
                "pixel_values": final_pixel_values,
                "image_grid_thw": final_image_grids,
            },
            tensor_type=return_tensors,
        )


class IsaacProcessor(ProcessorMixin):
    attributes = ["image_processor", "tokenizer"]

    def __init__(self, image_processor=None, tokenizer=None, **kwargs):
        self.image_token = kwargs.pop("image_token", "<image>")
        self.image_processor = image_processor
        self.tokenizer = tokenizer

    def __call__(self, text=None, images=None, **kwargs) -> BatchFeature:
        result = {}

        if images is not None:
            image_inputs = self.image_processor.preprocess(images, **kwargs)
            image_grid_thw = image_inputs["image_grid_thw"]
            result.update(image_inputs)

            if text is not None:
                if not isinstance(text, list):
                    text = [text]

                text = text.copy()  # below lines change text in-place
                merge_length = self.image_processor.pixel_shuffle_scale**2
                index = 0
                for i in range(len(text)):
                    while self.image_token in text[i]:
                        num_image_tokens = image_grid_thw[index].prod() // merge_length
                        text[i] = text[i].replace(
                            self.image_token, "<|placeholder|>" * num_image_tokens, 1
                        )
                        index += 1
                    text[i] = text[i].replace("<|placeholder|>", "<|image_pad|>")

        if text is not None:
            result.update(self.tokenizer(text, **kwargs))

        return BatchFeature(result)

    def apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        tokenize: bool = False,
        add_generation_prompt: bool = False,
        **kwargs,
    ) -> Any:
        # Convert mixed content messages to simple text format
        processed_messages = []

        for message in messages:
            if "content" in message and isinstance(message["content"], list):
                # Handle mixed content (text + image)
                text_parts = []
                for content_item in message["content"]:
                    if content_item.get("type") == "text":
                        text_parts.append(content_item.get("text", ""))
                    elif content_item.get("type") == "image":
                        # Replace image with vision token
                        text_parts.append(self.image_token)

                processed_message = {
                    "role": message.get("role", "user"),
                    "content": "".join(text_parts),
                }
                processed_messages.append(processed_message)
            else:
                # Regular text message
                processed_messages.append(message)

        kwargs["return_dict"] = False
        return self.tokenizer.apply_chat_template(
            processed_messages,
            tokenize=tokenize,
            add_generation_prompt=add_generation_prompt,
            **kwargs,
        )
