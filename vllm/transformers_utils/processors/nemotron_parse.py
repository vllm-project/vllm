# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Adapted from https://github.com/amalad/vllm/blob/nemotron_parse/vllm/model_executor/models/nemotron_parse.py
# that's based on https://huggingface.co/nvidia/NVIDIA-Nemotron-Parse-v1.1/blob/main/hf_nemotron_parse_modeling.py
from typing import TypeVar

import numpy as np
import torch
from PIL import Image
from timm.data.constants import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD
from torchvision import transforms as T
from transformers import BatchFeature, PretrainedConfig, TensorType

from vllm.tokenizers import TokenizerLike

_T = TypeVar("_T")

DEFAULT_FINAL_IMAGE_SIZE = (2048, 1648)


class NemotronParseImageProcessor:
    """
    NemotronParse Image Processor
    """

    def __init__(
        self,
        final_size: tuple = DEFAULT_FINAL_IMAGE_SIZE,
        **kwargs,
    ):
        # Ensure final_size is properly formatted
        if isinstance(final_size, (list, tuple)) and len(final_size) >= 2:
            self.final_size = (int(final_size[0]), int(final_size[1]))
        elif isinstance(final_size, (int, float)):
            self.final_size = (int(final_size), int(final_size))
        else:
            self.final_size = DEFAULT_FINAL_IMAGE_SIZE  # Default fallback

        self.norm_mean = torch.Tensor(OPENAI_CLIP_MEAN).reshape(1, 3, 1, 1)
        self.norm_std = torch.Tensor(OPENAI_CLIP_STD).reshape(1, 3, 1, 1)

        # Create transforms
        self._create_transforms()

    def _create_transforms(self):
        """Create transform objects."""
        try:
            import albumentations as A
        except ImportError as err:
            raise ImportError(
                "The package `albumentations` is required to use "
                "NemotronParse model. Please install it with `pip install "
                "albumentations`."
            ) from err

        # Ensure final_size is a tuple of integers
        if isinstance(self.final_size, (list, tuple)):
            self.target_height, self.target_width = (
                int(self.final_size[0]),
                int(self.final_size[1]),
            )
        else:
            self.target_height = self.target_width = int(self.final_size)

        import cv2

        self.transform = A.Compose(
            [
                A.PadIfNeeded(
                    min_height=self.target_height,
                    min_width=self.target_width,
                    border_mode=cv2.BORDER_CONSTANT,
                    fill=[255, 255, 255],
                    p=1.0,
                ),
            ]
        )

        self.torch_transform = T.Compose(
            [
                T.ToTensor(),
            ]
        )

    def _resize_with_aspect_ratio(self, image: np.ndarray) -> np.ndarray:
        """Resize image maintaining aspect ratio (exact replica of original
        LongestMaxSizeHW)."""
        height, width = image.shape[:2]
        max_size_height = self.target_height
        max_size_width = self.target_width

        # Original LongestMaxSizeHW algorithm from custom_augmentations.py
        aspect_ratio = width / height
        new_height = height
        new_width = width

        # If height too big then scale image down
        if height > max_size_height:
            new_height = max_size_height
            new_width = int(new_height * aspect_ratio)

        # If width too big, scale image down further
        if new_width > max_size_width:
            new_width = max_size_width
            new_height = int(new_width / aspect_ratio)

        # Use cv2.INTER_LINEAR like the original
        import cv2

        return cv2.resize(
            image, (new_width, new_height), interpolation=cv2.INTER_LINEAR
        )

    def _pad_to_size(self, image: np.ndarray) -> np.ndarray:
        """Pad image to target size with white padding (matches A.PadIfNeeded
        behavior)."""
        h, w = image.shape[:2]
        min_height, min_width = self.target_height, self.target_width

        # Only pad if image is smaller than target (matches A.PadIfNeeded logic)
        pad_h = max(0, min_height - h)
        pad_w = max(0, min_width - w)

        if pad_h == 0 and pad_w == 0:
            return image

        # A.PadIfNeeded pads to bottom-right with constant value
        if len(image.shape) == 3:
            # Color image - pad bottom and right with white (255, 255, 255)
            padded = np.pad(
                image,
                ((0, pad_h), (0, pad_w), (0, 0)),
                mode="constant",
                constant_values=255,
            )
        else:
            # Grayscale image - pad with white (255)
            padded = np.pad(
                image, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=255
            )

        return padded

    def preprocess(
        self,
        images: Image.Image | list[Image.Image],
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        """
        Preprocess an image or batch of images for the NemotronParse model.

        Args:
            images: Input image(s)
        """
        # Ensure images is a list
        if not isinstance(images, list):
            images = [images]

        # Convert PIL images to numpy arrays if needed
        processed_images = []
        for image in images:
            if isinstance(image, Image.Image):
                image = np.asarray(image)
            processed_images.append(image)

        # Apply NemotronParse-specific transforms
        pixel_values = []
        for image in processed_images:
            # Manual resize with aspect ratio preservation
            # (replaces LongestMaxSizeHW)
            processed_image = self._resize_with_aspect_ratio(image)

            # Apply remaining albumentations transforms if available
            if self.transform is not None:
                transformed = self.transform(image=processed_image)
                processed_image = transformed["image"]
            else:
                # Fallback: just pad to target size
                processed_image = self._pad_to_size(processed_image)

            # Convert to tensor
            pixel_values_tensor = self.torch_transform(processed_image)

            # Handle grayscale images
            if pixel_values_tensor.shape[0] == 1:
                pixel_values_tensor = pixel_values_tensor.expand(3, -1, -1)

            pixel_values.append(pixel_values_tensor)

        # Stack into batch
        pixel_values = torch.stack(pixel_values)

        # Normalize pixel values
        normalized_values = (pixel_values - self.norm_mean) / self.norm_std
        return {"pixel_values": normalized_values}

    def __call__(
        self, images: Image.Image | list[Image.Image], **kwargs
    ) -> dict[str, torch.Tensor]:
        return self.preprocess(images, **kwargs)


class NemotronParseProcessor:
    """
    NemotronParse Processor
    """

    def __init__(
        self,
        config: PretrainedConfig,
        tokenizer: TokenizerLike,
        **kwargs,
    ) -> None:
        super().__init__()

        self.config = config
        self.tokenizer = tokenizer

        self.image_processor = NemotronParseImageProcessor(final_size=config.image_size)

    def _make_batch_input(self, input_item: _T | list[_T] | None = None) -> list[_T]:
        if input_item is None:
            input_item = []
        if not isinstance(input_item, list):
            input_item = [input_item]
        return input_item

    def __call__(
        self,
        text: str | list[str] | None = None,
        images: Image.Image | list[Image.Image] | None = None,
        return_tensors: str | TensorType | None = None,
        **kwargs,
    ) -> BatchFeature:
        text = self._make_batch_input(text)
        images = self._make_batch_input(images)
        image_inputs = {} if len(images) == 0 else self.image_processor(images)

        text_inputs = self.tokenizer(text, add_special_tokens=False, **kwargs)
        combined_outputs = BatchFeature(
            data={**text_inputs, **image_inputs},
            tensor_type=return_tensors,
        )
        return combined_outputs
