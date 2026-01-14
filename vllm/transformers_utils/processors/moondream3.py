# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Custom processor for Moondream3 model."""

import math

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, BatchFeature
from transformers.image_utils import ImageInput
from transformers.processing_utils import ProcessingKwargs, ProcessorMixin, Unpack
from transformers.tokenization_utils_base import PreTokenizedInput, TextInput

from vllm.multimodal.image import convert_image_mode

__all__ = ["Moondream3Processor"]


class Moondream3ProcessorKwargs(ProcessingKwargs, total=False):  # type: ignore[call-arg]
    _defaults = {
        "text_kwargs": {
            "padding": False,
        },
        "images_kwargs": {
            "max_crops": 12,
            "overlap_margin": 4,
            "crop_size": 378,
            "patch_size": 14,
            "convert_to_rgb": True,
            "return_tensors": "pt",
        },
    }


def select_tiling(
    height: int, width: int, crop_size: int, max_crops: int
) -> tuple[int, int]:
    """Determine the optimal number of tiles to cover an image."""
    if height <= crop_size or width <= crop_size:
        return (1, 1)

    min_h = math.ceil(height / crop_size)
    min_w = math.ceil(width / crop_size)

    if min_h * min_w > max_crops:
        ratio = math.sqrt(max_crops / (min_h * min_w))
        return (max(1, math.floor(min_h * ratio)), max(1, math.floor(min_w * ratio)))

    h_tiles = math.floor(math.sqrt(max_crops * height / width))
    w_tiles = math.floor(math.sqrt(max_crops * width / height))

    h_tiles = max(h_tiles, min_h)
    w_tiles = max(w_tiles, min_w)

    if h_tiles * w_tiles > max_crops:
        if w_tiles > h_tiles:
            w_tiles = math.floor(max_crops / h_tiles)
        else:
            h_tiles = math.floor(max_crops / w_tiles)

    return (max(1, h_tiles), max(1, w_tiles))


class Moondream3Processor(ProcessorMixin):
    """
    Constructs a Moondream3 processor which handles image preprocessing
    and tokenization for the Moondream3 multimodal model.

    Args:
        tokenizer: The tokenizer to use for text processing.
        crop_size: Size of each image crop (default: 378).
        max_crops: Maximum number of crops per image (default: 12).
        overlap_margin: Margin for overlapping crops in patches (default: 4).
        patch_size: Size of each patch (default: 14).
    """

    attributes = ["tokenizer"]
    valid_kwargs = [
        "chat_template",
        "crop_size",
        "max_crops",
        "overlap_margin",
        "patch_size",
    ]

    tokenizer_class = "AutoTokenizer"
    # Use separate tokenizer repo
    _tokenizer_repo = "moondream/starmie-v1"

    def __init__(
        self,
        tokenizer=None,
        chat_template=None,
        crop_size: int = 378,
        max_crops: int = 12,
        overlap_margin: int = 4,
        patch_size: int = 14,
        **kwargs,
    ):
        self.image_token = "<image>"
        self.crop_size = crop_size
        self.max_crops = max_crops
        self.overlap_margin = overlap_margin
        self.patch_size = patch_size

        # Number of patches per crop (27x27 = 729 for 378/14)
        self.patches_per_crop = (crop_size // patch_size) ** 2

        super().__init__(tokenizer, chat_template=chat_template)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path,
        **kwargs,
    ):
        """
        Load the processor, using a separate tokenizer repo.

        The moondream3 model uses a custom tokenizer from 'moondream/starmie-v1'
        instead of having tokenizer files in the model repo.
        """
        from transformers import AutoTokenizer

        # Load tokenizer from the separate tokenizer repo
        tokenizer = AutoTokenizer.from_pretrained(
            cls._tokenizer_repo,
            trust_remote_code=kwargs.get("trust_remote_code", False),
        )

        # Extract processor-specific kwargs
        crop_size = kwargs.pop("crop_size", 378)
        max_crops = kwargs.pop("max_crops", 12)
        overlap_margin = kwargs.pop("overlap_margin", 4)
        patch_size = kwargs.pop("patch_size", 14)
        chat_template = kwargs.pop("chat_template", None)

        return cls(
            tokenizer=tokenizer,
            chat_template=chat_template,
            crop_size=crop_size,
            max_crops=max_crops,
            overlap_margin=overlap_margin,
            patch_size=patch_size,
        )

    def __call__(
        self,
        images: ImageInput = None,
        text: TextInput
        | PreTokenizedInput
        | list[TextInput]
        | list[PreTokenizedInput] = None,
        **kwargs: Unpack[Moondream3ProcessorKwargs],
    ) -> BatchFeature:
        """
        Process images and text for Moondream3 model.

        Args:
            images: Input images (PIL Image, numpy array, or list thereof).
            text: Input text or list of texts.
            **kwargs: Additional processing arguments.

        Returns:
            BatchFeature with processed inputs.
        """
        output_kwargs = self._merge_kwargs(
            Moondream3ProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        # Process images
        image_features = {}
        if images is not None:
            processed_images = []
            tilings = []

            images_list = images if isinstance(images, list) else [images]
            for image in images_list:
                pixel_values, tiling = self.preprocess_image(
                    image, **output_kwargs["images_kwargs"]
                )
                processed_images.append(pixel_values)
                tilings.append(tiling)

            if processed_images:
                image_features["pixel_values"] = processed_images
                image_features["tilings"] = tilings

        # Process text
        if text is not None:
            if not isinstance(text, list):
                text = [text]

            # Get text kwargs, remove return_tensors if present (we set it)
            text_kwargs = output_kwargs.get("text_kwargs", {}).copy()
            text_kwargs.pop("return_tensors", None)

            # Tokenize text
            tokenized = self.tokenizer(
                text,
                add_special_tokens=True,
                return_tensors="pt",
                **text_kwargs,
            )

            output = BatchFeature(data=dict(tokenized))

            # Add image features
            if image_features:
                output["pixel_values"] = image_features["pixel_values"]
                output["tilings"] = image_features["tilings"]

            return output

        # If only images were provided
        return BatchFeature(data=image_features)

    def preprocess_image(
        self,
        image: Image.Image,
        max_crops: int = 12,
        overlap_margin: int = 4,
        crop_size: int = 378,
        patch_size: int = 14,
        convert_to_rgb: bool = True,
        return_tensors: str = "pt",
    ) -> tuple[torch.Tensor, tuple[int, int]]:
        """
        Preprocess an image using overlap-and-resize cropping strategy.

        Args:
            image: Input PIL Image.
            max_crops: Maximum number of crops.
            overlap_margin: Margin for overlapping in patches.
            crop_size: Size of each crop.
            patch_size: Size of each patch.
            convert_to_rgb: Whether to convert to RGB.
            return_tensors: Return type ("pt" for PyTorch).

        Returns:
            Tuple of (pixel_values tensor, tiling tuple).
        """
        if convert_to_rgb:
            image = convert_image_mode(image, "RGB")

        # Convert to numpy array
        image_array = np.array(image)
        original_h, original_w = image_array.shape[:2]

        margin_pixels = patch_size * overlap_margin
        total_margin_pixels = margin_pixels * 2

        crop_patches = crop_size // patch_size
        crop_window_patches = crop_patches - (2 * overlap_margin)
        crop_window_size = crop_window_patches * patch_size

        tiling = select_tiling(
            original_h - total_margin_pixels,
            original_w - total_margin_pixels,
            crop_window_size,
            max_crops,
        )

        n_crops = tiling[0] * tiling[1] + 1
        crops = np.zeros((n_crops, crop_size, crop_size, 3), dtype=np.uint8)

        target_size = (
            tiling[0] * crop_window_size + total_margin_pixels,
            tiling[1] * crop_window_size + total_margin_pixels,
        )

        # Resize image
        pil_img = Image.fromarray(image_array)
        resized = pil_img.resize(
            (int(target_size[1]), int(target_size[0])),
            resample=Image.Resampling.LANCZOS,
        )
        resized_array = np.asarray(resized)

        # Create global crop
        global_pil = pil_img.resize(
            (crop_size, crop_size), resample=Image.Resampling.LANCZOS
        )
        crops[0] = np.asarray(global_pil)

        # Create local crops
        for i in range(tiling[0]):
            for j in range(tiling[1]):
                y0 = i * crop_window_size
                x0 = j * crop_window_size
                y_end = min(y0 + crop_size, resized_array.shape[0])
                x_end = min(x0 + crop_size, resized_array.shape[1])

                crop_region = resized_array[y0:y_end, x0:x_end]
                crop_idx = 1 + i * tiling[1] + j
                h_slice = slice(None, crop_region.shape[0])
                w_slice = slice(None, crop_region.shape[1])
                crops[crop_idx, h_slice, w_slice] = crop_region

        # Normalize: (x - 0.5) / 0.5 = 2*x - 1
        # Convert to float and normalize to [-1, 1]
        pixel_values = crops.astype(np.float32) / 255.0
        pixel_values = (pixel_values - 0.5) / 0.5

        # Convert to tensor: (n_crops, H, W, C) -> (n_crops, C, H, W)
        pixel_values = np.transpose(pixel_values, (0, 3, 1, 2))

        if return_tensors == "pt":
            pixel_values = torch.from_numpy(pixel_values)

        return pixel_values, tiling

    def get_num_image_tokens(self) -> int:
        """Return the number of image tokens (729 = 27x27 patches)."""
        return self.patches_per_crop

    def batch_decode(self, *args, **kwargs):
        """Forward to tokenizer's batch_decode."""
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """Forward to tokenizer's decode."""
        return self.tokenizer.decode(*args, **kwargs)

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        return tokenizer_input_names + ["pixel_values", "tilings"]


AutoProcessor.register("Moondream3Processor", Moondream3Processor)
