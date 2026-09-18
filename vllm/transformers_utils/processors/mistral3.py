# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterable
from typing import Any

import numpy as np
import torch
from transformers.models.pixtral import PixtralProcessor
from transformers.models.pixtral.image_processing_pixtral import (
    PixtralImageProcessor,
)

from vllm.utils.import_utils import is_numba_available

_NUMBA_AVAILABLE = is_numba_available()

if _NUMBA_AVAILABLE:
    from numba import njit

    @njit(cache=True, nogil=True)
    def _normalize_uint8_nchw(
        images: np.ndarray,
        normalize_lut: np.ndarray,
        output: np.ndarray,
    ) -> None:
        batch_size, num_channels, height, width = images.shape
        for batch_idx in range(batch_size):
            for channel_idx in range(num_channels):
                for y in range(height):
                    for x in range(width):
                        output[batch_idx, channel_idx, y, x] = normalize_lut[
                            images[batch_idx, channel_idx, y, x], channel_idx
                        ]

else:

    def _normalize_uint8_nchw(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("numba is required for optimized Mistral3 preprocessing")


class Mistral3ImageProcessor(PixtralImageProcessor):
    def rescale_and_normalize(
        self,
        images: torch.Tensor,
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: float | Iterable[float],
        image_std: float | Iterable[float],
    ) -> torch.Tensor:
        if (
            not do_normalize
            or not _NUMBA_AVAILABLE
            or images.device.type != "cpu"
            or images.layout != torch.strided
            or images.dtype != torch.uint8
            or images.ndim != 4
        ):
            return super().rescale_and_normalize(
                images,
                do_rescale,
                rescale_factor,
                do_normalize,
                image_mean,
                image_std,
            )

        original_image_mean = image_mean
        original_image_std = image_std
        fused_mean, fused_std, _ = self._fuse_mean_std_and_rescale_factor(
            do_normalize=do_normalize,
            image_mean=image_mean,
            image_std=image_std,
            do_rescale=do_rescale,
            rescale_factor=rescale_factor,
            device=images.device,
        )
        assert isinstance(fused_mean, torch.Tensor)
        assert isinstance(fused_std, torch.Tensor)
        num_channels = images.shape[1]
        if fused_mean.numel() == 1:
            fused_mean = fused_mean.expand(num_channels)
        if fused_std.numel() == 1:
            fused_std = fused_std.expand(num_channels)
        if fused_mean.numel() != num_channels or fused_std.numel() != num_channels:
            return super().rescale_and_normalize(
                images,
                do_rescale,
                rescale_factor,
                do_normalize,
                original_image_mean,
                original_image_std,
            )
        if (fused_std == 0).any():
            raise ValueError(
                "std evaluated to zero after conversion to "
                f"{torch.float32}, leading to division by zero."
            )

        values = torch.arange(256, dtype=torch.float32).view(-1, 1)
        normalize_lut = values.sub(fused_mean.view(1, -1)).div(fused_std.view(1, -1))
        output = torch.empty(images.shape, dtype=torch.float32)
        _normalize_uint8_nchw(
            images.numpy(),
            normalize_lut.numpy(),
            output.numpy(),
        )
        return output


class Mistral3Processor(PixtralProcessor):
    @classmethod
    def _get_arguments_from_pretrained(
        cls,
        pretrained_model_name_or_path,
        processor_dict=None,
        **kwargs,
    ):
        # Transformers selects Pixtral's tokenizer loader by class name.
        return PixtralProcessor._get_arguments_from_pretrained(
            pretrained_model_name_or_path,
            processor_dict,
            **kwargs,
        )

    def __init__(
        self,
        image_processor=None,
        tokenizer=None,
        patch_size: int = 16,
        spatial_merge_size: int = 1,
        chat_template=None,
        image_token: str = "[IMG]",
        image_break_token: str = "[IMG_BREAK]",
        image_end_token: str = "[IMG_END]",
        **kwargs,
    ) -> None:
        if image_processor is not None and not isinstance(
            image_processor, Mistral3ImageProcessor
        ):
            image_processor = Mistral3ImageProcessor.from_dict(
                image_processor.to_dict()
            )

        super().__init__(
            image_processor=image_processor,
            tokenizer=tokenizer,
            patch_size=patch_size,
            spatial_merge_size=spatial_merge_size,
            chat_template=chat_template,
            image_token=image_token,
            image_break_token=image_break_token,
            image_end_token=image_end_token,
            **kwargs,
        )
