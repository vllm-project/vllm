# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Multimodal preprocessing for Dots3 NOTE Omni checkpoints."""

import math
from collections.abc import Mapping, Sequence
from functools import cached_property
from typing import Any

import numpy as np
import torch
from PIL import Image
from transformers import BatchFeature

from vllm.config.multimodal import BaseDummyOptions
from vllm.inputs import MultiModalDataDict
from vllm.multimodal.inputs import MultiModalFieldConfig, MultiModalKwargsItems
from vllm.multimodal.parse import MultiModalDataItems, MultiModalDataParser
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder,
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    PromptReplacement,
    PromptUpdate,
    PromptUpdateDetails,
)
from vllm.transformers_utils.repo_utils import get_hf_file_to_dict

IMAGE_START = "<|img|>"
IMAGE_PAD = "<|imgpad|>"
IMAGE_END = "<|endofimg|>"
AUDIO_START = "<|audio_comp_start|>"
AUDIO_PAD = "<|audio_comp_pad|>"
AUDIO_END = "<|audio_comp_end|>"

_HOP_LENGTH = 160
_DEFAULT_SAMPLE_RATE = 16000


def load_note_subconfig(
    model: str,
    revision: str | None,
    subfolder: str,
) -> dict[str, Any] | None:
    return get_hf_file_to_dict(f"{subfolder}/config.json", model, revision)


class DotsNoteImageProcessor:
    """CPU image preprocessing matching the native Cybertron ViT."""

    def __init__(
        self,
        config: Mapping[str, Any],
        image_details: Mapping[str, Any] | None = None,
    ) -> None:
        self.min_pixels = int(config["min_pixels"])
        self.max_pixels = int(config["max_pixels"])
        self.patch_size = int(config["patch_size"])
        self.temporal_patch_size = int(config["temporal_patch_size"])
        self.merge_size = int(config["merge_size"])
        self.pre_pixel_shuffle = bool(config.get("pre_pixel_shuffle", True))
        self.image_mean = np.asarray(config["image_mean"], dtype=np.float32)
        self.image_std = np.asarray(config["image_std"], dtype=np.float32)
        self.image_details = dict(image_details or {})

    @property
    def factor(self) -> int:
        return self.patch_size * self.merge_size

    @staticmethod
    def _round_by_factor(value: int, factor: int) -> int:
        return round(value / factor) * factor

    @staticmethod
    def _ceil_by_factor(value: float, factor: int) -> int:
        return math.ceil(value / factor) * factor

    @staticmethod
    def _floor_by_factor(value: float, factor: int) -> int:
        return math.floor(value / factor) * factor

    def resized_size(
        self,
        width: int,
        height: int,
        *,
        detail: str = "auto",
        min_pixels: int | None = None,
        max_pixels: int | None = None,
        target_height: int | None = None,
        target_width: int | None = None,
    ) -> tuple[int, int]:
        detail_config = self.image_details.get(detail, {})
        min_pixels = int(
            min_pixels
            if min_pixels is not None
            else detail_config.get("min_pixels", self.min_pixels)
        )
        max_pixels = int(
            max_pixels
            if max_pixels is not None
            else detail_config.get("max_pixels", self.max_pixels)
        )
        height = int(target_height or detail_config.get("target_height") or height)
        width = int(target_width or detail_config.get("target_width") or width)

        factor = self.factor
        if min(height, width) < factor // 4:
            raise ValueError(
                f"Image height and width must be at least {factor // 4}, "
                f"got {height}x{width}"
            )
        if max(height, width) / min(height, width) > 200:
            raise ValueError("Image aspect ratio must be smaller than 200")

        resized_h = max(factor, self._round_by_factor(height, factor))
        resized_w = max(factor, self._round_by_factor(width, factor))
        if resized_h * resized_w > max_pixels:
            beta = math.sqrt(height * width / max_pixels)
            resized_h = max(factor, self._floor_by_factor(height / beta, factor))
            resized_w = max(factor, self._floor_by_factor(width / beta, factor))
        elif resized_h * resized_w < min_pixels:
            beta = math.sqrt(min_pixels / (height * width))
            resized_h = self._ceil_by_factor(height * beta, factor)
            resized_w = self._ceil_by_factor(width * beta, factor)
            if resized_h * resized_w > max_pixels:
                beta = math.sqrt(resized_h * resized_w / max_pixels)
                resized_h = max(factor, self._floor_by_factor(resized_h / beta, factor))
                resized_w = max(factor, self._floor_by_factor(resized_w / beta, factor))
        return resized_h, resized_w

    def preprocess(
        self,
        image: Image.Image,
        *,
        detail: str = "auto",
        **size_overrides: object,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not isinstance(image, Image.Image):
            raise TypeError(f"Expected a PIL image, got {type(image)}")
        if image.mode == "RGBA":
            background = Image.new("RGB", image.size, (255, 255, 255))
            background.paste(image, mask=image.getchannel("A"))
            image = background
        elif image.mode != "RGB":
            image = image.convert("RGB")

        resized_h, resized_w = self.resized_size(
            *image.size,
            detail=detail,
            **size_overrides,
        )
        image = image.resize((resized_w, resized_h), Image.Resampling.BICUBIC)
        array = np.asarray(image, dtype=np.float32) / 255.0
        array = (array - self.image_mean) / self.image_std
        patches = array.transpose(2, 0, 1)[None]
        if patches.shape[0] == 1:
            patches = np.tile(patches, (self.temporal_patch_size, 1, 1, 1))

        channel = patches.shape[1]
        grid_t = patches.shape[0] // self.temporal_patch_size
        grid_h = resized_h // self.patch_size
        grid_w = resized_w // self.patch_size
        if self.pre_pixel_shuffle:
            patches = patches.reshape(
                grid_t,
                self.temporal_patch_size,
                channel,
                grid_h // self.merge_size,
                self.merge_size,
                self.patch_size,
                grid_w // self.merge_size,
                self.merge_size,
                self.patch_size,
            )
            patches = patches.transpose(0, 3, 6, 4, 7, 2, 1, 5, 8)
        else:
            patches = patches.reshape(
                grid_t,
                self.temporal_patch_size,
                channel,
                grid_h,
                self.patch_size,
                grid_w,
                self.patch_size,
            )
            patches = patches.transpose(0, 3, 5, 2, 1, 4, 6)

        pixel_values = torch.from_numpy(
            patches.reshape(
                grid_t * grid_h * grid_w,
                channel * self.temporal_patch_size * self.patch_size * self.patch_size,
            )
        )
        grid_thw = torch.tensor([grid_t, grid_h, grid_w], dtype=torch.long)
        return pixel_values, grid_thw


class DotsNoteOmniProcessor:
    """Small HF-like processor used by vLLM's multimodal frontend."""

    def __init__(
        self,
        tokenizer,
        image_processor: DotsNoteImageProcessor | None,
    ) -> None:
        self.tokenizer = tokenizer
        self.image_processor = image_processor

    def __call__(self, text: str, **kwargs: object) -> BatchFeature:
        modality_order = [key for key in kwargs if key in ("images", "audios")]
        images = list(kwargs.pop("images", []) or [])
        audios = list(kwargs.pop("audios", []) or [])
        detail = kwargs.pop("image_detail", "auto")
        details = list(detail) if isinstance(detail, (list, tuple)) else None
        size_overrides = {
            key: kwargs.pop(key)
            for key in ("min_pixels", "max_pixels", "target_height", "target_width")
            if key in kwargs
        }

        tokenized = self.tokenizer(text, **kwargs)
        input_ids = tokenized["input_ids"]
        if isinstance(input_ids, torch.Tensor):
            if input_ids.ndim == 1:
                input_ids = input_ids.unsqueeze(0)
        elif input_ids and isinstance(input_ids[0], int):
            input_ids = [input_ids]
        data: dict[str, object] = {"input_ids": input_ids}

        for modality in modality_order:
            if modality == "images" and images:
                if self.image_processor is None:
                    raise ValueError("This NOTE checkpoint has no vision encoder")
                pixel_values = []
                grids = []
                for idx, image in enumerate(images):
                    image_detail = details[idx] if details is not None else str(detail)
                    pixels, grid = self.image_processor.preprocess(
                        image,
                        detail=image_detail,
                        **size_overrides,
                    )
                    pixel_values.append(pixels)
                    grids.append(grid)
                data["pixel_values"] = torch.cat(pixel_values)
                data["image_grid_thw"] = torch.stack(grids)
            elif modality == "audios" and audios:
                waveforms = [
                    torch.from_numpy(
                        np.ascontiguousarray(audio, dtype=np.float32)
                    ).view(torch.int32)
                    for audio in audios
                ]
                data["audio_values"] = torch.cat(waveforms)
                data["audio_lengths"] = torch.tensor(
                    [waveform.numel() for waveform in waveforms],
                    dtype=torch.long,
                )
        return BatchFeature(data=data)


class DotsNoteOmniProcessingInfo(BaseProcessingInfo):
    @cached_property
    def vision_config(self) -> dict[str, Any] | None:
        model_config = self.ctx.model_config
        return load_note_subconfig(
            model_config.model,
            model_config.revision,
            "new_ve",
        )

    @cached_property
    def audio_config(self) -> dict[str, Any] | None:
        model_config = self.ctx.model_config
        return load_note_subconfig(
            model_config.model,
            model_config.revision,
            "new_ae",
        )

    @cached_property
    def image_processor(self) -> DotsNoteImageProcessor | None:
        if self.vision_config is None:
            return None
        model_config = self.ctx.model_config
        preprocessor_config = get_hf_file_to_dict(
            "new_ve/preprocessor_config.json",
            model_config.model,
            model_config.revision,
        )
        if preprocessor_config is None:
            raise ValueError(
                "NOTE vision checkpoint is missing preprocessor_config.json"
            )
        image_detail_config = get_hf_file_to_dict(
            "new_ve/image_detail.json",
            model_config.model,
            model_config.revision,
        )
        image_details = (image_detail_config or {}).get("image_details", {})
        return DotsNoteImageProcessor(preprocessor_config, image_details)

    @cached_property
    def processor(self) -> DotsNoteOmniProcessor:
        return DotsNoteOmniProcessor(self.get_tokenizer(), self.image_processor)

    def get_hf_processor(self, **kwargs: object) -> DotsNoteOmniProcessor:
        return self.processor

    def get_data_parser(self) -> MultiModalDataParser:
        sample_rate = int((self.audio_config or {}).get("sampling_rate", 16000))
        return MultiModalDataParser(target_sr=float(sample_rate), target_channels=1)

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        limits: dict[str, int | None] = {}
        if self.vision_config is not None:
            limits["image"] = None
        if self.audio_config is not None:
            limits["audio"] = None
        return limits

    def get_max_image_size(self) -> tuple[int, int]:
        image_processor = self.image_processor
        if image_processor is None:
            return 0, 0
        factor = image_processor.factor
        max_tokens = image_processor.max_pixels // (factor * factor)
        for height_factor in range(math.isqrt(max_tokens), 0, -1):
            if max_tokens % height_factor == 0:
                width_factor = max_tokens // height_factor
                if width_factor / height_factor <= 200:
                    return factor * width_factor, factor * height_factor
        return factor * max_tokens, factor

    def get_mm_max_tokens_per_item(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> Mapping[str, int]:
        result: dict[str, int] = {}
        if self.image_processor is not None:
            factor = self.image_processor.factor
            result["image"] = self.image_processor.max_pixels // (factor * factor)
        if self.audio_config is not None:
            stride = (
                _HOP_LENGTH
                * (8 if self.audio_config.get("use_conv2d_stem", True) else 2)
                * int(self.audio_config.get("merge_factor", 1))
            )
            chunk_samples = int(
                self.audio_config.get("chunk_seconds", 60)
                * self.audio_config.get("sampling_rate", _DEFAULT_SAMPLE_RATE)
            )
            result["audio"] = min(seq_len, math.ceil(chunk_samples / stride))
        return result


class DotsNoteOmniDummyInputsBuilder(
    BaseDummyInputsBuilder[DotsNoteOmniProcessingInfo]
):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        return f"{IMAGE_START}{IMAGE_PAD}{IMAGE_END}" * mm_counts.get(
            "image", 0
        ) + f"{AUDIO_START}{AUDIO_PAD}{AUDIO_END}" * mm_counts.get("audio", 0)

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions],
    ) -> MultiModalDataDict:
        data: MultiModalDataDict = {}
        num_images = mm_counts.get("image", 0)
        if num_images:
            width, height = self.info.get_max_image_size()
            data["image"] = self._get_dummy_images(
                width=width,
                height=height,
                num_images=num_images,
                overrides=mm_options.get("image"),
            )
        num_audios = mm_counts.get("audio", 0)
        if num_audios:
            audio_config = self.info.audio_config or {}
            stride = (
                _HOP_LENGTH
                * (8 if audio_config.get("use_conv2d_stem", True) else 2)
                * int(audio_config.get("merge_factor", 1))
            )
            chunk_samples = int(
                audio_config.get("chunk_seconds", 60)
                * audio_config.get("sampling_rate", _DEFAULT_SAMPLE_RATE)
            )
            data["audio"] = self._get_dummy_audios(
                length=min(chunk_samples, max(1, seq_len) * stride),
                num_audios=num_audios,
                overrides=mm_options.get("audio"),
            )
        return data


class DotsNoteOmniMultiModalProcessor(
    BaseMultiModalProcessor[DotsNoteOmniProcessingInfo]
):
    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        return super()._call_hf_processor(prompt, mm_data, mm_kwargs, tok_kwargs)

    def _hf_processor_applies_updates(
        self,
        prompt_text: str,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        tokenization_kwargs: Mapping[str, object],
    ) -> bool:
        return False

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        fields: dict[str, MultiModalFieldConfig] = {}
        if "pixel_values" in hf_inputs:
            grid_thw = hf_inputs["image_grid_thw"]
            fields["pixel_values"] = MultiModalFieldConfig.flat_from_sizes(
                "image", grid_thw.prod(-1)
            )
            fields["image_grid_thw"] = MultiModalFieldConfig.batched(
                "image", keep_on_cpu=True
            )
        if "audio_values" in hf_inputs:
            fields["audio_values"] = MultiModalFieldConfig.flat_from_sizes(
                "audio", hf_inputs["audio_lengths"]
            )
            fields["audio_lengths"] = MultiModalFieldConfig.batched(
                "audio", keep_on_cpu=True
            )
        return fields

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        vocab = self.info.get_tokenizer().get_vocab()
        updates: list[PromptUpdate] = []

        if "image" in out_mm_kwargs:
            image_start_id = vocab[IMAGE_START]
            image_pad_id = vocab[IMAGE_PAD]
            image_end_id = vocab[IMAGE_END]
            merge_size = self.info.image_processor.merge_size  # type: ignore[union-attr]

            def image_replacement(item_idx: int) -> PromptUpdateDetails[list[int]]:
                grid = out_mm_kwargs["image"][item_idx]["image_grid_thw"].data
                num_tokens = int(grid.prod()) // merge_size**2
                return PromptUpdateDetails.select_token_id(
                    [image_start_id] + [image_pad_id] * num_tokens + [image_end_id],
                    image_pad_id,
                )

            updates.append(
                PromptReplacement(
                    modality="image",
                    target=[image_start_id, image_pad_id, image_end_id],
                    replacement=image_replacement,
                )
            )

        if "audio" in out_mm_kwargs:
            audio_start_id = vocab[AUDIO_START]
            audio_pad_id = vocab[AUDIO_PAD]
            audio_end_id = vocab[AUDIO_END]
            config = self.info.audio_config or {}
            stride = (
                _HOP_LENGTH
                * (8 if config.get("use_conv2d_stem", True) else 2)
                * int(config.get("merge_factor", 1))
            )

            def audio_replacement(item_idx: int) -> PromptUpdateDetails[list[int]]:
                length = out_mm_kwargs["audio"][item_idx]["audio_lengths"].data
                num_tokens = math.ceil(int(length) / stride)
                return PromptUpdateDetails.select_token_id(
                    [audio_start_id] + [audio_pad_id] * num_tokens + [audio_end_id],
                    audio_pad_id,
                )

            updates.append(
                PromptReplacement(
                    modality="audio",
                    target=[audio_start_id, audio_pad_id, audio_end_id],
                    replacement=audio_replacement,
                )
            )
        return updates
