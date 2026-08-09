# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Multimodal preprocessing for Dots3Note checkpoints."""

import math
from collections.abc import Mapping, Sequence
from functools import cached_property
from typing import Any, cast

import numpy as np
import torch
from PIL import Image
from transformers import BatchFeature

from vllm.config.multimodal import (
    AudioDummyOptions,
    BaseDummyOptions,
    ImageDummyOptions,
    VideoDummyOptions,
)
from vllm.inputs import MultiModalDataDict
from vllm.multimodal.inputs import MultiModalFieldConfig, MultiModalKwargsItems
from vllm.multimodal.media import MediaWithBytes
from vllm.multimodal.parse import (
    MultiModalDataItems,
    MultiModalDataParser,
    VideoProcessorItems,
)
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder,
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    PromptReplacement,
    PromptUpdate,
    PromptUpdateDetails,
)
from vllm.transformers_utils.repo_utils import get_hf_file_to_dict

from .video import preprocess_dots3note_video

IMAGE_START = "<|img|>"
IMAGE_PAD = "<|imgpad|>"
IMAGE_END = "<|endofimg|>"
AUDIO_START = "<|audio_comp_start|>"
AUDIO_PAD = "<|audio_comp_pad|>"
AUDIO_END = "<|audio_comp_end|>"
VIDEO_PLACEHOLDER = "<|video|>"

_HOP_LENGTH = 160
_DEFAULT_SAMPLE_RATE = 16000


def load_note_config_section(
    model: str,
    revision: str | None,
    section: str,
) -> dict[str, Any] | None:
    config = get_hf_file_to_dict("config.json", model, revision)
    value = (config or {}).get(section)
    return value if isinstance(value, dict) else None


class Dots3NoteImageProcessor:
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
        min_pixels: int | None = None,
        max_pixels: int | None = None,
        target_height: int | None = None,
        target_width: int | None = None,
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
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            target_height=target_height,
            target_width=target_width,
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


class Dots3NoteProcessor:
    """Small HF-like processor used by vLLM's multimodal frontend."""

    def __init__(
        self,
        tokenizer,
        image_processor: Dots3NoteImageProcessor | None,
        *,
        max_model_len: int,
        video_audio_enabled: bool,
        audio_token_stride: int,
    ) -> None:
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.max_model_len = max_model_len
        self.video_audio_enabled = video_audio_enabled
        self.audio_token_stride = audio_token_stride

    def _token_ids(self, text: str) -> list[int]:
        if hasattr(self.tokenizer, "encode"):
            return self.tokenizer.encode(text, add_special_tokens=False)
        token_ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        if token_ids and isinstance(token_ids[0], list):
            return token_ids[0]
        return token_ids

    def _process_video(
        self,
        video,
        *,
        detail: str,
        size_overrides: Mapping[str, int | None],
        seq: int,
        output_reserve: int | None,
        audio_cap: float,
        audio_sample_rate: int,
        k_mode: str,
        max_new_tokens: int,
        question: str,
    ) -> dict[str, torch.Tensor]:
        if self.image_processor is None:
            raise ValueError("This NOTE checkpoint has no vision encoder")
        if not self.video_audio_enabled:
            audio_cap = 0.0
        parts = preprocess_dots3note_video(
            video,
            tokenizer=self.tokenizer,
            question=question,
            seq=seq,
            output_reserve=output_reserve,
            audio_cap=audio_cap,
            audio_sample_rate=audio_sample_rate,
            k_mode=k_mode,
            max_new_tokens=max_new_tokens,
        )

        fragments: list[str] = []
        pixel_values: list[torch.Tensor] = []
        image_grids: list[torch.Tensor] = []
        audio_values: list[torch.Tensor] = []
        modalities: list[int] = []
        image_pad_count = 0
        audio_pad_count = 0

        for part in parts:
            if part.kind == "text":
                fragments.append(cast(str, part.value))
                continue
            if part.kind == "image":
                pixels, grid = self.image_processor.preprocess(
                    cast(Image.Image, part.value),
                    detail=detail,
                    min_pixels=size_overrides.get("min_pixels"),
                    max_pixels=size_overrides.get("max_pixels"),
                    target_height=size_overrides.get("target_height"),
                    target_width=size_overrides.get("target_width"),
                )
                num_tokens = int(grid.prod()) // self.image_processor.merge_size**2
                fragments.append(f"{IMAGE_START}{IMAGE_PAD * num_tokens}{IMAGE_END}")
                pixel_values.append(pixels)
                image_grids.append(grid)
                modalities.append(0)
                image_pad_count += num_tokens
                continue

            waveform = torch.from_numpy(
                np.ascontiguousarray(part.value, dtype=np.float32)
            ).view(torch.int32)
            num_tokens = math.ceil(waveform.numel() / self.audio_token_stride)
            fragments.append(f"{AUDIO_START}{AUDIO_PAD * num_tokens}{AUDIO_END}")
            audio_values.append(waveform)
            modalities.append(1)
            audio_pad_count += num_tokens

        if not pixel_values:
            raise ValueError("NOTE video preprocessing produced no frames")
        fragment_ids = self._token_ids("".join(fragments))
        vocab = self.tokenizer.get_vocab()
        embed_mask = torch.isin(
            torch.tensor(fragment_ids, dtype=torch.long),
            torch.tensor([vocab[IMAGE_PAD], vocab[AUDIO_PAD]], dtype=torch.long),
        )
        expected_embeds = image_pad_count + audio_pad_count
        if int(embed_mask.sum()) != expected_embeds:
            raise ValueError(
                "NOTE video placeholder expansion produced an invalid embedding mask: "
                f"expected={expected_embeds}, actual={int(embed_mask.sum())}"
            )

        empty_audio = torch.empty(0, dtype=torch.int32)
        return {
            "video_pixel_values": torch.cat(pixel_values),
            "video_image_grid_thw": torch.stack(image_grids),
            "video_audio_values": (
                torch.cat(audio_values) if audio_values else empty_audio
            ),
            "video_audio_lengths": torch.tensor(
                [value.numel() for value in audio_values],
                dtype=torch.long,
            ),
            "video_modalities": torch.tensor(modalities, dtype=torch.uint8),
            "video_input_ids": torch.tensor(fragment_ids, dtype=torch.long),
            "video_embed_mask": embed_mask,
            "video_frame_counts": torch.tensor([len(image_grids)], dtype=torch.long),
            "video_audio_counts": torch.tensor([len(audio_values)], dtype=torch.long),
            "video_emission_counts": torch.tensor([len(modalities)], dtype=torch.long),
            "video_prompt_lengths": torch.tensor([len(fragment_ids)], dtype=torch.long),
            "video_patch_counts": torch.tensor(
                [sum(int(grid.prod()) for grid in image_grids)], dtype=torch.long
            ),
            "video_audio_sample_counts": torch.tensor(
                [sum(value.numel() for value in audio_values)], dtype=torch.long
            ),
        }

    def __call__(self, text: str, **kwargs: object) -> BatchFeature:
        modality_order = [
            key for key in kwargs if key in ("images", "audios", "videos")
        ]
        raw_images = kwargs.pop("images", None)
        raw_audios = kwargs.pop("audios", None)
        raw_videos = kwargs.pop("videos", None)
        images = (
            list(cast(Sequence[Image.Image], raw_images))
            if raw_images is not None
            else []
        )
        audios = (
            list(cast(Sequence[np.ndarray], raw_audios))
            if raw_audios is not None
            else []
        )
        videos = (
            list(cast(Sequence[object], raw_videos)) if raw_videos is not None else []
        )
        if videos and (images or audios):
            raise ValueError(
                "Dots3Note does not support mixing a native video with "
                "separate image/audio inputs"
            )
        if len(videos) > 1:
            raise ValueError("Dots3Note supports one video per request")
        detail = kwargs.pop("image_detail", "auto")
        details = list(detail) if isinstance(detail, (list, tuple)) else None
        size_overrides: dict[str, int | None] = {}
        for key in ("min_pixels", "max_pixels", "target_height", "target_width"):
            if key in kwargs:
                value = kwargs.pop(key)
                size_overrides[key] = None if value is None else int(cast(Any, value))
        video_seq = int(cast(Any, kwargs.pop("seq", self.max_model_len)))
        reserve_value = kwargs.pop("output_reserve", None)
        output_reserve = (
            None if reserve_value is None else int(cast(Any, reserve_value))
        )
        audio_cap = float(cast(Any, kwargs.pop("audio_cap", 1.0)))
        audio_sample_rate = int(cast(Any, kwargs.pop("audio_sr", _DEFAULT_SAMPLE_RATE)))
        k_mode = str(kwargs.pop("k_mode", "eval_ek"))
        max_new_tokens = int(cast(Any, kwargs.pop("max_new_tokens", 0)))
        question = str(kwargs.pop("video_question", text))

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
                        min_pixels=size_overrides.get("min_pixels"),
                        max_pixels=size_overrides.get("max_pixels"),
                        target_height=size_overrides.get("target_height"),
                        target_width=size_overrides.get("target_width"),
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
            elif modality == "videos" and videos:
                video_detail = details[0] if details is not None else str(detail)
                data.update(
                    self._process_video(
                        videos[0],
                        detail=video_detail,
                        size_overrides=size_overrides,
                        seq=video_seq,
                        output_reserve=output_reserve,
                        audio_cap=audio_cap,
                        audio_sample_rate=audio_sample_rate,
                        k_mode=k_mode,
                        max_new_tokens=max_new_tokens,
                        question=question,
                    )
                )
        return BatchFeature(data=data)


class Dots3NoteProcessingInfo(BaseProcessingInfo):
    @cached_property
    def vision_config(self) -> dict[str, Any] | None:
        model_config = self.ctx.model_config
        return load_note_config_section(
            model_config.model,
            model_config.revision,
            "vision_config",
        )

    @cached_property
    def audio_config(self) -> dict[str, Any] | None:
        model_config = self.ctx.model_config
        return load_note_config_section(
            model_config.model,
            model_config.revision,
            "audio_config",
        )

    @cached_property
    def image_processor(self) -> Dots3NoteImageProcessor | None:
        if self.vision_config is None:
            return None
        model_config = self.ctx.model_config
        processor_config = get_hf_file_to_dict(
            "preprocessor_config.json",
            model_config.model,
            model_config.revision,
        )
        preprocessor_config = (processor_config or {}).get("vision_config")
        if not isinstance(preprocessor_config, dict):
            raise ValueError(
                "NOTE vision checkpoint is missing preprocessor_config.json"
            )
        image_details = preprocessor_config.get("image_details", {})
        return Dots3NoteImageProcessor(preprocessor_config, image_details)

    @cached_property
    def processor(self) -> Dots3NoteProcessor:
        mm_config = self.ctx.get_mm_config()
        audio_config = self.audio_config or {}
        audio_token_stride = (
            _HOP_LENGTH
            * (8 if audio_config.get("use_conv2d_stem", True) else 2)
            * int(audio_config.get("merge_factor", 1))
        )
        return Dots3NoteProcessor(
            self.get_tokenizer(),
            self.image_processor,
            max_model_len=self.ctx.model_config.max_model_len,
            video_audio_enabled=(
                self.audio_config is not None
                and (
                    mm_config.get_limit_per_prompt("audio") > 0
                    or mm_config.get_limit_per_prompt("video") > 0
                )
            ),
            audio_token_stride=audio_token_stride,
        )

    def get_hf_processor(self, **kwargs: object) -> Dots3NoteProcessor:
        return self.processor

    def get_data_parser(self) -> MultiModalDataParser:
        sample_rate = int((self.audio_config or {}).get("sampling_rate", 16000))
        return MultiModalDataParser(target_sr=float(sample_rate), target_channels=1)

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        limits: dict[str, int | None] = {}
        if self.vision_config is not None:
            limits["image"] = 512
            limits["video"] = 1
        if self.audio_config is not None:
            limits["audio"] = 128
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
        if self.image_processor is not None:
            result["video"] = max(1, seq_len - seq_len // 4)
        return result


class Dots3NoteDummyInputsBuilder(BaseDummyInputsBuilder[Dots3NoteProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        return (
            f"{IMAGE_START}{IMAGE_PAD}{IMAGE_END}" * mm_counts.get("image", 0)
            + f"{AUDIO_START}{AUDIO_PAD}{AUDIO_END}" * mm_counts.get("audio", 0)
            + VIDEO_PLACEHOLDER * mm_counts.get("video", 0)
        )

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions],
    ) -> MultiModalDataDict:
        data: dict[str, Any] = {}
        num_images = mm_counts.get("image", 0)
        if num_images:
            width, height = self.info.get_max_image_size()
            data["image"] = self._get_dummy_images(
                width=width,
                height=height,
                num_images=num_images,
                overrides=cast(ImageDummyOptions | None, mm_options.get("image")),
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
                overrides=cast(AudioDummyOptions | None, mm_options.get("audio")),
            )
        num_videos = mm_counts.get("video", 0)
        if num_videos:
            width, height = self.info.get_max_image_size()
            factor = self.info.image_processor.factor  # type: ignore[union-attr]
            frame_tokens = max(1, width * height // (factor * factor))
            video_tokens = max(1, seq_len - seq_len // 4)
            num_frames = max(4, math.ceil(video_tokens / frame_tokens))
            data["video"] = self._get_dummy_videos(
                width=width,
                height=height,
                num_frames=num_frames,
                num_videos=num_videos,
                overrides=cast(VideoDummyOptions | None, mm_options.get("video")),
            )
        return data


class Dots3NoteMultiModalProcessor(BaseMultiModalProcessor[Dots3NoteProcessingInfo]):
    def _get_hf_mm_data(
        self,
        mm_items: MultiModalDataItems,
    ) -> tuple[Mapping[str, object], Mapping[str, object]]:
        processor_data, passthrough_data = super()._get_hf_mm_data(mm_items)
        if "video" not in mm_items:
            return processor_data, passthrough_data

        videos = mm_items.get_items("video", VideoProcessorItems)
        raw_videos: list[object] = []
        for index, item in enumerate(videos.data):
            if isinstance(item, MediaWithBytes):
                raw_videos.append(item.original_bytes)
            else:
                raw_videos.append(videos.get(index))
        processor_data = dict(processor_data)
        processor_data["videos"] = raw_videos
        return processor_data, passthrough_data

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
        if "video_input_ids" in hf_inputs:
            prompt_lengths = hf_inputs["video_prompt_lengths"]
            frame_counts = hf_inputs["video_frame_counts"]
            audio_counts = hf_inputs["video_audio_counts"]
            emission_counts = hf_inputs["video_emission_counts"]
            fields["video_pixel_values"] = MultiModalFieldConfig.flat_from_sizes(
                "video", hf_inputs["video_patch_counts"]
            )
            fields["video_image_grid_thw"] = MultiModalFieldConfig.flat_from_sizes(
                "video", frame_counts, keep_on_cpu=True
            )
            fields["video_audio_values"] = MultiModalFieldConfig.flat_from_sizes(
                "video", hf_inputs["video_audio_sample_counts"]
            )
            fields["video_audio_lengths"] = MultiModalFieldConfig.flat_from_sizes(
                "video", audio_counts, keep_on_cpu=True
            )
            fields["video_modalities"] = MultiModalFieldConfig.flat_from_sizes(
                "video", emission_counts, keep_on_cpu=True
            )
            fields["video_input_ids"] = MultiModalFieldConfig.flat_from_sizes(
                "video", prompt_lengths, keep_on_cpu=True
            )
            fields["video_embed_mask"] = MultiModalFieldConfig.flat_from_sizes(
                "video", prompt_lengths, keep_on_cpu=True
            )
            for key in (
                "video_frame_counts",
                "video_audio_counts",
                "video_emission_counts",
            ):
                fields[key] = MultiModalFieldConfig.batched("video", keep_on_cpu=True)
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
                assert isinstance(grid, torch.Tensor)
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
                assert isinstance(length, torch.Tensor)
                num_tokens = math.ceil(int(length.item()) / stride)
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
        if "video" in out_mm_kwargs:

            def video_replacement(item_idx: int) -> PromptUpdateDetails[list[int]]:
                item = out_mm_kwargs["video"][item_idx]
                input_ids_data = item["video_input_ids"].data
                embed_mask_data = item["video_embed_mask"].data
                assert isinstance(input_ids_data, torch.Tensor)
                assert isinstance(embed_mask_data, torch.Tensor)
                input_ids = input_ids_data.tolist()
                embed_mask = embed_mask_data.bool()

                def select_video_embeds(tokenizer, full) -> torch.Tensor:
                    del tokenizer, full
                    return embed_mask

                return PromptUpdateDetails(
                    full=input_ids,
                    is_embed=select_video_embeds,
                )

            updates.append(
                PromptReplacement(
                    modality="video",
                    target=VIDEO_PLACEHOLDER,
                    replacement=video_replacement,
                )
            )
        return updates
