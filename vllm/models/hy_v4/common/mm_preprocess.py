# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright 2025 The HunYuan team.

import math
from collections.abc import Mapping
from dataclasses import replace
from functools import partial

import torch

from vllm.multimodal.hasher import MultiModalHasher
from vllm.multimodal.inputs import MultiModalFieldConfig
from vllm.multimodal.parse import (
    DictEmbeddingItems,
    ImageSize,
    MultiModalDataItems,
    MultiModalDataParser,
    VideoProcessorItems,
)
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder,
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    PromptReplacement,
    PromptUpdateDetails,
    cached_encode,
)
from vllm.multimodal.processing.inputs import EPD_VIDEO_PARTITION_KEY
from vllm.multimodal.processing.processor import replace_token_matches
from vllm.transformers_utils.configs.hy_v4 import HYV4VLConfig
from vllm.transformers_utils.processors.hy_v4 import (
    HYV4VLProcessor,
    adaptive_resize,
    smart_resize,
    vision_token_count,
)


def _get_vision_fields(inputs, temporal_patch_size):
    result = {}
    for modality, pixel_key in (
        ("image", "pixel_values"),
        ("video", "pixel_values_videos"),
    ):
        grid_key = f"{modality}_grid_thw"
        grid = inputs.get(grid_key, torch.empty((0, 3), dtype=torch.long))
        result[pixel_key] = MultiModalFieldConfig.flat_from_sizes(
            modality, grid.prod(-1)
        )
        result[grid_key] = MultiModalFieldConfig.batched(modality, keep_on_cpu=True)
        if modality == "video":
            tau = temporal_patch_size
            result["video_timestamps"] = MultiModalFieldConfig.flat_from_sizes(
                "video", (grid[:, 0] + tau - 1) // tau, keep_on_cpu=True
            )
    return result


class HYV4VLDataParser(MultiModalDataParser):
    embedding_fields = {
        "image": {"image_grid_thw": "metadata"},
        "video": {"video_grid_thw": "metadata", "video_timestamps": "metadata"},
    }

    def __init__(self, vision_config, max_model_len: int, **kwargs):
        super().__init__(video_needs_metadata=True, **kwargs)
        self.vision_config = vision_config
        self.max_model_len = max_model_len

    def _parse_metadata(self, data, modality: str) -> DictEmbeddingItems:
        fields = self.placeholder_metadata_fields(modality)
        if not self.allow_missing_mm_embeddings or data.keys() != fields:
            raise ValueError(f"HYV4 VL expects only {sorted(fields)} on the consumer")
        grid = data[f"{modality}_grid_thw"]
        if (
            not isinstance(grid, torch.Tensor)
            or grid.dtype not in (torch.int32, torch.int64)
            or grid.ndim != 2
            or grid.shape[1] != 3
            or not len(grid)
        ):
            raise ValueError(f"Invalid HYV4 {modality} grid")
        tau = self.vision_config.temporal_patch_size
        total_tokens = total_frames = 0
        for item in grid.tolist():
            total_tokens += vision_token_count(item, self.vision_config, modality)
            total_frames += (item[0] + tau - 1) // tau
            if total_tokens > self.max_model_len:
                raise ValueError(f"HYV4 {modality} metadata exceeds the model context")
        if modality == "video":
            timestamps = data["video_timestamps"]
            if (
                not isinstance(timestamps, torch.Tensor)
                or timestamps.shape != (total_frames,)
                or not torch.isfinite(timestamps).all()
            ):
                raise ValueError("HYV4 video timestamps must match the merged frames")
        return DictEmbeddingItems(
            data,
            modality=modality,
            required_fields=fields,
            fields_factory=partial(_get_vision_fields, temporal_patch_size=tau),
        )

    def _parse_image_data(self, data):
        if isinstance(data, dict):
            return self._parse_metadata(data, "image")
        if self.is_embeddings(data):
            raise ValueError("HYV4 VL does not support precomputed image embeddings")
        return super()._parse_image_data(data)

    def _parse_video_data(self, data):
        if isinstance(data, dict):
            return self._parse_metadata(data, "video")
        if self.is_embeddings(data):
            raise ValueError("HYV4 VL does not support precomputed video embeddings")
        return super()._parse_video_data(data)


class HYV4VLProcessingInfo(BaseProcessingInfo):
    def get_hf_config(self) -> HYV4VLConfig:
        return self.ctx.get_hf_config(HYV4VLConfig)

    def get_hf_processor(self, **kwargs) -> HYV4VLProcessor:
        return self.ctx.get_hf_processor(HYV4VLProcessor, **kwargs)

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"image": None, "video": None}

    def get_max_frames_per_video(self) -> int:
        processor = self.get_hf_processor().video_processor
        options = self._merge_and_resolve_mm_processor_kwargs({}).get(
            "videos_kwargs", {}
        )
        return int(options.get("max_frames", processor.max_frames))

    def get_data_parser(self):
        return HYV4VLDataParser(
            self.get_hf_config().vision_config,
            self.ctx.model_config.max_model_len,
            allow_missing_mm_embeddings=self.allow_missing_mm_embeddings,
        )

    def get_image_size_with_most_features(self):
        processor = self.get_hf_processor().image_processor
        options = self._merge_and_resolve_mm_processor_kwargs({}).get(
            "images_kwargs", {}
        )
        factor = processor.patch_size * processor.merge_size
        maximum = options.get("max_pixels", processor.size["longest_edge"])
        minimum = options.get("min_pixels", processor.size["shortest_edge"])
        cells = max(1, maximum // factor**2)
        candidates = []
        for rows in range(1, math.isqrt(cells * 200) + 1):
            cols = min(cells // rows, rows * 200)
            if cols and rows <= cols * 200:
                candidates.append((rows * (cols + 1), rows, cols))
        _, rows, cols = max(candidates)
        height, width = smart_resize(
            rows * factor, cols * factor, factor, minimum, maximum
        )
        return ImageSize(width=width, height=height)

    def get_dummy_video_shape(self, seq_len, mm_counts):
        processor = self.get_hf_processor().video_processor
        options = self._merge_and_resolve_mm_processor_kwargs({}).get(
            "videos_kwargs", {}
        )
        config = self.get_hf_config().vision_config
        maximum = options.get("max_pixels", processor.size["longest_edge"])
        max_frames = options.get("max_frames", processor.max_frames)
        side = options.get("video_max_image_size", processor.video_max_image_size)
        available = max(1, seq_len // max(1, sum(mm_counts.values())))
        best = (
            0,
            1,
            config.patch_size * config.spatial_merge_size,
            config.patch_size * config.spatial_merge_size,
        )
        for frames in range(1, min(max_frames, available) + 1):
            height, width = adaptive_resize(
                side,
                side,
                frames,
                config.patch_size,
                config.spatial_merge_size,
                maximum,
                side,
                options.get(
                    "timestamp_token_per_frame", processor.timestamp_token_per_frame
                ),
                options.get(
                    "timestamp_token_reserve", processor.timestamp_token_reserve
                ),
            )
            grid = (frames, height // config.patch_size, width // config.patch_size)
            count = vision_token_count(grid, config, "video")
            overhead = frames * (processor.timestamp_token_per_frame + 2) + 2
            if count + overhead <= available and count > best[0]:
                best = (count, frames, height, width)
        return best[1:]

    def get_mm_max_tokens_per_item(self, seq_len, mm_counts):
        config = self.get_hf_config().vision_config
        size = self.get_image_size_with_most_features()
        image_tokens = vision_token_count(
            (1, size.height // config.patch_size, size.width // config.patch_size),
            config,
            "image",
        )
        video = self.get_hf_processor().video_processor
        options = self._merge_and_resolve_mm_processor_kwargs({}).get(
            "videos_kwargs", {}
        )
        factor = config.patch_size * config.spatial_merge_size
        max_frames = options.get("max_frames", video.max_frames)
        max_pixels = options.get("max_pixels", video.size["longest_edge"])
        video_tokens = 2 * (max_pixels // factor**2 + max_frames)
        video_tokens += 2 * max_frames * int(bool(config.cat_extra_token))
        return {"image": image_tokens, "video": video_tokens}


class HYV4VLDummyInputsBuilder(BaseDummyInputsBuilder[HYV4VLProcessingInfo]):
    def get_dummy_text(self, mm_counts):
        processor = self.info.get_hf_processor()
        return "".join(
            processor.get_placeholder(modality) * mm_counts.get(modality, 0)
            for modality in ("image", "video")
        )

    def get_dummy_mm_data(self, seq_len, mm_counts, mm_options):
        size = self.info.get_image_size_with_most_features()
        frames, height, width = self.info.get_dummy_video_shape(seq_len, mm_counts)
        videos = self._get_dummy_videos(
            width=width,
            height=height,
            num_frames=frames,
            num_videos=mm_counts.get("video", 0),
            overrides=mm_options.get("video"),
        )
        return {
            "image": self._get_dummy_images(
                width=size.width,
                height=size.height,
                num_images=mm_counts.get("image", 0),
                overrides=mm_options.get("image"),
            ),
            "video": [
                (
                    video,
                    {
                        "fps": 2.0,
                        "duration": len(video) / 2,
                        "total_num_frames": len(video),
                        "frames_indices": list(range(len(video))),
                        "do_sample_frames": False,
                        "video_backend": "opencv",
                    },
                )
                for video in videos
            ],
        }


class HYV4VLMultiModalProcessor(BaseMultiModalProcessor[HYV4VLProcessingInfo]):
    def apply(self, inputs, timing_ctx):
        kwargs = {
            key: value
            for key, value in inputs.hf_processor_mm_kwargs.items()
            if key != EPD_VIDEO_PARTITION_KEY
        }
        processor = self.info.get_hf_processor(**kwargs)
        prompt = inputs.prompt
        for modality in ("image", "video"):
            if inputs.mm_data_items.get_count(modality, strict=False):
                target = [
                    getattr(processor, f"{modality}_start_token_id"),
                    getattr(processor, f"{modality}_token_id"),
                    getattr(processor, f"{modality}_end_token_id"),
                ]
                prompt = replace_token_matches(
                    prompt,
                    target,
                    cached_encode(
                        processor.tokenizer, f"<{modality}>", add_special_tokens=False
                    ),
                )
        return super().apply(replace(inputs, prompt=prompt), timing_ctx)

    def _get_video_intervals(self, inputs, num_videos):
        options = self.info.ctx.get_merged_mm_kwargs(inputs.hf_processor_mm_kwargs)
        intervals = options.get("second_per_grid_ts")
        if intervals is None or not num_videos:
            return None
        if not isinstance(intervals, (list, tuple)) or len(intervals) != num_videos:
            raise ValueError("second_per_grid_ts must match video count")
        if any(
            type(value) not in (int, float) or not math.isfinite(value) or value <= 0
            for value in intervals
        ):
            raise ValueError(
                "second_per_grid_ts must contain positive finite intervals"
            )
        return intervals

    def _partition_video_kwargs(self, inputs, video_index, num_videos):
        intervals = self._get_video_intervals(inputs, num_videos)
        if intervals is None:
            return inputs
        return replace(
            inputs,
            hf_processor_mm_kwargs={
                **inputs.hf_processor_mm_kwargs,
                "second_per_grid_ts": [intervals[video_index]],
            },
        )

    def _cached_apply_hf_processor(self, inputs, timing_ctx):
        videos = inputs.mm_data_items.get("video", ())
        intervals = self._get_video_intervals(inputs, len(videos))
        if intervals is not None:
            mm_items = inputs.mm_data_items
            if not isinstance(videos, DictEmbeddingItems):
                normalized = []
                for item, interval in zip(videos, intervals):
                    if item is None:
                        raise ValueError(
                            "Video timing overrides require media metadata"
                        )
                    frames, metadata = item
                    normalized.append(
                        (frames, {**metadata, "second_per_grid_t": float(interval)})
                    )
                mm_items = MultiModalDataItems(
                    {**inputs.mm_data_items, "video": VideoProcessorItems(normalized)}
                )
            inputs = replace(inputs, mm_data_items=mm_items)
        if "second_per_grid_ts" in inputs.hf_processor_mm_kwargs:
            kwargs = dict(inputs.hf_processor_mm_kwargs)
            del kwargs["second_per_grid_ts"]
            inputs = replace(inputs, hf_processor_mm_kwargs=kwargs)
        if intervals is not None:
            # Explicit UUIDs bypass the media metadata when computing base keys.
            # Fold the effective interval in for both raw and metadata-only items.
            hashes = dict(self._get_mm_hashes(inputs))
            algorithm = self.info.ctx.get_mm_config().mm_hasher_algorithm
            hashes["video"] = [
                MultiModalHasher.hash_kwargs(
                    algorithm, mm_hash=key, second_per_grid_t=float(interval)
                )
                for key, interval in zip(hashes["video"], intervals, strict=True)
            ]
            inputs = replace(inputs, mm_hashes_override=hashes)
        return super()._cached_apply_hf_processor(inputs, timing_ctx)

    def _get_mm_fields_config(self, hf_inputs, hf_processor_mm_kwargs):
        return _get_vision_fields(
            hf_inputs, self.info.get_hf_config().vision_config.temporal_patch_size
        )

    def _get_prompt_updates(self, mm_items, hf_processor_mm_kwargs, out_mm_kwargs):
        processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)

        def replacement(index: int, modality: str) -> PromptUpdateDetails:
            item = out_mm_kwargs[modality][index]
            grid = item[f"{modality}_grid_thw"].data
            timestamps = item["video_timestamps"].data if modality == "video" else None
            tokens = [
                getattr(processor, f"{modality}_start_token_id"),
                *processor.get_replacement(modality, grid, timestamps),
                getattr(processor, f"{modality}_end_token_id"),
            ]
            token_id = getattr(processor, f"{modality}_token_id")
            return PromptUpdateDetails.select_token_id(tokens, token_id)

        return [
            PromptReplacement(
                modality=modality,
                target=cached_encode(
                    processor.tokenizer, f"<{modality}>", add_special_tokens=False
                ),
                replacement=partial(replacement, modality=modality),
            )
            for modality in ("image", "video")
        ]
