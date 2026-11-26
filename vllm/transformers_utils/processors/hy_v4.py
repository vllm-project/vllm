# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright 2025 The HunYuan team.

import math
from copy import deepcopy
from dataclasses import fields

import numpy as np
import regex as re
import torch
from transformers import AutoTokenizer
from transformers.feature_extraction_utils import BatchFeature
from transformers.image_processing_backends import TorchvisionBackend
from transformers.image_transforms import group_images_by_shape, reorder_images
from transformers.image_utils import (
    OPENAI_CLIP_MEAN,
    OPENAI_CLIP_STD,
    PILImageResampling,
    SizeDict,
)
from transformers.processing_utils import (
    ImagesKwargs,
    ProcessingKwargs,
    ProcessorMixin,
    Unpack,
    VideosKwargs,
)
from transformers.video_processing_utils import BaseVideoProcessor
from transformers.video_utils import VideoMetadata, make_batched_videos

from vllm.transformers_utils.configs.hy_v4 import HYV4VLConfig


def smart_resize(height, width, factor=32, min_pixels=4096, max_pixels=4194304):
    if min(height, width, factor, min_pixels) <= 0 or max_pixels < min_pixels:
        raise ValueError(
            "Image dimensions and pixel limits must be positive and ordered"
        )
    if max(height, width) / min(height, width) > 200:
        raise ValueError("Image aspect ratio must not exceed 200")
    h, w = round(height / factor) * factor, round(width / factor) * factor
    if h * w > max_pixels:
        scale = math.sqrt(height * width / max_pixels)
        h = max(factor, math.floor(height / scale / factor) * factor)
        w = max(factor, math.floor(width / scale / factor) * factor)
    elif h * w < min_pixels:
        scale = math.sqrt(min_pixels / (height * width))
        h = math.ceil(height * scale / factor) * factor
        w = math.ceil(width * scale / factor) * factor
    h, w = max(h, factor), max(w, factor)
    if h * w > max_pixels:
        raise ValueError("Image aspect ratio and patch rounding exceed max_pixels")
    return h, w


def patchify(frames: torch.Tensor, patch_size: int) -> torch.Tensor:
    t, channels, height, width = frames.shape
    if height % patch_size or width % patch_size:
        raise ValueError("Image dimensions must be divisible by patch_size")
    return (
        frames.reshape(
            t,
            channels,
            height // patch_size,
            patch_size,
            width // patch_size,
            patch_size,
        )
        .permute(0, 2, 4, 1, 3, 5)
        .reshape(-1, channels * patch_size**2)
    )


class HYV4ImageProcessorKwargs(ImagesKwargs, total=False):  # type: ignore[call-arg]
    min_pixels: int
    max_pixels: int
    patch_size: int
    merge_size: int
    temporal_patch_size: int
    spatial_patch_size: int


class HYV4VLImageProcessor(TorchvisionBackend):
    """Row-major per-frame patches used by the HYV4 vision tower."""

    do_resize = True
    do_rescale = True
    do_normalize = True
    do_convert_rgb = True
    resample = PILImageResampling.BICUBIC
    size = {"shortest_edge": 512 * 512, "longest_edge": 2048 * 2048}
    default_to_square = False
    image_mean = OPENAI_CLIP_MEAN
    image_std = OPENAI_CLIP_STD
    patch_size = 16
    merge_size = 2
    temporal_patch_size = 1
    spatial_patch_size = 1
    min_pixels = None
    max_pixels = None
    valid_kwargs = HYV4ImageProcessorKwargs
    model_input_names = ["pixel_values", "image_grid_thw"]

    def __init__(self, **kwargs: Unpack[HYV4ImageProcessorKwargs]):
        kwargs["size"] = dict(kwargs.get("size") or self.size)
        super().__init__(**kwargs)

    def _standardize_kwargs(
        self, size=None, min_pixels=None, max_pixels=None, **kwargs
    ):
        size = dict(size or self.size)
        if min_pixels is not None:
            size["shortest_edge"] = min_pixels
        if max_pixels is not None:
            size["longest_edge"] = max_pixels
        if kwargs.get("resample") == PILImageResampling.LANCZOS:
            kwargs["resample"] = PILImageResampling.BICUBIC
        return super()._standardize_kwargs(
            size=size, min_pixels=min_pixels, max_pixels=max_pixels, **kwargs
        )

    def _preprocess(
        self,
        images,
        do_resize,
        size,
        resample,
        do_rescale,
        rescale_factor,
        do_normalize,
        image_mean,
        image_std,
        patch_size,
        merge_size,
        return_tensors,
        disable_grouping=None,
        **kwargs,
    ):
        grouped, indices = group_images_by_shape(
            images, disable_grouping=disable_grouping
        )
        outputs, grids = {}, {}
        for shape, batch in grouped.items():
            height, width = batch.shape[-2:]
            if do_resize:
                height, width = smart_resize(
                    height,
                    width,
                    patch_size * merge_size,
                    size.shortest_edge,
                    size.longest_edge,
                )
                batch = self.resize(
                    batch, SizeDict(height=height, width=width), resample
                )
            if height % (patch_size * merge_size) or width % (patch_size * merge_size):
                raise ValueError(
                    "Image dimensions must align with patch_size * merge_size"
                )
            if height * width > size.longest_edge:
                raise ValueError("Image exceeds max_pixels with resizing disabled")
            batch = self.rescale_and_normalize(
                batch, do_rescale, rescale_factor, do_normalize, image_mean, image_std
            )
            outputs[shape] = patchify(batch, patch_size).reshape(
                len(batch), -1, 3 * patch_size**2
            )
            grids[shape] = [[1, height // patch_size, width // patch_size]] * len(batch)
        return BatchFeature(
            {
                "pixel_values": torch.cat(reorder_images(outputs, indices)),
                "image_grid_thw": torch.tensor(reorder_images(grids, indices)),
            },
            tensor_type=return_tensors,
        )

    def get_number_of_image_patches(self, height, width, images_kwargs=None):
        options = images_kwargs or {}
        patch = options.get("patch_size", self.patch_size)
        height, width = smart_resize(
            height,
            width,
            patch * options.get("merge_size", self.merge_size),
            options.get("min_pixels", self.size["shortest_edge"]),
            options.get("max_pixels", self.size["longest_edge"]),
        )
        return height // patch, width // patch


def adaptive_resize(
    height,
    width,
    num_frames,
    patch_size=16,
    merge_size=2,
    max_pixels=33554432,
    video_max_image_size=1024,
    timestamp_token_per_frame=6,
    timestamp_token_reserve=2000,
):
    factor = patch_size * merge_size
    if min(height, width, num_frames, factor, max_pixels) <= 0:
        raise ValueError(
            "Video dimensions, frame count and pixel budget must be positive"
        )
    frame_tokens = height * width // factor**2 + max(height, width) // factor
    budget = max(
        1,
        max_pixels // factor**2
        - timestamp_token_reserve
        - timestamp_token_per_frame * num_frames,
    )
    if frame_tokens * num_frames > budget:
        scale = math.sqrt(budget / (frame_tokens * num_frames))
    elif height * width > video_max_image_size**2:
        scale = video_max_image_size / math.sqrt(height * width)
    else:
        scale = 1.0
    return (
        max(factor, int(height * scale) // factor * factor),
        max(factor, int(width * scale) // factor * factor),
    )


class HYV4VideoProcessorKwargs(VideosKwargs, total=False):  # type: ignore[call-arg]
    min_pixels: int
    max_pixels: int
    patch_size: int
    merge_size: int
    spatial_patch_size: int
    temporal_patch_size: int
    min_frames: int
    max_frames: int
    timestamp_token_per_frame: int
    timestamp_token_reserve: int
    video_max_image_size: int


class HYV4VLVideoProcessor(BaseVideoProcessor):
    """HYV4 frame sampling and budgeted per-frame vision patches."""

    do_resize = True
    do_rescale = True
    do_normalize = True
    do_convert_rgb = True
    do_sample_frames = False
    resample = PILImageResampling.BICUBIC
    size = {"shortest_edge": 4096, "longest_edge": 33554432}
    image_mean = [0.5, 0.5, 0.5]
    image_std = [0.5, 0.5, 0.5]
    min_pixels = None
    max_pixels = None
    patch_size = 16
    merge_size = 2
    temporal_patch_size = 1
    spatial_patch_size = 1
    fps = 2
    min_frames = 4
    max_frames = 100
    timestamp_token_per_frame = 4
    timestamp_token_reserve = 2000
    video_max_image_size = 1024
    valid_kwargs = HYV4VideoProcessorKwargs
    model_input_names = ["pixel_values_videos", "video_grid_thw"]

    def __init__(self, **kwargs: Unpack[HYV4VideoProcessorKwargs]):
        kwargs["size"] = dict(kwargs.get("size") or self.size)
        super().__init__(**kwargs)

    def _standardize_kwargs(
        self, size=None, min_pixels=None, max_pixels=None, **kwargs
    ):
        size = dict(size or self.size)
        if min_pixels is not None:
            size["shortest_edge"] = min_pixels
        if max_pixels is not None:
            size["longest_edge"] = max_pixels
        return super()._standardize_kwargs(
            size=size, min_pixels=min_pixels, max_pixels=max_pixels, **kwargs
        )

    def preprocess(self, videos, **kwargs: Unpack[HYV4VideoProcessorKwargs]):
        if kwargs.get("num_frames") is not None and "fps" not in kwargs:
            kwargs["fps"] = None
        return super().preprocess(videos, **kwargs)

    def sample_frames(self, metadata, num_frames=None, fps=None, **kwargs):
        if fps is not None and num_frames is not None:
            raise ValueError("num_frames and fps are mutually exclusive")
        total = metadata.total_num_frames
        if total <= 0 or metadata.fps is None or metadata.fps <= 0:
            raise ValueError("Video metadata must contain positive frame count and fps")
        maximum = kwargs.get("max_frames") or self.max_frames
        minimum = kwargs.get("min_frames") or self.min_frames
        if num_frames is None:
            fps = self.fps if fps is None else fps
            desired = math.floor(total / metadata.fps * fps) if fps > 0 else maximum
            num_frames = min(total, max(minimum, min(maximum, desired)))
        if not 0 < num_frames <= total:
            raise ValueError(
                "num_frames must be positive and not exceed available frames"
            )
        segment = (total - 1) / num_frames
        return np.array(
            [
                min(
                    (int(np.round(segment * i)) + int(np.round(segment * (i + 1))))
                    // 2,
                    total - 1,
                )
                for i in range(num_frames)
            ],
            dtype=np.intp,
        )

    def _preprocess(
        self,
        videos,
        do_resize=True,
        size=None,
        do_convert_rgb=True,
        do_rescale=True,
        rescale_factor=1 / 255,
        do_normalize=True,
        image_mean=None,
        image_std=None,
        patch_size=16,
        merge_size=2,
        timestamp_token_per_frame=None,
        timestamp_token_reserve=None,
        video_max_image_size=None,
        return_tensors=None,
        **kwargs,
    ):
        pixels, grids = [], []
        for video in videos:
            t, _, h, w = video.shape
            if t == 0:
                raise ValueError("Video must contain at least one frame")
            if t > (kwargs.get("max_frames") or self.max_frames):
                raise ValueError(
                    "Video exceeds max_frames; sample frames before encoding"
                )
            if do_convert_rgb:
                video = self.convert_to_rgb(video)
            if do_resize:
                h, w = adaptive_resize(
                    h,
                    w,
                    t,
                    patch_size,
                    merge_size,
                    size.longest_edge,
                    video_max_image_size or self.video_max_image_size,
                    timestamp_token_per_frame
                    if timestamp_token_per_frame is not None
                    else self.timestamp_token_per_frame,
                    timestamp_token_reserve
                    if timestamp_token_reserve is not None
                    else self.timestamp_token_reserve,
                )
                video = self.resize(
                    video, SizeDict(height=h, width=w), PILImageResampling.BICUBIC
                )
            if h % (patch_size * merge_size) or w % (patch_size * merge_size):
                raise ValueError(
                    "Video frame dimensions must align with patch and merge"
                )
            if t * h * w > size.longest_edge:
                raise ValueError("Processed video exceeds its total max_pixels budget")
            video = self.rescale_and_normalize(
                video, do_rescale, rescale_factor, do_normalize, image_mean, image_std
            )
            pixels.append(patchify(video, patch_size))
            grids.append([t, h // patch_size, w // patch_size])
        return BatchFeature(
            {
                "pixel_values_videos": torch.cat(pixels),
                "video_grid_thw": torch.tensor(grids),
            },
            tensor_type=return_tensors,
        )


def vision_token_grid(grid_thw, config, modality):
    t, h, w = map(int, grid_thw)
    if min(t, h, w) <= 0:
        raise ValueError("Vision grid dimensions must be positive")
    merge = config.spatial_merge_size
    if modality == "video":
        merge *= config.spatial_patch_size
        if t == 1 and config.spatial_patch_size != 1:
            raise ValueError("Single-frame spatial video compression is unsupported")
    elif t != 1:
        raise ValueError("An image must have exactly one temporal grid entry")
    if h % merge or w % merge:
        raise ValueError("Vision grid must be divisible by its spatial merge factor")
    frames = math.ceil(t / config.temporal_patch_size) if modality == "video" else 1
    rows, cols = h // merge, w // merge + int(config.use_image_newline_token)
    return frames, rows, cols


def vision_token_count(grid_thw, config, modality):
    frames, rows, cols = vision_token_grid(grid_thw, config, modality)
    return frames * (rows * cols + 2 * int(bool(config.cat_extra_token)))


class HYV4VLProcessorKwargs(ProcessingKwargs, total=False):  # type: ignore[call-arg]
    images_kwargs: HYV4ImageProcessorKwargs
    videos_kwargs: HYV4VideoProcessorKwargs
    _defaults = {"text_kwargs": {"padding": False}}


class HYV4VLProcessor(ProcessorMixin):
    """Local HYV4 text, image and video processing without private HF classes."""

    valid_processor_kwargs = HYV4VLProcessorKwargs

    def __init__(
        self,
        image_processor,
        tokenizer,
        video_processor,
        config=None,
        chat_template=None,
    ):
        super().__init__(
            image_processor=image_processor,
            tokenizer=tokenizer,
            video_processor=video_processor,
            chat_template=chat_template,
        )
        self.config = config or HYV4VLConfig()
        for name in (
            "image_start_token_id",
            "image_end_token_id",
            "image_token_id",
            "image_newline_token_id",
            "video_start_token_id",
            "video_end_token_id",
            "video_token_id",
        ):
            value = getattr(self.config, name)
            token_name = name.removesuffix("_id")
            token = getattr(tokenizer, token_name, None)
            if token is None:
                token = tokenizer.convert_ids_to_tokens(value)
            if token is None:
                raise ValueError(f"HYV4 tokenizer is missing {name}={value}")
            if tokenizer.convert_tokens_to_ids(token) != value:
                raise ValueError(
                    f"HYV4 tokenizer {token_name} does not match {name}={value}"
                )
            setattr(self, token_name, token)
            setattr(self, name, value)
        for component in (image_processor, video_processor):
            for name, config_name in (
                ("patch_size", "patch_size"),
                ("merge_size", "spatial_merge_size"),
                ("spatial_patch_size", "spatial_patch_size"),
                ("temporal_patch_size", "temporal_patch_size"),
            ):
                if getattr(component, name) != getattr(
                    self.config.vision_config, config_name
                ):
                    raise ValueError(
                        f"HYV4 processor {name} disagrees with vision_config"
                    )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        tokenizer = kwargs.pop("tokenizer", None)
        config = kwargs.pop("config", None)
        load_kwargs = {
            key: kwargs.pop(key)
            for key in ("revision", "cache_dir", "local_files_only", "token")
            if key in kwargs
        }
        kwargs.pop("trust_remote_code", None)
        kwargs.pop("use_fast", None)
        kwargs.pop("backend", None)
        config = config or HYV4VLConfig.from_pretrained(
            pretrained_model_name_or_path, **load_kwargs
        )
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(
                pretrained_model_name_or_path, trust_remote_code=False, **load_kwargs
            )
        processor_dict, _ = HYV4VLImageProcessor.get_image_processor_dict(
            pretrained_model_name_or_path, **load_kwargs
        )
        image_kwargs = {**processor_dict, **kwargs.pop("images_kwargs", {})}
        video_kwargs = {**processor_dict, **kwargs.pop("videos_kwargs", {})}
        for key, value in kwargs.items():
            if key in HYV4ImageProcessorKwargs.__annotations__:
                image_kwargs[key] = value
            if key in HYV4VideoProcessorKwargs.__annotations__:
                video_kwargs[key] = value
        return cls(
            HYV4VLImageProcessor(**image_kwargs),
            tokenizer,
            HYV4VLVideoProcessor(**video_kwargs),
            config=config,
            chat_template=getattr(tokenizer, "chat_template", None),
        )

    def get_placeholder(self, modality: str) -> str:
        if modality not in ("image", "video"):
            raise ValueError(f"Unsupported HYV4 VL modality: {modality}")
        return (
            getattr(self, f"{modality}_start_token")
            + getattr(self, f"{modality}_token")
            + getattr(self, f"{modality}_end_token")
        )

    def get_replacement(self, modality, grid, timestamps=None):
        config = self.config.vision_config
        if modality == "image":
            return [self.image_token_id] * vision_token_count(grid, config, modality)
        frames, _, _ = vision_token_grid(grid, config, modality)
        per_frame = vision_token_count(grid, config, modality) // frames
        if timestamps is None or len(timestamps) != frames:
            raise ValueError("Video timestamps must match merged frame count")
        tokens = []
        for timestamp in timestamps:
            tokens.extend(
                self.tokenizer.encode(
                    f"<{float(timestamp):.1f} seconds>", add_special_tokens=False
                )
            )
            tokens.extend([self.image_start_token_id])
            tokens.extend([self.video_token_id] * per_frame)
            tokens.extend([self.image_end_token_id])
        return tokens

    def __call__(
        self,
        images=None,
        text=None,
        videos=None,
        second_per_grid_ts=None,
        **kwargs: Unpack[HYV4VLProcessorKwargs],
    ):
        options = self._merge_kwargs(
            HYV4VLProcessorKwargs,
            tokenizer_init_kwargs=getattr(self.tokenizer, "init_kwargs", {}),
            **kwargs,
        )
        data = {}
        if images is not None:
            data.update(self.image_processor(images, **options["images_kwargs"]))
        if videos is not None:
            video_options = dict(options["videos_kwargs"])
            supplied_metadata = video_options.pop("video_metadata", None)
            video_options.pop("return_metadata", None)
            if isinstance(videos, list) and videos and isinstance(videos[0], tuple):
                items = videos
            else:
                batch = make_batched_videos(videos)
                metadata = supplied_metadata or [None] * len(batch)
                if len(metadata) != len(batch):
                    raise ValueError("Video metadata count must match video count")
                items = list(zip(batch, metadata))
            pixels, grids, timestamps = [], [], []
            for index, (frames, raw_metadata) in enumerate(items):
                metadata = deepcopy(raw_metadata or {})
                if isinstance(metadata, VideoMetadata):
                    metadata = {
                        field.name: getattr(metadata, field.name)
                        for field in fields(metadata)
                    }
                do_sample = metadata.pop(
                    "do_sample_frames", video_options.get("do_sample_frames", False)
                )
                offset = float(metadata.pop("timestamp_offset", 0.0))
                interval = metadata.pop("second_per_grid_t", None)
                if interval is None and second_per_grid_ts is not None:
                    if len(second_per_grid_ts) != len(items):
                        raise ValueError("second_per_grid_ts must match video count")
                    interval = float(second_per_grid_ts[index])
                if not math.isfinite(offset) or (
                    interval is not None
                    and (not math.isfinite(interval) or interval <= 0)
                ):
                    raise ValueError(
                        "Video timestamps require a finite offset and positive interval"
                    )
                metadata.setdefault("total_num_frames", len(frames))
                metadata.setdefault("fps", 24.0)
                if (
                    not len(frames)
                    or metadata["fps"] is None
                    or not math.isfinite(metadata["fps"])
                    or metadata["fps"] <= 0
                ):
                    raise ValueError(f"Video {index} requires frames and positive fps")
                if metadata.get("frames_indices") is None:
                    metadata["frames_indices"] = list(range(len(frames)))
                if do_sample and len(frames) != metadata["total_num_frames"]:
                    raise ValueError(
                        f"Video {index} was already sampled; set do_sample_frames=False"
                    )
                output = self.video_processor(
                    [frames],
                    **{**video_options, "do_sample_frames": do_sample},
                    video_metadata=[metadata],
                    return_metadata=True,
                )
                processed_metadata = output.pop("video_metadata")[0]
                grid = output["video_grid_thw"][0]
                t = int(grid[0])
                indices = list(processed_metadata.frames_indices)
                if len(indices) != t:
                    raise ValueError(
                        f"Video {index} frame indices do not match processed frames"
                    )
                tau = self.config.vision_config.temporal_patch_size
                indices += [indices[-1]] * (-len(indices) % tau)
                count = math.ceil(t / tau)
                if interval is not None:
                    times = [offset + i * interval for i in range(count)]
                else:
                    times = [
                        offset
                        + sum(indices[i : i + tau]) / tau / processed_metadata.fps
                        for i in range(0, len(indices), tau)
                    ]
                pixels.append(output["pixel_values_videos"])
                grids.append(grid)
                timestamps.append(torch.tensor(times, dtype=torch.float64))
            data.update(
                pixel_values_videos=torch.cat(pixels),
                video_grid_thw=torch.stack(grids),
                video_timestamps=torch.cat(timestamps),
            )
        if text is not None:
            texts = [text] if isinstance(text, str) else list(text)
            image_index = video_index = timestamp_index = 0
            text_options = dict(options["text_kwargs"])
            text_options.setdefault("add_special_tokens", False)
            image_pad = self.image_token
            video_pad = self.video_token
            pattern = f"({re.escape(image_pad)}|{re.escape(video_pad)})"
            expanded_texts = []
            for prompt in texts:
                expanded = []
                for part in re.split(pattern, prompt):
                    if part == image_pad and images is not None:
                        if image_index >= len(data["image_grid_thw"]):
                            raise ValueError("More image placeholders than images")
                        grid = data["image_grid_thw"][image_index]
                        expanded.append(
                            image_pad * len(self.get_replacement("image", grid))
                        )
                        image_index += 1
                    elif part == video_pad and videos is not None:
                        if video_index >= len(data["video_grid_thw"]):
                            raise ValueError("More video placeholders than videos")
                        grid = data["video_grid_thw"][video_index]
                        count, _, _ = vision_token_grid(
                            grid, self.config.vision_config, "video"
                        )
                        times = data["video_timestamps"][
                            timestamp_index : timestamp_index + count
                        ]
                        expanded.append(
                            self.tokenizer.decode(
                                self.get_replacement("video", grid, times),
                                skip_special_tokens=False,
                                clean_up_tokenization_spaces=False,
                            )
                        )
                        video_index += 1
                        timestamp_index += count
                    else:
                        expanded.append(part)
                expanded_texts.append("".join(expanded))
            if image_index != len(data.get("image_grid_thw", [])) or video_index != len(
                data.get("video_grid_thw", [])
            ):
                raise ValueError("Media count must match image/video placeholders")
            data.update(self.tokenizer(expanded_texts, **text_options))
        return BatchFeature(data)
