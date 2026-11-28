# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Nemotron Shortcut vision-language model.

The vision tower is vLLM's Pixtral encoder and the projector is Mistral3's
spatial merger, in front of the Nemotron Shortcut language model.
"""

import bisect
import copy
import math
from collections.abc import Iterable, Mapping, Sequence
from functools import cached_property, partial
from typing import Any, cast

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from transformers import BatchFeature, PreTrainedConfig, ProcessorMixin
from transformers.models.pixtral.configuration_pixtral import PixtralVisionConfig

from vllm.config import VllmConfig
from vllm.config.multimodal import MultiModalDummyOptions
from vllm.inputs import MultiModalDataDict
from vllm.logger import init_logger
from vllm.model_executor.layers.conv import Conv2dLayer
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.models.interfaces import (
    IsHybrid,
    MultiModalEmbeddings,
    SupportsMultiModal,
    SupportsPP,
    SupportsReplaySSM,
)
from vllm.model_executor.models.mistral3 import (
    Mistral3DummyInputsBuilder,
    Mistral3MultiModalProcessor,
    Mistral3MultiModalProjector,
)
from vllm.model_executor.models.module_mapping import MultiModelKeys
from vllm.model_executor.models.pixtral import (
    TRANSFORMERS_WITH_AXIAL_ROPE,
    PixtralHFVisionModel,
    pack_image_patches,
)
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    WeightsMapper,
    maybe_prefix,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalFieldConfig, MultiModalKwargsItems
from vllm.multimodal.parse import (
    ImageSize,
    MultiModalDataItems,
    MultiModalDataParser,
    VideoProcessorItems,
)
from vllm.multimodal.processing import (
    BaseProcessingInfo,
    PromptReplacement,
    PromptUpdate,
    PromptUpdateDetails,
)
from vllm.multimodal.video import VIDEO_LOADER_REGISTRY, DecodedFrames, VideoBackend
from vllm.multimodal.video_decoders import (
    VideoDecoderBackend,
    VideoSourceMetadata,
    VideoTargetMetadata,
)
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.processors.nano_nemotron_vl import (
    get_video_target_size_and_feature_size,
)

from .model import NemotronShortcutForCausalLM, check_sdsa_indexer_loaded
from .sdsa import SdsaKVCacheGrouping

logger = init_logger(__name__)

_DEFAULT_MAX_VIDEO_FRAMES = 16

_VIDEO_TOKEN = "<nemo:video>"
_LEGACY_VIDEO_TOKEN = "<video>"

# Splits the per-tubelet video prompt into text segments around each tubelet's
# visual tokens. Same private-use character as the training marker, so the
# segments are encoded separately in the same places as in training.
_VIDEO_TUBELET_MARKER = "\ue000"

# Same values as Megatron-Bridge SUPPORTED_VIDEO_TEMPORAL_PATCH_SIZES.
_SUPPORTED_VIDEO_TEMPORAL_PATCH_SIZES = (1, 2)

# Video loader backend that samples frames with the NeMo RL training rule (for
# T > 1 exports). Opt in with
# --media-io-kwargs '{"video": {"video_backend": "nemotron_shortcut_t2"}}'.
NEMOTRON_SHORTCUT_VIDEO_LOADER_BACKEND = "nemotron_shortcut_t2"
# Training defaults: video_max_num_frames, video_min_num_frames and
# video_default_fps.
_TRAINING_VIDEO_MAX_FRAMES = 64
_TRAINING_VIDEO_MIN_FRAMES = 8
_TRAINING_VIDEO_FPS = 2


def resize_video_frame(
    frame: object, width: int, height: int, image_processor: Any
) -> torch.Tensor:
    """Apply the training video transform at the planned frame size."""
    if isinstance(frame, torch.Tensor):
        frame = frame.detach().cpu().numpy()
    array = np.asarray(frame)
    if array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError(
            "Nemotron Shortcut video frames must be RGB (height, width, 3)"
        )
    array = array.astype(np.float32) / np.float32(1.0 / image_processor.rescale_factor)
    if array.shape[:2] != (height, width):
        array = cv2.resize(array, (width, height), interpolation=cv2.INTER_AREA)
    pixels = torch.from_numpy(np.ascontiguousarray(array)).permute(2, 0, 1)
    mean = torch.tensor(image_processor.image_mean, dtype=torch.float32)[:, None, None]
    std = torch.tensor(image_processor.image_std, dtype=torch.float32)[:, None, None]
    return (pixels - mean) / std


def get_video_temporal_patch_size(config: object) -> int:
    """Return the top-level `video_temporal_patch_size` T of the config.

    T consecutive video frames form one tubelet, which `video_patch_conv`
    embeds as one patch grid. A config without the field means T = 1.

    The checks are the same as Megatron-Bridge
    `check_video_temporal_compression`: T must be 1 or 2, and T > 1 needs
    `separate_video_embedder` (default True), because images keep
    `vision_tower.patch_conv` and tubelets use `vision_tower.video_patch_conv`.
    """
    value = getattr(config, "video_temporal_patch_size", None)
    if value is None:
        return 1
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value not in _SUPPORTED_VIDEO_TEMPORAL_PATCH_SIZES
    ):
        raise ValueError(
            "Nemotron Shortcut supports video_temporal_patch_size in "
            f"{_SUPPORTED_VIDEO_TEMPORAL_PATCH_SIZES}, got {value!r}."
        )
    separate_video_embedder = getattr(config, "separate_video_embedder", None)
    if separate_video_embedder is None:
        separate_video_embedder = True
    if value > 1 and separate_video_embedder is not True:
        raise ValueError(
            "Nemotron Shortcut with video_temporal_patch_size > 1 supports only "
            "separate_video_embedder=True (images keep patch_conv and video "
            f"tubelets use video_patch_conv), got {separate_video_embedder!r}."
        )
    return value


def select_video_frame_count(
    num_decoded: int, max_frames: int, temporal_patch_size: int
) -> int:
    """Number of decoded frames to keep for a video of tubelets of T frames.

    This is the rounding of the training frame sampling: the count is
    `min(num_decoded, max_frames)` rounded to a multiple of T. A round up
    would need more frames than the decoded count or the frame cap, so it
    rounds down, but never below T. When fewer than T frames exist, all
    are kept and the caller pads them by repeating the last frame (as the
    Megatron-Bridge encoder does).
    """
    if num_decoded < 1:
        raise ValueError("Nemotron Shortcut requires at least one video frame")
    count = min(num_decoded, max_frames)
    count = count // temporal_patch_size * temporal_patch_size
    if count < temporal_patch_size:
        count = min(num_decoded, temporal_patch_size)
    return count


def pad_to_temporal_multiple(items: Sequence, temporal_patch_size: int) -> list:
    """Repeat the last item until the length is a multiple of T.

    The same pad as the training encoder (Megatron-Bridge
    `NemotronShortcutMIMOVisionEncoder._embed_media`), which repeats the last frame.
    """
    items = list(items)
    if not items:
        raise ValueError("Nemotron Shortcut requires at least one video frame")
    remainder = len(items) % temporal_patch_size
    if remainder:
        items.extend([items[-1]] * (temporal_patch_size - remainder))
    return items


def group_frames_into_tubelets(
    frames: Sequence[torch.Tensor], temporal_patch_size: int
) -> list[torch.Tensor]:
    """Stack T consecutive `[C, H, W]` frames into `[T * C, H, W]` tubelets.

    The channels are frame-major (frame 0 RGB, then frame 1 RGB), which is the
    input channel order of `video_patch_conv`. A frame count that is not a
    multiple of T is padded with the last frame.
    """
    frames = pad_to_temporal_multiple(frames, temporal_patch_size)
    tubelets = []
    for start in range(0, len(frames), temporal_patch_size):
        group = frames[start : start + temporal_patch_size]
        if len({tuple(frame.shape) for frame in group}) != 1:
            raise ValueError(
                "All frames of a Nemotron Shortcut video tubelet must have one "
                f"shape, got {[tuple(frame.shape) for frame in group]}."
            )
        tubelets.append(torch.cat(list(group), dim=0))
    return tubelets


def training_video_frame_times(
    num_source_frames: int,
    source_fps: float,
    *,
    max_frames: int = _TRAINING_VIDEO_MAX_FRAMES,
    sampling_fps: int | float = _TRAINING_VIDEO_FPS,
    min_frames: int = _TRAINING_VIDEO_MIN_FRAMES,
    temporal_patch_size: int = 2,
) -> tuple[list[int], list[float]]:
    """Source frame index and time label of each frame that training samples.

    This is the training frame selection with no clip, no jitter and a
    resolution-only scale aug, as the model was trained. The duration is
    `num_source_frames / source_fps`, the value that the OpenCV decoder
    reports. Training reads the duration from the sample metadata instead.

    - Count N: `min(num_source_frames, max_frames)` when there are fewer
      than `min_frames` source frames, else
      `min(max(int(sampling_fps * D), min_frames), max_frames)`. N is then
      rounded to a multiple of T: up when the source and the cap allow it,
      else down, but never below T.
    - The cap counts the frames after the T padding: N is a multiple of T and
      at most `max_frames // T * T`, the frame count of the profiling video.
      `max_frames` also caps the short-video branch, which is the same as
      `min_frames = min(min_frames, max_frames)`. Training (and the
      Megatron-LM reference sampler) requires `max_frames >= min_frames` and
      `max_frames % T == 0`; for those settings this rule is the training
      rule.
    - Times: `t_i = (D - 1) / N * (i + 0.5)` in float32, as training
      computes them. The prompt labels are `t_i` with two decimals.
    - Frame: `floor(t_i * fps)`, the frame shown at time `t_i` for a
      constant frame rate video that starts at 0. A video shorter than 1 s
      has negative times (training does the same); its frame indices are
      clamped to `[0, num_source_frames - 1]` and the labels stay negative.

    Returns `(source_indices, timestamps)`, one entry per sampled frame, in
    time order. `source_indices` can repeat an index.
    """
    if num_source_frames < 1:
        raise ValueError("Nemotron Shortcut requires at least one video frame")
    if not source_fps > 0:
        raise ValueError(
            "Nemotron Shortcut frame sampling needs a known source fps, but the video "
            f"reports {source_fps!r}."
        )
    if not sampling_fps > 0:
        raise ValueError(
            f"Nemotron Shortcut sampling fps must be positive, got {sampling_fps}"
        )
    if min_frames < 1 or max_frames < temporal_patch_size or temporal_patch_size < 1:
        raise ValueError(
            "Nemotron Shortcut frame sampling needs min_frames >= 1 and max_frames >= "
            f"temporal_patch_size >= 1, got min_frames={min_frames}, "
            f"max_frames={max_frames}, temporal_patch_size={temporal_patch_size}."
        )
    duration = num_source_frames / source_fps
    if num_source_frames < min_frames:
        count = min(num_source_frames, max_frames)
    else:
        count = min(max(int(sampling_fps * duration), min_frames), max_frames)
    if count % temporal_patch_size:
        rounded_down = count // temporal_patch_size * temporal_patch_size
        rounded_up = rounded_down + temporal_patch_size
        if rounded_up <= num_source_frames and rounded_up <= max_frames:
            count = rounded_up
        else:
            count = max(temporal_patch_size, rounded_down)
    # The checks of the Megatron-LM reference sampler.
    assert count % temporal_patch_size == 0, count
    assert temporal_patch_size <= count <= max_frames, (count, max_frames)
    # Same float32 arithmetic as training, so the %.2f labels are the same.
    segment_size = float(duration - 1) / count
    timestamps = segment_size * (torch.arange(count, dtype=torch.float32) + 0.5)
    times = [float(value) for value in timestamps.tolist()]
    indices = [
        min(max(math.floor(time * source_fps), 0), num_source_frames - 1)
        for time in times
    ]
    return indices, times


@VIDEO_LOADER_REGISTRY.register(NEMOTRON_SHORTCUT_VIDEO_LOADER_BACKEND)
class NemotronShortcutVideoBackend(VideoBackend):
    """Video loader that samples frames with the NeMo RL training rule.

    For exports with either temporal patch size (1 or 2). It decodes only
    the frames that `training_video_frame_times` picks (unique, sorted) and
    adds to the metadata:

    - `frame_timestamps`: the training time label of each sampled frame;
    - `frame_source_indices`: the source frame of each label (an index
      repeats when two labels fall in one frame);
    - `frame_temporal_patch_size`: the T the count was rounded to.

    The Nemotron Shortcut processor then uses these frames and labels as they
    are. Options (`--media-io-kwargs '{"video": {...}}'`):
    `max_frames` (default 64, the training `video_max_num_frames`),
    `fps` (sampling rate, default 2), `min_frames` (default 8),
    `temporal_patch_size` (default 2), and the decoder `backend`.
    `num_frames` is not used for sampling. A video never gets more than
    `max_frames // T * T` frames; the processor rejects a video with more
    frames than the engine-level `max_frames` (the memory profile).
    """

    _sampling_suffix = "_nemotron_shortcut"

    @classmethod
    def _frame_times(
        cls,
        source: VideoSourceMetadata,
        target: VideoTargetMetadata,
        max_frames: int = _TRAINING_VIDEO_MAX_FRAMES,
        min_frames: int = _TRAINING_VIDEO_MIN_FRAMES,
        temporal_patch_size: int = 2,
        **kwargs,
    ) -> tuple[list[int], list[float]]:
        if kwargs:
            raise TypeError(
                f"Unknown {NEMOTRON_SHORTCUT_VIDEO_LOADER_BACKEND} video options: "
                f"{sorted(kwargs)}"
            )
        return training_video_frame_times(
            source.total_frames_num,
            source.original_fps,
            max_frames=max_frames,
            sampling_fps=target.fps,
            min_frames=min_frames,
            temporal_patch_size=temporal_patch_size,
        )

    @classmethod
    def compute_frames_index_to_sample(
        cls,
        source: VideoSourceMetadata,
        target: VideoTargetMetadata,
        **kwargs,
    ) -> list[int]:
        indices, _ = cls._frame_times(source, target, **kwargs)
        return sorted(set(indices))

    @classmethod
    def load_bytes(
        cls,
        data: bytes,
        num_frames: int = -1,
        fps: int = _TRAINING_VIDEO_FPS,
        max_duration: int = 300,
        frame_recovery: bool = False,
        *,
        backend: VideoDecoderBackend = "opencv",
        **kwargs: Any,
    ) -> tuple[DecodedFrames, dict[str, Any]]:
        frames, metadata = super().load_bytes(
            data,
            num_frames=num_frames,
            fps=fps,
            max_duration=max_duration,
            frame_recovery=frame_recovery,
            backend=backend,
            **kwargs,
        )
        sampling_kwargs = {
            key: value
            for key, value in kwargs.items()
            if key in ("max_frames", "min_frames", "temporal_patch_size")
        }
        source = VideoSourceMetadata(
            total_frames_num=metadata["total_num_frames"],
            original_fps=metadata["fps"],
            duration=metadata["duration"],
        )
        target = VideoTargetMetadata(
            num_frames=num_frames, fps=fps, max_duration=max_duration
        )
        indices, times = cls._frame_times(source, target, **sampling_kwargs)
        metadata = dict(metadata)
        metadata["frame_timestamps"] = times
        metadata["frame_source_indices"] = indices
        metadata["frame_temporal_patch_size"] = sampling_kwargs.get(
            "temporal_patch_size", 2
        )
        return frames, metadata


def loader_frame_selection(
    metadata: Mapping[str, object] | None,
    num_decoded: int,
    temporal_patch_size: int,
) -> tuple[list[int], list[float]] | None:
    """Positions in the decoded frames, and labels, of the training sample.

    Returns None when the metadata does not come from the `nemotron_shortcut_t2`
    loader (no `frame_timestamps`). Otherwise each label's source frame
    (`frame_source_indices`) is found in the decoded frames
    (`frames_indices`). A source frame that failed to decode is replaced
    by the last decoded frame before it (the first one if there is none);
    training also repeats its last decoded frame for missing frames.
    """
    if not metadata:
        return None
    timestamps = metadata.get("frame_timestamps")
    if timestamps is None:
        return None
    source_indices = metadata.get("frame_source_indices")
    decoded = metadata.get("frames_indices")
    if not isinstance(source_indices, Sequence) or not isinstance(decoded, Sequence):
        raise ValueError(
            "Nemotron Shortcut video metadata with frame_timestamps also needs "
            "frame_source_indices and frames_indices."
        )
    timestamps = [float(value) for value in cast(Sequence[float], timestamps)]
    if len(source_indices) != len(timestamps) or not timestamps:
        raise ValueError(
            f"Nemotron Shortcut video metadata has {len(timestamps)} frame_timestamps "
            f"and {len(source_indices)} frame_source_indices."
        )
    loader_t = metadata.get("frame_temporal_patch_size", temporal_patch_size)
    if loader_t != temporal_patch_size:
        raise ValueError(
            f"The {NEMOTRON_SHORTCUT_VIDEO_LOADER_BACKEND} video loader rounded the "
            f"frame count to a multiple of temporal_patch_size={loader_t}, but the "
            f"model has video_temporal_patch_size={temporal_patch_size}."
        )
    decoded = [int(index) for index in decoded]
    if len(decoded) != num_decoded or num_decoded == 0:
        raise ValueError(
            f"Nemotron Shortcut video has {num_decoded} decoded frames but "
            f"{len(decoded)} frames_indices."
        )
    if decoded != sorted(decoded):
        raise ValueError("Nemotron Shortcut video frames_indices must be sorted.")
    positions = []
    missing = 0
    for index in source_indices:
        position = bisect.bisect_right(decoded, int(index)) - 1
        if position < 0 or decoded[position] != index:
            missing += 1
        positions.append(max(position, 0))
    if missing:
        logger.warning(
            "Nemotron Shortcut video: %d of %d sampled frames were not decoded; "
            "using the previous decoded frame.",
            missing,
            len(source_indices),
        )
    return positions, timestamps


def video_frame_timestamps(
    frame_positions: Sequence[int], metadata: Mapping[str, object] | None
) -> list[float] | None:
    """Return the time in seconds of each selected frame of a decoded video.

    `frame_positions` index the decoded frames. The video loader metadata maps
    them to source frame indices (`frames_indices`) and gives the source
    `fps`. Returns None when the metadata has no usable fps.
    """
    if not metadata:
        return None
    fps = metadata.get("fps")
    if not isinstance(fps, (int, float)) or isinstance(fps, bool) or fps <= 0:
        return None
    frames_indices = metadata.get("frames_indices")
    if frames_indices is None:
        frames_indices = range(max(frame_positions, default=-1) + 1)
    source_indices = list(cast(Sequence[int], frames_indices))
    if any(position >= len(source_indices) for position in frame_positions):
        return None
    return [float(source_indices[position]) / fps for position in frame_positions]


def _video_metadata(
    videos: VideoProcessorItems, video_idx: int, video: object
) -> Mapping[str, object] | None:
    """Return the loader metadata of one video item, if the parser kept it."""
    if isinstance(video, tuple):
        return video[1]
    metadata = videos.metadata
    if isinstance(metadata, list):
        return metadata[video_idx] if video_idx < len(metadata) else None
    return metadata


def video_prompt_text(
    frame_timestamps: Sequence[float], temporal_patch_size: int, placeholder: str
) -> str:
    """Build the training video prompt with one line per tubelet.

    This is the training prompt formula with `frame_timestamps` given: a
    "This is a video:" header, then one line per tubelet, "Frame i sampled at
    X.XX seconds and frame i+1 sampled at Y.YY seconds: <placeholder>", and a
    final newline. Training passes
    "<img>{marker}</img>" as the placeholder.
    """
    frame_count = len(frame_timestamps)
    lines = ["This is a video:"]
    for frame_start in range(0, frame_count, temporal_patch_size):
        frame_end = min(frame_start + temporal_patch_size, frame_count)
        timestamps = [
            f"{'Frame' if frame_index == frame_start else 'frame'} "
            f"{frame_index + 1} sampled at "
            f"{float(frame_timestamps[frame_index]):.2f} seconds"
            for frame_index in range(frame_start, frame_end)
        ]
        lines.append(" and ".join(timestamps) + f": {placeholder}")
    return "\n".join(lines) + "\n"


def video_prompt_token_ids(
    encode,
    frame_timestamps: Sequence[float],
    temporal_patch_size: int,
    tubelet_token_ids: Sequence[Sequence[int]],
) -> list[int]:
    """Token IDs of the per-tubelet video prompt.

    `tubelet_token_ids` holds the visual tokens of each tubelet, from the
    first `<img>` through the closing `</img>`. The text between them is
    encoded segment by segment, as the training tokenizer does around its
    media marker.
    """
    text = video_prompt_text(
        frame_timestamps, temporal_patch_size, _VIDEO_TUBELET_MARKER
    )
    segments = text.split(_VIDEO_TUBELET_MARKER)
    if len(segments) - 1 != len(tubelet_token_ids):
        raise ValueError(
            f"Nemotron Shortcut video prompt has {len(segments) - 1} tubelets but "
            f"{len(tubelet_token_ids)} tubelet token sequences."
        )
    token_ids: list[int] = []
    for index, segment in enumerate(segments):
        if segment:
            token_ids.extend(encode(segment))
        if index < len(tubelet_token_ids):
            token_ids.extend(tubelet_token_ids[index])
    return token_ids


def profile_video_num_frames(
    seq_len: int, tokens_per_tubelet: int, max_frames: int, temporal_patch_size: int
) -> int:
    """Frame count of the profiling video when T frames share one token grid.

    Each frame costs `tokens_per_tubelet / T` tokens. The count is a multiple
    of T, at least T, and at most the tubelet count of the longest video that
    the processor makes (`select_video_frame_count` with many decoded
    frames, padded up to a multiple of T).
    """
    longest = select_video_frame_count(
        max(max_frames, temporal_patch_size), max_frames, temporal_patch_size
    )
    max_tubelets = -(-longest // temporal_patch_size)
    tubelets = min(max_tubelets, max(1, seq_len // tokens_per_tubelet))
    return tubelets * temporal_patch_size


class _TimedVideoProcessorItems(VideoProcessorItems):
    """Video items whose processor-cache key includes the loader metadata.

    For every T the prompt carries each frame's time, which comes from the
    metadata (`fps`, `frames_indices`). The base class adds the metadata
    to the key only for `MediaWithBytes` items, so two raw-frame inputs with
    the same frames and a different fps would share cached timestamps.
    """

    def get_item_for_hash(self, index: int) -> object:
        item = super().get_item_for_hash(index)
        if isinstance(item, tuple) or not isinstance(self.metadata, list):
            return item
        metadata = self.metadata[index]
        return item if metadata is None else (item, metadata)


class _NemotronShortcutTimedVideoDataParser(MultiModalDataParser):
    """Data parser whose video items use `_TimedVideoProcessorItems`."""

    def _parse_video_data(self, data):
        items = super()._parse_video_data(data)
        if type(items) is VideoProcessorItems:
            items = _TimedVideoProcessorItems(items.data, metadata=items.metadata)
        return items


class NemotronShortcutProcessingInfo(BaseProcessingInfo):
    def get_data_parser(self) -> MultiModalDataParser:
        return _NemotronShortcutTimedVideoDataParser(
            expected_hidden_size=self._get_expected_hidden_size(),
            allow_missing_mm_embeddings=self.allow_missing_mm_embeddings,
        )

    def get_hf_config(self) -> PreTrainedConfig:
        return self.ctx.get_hf_config()

    @cached_property
    def video_token(self) -> str:
        if _VIDEO_TOKEN in self.get_tokenizer().get_vocab():
            return _VIDEO_TOKEN
        return _LEGACY_VIDEO_TOKEN

    def get_hf_processor(self, **kwargs: object) -> ProcessorMixin:
        return self.ctx.get_hf_processor(**kwargs)

    def get_video_temporal_patch_size(self) -> int:
        return get_video_temporal_patch_size(self.get_hf_config())

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"image": None, "video": 1}

    def get_max_video_frames(self) -> int:
        """The frame cap of one video, which the memory profile uses.

        The cap counts the frames after the T padding, so it must be at least
        T; the processor never emits more frames than the cap.
        """
        video_kwargs = self.ctx.get_mm_config().media_io_kwargs.get("video", {})
        temporal_patch_size = self.get_video_temporal_patch_size()
        if video_kwargs.get("video_backend") == NEMOTRON_SHORTCUT_VIDEO_LOADER_BACKEND:
            # That loader caps the frame count with max_frames, not num_frames.
            name = "max_frames"
            cap = video_kwargs.get(name, _TRAINING_VIDEO_MAX_FRAMES)
        else:
            name = "num_frames"
            cap = video_kwargs.get(name, _DEFAULT_MAX_VIDEO_FRAMES)
        if not isinstance(cap, int) or isinstance(cap, bool) or cap <= 0:
            raise ValueError(
                f"Nemotron Shortcut video {name} must be a positive integer"
            )
        if cap < temporal_patch_size:
            raise ValueError(
                f"Nemotron Shortcut video {name}={cap} is below "
                f"video_temporal_patch_size={temporal_patch_size}; the frame cap "
                "counts the padded frames, so it must hold one tubelet."
            )
        return cap

    def get_image_size_with_most_features(self) -> ImageSize:
        processor = self.get_hf_processor()
        edge = processor.image_processor.size["longest_edge"]
        multiple = processor.patch_size * processor.spatial_merge_size
        aligned_edge = -(-edge // multiple) * multiple
        return ImageSize(width=aligned_edge, height=aligned_edge)

    def get_video_frame_size(self, width: int = 1, height: int = 1) -> ImageSize:
        config = self.get_hf_config()
        target = getattr(config, "video_target_num_patches", None)
        target = 1024 if target is None else target
        keep_aspect = getattr(config, "video_maintain_aspect_ratio", False)
        keep_aspect = False if keep_aspect is None else keep_aspect
        if type(target) is not int or target <= 0:
            raise ValueError("video_target_num_patches must be a positive integer")
        if not isinstance(keep_aspect, bool):
            raise ValueError("video_maintain_aspect_ratio must be a boolean")
        if not keep_aspect and math.isqrt(target) ** 2 != target:
            raise ValueError("video_target_num_patches must be square")
        get_size = partial(
            get_video_target_size_and_feature_size,
            target_patches=target,
            maintain_aspect_ratio=keep_aspect,
            patch_size=config.vision_config.patch_size,
            downsample_ratio=1 / config.spatial_merge_size,
        )
        frame_width, frame_height, _ = get_size(width, height)
        profile_width, profile_height, _ = get_size(1, 1)
        if frame_width * frame_height > profile_width * profile_height:
            raise ValueError(
                "Video aspect ratio exceeds the profiled frame patch budget"
            )
        return ImageSize(width=frame_width, height=frame_height)


class NemotronShortcutDummyInputsBuilder(Mistral3DummyInputsBuilder):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        num_videos = mm_counts.get("video", 0)
        return super().get_dummy_text(mm_counts) + self.info.video_token * num_videos

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: MultiModalDummyOptions,
    ) -> MultiModalDataDict:
        data = dict(super().get_dummy_mm_data(seq_len, mm_counts, mm_options))
        num_videos = mm_counts.get("video", 0)
        if num_videos:
            width, height = self.info.get_video_frame_size()
            processor = self.info.get_hf_processor()
            multiple = processor.patch_size * processor.spatial_merge_size
            tokens_per_frame = (width // multiple + 1) * (height // multiple)
            temporal_patch_size = self.info.get_video_temporal_patch_size()
            if temporal_patch_size == 1:
                num_frames = min(
                    self.info.get_max_video_frames(),
                    max(1, seq_len // tokens_per_frame),
                )
            else:
                # One token grid per tubelet of T frames.
                num_frames = profile_video_num_frames(
                    seq_len,
                    tokens_per_frame,
                    self.info.get_max_video_frames(),
                    temporal_patch_size,
                )
            data["video"] = self._get_dummy_videos(
                width=width,
                height=height,
                num_frames=num_frames,
                num_videos=num_videos,
                overrides=mm_options.get("video"),
            )
        return data


class NemotronShortcutMultiModalProcessor(Mistral3MultiModalProcessor):
    def _apply_prompt_updates_via_text(
        self, token_ids, mm_prompt_updates, *, encode_segments_separately=False
    ):
        # Keep BPE merges across chat text from changing the video header.
        return super()._apply_prompt_updates_via_text(
            token_ids, mm_prompt_updates, encode_segments_separately=True
        )

    def _get_cache_missing_items(self, cache, mm_data_items, mm_hashes):
        """Keep the loader metadata of the videos that miss the processor cache.

        The base class rebuilds the missing items from `items[idx]`, which
        holds the frames only, and parses them again. The metadata (`fps`,
        `frames_indices`, the loader labels) is lost there, so a served video
        (the processor cache is on by default) would get the fallback labels
        "frame i at i seconds" instead of its real timestamps.
        """
        mm_is_cached, mm_missing_items = super()._get_cache_missing_items(
            cache, mm_data_items, mm_hashes
        )
        videos = mm_data_items.get("video")
        missing = mm_missing_items.get("video")
        if (
            isinstance(videos, VideoProcessorItems)
            and isinstance(missing, VideoProcessorItems)
            and isinstance(videos.metadata, list)
        ):
            idxs = [i for i, hit in enumerate(mm_is_cached["video"]) if not hit]
            if len(idxs) != missing.get_count():
                raise ValueError(
                    f"Nemotron Shortcut video cache: {len(idxs)} missing indices for "
                    f"{missing.get_count()} missing items."
                )
            missing.metadata = [videos.metadata[i] for i in idxs]
        return mm_is_cached, mm_missing_items

    def _apply_hf_processor_main(
        self,
        mm_items: MultiModalDataItems,
        hf_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        image_items = mm_items.select(set(mm_items) - {"video"})
        output = super()._apply_hf_processor_main(image_items, hf_kwargs)
        if "video" not in mm_items:
            return output

        videos = mm_items.get_items("video", VideoProcessorItems)
        image_processor = self.info.get_hf_processor(**hf_kwargs).image_processor
        temporal_patch_size = cast(
            NemotronShortcutProcessingInfo, self.info
        ).get_video_temporal_patch_size()
        pixel_values_videos = []
        video_timestamps = []
        for video_idx, video in enumerate(videos.get_all()):
            frames = video[0] if isinstance(video, tuple) else video
            if frames is None or len(frames) == 0:
                raise ValueError("Nemotron Shortcut requires at least one video frame")
            max_frames = cast(
                NemotronShortcutProcessingInfo, self.info
            ).get_max_video_frames()
            # Frames and labels sampled by the nemotron_shortcut_t2 loader with the
            # training rule (any T): use them as they are.
            metadata = _video_metadata(videos, video_idx, video)
            training_selection = loader_frame_selection(
                metadata, len(frames), temporal_patch_size
            )
            if training_selection is not None:
                frame_indices, timestamps = training_selection
                selected_frames = [frames[i] for i in frame_indices]
            else:
                if temporal_patch_size == 1:
                    num_selected = min(len(frames), max_frames)
                else:
                    # A multiple of T distinct frames, as in training; pad
                    # only a video shorter than T.
                    num_selected = select_video_frame_count(
                        len(frames), max_frames, temporal_patch_size
                    )
                frame_indices = (
                    torch.linspace(0, len(frames) - 1, num_selected).long().tolist()
                )
                selected_frames = [frames[i] for i in frame_indices]
                # The training prompt labels every frame with its time, for
                # every T (nemo_rl nemotron4_visual._video_prompt).
                frame_timestamps = video_frame_timestamps(frame_indices, metadata)
                if frame_timestamps is None:
                    logger.warning_once(
                        "Nemotron Shortcut video has no fps metadata; the prompt "
                        "labels frame i with i seconds."
                    )
                    frame_timestamps = [float(i) for i in frame_indices]
                timestamps = frame_timestamps
            # Only a video with fewer than T frames gets here with a count that
            # is not a multiple of T. The training encoder pads it by repeating
            # the last frame; the label repeats with it. No-op for T = 1.
            selected_frames = pad_to_temporal_multiple(
                selected_frames, temporal_patch_size
            )
            timestamps = pad_to_temporal_multiple(timestamps, temporal_patch_size)
            if len(selected_frames) > max_frames:
                # Loader frames sampled with a larger max_frames than the
                # engine cap (for example a per-request media_io_kwargs). More
                # frames than the profiled video would exceed the encoder
                # budget, so fail instead of serving them.
                raise ValueError(
                    f"Nemotron Shortcut video has {len(selected_frames)} frames, "
                    f"more than the frame cap {max_frames} of the memory "
                    "profile. Sample it with max_frames <= "
                    f"{max_frames}, or start the server with a larger "
                    "video max_frames."
                )
            video_timestamps.append(torch.tensor(timestamps, dtype=torch.float64))
            frame_pixels = []
            for frame in selected_frames:
                if isinstance(frame, torch.Tensor):
                    frame = frame.detach().cpu().numpy()
                array = np.asarray(frame)
                if (
                    array.ndim == 3
                    and array.shape[-1] not in (1, 3, 4)
                    and array.shape[0] == 3
                ):
                    array = np.moveaxis(array, 0, -1)
                height, width = array.shape[:2]
                size = cast(
                    NemotronShortcutProcessingInfo, self.info
                ).get_video_frame_size(width, height)
                frame_pixels.append(
                    resize_video_frame(
                        array, size.width, size.height, image_processor
                    ).to(dtype=self.info.ctx.model_config.dtype)
                )
            pixel_values_videos.append(frame_pixels)
        output["pixel_values_videos"] = pixel_values_videos
        output["video_timestamps"] = video_timestamps
        return output

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        fields = {
            **super()._get_mm_fields_config(hf_inputs, hf_processor_mm_kwargs),
            "pixel_values_videos": MultiModalFieldConfig.batched("video"),
        }
        if "video_timestamps" in hf_inputs:
            # The video prompt labels each frame with its time.
            fields["video_timestamps"] = MultiModalFieldConfig.batched("video")
        return fields

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)

        def image_replacement(item_idx: int):
            pixels = out_mm_kwargs["image"][item_idx]["pixel_values"].data
            assert isinstance(pixels, torch.Tensor)
            height, width = pixels.shape[-2:]
            image_tokens = processor.replace_image_token(
                {"image_sizes": [(height, width)]},
                0,
            )
            tokens = processor.tokenizer.encode(image_tokens, add_special_tokens=False)
            return PromptUpdateDetails.select_token_id(tokens, processor.image_token_id)

        temporal_patch_size = cast(
            NemotronShortcutProcessingInfo, self.info
        ).get_video_temporal_patch_size()

        def encode(text: str) -> list[int]:
            return processor.tokenizer.encode(text, add_special_tokens=False)

        def tubelet_video_replacement(item_idx: int):
            item = out_mm_kwargs["video"][item_idx]
            frames = item["pixel_values_videos"].data
            timestamps = item["video_timestamps"].data
            assert isinstance(frames, (list, tuple))
            assert isinstance(timestamps, torch.Tensor)
            if len(frames) % temporal_patch_size or len(frames) != len(timestamps):
                raise ValueError(
                    f"Nemotron Shortcut video has {len(frames)} frames and "
                    f"{len(timestamps)} timestamps; both must be the same "
                    f"multiple of video_temporal_patch_size={temporal_patch_size}."
                )
            tubelet_tokens = []
            for start in range(0, len(frames), temporal_patch_size):
                pixels = frames[start]
                assert isinstance(pixels, torch.Tensor)
                height, width = pixels.shape[-2:]
                # A tubelet embeds as one frame's patch grid.
                image_tokens = processor.replace_image_token(
                    {"image_sizes": [(height, width)]}, 0
                )
                tubelet_tokens.append(encode(image_tokens))
            tokens = video_prompt_token_ids(
                encode, timestamps.tolist(), temporal_patch_size, tubelet_tokens
            )
            return PromptUpdateDetails.select_token_id(tokens, processor.image_token_id)

        return [
            PromptReplacement(
                modality="image",
                target=[processor.image_token_id],
                replacement=image_replacement,
            ),
            PromptReplacement(
                modality="video",
                target=processor.tokenizer.encode(
                    cast(NemotronShortcutProcessingInfo, self.info).video_token,
                    add_special_tokens=False,
                ),
                replacement=tubelet_video_replacement,
            ),
        ]


class NemotronShortcutVisionModel(PixtralHFVisionModel):
    """vLLM's Pixtral encoder, plus `video_patch_conv` for video tubelets.

    A tubelet stacks T frames frame-major on the channel axis.
    `video_patch_conv` embeds it as one patch grid, and the rest of the encoder
    runs as for one frame of the same size, with the same RoPE positions."""

    def __init__(
        self,
        config: PixtralVisionConfig,
        video_temporal_patch_size: int,
        quant_config=None,
        prefix: str = "",
    ) -> None:
        super().__init__(config, quant_config=quant_config, prefix=prefix)
        self.video_patch_conv: Conv2dLayer | None = None
        if video_temporal_patch_size > 1:
            self.video_patch_conv = Conv2dLayer(
                in_channels=config.num_channels * video_temporal_patch_size,
                out_channels=config.hidden_size,
                kernel_size=config.patch_size,
                stride=config.patch_size,
                bias=False,
            )

    def encode_tubelets(self, tubelets: list[torch.Tensor]) -> tuple[torch.Tensor, ...]:
        """Encode the tubelets of a video; `forward` encodes images."""
        assert self.video_patch_conv is not None
        tubelets = [tubelet.to(device=self.device) for tubelet in tubelets]
        patches, grid_sizes = pack_image_patches(
            tubelets, (self.patch_size, self.patch_size)
        )
        weight = self.video_patch_conv.weight
        patch_embeds = F.linear(
            patches.to(weight.dtype), weight.reshape(weight.shape[0], -1)
        )
        patch_embeds = self.ln_pre(patch_embeds.unsqueeze(0))
        positions = torch.cat(
            [
                torch.cartesian_prod(torch.arange(rows), torch.arange(cols))
                for rows, cols in grid_sizes
            ]
        ).to(self.device)
        if not TRANSFORMERS_WITH_AXIAL_ROPE:
            max_width = self.config.image_size // self.config.patch_size
            positions = positions[:, 0] * max_width + positions[:, 1]
        embed_sizes = [rows * cols for rows, cols in grid_sizes]
        out = self.transformer(
            patch_embeds,
            self.patch_positional_embedding(patch_embeds, positions),
            **self.packed_attention_metadata(embed_sizes, self.device),
            return_all_hidden_states=False,
        )
        return torch.split(out.squeeze(0), embed_sizes)


class NemotronShortcutMultiModalProjector(Mistral3MultiModalProjector):
    def __init__(self, config: PreTrainedConfig, quant_config=None, prefix: str = ""):
        if config.vision_feature_layer != -1:
            raise ValueError(
                "Nemotron Shortcut currently only supports vision_feature_layer=-1"
            )
        assert isinstance(config.vision_config, PixtralVisionConfig)
        vision_size = config.vision_config.hidden_size
        text_size = config.text_config.hidden_size
        super().__init__(
            vision_hidden_size=vision_size,
            text_hidden_size=text_size,
            spatial_merge_size=config.spatial_merge_size,
            patch_size=config.vision_config.patch_size,
            projector_hidden_act=config.projector_hidden_act,
            multimodal_projector_bias=config.multimodal_projector_bias,
            quant_config=quant_config,
            prefix=prefix,
        )
        self.norm = RMSNorm(vision_size, eps=config.projector_norm_eps)


@MULTIMODAL_REGISTRY.register_processor(
    NemotronShortcutMultiModalProcessor,
    info=NemotronShortcutProcessingInfo,
    dummy_inputs=NemotronShortcutDummyInputsBuilder,
)
class NemotronShortcutForConditionalGeneration(
    nn.Module,
    SupportsMultiModal,
    SupportsPP,
    IsHybrid,
    SdsaKVCacheGrouping,
    SupportsReplaySSM,
):
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "model.language_model.mtp.": None,
            "model.language_model.": "language_model.model.",
            "model.vision_tower.": "vision_tower.",
            "model.multi_modal_projector.": "multi_modal_projector.",
            "model.video_patch_conv.": "vision_tower.video_patch_conv.",
            "lm_head.": "language_model.lm_head.",
        }
    )

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str:
        if modality.startswith("image"):
            return "<img>"
        if modality.startswith("video"):
            return "<video>"
        raise ValueError("Nemotron Shortcut supports only image and video inputs")

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        self.config = config = vllm_config.model_config.hf_config
        self.quant_config = vllm_config.quant_config
        self.configure_mm_token_handling(
            config.text_config.vocab_size,
            [config.image_token_id],
        )
        self.video_temporal_patch_size = get_video_temporal_patch_size(config)
        with self._mark_tower_model(vllm_config, {"image", "video"}):
            vision_config = config.vision_config
            rope = dict(vision_config.rope_parameters or {})
            if rope.get("rope_type") == "axial" and not TRANSFORMERS_WITH_AXIAL_ROPE:
                # Before 5.17, Pixtral's default RoPE computes the same 2D angles.
                vision_config = copy.deepcopy(vision_config)
                vision_config.rope_parameters = {**rope, "rope_type": "default"}
            self.vision_tower = NemotronShortcutVisionModel(
                vision_config,
                self.video_temporal_patch_size,
                quant_config=vllm_config.quant_config,
                prefix=maybe_prefix(prefix, "vision_tower"),
            )
            self.multi_modal_projector = NemotronShortcutMultiModalProjector(
                config,
                vllm_config.quant_config,
                prefix=maybe_prefix(prefix, "multi_modal_projector"),
            )
        with self._mark_language_model(vllm_config):
            self.language_model = NemotronShortcutForCausalLM(
                vllm_config=vllm_config,
                prefix=maybe_prefix(prefix, "language_model"),
            )
        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        embeddings: list[torch.Tensor] = []
        for key, pixels in kwargs.items():
            if key == "pixel_values":
                embeddings.extend(self._embed_images(pixels))
            elif key == "pixel_values_videos":
                assert isinstance(pixels, (list, tuple))
                for frames in pixels:
                    if self.video_temporal_patch_size == 1:
                        embeddings.append(torch.cat(self._embed_images(frames)))
                        continue
                    tubelets = group_frames_into_tubelets(
                        list(frames), self.video_temporal_patch_size
                    )
                    embeddings.append(
                        torch.cat(self._embed_images(tubelets, video=True))
                    )
        return embeddings

    def _embed_images(
        self, pixels: object, video: bool = False
    ) -> tuple[torch.Tensor, ...]:
        assert isinstance(pixels, (torch.Tensor, list, tuple))
        images = list(pixels)
        sizes = torch.tensor([image.shape[-2:] for image in images])
        if video:
            features = self.vision_tower.encode_tubelets(images)
        else:
            features = self.vision_tower(images)
        embeds = self.multi_modal_projector(torch.cat(features), sizes)
        multiple = self.config.vision_config.patch_size * self.config.spatial_merge_size
        lengths = (sizes // multiple).prod(-1).tolist()
        return embeds.split(lengths)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors:
        return self.language_model(
            input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds=inputs_embeds if intermediate_tensors is None else None,
        )

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.language_model.compute_logits(hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # The loader hands the language model its weights in runs, so check the
        # SDSA indexer once, over the whole load.
        self.language_model.check_sdsa_indexer_on_load = False
        loaded = AutoWeightsLoader(self).load_weights(
            weights, mapper=self.hf_to_vllm_mapper
        )
        check_sdsa_indexer_loaded(self, loaded)
        return loaded

    def get_mm_mapping(self) -> MultiModelKeys:
        return MultiModelKeys.from_string_field(
            language_model="language_model",
            connector="multi_modal_projector",
            tower_model="vision_tower",
        )

    get_mamba_state_dtype_from_config = (
        NemotronShortcutForCausalLM.get_mamba_state_dtype_from_config
    )
    get_mamba_state_shape_from_config = (
        NemotronShortcutForCausalLM.get_mamba_state_shape_from_config
    )
    get_mamba_state_copy_func = NemotronShortcutForCausalLM.get_mamba_state_copy_func
