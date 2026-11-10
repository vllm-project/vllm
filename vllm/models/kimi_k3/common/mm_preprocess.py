# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared Kimi-K3 multimodal preprocessing."""

import dataclasses
import functools
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import torch
from PIL import Image as PILImage
from transformers import BatchFeature

from vllm.config import ModelConfig
from vllm.config.multimodal import (
    ImageDummyOptions,
    MultiModalDummyOptions,
    VideoDummyOptions,
)
from vllm.inputs import MultiModalDataDict
from vllm.logger import init_logger
from vllm.multimodal.inputs import (
    MultiModalFieldConfig,
    MultiModalKwargsItems,
)
from vllm.multimodal.parse import (
    ImageProcessorItems,
    ImageSize,
    MultiModalDataItems,
    MultiModalDataParser,
)
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder,
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    InputProcessingContext,
    PromptReplacement,
    PromptUpdate,
    PromptUpdateDetails,
    cached_encode,
)
from vllm.transformers_utils.configs.kimi_k3 import KimiK3Config
from vllm.transformers_utils.processor import cached_get_image_processor
from vllm.transformers_utils.processors.kimi_k3 import KimiK3Processor
from vllm.transformers_utils.processors.kimi_k25_vision_fused import (
    KimiK25FusedVisionProcessor,
)

logger = init_logger(__name__)


def navit_resize_image(
    width: int,
    height: int,
    patch_size: int,
    merge_kernel_size: int,
    in_patch_limit: int,
    patch_limit_on_one_side: int,
    fixed_output_tokens: int | None,
):
    # Apply the patch limits.
    s1 = math.sqrt(
        in_patch_limit
        / (max(1.0, width // patch_size) * max(1.0, height // patch_size))
    )
    s2 = patch_limit_on_one_side * patch_size / width
    s3 = patch_limit_on_one_side * patch_size / height
    scale = min(1.0, s1, s2, s3)
    new_w, new_h = max(1, int(width * scale)), max(1, int(height * scale))
    new_w = min(new_w, patch_limit_on_one_side * patch_size)
    new_h = min(new_h, patch_limit_on_one_side * patch_size)

    factor = merge_kernel_size * patch_size

    pad_height = (factor - new_h % factor) % factor
    pad_width = (factor - new_w % factor) % factor

    if fixed_output_tokens is not None:
        num_tokens = fixed_output_tokens
    else:
        # Calculate new dimensions after padding and patching
        token_height = (new_h + pad_height) // factor
        token_width = (new_w + pad_width) // factor

        assert token_height * merge_kernel_size <= patch_limit_on_one_side, (
            f"token_height {token_height} * merge_kernel_size {merge_kernel_size} > "
            f"patch_limit_on_one_side {patch_limit_on_one_side}"
        )
        assert token_width * merge_kernel_size <= patch_limit_on_one_side, (
            f"token_width {token_width} * merge_kernel_size {merge_kernel_size} > "
            f"patch_limit_on_one_side {patch_limit_on_one_side}"
        )

        num_tokens = token_height * token_width
    return {
        "num_tokens": num_tokens,
        "new_width": new_w,
        "new_height": new_h,
        "pad_width": pad_width,
        "pad_height": pad_height,
        "sampled_nframes": 1,
    }


KIMI_K3_VIDEO_PLACEHOLDER = "<|kimi_video_placeholder|>"
"""Text placeholder for one video in a Kimi-K3 prompt.

The chat renderer writes one per `video_url` part. Like the image placeholder
it is plain text, not a vocabulary token, and the processor replaces it with
the video's timestamped chunk blocks.
"""


def format_kimi_k3_timestamp(seconds: float, mode: str) -> str:
    """Format a chunk start time the way the Kimi vision processors do."""
    seconds = max(0.0, float(seconds))
    total = int(seconds)
    millis = int((seconds % 1) * 1000)
    hours, minutes, secs = (total // 3600) % 24, (total // 60) % 60, total % 60
    if mode == "hh:mm:ss.fff":
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"
    if mode == "mm:ss.fff":
        return f"{minutes:02d}:{secs:02d}.{millis:03d}"
    if mode == "mm:ss":
        return f"{minutes:02d}:{secs:02d}"
    raise ValueError(f"Invalid Kimi-K3 timestamp mode: {mode}")


@dataclass(frozen=True)
class KimiK3VideoPlan:
    """How one decoded video becomes MoonViT3d chunks.

    `frame_indices` index the decoded frames handed to the processor;
    `chunk_starts` are the start times, in seconds, of each chunk of
    `temporal_merge_kernel_size` frames (the last chunk may be shorter).
    """

    frame_indices: tuple[int, ...]
    chunk_starts: tuple[float, ...]
    frames_per_chunk: int
    in_patch_limit_each_frame: int

    @property
    def num_chunks(self) -> int:
        return len(self.chunk_starts)


def plan_kimi_k3_video(
    num_decoded_frames: int,
    metadata: Mapping[str, Any] | None,
    media_proc_cfg: Mapping[str, Any],
) -> KimiK3VideoPlan:
    """Sample frames and split them into chunks per the checkpoint config.

    The video loader may already have sampled frames (`frames_indices` in
    the metadata point into the source video). The checkpoint asks for
    `sample_fps` frames per second; when the decoded frames are denser than
    that, they are thinned uniformly. The video's total patch budget
    (`in_patch_limit_video`) is split evenly across the kept frames.
    """
    if num_decoded_frames <= 0:
        raise ValueError("Kimi-K3 video input must contain at least one frame.")
    metadata = metadata or {}

    cfg_fps = float(media_proc_cfg.get("sample_fps") or 1.0)
    src_fps = float(metadata.get("fps") or 0.0)
    src_indices = metadata.get("frames_indices")
    if src_fps <= 0 or src_indices is None or len(src_indices) != num_decoded_frames:
        src_indices = list(range(num_decoded_frames))
    if src_fps <= 0:
        # Unknown source rate: treat the decoded frames as already sampled at
        # the checkpoint rate.
        src_fps = cfg_fps
        total_src_frames = num_decoded_frames
    else:
        total_src_frames = int(metadata.get("total_num_frames") or 0)
        total_src_frames = max(total_src_frames, int(src_indices[-1]) + 1)

    sample_fps = min(cfg_fps, src_fps)
    wanted = max(round(total_src_frames * sample_fps / src_fps), 1)
    max_frames = media_proc_cfg.get("max_num_frames_each_video")
    if max_frames is not None:
        wanted = min(wanted, max(1, int(max_frames)))
    wanted = min(wanted, num_decoded_frames)

    if wanted == num_decoded_frames:
        frame_indices = list(range(num_decoded_frames))
    else:
        frame_indices = (
            np.linspace(0, num_decoded_frames - 1, wanted).round().astype(int).tolist()
        )

    frames_per_chunk = int(media_proc_cfg.get("temporal_merge_kernel_size") or 1)
    if not 1 <= frames_per_chunk <= 4:
        raise ValueError(
            "Kimi-K3 temporal_merge_kernel_size must be between 1 and 4, "
            f"got {frames_per_chunk}"
        )

    per_frame = int(
        media_proc_cfg.get("in_patch_limit_each_frame")
        or media_proc_cfg["in_patch_limit"]
    )
    total_limit = media_proc_cfg.get("in_patch_limit_video")
    if total_limit is not None:
        per_frame = min(per_frame, max(1, round(int(total_limit) / wanted)))

    chunk_starts = tuple(
        float(src_indices[frame_indices[start]]) / src_fps
        for start in range(0, wanted, frames_per_chunk)
    )
    return KimiK3VideoPlan(
        frame_indices=tuple(frame_indices),
        chunk_starts=chunk_starts,
        frames_per_chunk=frames_per_chunk,
        in_patch_limit_each_frame=per_frame,
    )


def _iter_video_chunk_batches(
    grid_thws: torch.Tensor, max_patches: int
) -> list[tuple[int, int, int, int]]:
    """Group consecutive chunk grids so each group has at most `max_patches`.

    Returns `(row_start, row_end, patch_start, patch_end)` per group. A
    single chunk larger than `max_patches` forms its own group.
    """
    sizes = grid_thws.prod(-1).tolist()
    groups = []
    row_start = patch_start = patch = 0
    for row, size in enumerate(sizes):
        if row > row_start and patch + size - patch_start > max_patches:
            groups.append((row_start, row, patch_start, patch))
            row_start, patch_start = row, patch
        patch += size
    if sizes:
        groups.append((row_start, len(sizes), patch_start, patch))
    return groups


def _get_image_processor(model_config: ModelConfig) -> KimiK25FusedVisionProcessor:
    return cached_get_image_processor(
        model_config.model,
        revision=model_config.revision,
        trust_remote_code=model_config.trust_remote_code,
        processor_cls_overrides=KimiK25FusedVisionProcessor,
    )


def get_kimi_k3_image_patch_limit(model_config: ModelConfig) -> int:
    """ViT patches of the largest single image (`in_patch_limit`)."""
    return int(_get_image_processor(model_config).media_proc_cfg["in_patch_limit"])


def encode_kimi_k3_videos(
    encode_chunks: Callable[[torch.Tensor, torch.Tensor], Sequence[torch.Tensor]],
    pixel_values: torch.Tensor,
    grid_thws: torch.Tensor,
    num_chunks: object,
    merge_kernel_size: Sequence[int],
    max_patches_per_forward: int,
) -> list[torch.Tensor]:
    """Encode the chunks of a batch of videos and return one embedding per video.

    `encode_chunks(pixel_values, grid_thws)` runs the vision tower and
    projector and returns one feature tensor per grid row. Each chunk is its
    own `[t, h, w]` grid, so the tower (and its encoder-DP sharding) treats
    chunks as independent items.

    A chunk costs `t` times the patches of an image with the same number of
    output tokens, so encoding every chunk in one call would exceed the
    activation peak the startup profile measured for the largest image. The
    chunks are therefore encoded in groups of at most
    `max_patches_per_forward` patches, and each group's features are written
    into one preallocated buffer: the peak is the output plus one group,
    not the output twice (per-chunk features plus their concatenation).
    """
    merge_h, merge_w = merge_kernel_size
    chunk_tokens = (
        (grid_thws[:, 1] // merge_h) * (grid_thws[:, 2] // merge_w)
    ).tolist()
    out: torch.Tensor | None = None
    offset = 0
    for row0, row1, patch0, patch1 in _iter_video_chunk_batches(
        grid_thws, max_patches_per_forward
    ):
        for feat in encode_chunks(pixel_values[patch0:patch1], grid_thws[row0:row1]):
            if out is None:
                out = feat.new_empty((sum(chunk_tokens), feat.shape[-1]))
            out[offset : offset + feat.shape[0]] = feat
            offset += feat.shape[0]
    if out is None or offset != out.shape[0]:
        raise ValueError(
            f"Kimi-K3 video encoder returned {offset} rows, "
            f"expected {sum(chunk_tokens)}"
        )

    counts = torch.as_tensor(num_chunks).flatten().tolist()
    if sum(counts) != len(chunk_tokens):
        raise ValueError(
            f"Kimi-K3 video chunk count mismatch: {sum(counts)} expected, "
            f"{len(chunk_tokens)} encoded"
        )
    per_video, start = [], 0
    for n in counts:
        per_video.append(sum(chunk_tokens[start : start + n]))
        start += n
    return list(out.split(per_video))


class KimiK3ProcessingInfo(BaseProcessingInfo):
    """Processing information for the Kimi-K3 model (images and videos).

    K3 uses the standard `image` modality (unlike K2.5's unified
    `vision_chunk`), so it builds its own `KimiK3Processor` wrapper around
    the checkpoint's image processor and resolves the `<|media_pad|>` token
    id the same way K2.5 does.
    """

    def __init__(self, ctx: InputProcessingContext) -> None:
        super().__init__(ctx)

        self.hf_config = hf_config = self.get_hf_config()

        tokenizer = self.get_tokenizer()
        image_processor = _get_image_processor(self.ctx.model_config)

        # Resolve token ID from the tokenizer because transformers v5
        # may remap token IDs vs config.json.
        config_token_id = hf_config.media_placeholder_token_id
        resolved_token_id = tokenizer.convert_tokens_to_ids("<|media_pad|>")
        unk_token_id = getattr(tokenizer, "unk_token_id", None)
        is_valid_resolved = isinstance(resolved_token_id, int) and (
            unk_token_id is None or resolved_token_id != unk_token_id
        )
        if is_valid_resolved and resolved_token_id != config_token_id:
            logger.warning_once(
                "Kimi-K3 config.media_placeholder_token_id (%d) disagrees "
                "with tokenizer mapping for <|media_pad|> (%d). "
                "Using tokenizer value.",
                config_token_id,
                resolved_token_id,
            )
            media_token_id = resolved_token_id
            # Patch config so downstream code also sees the correct ID.
            hf_config.media_placeholder_token_id = resolved_token_id
        else:
            media_token_id = config_token_id

        self.media_token_id = media_token_id
        self.media_token = tokenizer.decode(media_token_id)

        self.image_processor = image_processor
        self.hf_processor = KimiK3Processor(
            tokenizer=tokenizer,
            image_processor=image_processor,
        )
        self.media_tokens_calculator = image_processor.media_tokens_calculator

    def get_hf_processor(self, **kwargs: object) -> KimiK3Processor:
        return self.hf_processor

    def get_hf_config(self) -> KimiK3Config:
        return self.ctx.get_hf_config(KimiK3Config)

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        # None means unlimited
        return {"image": None, "video": None}

    def get_data_parser(self) -> MultiModalDataParser:
        # Keep each video's metadata (source fps, decoded frame indices) next
        # to its frames: sampling and chunk timestamps depend on it.
        return MultiModalDataParser(
            video_needs_metadata=True,
            expected_hidden_size=self._get_expected_hidden_size(),
            allow_missing_mm_embeddings=self.allow_missing_mm_embeddings,
        )

    @property
    def media_proc_cfg(self) -> Mapping[str, Any]:
        """The checkpoint's `media_proc_cfg` (image and video settings)."""
        return self.image_processor.media_proc_cfg

    def dummy_video_shape(self) -> tuple[int, int, int]:
        """`(num_frames, width, height)` of the video with the most tokens.

        The token count of a video grows with its frame count until the
        per-video patch budget, not the per-frame one, binds; that happens at
        `in_patch_limit_video / in_patch_limit_each_frame` frames (rounded
        up to whole chunks), at the largest per-frame size.
        """
        cfg = self.media_proc_cfg
        per_frame = int(cfg.get("in_patch_limit_each_frame") or cfg["in_patch_limit"])
        frames_per_chunk = int(cfg.get("temporal_merge_kernel_size") or 1)
        total_limit = cfg.get("in_patch_limit_video")
        num_frames = (
            max(1, math.ceil(int(total_limit) / per_frame)) if total_limit else 1
        )
        num_frames = frames_per_chunk * math.ceil(num_frames / frames_per_chunk)
        if cfg.get("max_num_frames_each_video") is not None:
            num_frames = min(num_frames, int(cfg["max_num_frames_each_video"]))
        size = self.get_max_image_size(
            cfg["patch_size"],
            cfg["merge_kernel_size"],
            per_frame,
            cfg["patch_limit_on_one_side"],
            cfg["fixed_output_tokens"],
        )
        return num_frames, size.width, size.height

    def video_tokens(self, width: int, height: int, plan: KimiK3VideoPlan) -> int:
        """Encoder tokens of a planned video whose frames are `width x height`."""
        cfg = self.media_proc_cfg
        per_chunk = navit_resize_image(
            width,
            height,
            cfg["patch_size"],
            cfg["merge_kernel_size"],
            plan.in_patch_limit_each_frame,
            cfg["patch_limit_on_one_side"],
            cfg["fixed_output_tokens"],
        )["num_tokens"]
        return per_chunk * plan.num_chunks

    @functools.cached_property
    def max_video_tokens(self) -> int:
        """Tokens of the profiling video: the encoder cache is sized for it.

        Frames are padded to whole merge windows, so a real video can come out
        slightly larger than the profiling one (or, for very elongated
        frames, much larger); `get_video_chunks` shrinks such videos to fit.
        """
        num_frames, width, height = self.dummy_video_shape()
        fps = float(self.media_proc_cfg.get("sample_fps") or 1.0)
        plan = plan_kimi_k3_video(
            num_frames,
            {"fps": fps, "total_num_frames": num_frames},
            self.media_proc_cfg,
        )
        return self.video_tokens(width, height, plan)

    def get_video_chunks(
        self, item: Any
    ) -> tuple[KimiK3VideoPlan, list[dict[str, Any]]]:
        """Plan one decoded video and build its `video_chunk` media dicts."""
        frames, metadata = item if isinstance(item, tuple) else (item, None)
        plan = plan_kimi_k3_video(len(frames), metadata, self.media_proc_cfg)
        first = frames[0]
        if isinstance(first, PILImage.Image):
            width, height = first.size
        else:
            height, width = first.shape[:2]  # HWC array or tensor
        # Keep every video within the encoder cache the profile sized: lower
        # the per-frame patch limit until the padded token count fits.
        limit = self.max_video_tokens
        tokens = self.video_tokens(width, height, plan)
        while tokens > limit and plan.in_patch_limit_each_frame > 1:
            per_frame = max(
                1,
                min(
                    plan.in_patch_limit_each_frame - 1,
                    int(plan.in_patch_limit_each_frame * limit / tokens),
                ),
            )
            plan = dataclasses.replace(plan, in_patch_limit_each_frame=per_frame)
            tokens = self.video_tokens(width, height, plan)
        chunks = []
        for start in range(0, len(plan.frame_indices), plan.frames_per_chunk):
            indices = plan.frame_indices[start : start + plan.frames_per_chunk]
            chunks.append(
                {
                    "type": "video_chunk",
                    "video_chunk": [frames[i] for i in indices],
                    "in_patch_limit_each_frame": plan.in_patch_limit_each_frame,
                }
            )
        return plan, chunks

    @classmethod
    def get_max_image_size(
        cls,
        patch_size: int,
        merge_kernel_size: int,
        in_patch_limit: int,
        patch_limit_on_one_side: int,
        fixed_output_tokens: int | None,
    ) -> ImageSize:
        max_side = patch_limit_on_one_side * patch_size
        best_score = (-1, -1)
        best_size = (max_side, max_side)

        for width_patches in range(patch_limit_on_one_side + 1):
            width = min((width_patches + 1) * patch_size - 1, max_side)
            for height_patches in range(width_patches, patch_limit_on_one_side + 1):
                height = min((height_patches + 1) * patch_size - 1, max_side)
                resize_config = navit_resize_image(
                    width,
                    height,
                    patch_size,
                    merge_kernel_size,
                    in_patch_limit,
                    patch_limit_on_one_side,
                    fixed_output_tokens,
                )
                padded_width = resize_config["new_width"] + resize_config["pad_width"]
                padded_height = (
                    resize_config["new_height"] + resize_config["pad_height"]
                )
                num_patches = padded_width // patch_size * (padded_height // patch_size)
                score = (resize_config["num_tokens"], num_patches)
                if score > best_score:
                    best_score = score
                    best_size = (width, height)
        return ImageSize(width=best_size[0], height=best_size[1])


class KimiK3DummyInputsBuilder(BaseDummyInputsBuilder[KimiK3ProcessingInfo]):
    """Builds dummy inputs for K3 profiling.

    The dummy text is made of `<|kimi_image_placeholder|>` and
    `<|kimi_video_placeholder|>` strings — exactly the placeholders that K3's
    `_get_prompt_updates` expands — and the dummy mm data is a list of PIL
    images under `image` and `(frames, metadata)` pairs under `video`.
    """

    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        num_images = mm_counts.get("image", 0)
        num_videos = mm_counts.get("video", 0)
        return (
            self.info.get_hf_config().image_placeholder * num_images
            + KIMI_K3_VIDEO_PLACEHOLDER * num_videos
        )

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: MultiModalDummyOptions | None = None,
    ) -> MultiModalDataDict:
        media_proc_cfg = self.info.image_processor.media_proc_cfg
        max_size = self.info.get_max_image_size(
            media_proc_cfg["patch_size"],
            media_proc_cfg["merge_kernel_size"],
            media_proc_cfg["in_patch_limit"],
            media_proc_cfg["patch_limit_on_one_side"],
            media_proc_cfg["fixed_output_tokens"],
        )
        image_overrides = cast(
            ImageDummyOptions | None,
            mm_options.get("image") if mm_options else None,
        )
        dummy: dict[str, Any] = {
            "image": self._get_dummy_images(
                width=max_size.width,
                height=max_size.height,
                num_images=mm_counts.get("image", 0),
                overrides=image_overrides,
            )
        }
        num_videos = mm_counts.get("video", 0)
        if num_videos:
            dummy["video"] = self._get_dummy_kimi_videos(
                num_videos,
                cast(
                    VideoDummyOptions | None,
                    mm_options.get("video") if mm_options else None,
                ),
            )
        return dummy

    def _get_dummy_kimi_videos(
        self, num_videos: int, overrides: VideoDummyOptions | None
    ) -> list[tuple[np.ndarray, dict[str, Any]]]:
        """The video with the most encoder tokens the config allows, sampled
        at `sample_fps` so none of its frames are dropped."""
        cfg = self.info.media_proc_cfg
        num_frames, width, height = self.info.dummy_video_shape()
        videos = self._get_dummy_videos(
            width=width,
            height=height,
            num_frames=num_frames,
            num_videos=num_videos,
            overrides=overrides,
        )
        fps = float(cfg.get("sample_fps") or 1.0)
        return [
            (
                video,
                {
                    "fps": fps,
                    "duration": len(video) / fps,
                    "total_num_frames": len(video),
                    "frames_indices": list(range(len(video))),
                },
            )
            for video in videos
        ]


class KimiK3MultiModalProcessor(BaseMultiModalProcessor[KimiK3ProcessingInfo]):
    """Image and video multi-modal processor for Kimi-K3.

    A video is sampled at the checkpoint's `sample_fps` and split into
    chunks of `temporal_merge_kernel_size` frames. Each chunk is one
    MoonViT3d grid `[t, h, w]` that the tower pools over time into
    `h * w / merge**2` tokens, and is announced in the prompt as
    `{start}<|media_begin|>video<|media_content|>{pads}<|media_end|>`
    (the Kimi-K2.5 video format, which the K3 config's video keys mirror).
    """

    def _call_hf_processor(
        self,
        hf_data: Mapping[str, object],
        hf_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        hf_data = dict(hf_data)
        videos = cast(list[Any] | None, hf_data.pop("videos", None))
        if hf_data.get("images"):
            outputs = super()._call_hf_processor(hf_data, hf_kwargs)
        else:
            outputs = BatchFeature()
        if videos:
            outputs.update(self._process_videos(videos, hf_kwargs))
        return outputs

    def _process_videos(
        self,
        videos: list[Any],
        hf_kwargs: Mapping[str, object],
    ) -> dict[str, torch.Tensor]:
        image_processor = self.info.image_processor
        # Same kwarg resolution as the image path (`call_hf_processor`): with
        # mm_device_do_normalize the merged kwargs carry do_normalize=False and
        # raw uint8 pixels go to the device, which normalizes them in the
        # tower. Reading `hf_kwargs` alone would normalize here as well and the
        # tower would normalize twice.
        merged = self.info.ctx.get_merged_mm_kwargs(hf_kwargs)
        do_normalize = bool(merged.get("do_normalize", True))
        pixel_values, grid_thws, starts, num_chunks = [], [], [], []
        for video in videos:
            plan, chunks = self.info.get_video_chunks(video)
            out = image_processor.preprocess(
                chunks,
                return_tensors="pt",
                do_rescale=do_normalize,
                do_normalize=do_normalize,
            )
            pixel_values.append(out["pixel_values"])
            grid_thws.append(out["grid_thws"])
            starts.append(torch.tensor(plan.chunk_starts, dtype=torch.float64))
            num_chunks.append(plan.num_chunks)
        return {
            "video_pixel_values": torch.cat(pixel_values),
            "video_grid_thws": torch.cat(grid_thws),
            "video_chunk_starts": torch.cat(starts),
            "video_num_chunks": torch.tensor(num_chunks, dtype=torch.int64),
        }

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        """Slice the flattened patch tensors back into per-item data.

        `pixel_values` holds all patches from every image concatenated; each
        image's patch count is `prod(grid_thws[i])`. `grid_thws` is one
        `[N_t, N_h, N_w]` row per image. A video owns `video_num_chunks`
        consecutive rows of `video_grid_thws` / `video_chunk_starts` and
        the patches of all of those grids in `video_pixel_values`.
        """
        grid_thws = hf_inputs.get("grid_thws", torch.empty((0, 3)))
        grid_sizes = grid_thws.prod(-1)

        fields = dict(
            pixel_values=MultiModalFieldConfig.flat_from_sizes("image", grid_sizes),
            grid_thws=MultiModalFieldConfig.batched("image", keep_on_cpu=True),
        )

        num_chunks = hf_inputs.get("video_num_chunks")
        if num_chunks is not None:
            video_grids = hf_inputs["video_grid_thws"]
            chunk_patches = video_grids.prod(-1)
            bounds = torch.cumsum(num_chunks, 0)
            video_patches = torch.stack(
                [
                    chunk_patches[int(end - n) : int(end)].sum()
                    for n, end in zip(num_chunks, bounds)
                ]
            )
            fields.update(
                video_pixel_values=MultiModalFieldConfig.flat_from_sizes(
                    "video", video_patches
                ),
                video_grid_thws=MultiModalFieldConfig.flat_from_sizes(
                    "video", num_chunks, keep_on_cpu=True
                ),
                video_chunk_starts=MultiModalFieldConfig.flat_from_sizes(
                    "video", num_chunks, keep_on_cpu=True
                ),
                video_num_chunks=MultiModalFieldConfig.batched(
                    "video", keep_on_cpu=True
                ),
            )
        return fields

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, Any],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        """Expand each K3 image placeholder into a resolution-aware update.

        K3's prompt carries a single `<|kimi_image_placeholder|>` token per
        image. This replaces that token with
        `<|media_begin|>image {w}x{h}<|media_content|>{pads}<|media_end|>`,
        embedding the per-image resolution in the prompt and marking only the
        `<|media_pad|>` positions as embedding slots (the number of pads is
        the feature size returned by `media_tokens_calculator`).
        """
        media_token_id = self.info.media_token_id
        media_token = self.info.media_token
        image_placeholder = self.info.get_hf_config().image_placeholder
        tokenizer = self.info.get_tokenizer()

        def get_replacement(item_idx: int) -> PromptUpdateDetails:
            images = mm_items.get_items("image", ImageProcessorItems)
            image = images.get(item_idx)
            if image is None:
                raise ValueError(f"Missing image data at index {item_idx}")

            # The checkpoint image processor works on media dicts, so wrap the
            # PIL image before asking it for the token count.
            num_media_token = self.info.media_tokens_calculator(
                {"type": "image", "image": image}
            )
            pads = media_token * num_media_token

            # NOTE: `width`/`height` are the ORIGINAL upload dimensions, not the
            # post-preprocess (smart-resized) ones. `image` comes from the
            # untouched parsed `mm_items`; the checkpoint image processor
            # (`KimiK3VisionProcessor.preprocess`) only produces new tensors via
            # `image.resize(...)` and never mutates the stored PIL. This matches
            # the reference HF processor (`KimiK3Processor.preprocess_medias`),
            # which also builds the prompt from the original `img.size`. The
            # resize is reflected only in the pad count above.
            width, height = images.get_image_size(item_idx)
            full = (
                f"<|media_begin|>image {width}x{height}<|media_content|>"
                f"{pads}<|media_end|>"
            )

            return PromptUpdateDetails.select_token_id(
                cached_encode(tokenizer, full, add_special_tokens=False),
                media_token_id,
            )

        merge = int(self.info.media_proc_cfg["merge_kernel_size"])
        timestamp_mode = str(
            self.info.media_proc_cfg.get("timestamp_mode") or "hh:mm:ss.fff"
        )

        def get_video_replacement(item_idx: int) -> PromptUpdateDetails:
            out_item = out_mm_kwargs["video"][item_idx]
            grids = out_item["video_grid_thws"].data
            starts = out_item["video_chunk_starts"].data
            assert isinstance(grids, torch.Tensor)
            assert isinstance(starts, torch.Tensor)
            token_ids: list[int] = []
            for (_, grid_h, grid_w), start in zip(grids.tolist(), starts.tolist()):
                num_tokens = (int(grid_h) // merge) * (int(grid_w) // merge)
                prefix = (
                    format_kimi_k3_timestamp(start, timestamp_mode)
                    + "<|media_begin|>video<|media_content|>"
                )
                token_ids += cached_encode(tokenizer, prefix, add_special_tokens=False)
                token_ids += [media_token_id] * num_tokens
                token_ids += cached_encode(
                    tokenizer, "<|media_end|>", add_special_tokens=False
                )
            return PromptUpdateDetails.select_token_id(token_ids, media_token_id)

        return [
            PromptReplacement(
                modality="image",
                target=cached_encode(
                    tokenizer, image_placeholder, add_special_tokens=False
                ),
                replacement=get_replacement,
            ),
            PromptReplacement(
                modality="video",
                target=cached_encode(
                    tokenizer, KIMI_K3_VIDEO_PLACEHOLDER, add_special_tokens=False
                ),
                replacement=get_video_replacement,
            ),
        ]
