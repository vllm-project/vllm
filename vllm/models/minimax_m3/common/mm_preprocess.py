# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
from collections.abc import Mapping, Sequence
from typing import Any, cast

import numpy.typing as npt
import torch
from transformers import BatchFeature
from transformers.video_utils import VideoMetadata

from vllm.config.multimodal import (
    MultiModalDummyOptions,
)
from vllm.inputs import MultiModalDataDict
from vllm.multimodal.inputs import (
    MultiModalFieldConfig,
    MultiModalKwargsItems,
)
from vllm.multimodal.parse import (
    ImageSize,
    MultiModalDataItems,
)
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder,
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    PromptReplacement,
    PromptUpdate,
    PromptUpdateDetails,
)
from vllm.multimodal.processing.processor import HFMultiModalInputs
from vllm.multimodal.video import (
    VIDEO_LOADER_REGISTRY,
    VideoBackend,
    VideoDecoderBackend,
    VideoSourceMetadata,
    VideoTargetMetadata,
)
from vllm.transformers_utils.configs.minimax_m3 import MiniMaxM3Config
from vllm.transformers_utils.processors.minimax_m3 import (
    MiniMaxM3VLImageProcessor,
    MiniMaxM3VLVideoProcessor,
    MiniMaxVLProcessor,
    compute_aligned_size,
    resolve_max_long_side_pixel,
)

# Worst-case long-side caps in pixels, matching the processor's detail=high
# tiers. Profiling budgets are sized against these; request-level
# ``detail``/``max_long_side_pixel`` kwargs cannot exceed them.
_IMAGE_MAX_LONG_SIDE_PIXEL = 3584
_VIDEO_MAX_LONG_SIDE_PIXEL = 2188

# Upper bound on the number of frames used to build the dummy video during
# memory profiling. Sized to the worst-case video the processor accepts:
# ``max_total_pixels // max_pixels_per_frame`` = 301,056,000 // (2184 * 2184)
# = 63 frames, each a square at the video long-side cap (2188 px floored to
# the patch*merge factor, 28). Without a cap,
# ``_get_max_video_frames(seq_len)`` with M3's large ``max_model_len`` yields
# ~1400 frames, producing a multi-GB dummy tensor that overflows the
# multimodal encoder cache.
_MAX_FRAMES_PER_VIDEO = 63


class MiniMaxM3VLProcessingInfo(BaseProcessingInfo):
    IMAGE_TOKEN = "]<]image[>["
    VIDEO_TOKEN = "]<]video[>["
    VISION_START_TOKEN = "]<]start of image[>["
    VISION_END_TOKEN = "]<]end of image[>["

    def get_hf_config(self) -> MiniMaxM3Config:
        return self.ctx.get_hf_config(MiniMaxM3Config)

    def get_hf_processor(self, **kwargs: object) -> MiniMaxVLProcessor:
        # The released checkpoint only ships the processor as remote code
        # (via ``auto_map``). Construct the vendored processor directly so the
        # model loads without ``--trust-remote-code``.
        return self.ctx.get_hf_processor(MiniMaxVLProcessor, **kwargs)

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"image": None, "video": None}

    def get_mm_max_tokens_per_item(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> Mapping[str, int]:
        return {
            "image": self.get_max_image_tokens(),
            "video": self.get_max_video_tokens(seq_len, mm_counts),
        }

    def get_image_processor(self, **kwargs: object) -> MiniMaxM3VLImageProcessor:
        return self.get_hf_processor(**kwargs).image_processor

    def get_video_processor(self, **kwargs: object) -> MiniMaxM3VLVideoProcessor:
        return self.get_hf_processor(**kwargs).video_processor

    def _get_vision_info(
        self,
        *,
        image_width: int,
        image_height: int,
        num_frames: int,
        image_processor,
        mm_kwargs: Mapping[str, object],
        modality: str,
    ) -> tuple[ImageSize, int]:
        """Compute resized image size and number of vision tokens.

        Mirrors the processor's ``compute_aligned_size`` so token counts
        match the actual processor output.
        """
        patch_size: int = image_processor.patch_size
        merge_size: int = image_processor.merge_size
        temporal_patch_size: int = image_processor.temporal_patch_size
        factor = patch_size * merge_size

        # ``image_processor`` is the *video* processor when counting video
        # tokens, so modality-scoped request kwargs (flat keys overlaid with
        # ``images_kwargs``/``videos_kwargs`` via ``modality=``) and processor
        # defaults are read per modality. Per-key fallback mirrors the
        # processor, whose own defaults feed ``resolve_max_long_side_pixel``
        # the same way.
        merged = self.ctx.get_merged_mm_kwargs(mm_kwargs)
        # Kwargs scoped to this modality (``images_kwargs`` / ``videos_kwargs``)
        # override the flat ones.
        scoped = merged.get("images_kwargs" if modality == "image" else "videos_kwargs")
        if isinstance(scoped, Mapping):
            merged = {**merged, **scoped}
        detail = merged.get("detail", getattr(image_processor, "detail", None))
        max_long_side_pixel = merged.get(
            "max_long_side_pixel", getattr(image_processor, "max_long_side_pixel", None)
        )
        resolved_long_side = resolve_max_long_side_pixel(
            detail, max_long_side_pixel, modality=modality
        )

        # No volumetric cap here: token counting must not raise; the video
        # processor enforces its volume cap on the real inputs.
        new_h, new_w = compute_aligned_size(
            image_height,
            image_width,
            factor,
            max_long_side_pixel=resolved_long_side,
        )
        grid_h = new_h // patch_size
        grid_w = new_w // patch_size

        # Pad frames to be divisible by temporal_patch_size
        padded_frames = num_frames + (-num_frames % temporal_patch_size)
        grid_t = max(padded_frames // temporal_patch_size, 1)

        num_tokens = grid_t * grid_h * grid_w // (merge_size**2)
        return ImageSize(width=new_w, height=new_h), num_tokens

    def get_num_image_tokens(
        self,
        *,
        image_width: int,
        image_height: int,
        image_processor,
        mm_kwargs: Mapping[str, object],
    ) -> int:
        _, n = self._get_vision_info(
            image_width=image_width,
            image_height=image_height,
            num_frames=1,
            image_processor=image_processor,
            mm_kwargs=mm_kwargs,
            modality="image",
        )
        return n

    def get_num_video_tokens(
        self,
        *,
        image_width: int,
        image_height: int,
        num_frames: int,
        image_processor,
        mm_kwargs: Mapping[str, object],
    ) -> int:
        _, n = self._get_vision_info(
            image_width=image_width,
            image_height=image_height,
            num_frames=num_frames,
            image_processor=image_processor,
            mm_kwargs=mm_kwargs,
            modality="video",
        )
        return n

    def get_image_size_with_most_features(self) -> ImageSize:
        # Worst case is a detail=high image: a square whose side is the max
        # long side (3584 px, a multiple of patch_size*merge_size).
        image_processor = self.get_image_processor()
        factor = image_processor.patch_size * image_processor.merge_size
        side = (_IMAGE_MAX_LONG_SIDE_PIXEL // factor) * factor
        return ImageSize(width=side, height=side)

    def get_video_size_with_most_features(self) -> ImageSize:
        # Worst single frame is a detail=high square capped by the video
        # long-side max (2188 px), floored to a patch_size*merge_size
        # multiple (2184).
        video_processor = self.get_video_processor()
        factor = video_processor.patch_size * video_processor.merge_size
        side = (_VIDEO_MAX_LONG_SIDE_PIXEL // factor) * factor
        return ImageSize(width=side, height=side)

    def get_max_image_tokens(self) -> int:
        image_processor = self.get_image_processor()
        size = self.get_image_size_with_most_features()
        return self.get_num_image_tokens(
            image_width=size.width,
            image_height=size.height,
            image_processor=image_processor,
            # Budgets must cover the worst case (detail=high); the default
            # detail=auto cap would undercount.
            mm_kwargs={"max_long_side_pixel": _IMAGE_MAX_LONG_SIDE_PIXEL},
        )

    def _get_max_video_frames(self, max_tokens: int) -> int:
        video_processor = self.get_video_processor()
        size = self.get_video_size_with_most_features()
        num_frames = 1
        while True:
            next_n = self.get_num_video_tokens(
                image_width=size.width,
                image_height=size.height,
                num_frames=num_frames + 1,
                image_processor=video_processor,
                # Worst case (detail=high); see get_max_image_tokens.
                mm_kwargs={"max_long_side_pixel": _VIDEO_MAX_LONG_SIDE_PIXEL},
            )
            if next_n > max_tokens:
                break
            num_frames += 1
        return num_frames

    def get_num_frames_with_most_features(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        max_frames_per_video: int = _MAX_FRAMES_PER_VIDEO,
    ) -> int:
        max_videos = mm_counts.get("video", 0)
        max_total_frames = self._get_max_video_frames(seq_len)
        max_frames_per_video = min(
            max_total_frames // max(max_videos, 1), max_frames_per_video
        )
        return max(max_frames_per_video, 1)

    def get_max_video_tokens(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> int:
        video_processor = self.get_video_processor()
        size = self.get_video_size_with_most_features()
        return self.get_num_video_tokens(
            image_width=size.width,
            image_height=size.height,
            num_frames=self.get_num_frames_with_most_features(seq_len, mm_counts),
            image_processor=video_processor,
            # Worst case (detail=high); see get_max_image_tokens.
            mm_kwargs={"max_long_side_pixel": _VIDEO_MAX_LONG_SIDE_PIXEL},
        )


class MiniMaxM3VLDummyInputsBuilder(BaseDummyInputsBuilder[MiniMaxM3VLProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        num_images = mm_counts.get("image", 0)
        num_videos = mm_counts.get("video", 0)
        image_token: str = self.info.IMAGE_TOKEN
        video_token: str = self.info.VIDEO_TOKEN
        return image_token * num_images + video_token * num_videos

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: MultiModalDummyOptions,
    ) -> MultiModalDataDict:
        size = self.info.get_image_size_with_most_features()
        video_size = self.info.get_video_size_with_most_features()
        num_frames = self.info.get_num_frames_with_most_features(seq_len, mm_counts)
        return {
            "image": self._get_dummy_images(
                width=size.width,
                height=size.height,
                num_images=mm_counts.get("image", 0),
                overrides=mm_options.get("image"),
            ),
            "video": self._get_dummy_videos(
                width=video_size.width,
                height=video_size.height,
                num_frames=num_frames,
                num_videos=mm_counts.get("video", 0),
                overrides=mm_options.get("video"),
            ),
        }


class MiniMaxM3VLMultiModalProcessor(
    BaseMultiModalProcessor[MiniMaxM3VLProcessingInfo]
):
    def _get_hf_mm_text(self, mm_counts: Mapping[str, int]) -> str:
        return self.dummy_inputs.get_dummy_text(mm_counts)

    def _get_hf_mm_inputs(
        self,
        mm_items: MultiModalDataItems,
        hf_kwargs: Mapping[str, object],
    ) -> HFMultiModalInputs:
        hf_inputs = super()._get_hf_mm_inputs(mm_items, hf_kwargs)
        hf_data = hf_inputs.hf_data

        # With ``video_needs_metadata=True`` each video arrives as a
        # ``(frames, metadata)`` tuple. Split the frames back out and forward the
        # metadata as ``VideoMetadata`` so the processor emits timestamps.
        videos = cast(list | None, hf_data.get("videos"))
        video_metadata: list[VideoMetadata] | None = None
        if videos:
            frames_only = []
            video_metadata = []
            for item in videos:
                if isinstance(item, tuple) and len(item) == 2:
                    frames, meta = item
                else:
                    frames, meta = item, {}
                frames_only.append(frames)
                meta = {
                    k: v for k, v in (meta or {}).items() if k != "do_sample_frames"
                }
                # VideoMetadata requires total_num_frames; derive it for
                # dummy/profiling videos whose metadata omits it. fps and
                # frames_indices default to None there → no timestamps, which
                # stays consistent with _get_prompt_updates.
                meta.setdefault("total_num_frames", len(frames))
                video_metadata.append(VideoMetadata(**meta))
            hf_data["videos"] = frames_only

        if video_metadata is not None:
            hf_data["video_metadata"] = video_metadata

        return hf_inputs

    def _call_hf_processor(
        self,
        hf_data: Mapping[str, object],
        hf_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        # Override the video processor's default for vLLM's raw-frame inputs.
        merged = dict(do_resize=True, **hf_kwargs)
        return self.info.ctx.call_hf_processor(
            self.info.get_hf_processor(**hf_kwargs),
            hf_data,
            merged,
        )

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        image_grid_thw = hf_inputs.get("image_grid_thw")
        video_grid_thw = hf_inputs.get("video_grid_thw")

        # Total patches per item (grid_t * grid_h * grid_w)
        image_grid_sizes = (
            image_grid_thw.prod(-1)
            if image_grid_thw is not None
            else torch.empty(0, dtype=torch.long)
        )
        video_grid_sizes = (
            video_grid_thw.prod(-1)
            if video_grid_thw is not None
            else torch.empty(0, dtype=torch.long)
        )

        return {
            "pixel_values": MultiModalFieldConfig.flat_from_sizes(
                "image", image_grid_sizes
            ),
            "image_grid_thw": MultiModalFieldConfig.batched("image", keep_on_cpu=True),
            "pixel_values_videos": MultiModalFieldConfig.flat_from_sizes(
                "video", video_grid_sizes
            ),
            "video_grid_thw": MultiModalFieldConfig.batched("video", keep_on_cpu=True),
        }

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        hf_processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)
        tokenizer = self.info.get_tokenizer()
        vocab = tokenizer.get_vocab()

        image_token_id: int = vocab[self.info.IMAGE_TOKEN]
        video_token_id: int = vocab[self.info.VIDEO_TOKEN]
        start_token_id: int = vocab[self.info.VISION_START_TOKEN]
        end_token_id: int = vocab[self.info.VISION_END_TOKEN]
        merge_length: int = hf_processor.image_processor.merge_size**2

        def get_image_replacement(item_idx: int):
            grid_thw: torch.Tensor = out_mm_kwargs["image"][item_idx][
                "image_grid_thw"
            ].data
            # grid_thw shape: (3,) = [1, grid_h, grid_w]
            N = int(grid_thw.prod().item()) // merge_length
            full = [start_token_id] + [image_token_id] * N + [end_token_id]
            return PromptUpdateDetails.select_token_id(full, image_token_id)

        # Per-video metadata (fps + sampled frame indices) is carried on the
        # parsed video items; used to reproduce the HF processor's timestamps.
        video_items = mm_items.get("video")
        video_metadata = getattr(video_items, "metadata", None)
        temporal_patch_size: int = hf_processor.video_processor.temporal_patch_size

        def get_video_replacement(item_idx: int):
            grid_thw: torch.Tensor = out_mm_kwargs["video"][item_idx][
                "video_grid_thw"
            ].data
            # grid_thw shape: (3,) = [grid_t, grid_h, grid_w]
            # HF model uses VIDEO_TOKEN (not IMAGE_TOKEN) for video frame content:
            # processing_minimax.py L245: replace(placeholder, self.VIDEO_TOKEN)
            T = int(grid_thw[0].item())
            M = int(grid_thw[1].item() * grid_thw[2].item()) // merge_length

            # Reproduce the HF processor's per-frame timestamp markers
            # (processing_minimax.py: ts = frames_indices[frame*tps] / fps,
            # rendered as "]<]X.X seconds[>["). Falls back to no timestamps when
            # metadata is unavailable (keeping the replacement aligned with the
            # processor output in both cases).
            meta = (
                video_metadata[item_idx]
                if video_metadata is not None and item_idx < len(video_metadata)
                else None
            )
            fps = meta.get("fps") if meta else None
            frames_indices = meta.get("frames_indices") if meta else None

            full: list[int] = []
            for frame_idx in range(T):
                if fps is not None and frames_indices is not None:
                    idx = min(frame_idx * temporal_patch_size, len(frames_indices) - 1)
                    ts = frames_indices[idx] / fps
                    full += tokenizer.encode(
                        f"]<]{ts:.1f} seconds[>[", add_special_tokens=False
                    )
                full += [start_token_id] + [video_token_id] * M + [end_token_id]
            return PromptUpdateDetails.select_token_id(full, video_token_id)

        return [
            PromptReplacement(
                modality="image",
                target=[image_token_id],
                replacement=get_image_replacement,
            ),
            PromptReplacement(
                modality="video",
                target=[video_token_id],
                replacement=get_video_replacement,
            ),
        ]


@VIDEO_LOADER_REGISTRY.register(
    name="minimax_m3_vl",
    video_processor="MiniMaxM3VLVideoProcessor",
)
class MiniMaxM3VideoBackend(VideoBackend):
    @classmethod
    def load_bytes(
        cls,
        data: bytes,
        num_frames: int = -1,
        fps: int = 1,
        max_duration: int = 300,
        frame_recovery: bool = False,
        *,
        backend: VideoDecoderBackend = "opencv",
        **kwargs,
    ) -> tuple[npt.NDArray, dict[str, Any]]:
        # fps=-1 (unspecified) falls back to the default 1 fps.
        if fps == -1:
            fps = 1
        if not 0.2 <= fps <= 5:
            raise ValueError(f"video fps must be in [0.2, 5], got {fps}")
        return super().load_bytes(
            data,
            num_frames=num_frames,
            fps=fps,
            max_duration=max_duration,
            frame_recovery=frame_recovery,
            backend=backend,
            **kwargs,
        )

    @classmethod
    def compute_frames_index_to_sample(
        cls,
        source: VideoSourceMetadata,
        target: VideoTargetMetadata,
        **kwargs,
    ) -> list[int]:
        total_frames = source.total_frames_num
        video_fps = source.original_fps
        fps = target.fps
        if total_frames <= 0 or video_fps <= 0 or fps <= 0:
            return [0] if total_frames > 0 else []

        read_time_interval = 1.0 / fps
        eps = 1e-4

        indices: list[int] = []
        prev_kept_ts = -float("inf")
        while True:
            if not indices:
                target_frame = 0
            else:
                target_ts = prev_kept_ts + read_time_interval - eps
                target_frame = math.ceil(target_ts * video_fps)
                target_frame = max(target_frame, indices[-1] + 1)
            if target_frame >= total_frames:
                break
            indices.append(target_frame)
            prev_kept_ts = target_frame / video_fps

        if not indices:
            indices = [0]
        return indices
