# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
from collections.abc import Mapping, Sequence
from typing import cast

import torch
from transformers import BatchFeature
from transformers.video_utils import VideoMetadata

from vllm.config.multimodal import (
    BaseDummyOptions,
    ImageDummyOptions,
    VideoDummyOptions,
)
from vllm.inputs import MultiModalDataDict
from vllm.multimodal.inputs import (
    MultiModalFieldConfig,
    MultiModalKwargsItems,
)
from vllm.multimodal.parse import (
    ImageSize,
    MultiModalDataItems,
    MultiModalDataParser,
)
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder,
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    PromptReplacement,
    PromptUpdate,
    PromptUpdateDetails,
)
from vllm.multimodal.video import (
    VIDEO_LOADER_REGISTRY,
    VideoBackend,
    VideoSourceMetadata,
    VideoTargetMetadata,
)
from vllm.transformers_utils.configs.minimax_m3 import MiniMaxM3Config
from vllm.transformers_utils.processors.minimax_m3 import (
    MIN_SHORT_SIDE_PIXEL,
    MiniMaxM3VLImageProcessor,
    MiniMaxM3VLVideoProcessor,
    MiniMaxVLProcessor,
    smart_resize,
)

# Upper bound on the number of frames used to build the dummy video during
# memory profiling. Sized to the worst-case video the processor accepts:
# ``max_total_pixels // max_pixels_per_frame`` = 301,056,000 // 602,112 = 500
# frames, each at the video processor's per-frame ``max_pixels`` (768 * 28 * 28
# = 602,112). This reaches the true worst-case ~192,000 vision tokens, but only
# because the dummy video is sized via ``get_video_size_with_most_features()``
# (the video ``max_pixels`` bound), not the smaller image bound. Without a cap,
# ``_get_max_video_frames(seq_len)`` with M3's large ``max_model_len`` yields
# ~1400 frames, producing a multi-GB dummy tensor that overflows the
# multimodal encoder cache.
_MAX_FRAMES_PER_VIDEO = 500


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
    ) -> tuple[ImageSize, int]:
        """Compute resized image size and number of vision tokens.

        Mirrors the processor's Qwen-style ``smart_resize`` (area bound by
        ``max_pixels``) so token counts match the actual processor output.
        """
        patch_size: int = image_processor.patch_size
        merge_size: int = image_processor.merge_size
        temporal_patch_size: int = image_processor.temporal_patch_size
        factor = patch_size * merge_size
        max_pixels: int = image_processor.max_pixels
        # Long-side resize spec (opt-in). ``image_processor`` is the *video*
        # processor when counting video tokens, so read the bounds off it.
        max_long_side_pixel = getattr(image_processor, "max_long_side_pixel", None)
        min_short_side_pixel = getattr(
            image_processor, "min_short_side_pixel", MIN_SHORT_SIDE_PIXEL
        )

        new_h, new_w = smart_resize(
            image_height,
            image_width,
            factor=factor,
            max_pixels=max_pixels,
            max_long_side_pixel=max_long_side_pixel,
            min_short_side_pixel=min_short_side_pixel,
            # Token counting must not raise; the volumetric/area cap is enforced
            # in the processor's _preprocess on the real inputs.
            max_total_pixels=None,
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
        )
        return n

    def get_image_size_with_most_features(self) -> ImageSize:
        # Largest square (a multiple of patch_size*merge_size) whose area is
        # within the image processor's bound — this yields the most vision
        # tokens for one image. With the long-side spec the square side is
        # capped by ``max_long_side_pixel`` (and the fixed ``max_total_pixels``);
        # otherwise it is bound by the ``max_pixels`` area.
        image_processor = self.get_image_processor()
        factor = image_processor.patch_size * image_processor.merge_size
        max_long_side_pixel = getattr(image_processor, "max_long_side_pixel", None)
        if max_long_side_pixel is not None:
            side_px = min(
                max_long_side_pixel,
                math.isqrt(image_processor.max_total_pixels),
            )
        else:
            side_px = math.isqrt(image_processor.max_pixels)
        side = max(factor, (side_px // factor) * factor)
        return ImageSize(width=side, height=side)

    def get_video_size_with_most_features(self) -> ImageSize:
        # Per-frame size that yields the most vision tokens, bound by the
        # *video* processor's ``max_pixels`` (which differs from the image
        # bound). Token count depends only on area, so maximize the area
        # achievable with both sides a multiple of patch_size*merge_size rather
        # than picking the largest square — a square (e.g. 756x756 for M3's
        # 602,112 bound) leaves area on the table, undercounting frames.
        video_processor = self.get_video_processor()
        factor = video_processor.patch_size * video_processor.merge_size
        per_frame_pixels = video_processor.max_pixels
        max_long_side_pixel = getattr(video_processor, "max_long_side_pixel", None)
        if max_long_side_pixel is not None:
            # Long-side spec: a frame's worst case is a square capped by
            # ``max_long_side_pixel`` (per-frame area, not the volumetric cap).
            per_frame_pixels = min(per_frame_pixels, max_long_side_pixel**2)
        units = per_frame_pixels // (factor * factor)  # h_u * w_u
        h_u = math.isqrt(units)
        while units % h_u:
            h_u -= 1
        return ImageSize(width=(units // h_u) * factor, height=h_u * factor)

    def get_max_image_tokens(self) -> int:
        image_processor = self.get_image_processor()
        size = self.get_image_size_with_most_features()
        return self.get_num_image_tokens(
            image_width=size.width,
            image_height=size.height,
            image_processor=image_processor,
            mm_kwargs={},
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
                mm_kwargs={},
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
            mm_kwargs={},
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
        mm_options: Mapping[str, BaseDummyOptions],
    ) -> MultiModalDataDict:
        size = self.info.get_image_size_with_most_features()
        video_size = self.info.get_video_size_with_most_features()
        num_frames = self.info.get_num_frames_with_most_features(seq_len, mm_counts)
        return {
            "image": self._get_dummy_images(
                width=size.width,
                height=size.height,
                num_images=mm_counts.get("image", 0),
                overrides=cast(ImageDummyOptions | None, mm_options.get("image")),
            ),
            "video": self._get_dummy_videos(
                width=video_size.width,
                height=video_size.height,
                num_frames=num_frames,
                num_videos=mm_counts.get("video", 0),
                overrides=cast(VideoDummyOptions | None, mm_options.get("video")),
            ),
        }


class MiniMaxM3VLMultiModalProcessor(
    BaseMultiModalProcessor[MiniMaxM3VLProcessingInfo]
):
    def _get_data_parser(self) -> MultiModalDataParser:
        # Request video metadata (fps + sampled frame indices) so the HF
        # processor can emit per-frame ``]<]X.X seconds[>[`` timestamp markers,
        # matching MiniMax's reference video token stream. ``_get_prompt_updates``
        # reconstructs the same markers from the metadata to keep the prompt
        # replacement aligned with the processor output.
        return MultiModalDataParser(video_needs_metadata=True)

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        mm_data = dict(mm_data)
        # With ``video_needs_metadata=True`` each video arrives as a
        # ``(frames, metadata)`` tuple. Split the frames back out and forward the
        # metadata as ``VideoMetadata`` so the processor emits timestamps.
        videos = cast(list | None, mm_data.get("videos"))
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
            mm_data["videos"] = frames_only

        # Override the video processor's default do_resize=False (set for a
        # pre-resized pipeline) to True for vLLM's raw-frame inputs.
        merged = dict(do_resize=True, **mm_kwargs, **tok_kwargs)
        data = dict(text=prompt, **mm_data)
        if video_metadata is not None:
            data["video_metadata"] = video_metadata
        return self.info.ctx.call_hf_processor(
            self.info.get_hf_processor(**mm_kwargs),
            data,
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


# TODO(Isotr0py): Tie with MinimaxVideoProcessor
# after https://github.com/vllm-project/vllm/pull/44126
@VIDEO_LOADER_REGISTRY.register("minimax_m3_vl")
class MiniMaxM3VideoBackend(VideoBackend):
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

        last_frame_idx = total_frames - 1
        last_ts = last_frame_idx / video_fps
        if indices and indices[-1] != last_frame_idx and last_ts - prev_kept_ts > eps:
            indices.append(last_frame_idx)

        if not indices:
            indices = [0]
        return indices
