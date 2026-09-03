# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""vLLM-native multimodal processor for GLM-5.3-Flash."""

import math

import numpy as np
import torch
from torchvision.transforms.v2 import functional as tvF
from transformers.image_processing_utils import BatchFeature
from transformers.image_processing_utils_fast import (
    BaseImageProcessorFast,
    group_images_by_shape,
    reorder_images,
)
from transformers.image_utils import (
    OPENAI_CLIP_MEAN,
    OPENAI_CLIP_STD,
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    SizeDict,
    get_image_size,
)
from transformers.models.auto.image_processing_auto import get_image_processor_config
from transformers.processing_utils import (
    ImagesKwargs,
    MultiModalData,
    ProcessingKwargs,
    ProcessorMixin,
    Unpack,
    VideosKwargs,
)
from transformers.tokenization_utils_base import PreTokenizedInput, TextInput
from transformers.utils import TensorType, logging
from transformers.video_processing_utils import BaseVideoProcessor
from transformers.video_utils import (
    VideoInput,
    VideoMetadata,
    group_videos_by_shape,
    reorder_videos,
)

from vllm.transformers_utils.repo_utils import get_hf_file_to_dict

logger = logging.get_logger(__name__)

# Cap video inputs at 30,000 vision tokens to keep encoder profiling from
# starving the KV cache. Image inputs retain their checkpoint-defined budget.
_MAX_VIDEO_TOKENS = 30000

# Frame-sampler fallbacks (mirror the checkpoint's fps_interval=2 /
# max_frame_count_dynamic=2048); used when neither the request nor the
# processor config overrides them.
GLM_VIDEO_DEFAULT_FPS = 2.0
GLM_VIDEO_DEFAULT_MAX_FRAMES = 2048


def glm_sample_frame_indices(
    total_frames: int,
    fps: float,
    duration: float,
    *,
    target_fps: float | None = None,
    max_frame_count: int | None = None,
    temporal_patch_size: int = 2,
) -> list[int]:
    """GLM video frame sampling (training-reference parity).

    ``target_fps`` is the ``fps_interval`` request knob. The greedy walk
    advances at ``1 / (temporal_patch_size * target_fps)`` seconds, so on
    frame-dense sources it collects more candidates than ``extract_t`` and
    the ``> extract_t`` fixup re-spreads the picks uniformly with
    ``np.linspace`` -- that fallback is the intended reference behavior, not
    an accident. Short clips (fewer frames than ``extract_t``) are spread at
    evenly spaced timestamps (``floor`` sampling; the linspace variant
    samples frames unevenly and cost 4 points on video grounding evals).
    Request overrides: ``target_fps`` -> fps interval, ``max_frame_count``
    -> frame cap.
    """
    max_frame_idx = total_frames - 1
    if not duration:
        duration = (round(max_frame_idx / fps) + 1) if fps else 0
    if max_frame_count is None:
        max_frame_count = GLM_VIDEO_DEFAULT_MAX_FRAMES
    if target_fps is None:
        target_fps = GLM_VIDEO_DEFAULT_FPS

    extract_t = int(duration * target_fps)
    extract_t = min(extract_t, int(max_frame_count))

    duration_per_frame = 1 / fps
    timestamps = [i * duration_per_frame for i in range(total_frames)]
    max_second = int(duration)

    if total_frames < extract_t:
        frame_indices = [
            math.floor(_i * total_frames / extract_t) for _i in range(extract_t)
        ]
    else:
        frame_indices = []
        current_second = 0.0
        inv_fps = 1 / (temporal_patch_size * target_fps)
        for frame_index in range(total_frames):
            if timestamps[frame_index] >= current_second:
                current_second += inv_fps
                frame_indices.append(frame_index)
                if current_second >= max_second:
                    break

    if len(frame_indices) < extract_t:
        if len(frame_indices) == 0:
            start, end = 0, max(total_frames - 1, 0)
        else:
            start, end = frame_indices[0], frame_indices[-1]
        frame_indices = np.linspace(start, end, extract_t, dtype=int).tolist()
    elif len(frame_indices) > extract_t:
        frame_indices = np.linspace(0, total_frames - 1, extract_t, dtype=int).tolist()

    seen, uniq = set(), []
    for idx in frame_indices:
        if idx not in seen:
            seen.add(idx)
            uniq.append(int(idx))

    if len(uniq) & 1:
        uniq.append(uniq[-1])

    return uniq


def _ceil_to_factor(value: int, factor: int) -> int:
    """Round a positive integer upward to the nearest multiple of factor."""
    return math.ceil(value / factor) * factor


def _fit_aligned_size_within_budget(
    t: int,
    h: int,
    w: int,
    h_factor: int,
    w_factor: int,
    max_pixels: int,
) -> tuple[int, int]:
    """Largest proportional size whose upward-aligned canvas fits the budget.

    Binary search on the unaligned content height; each candidate is rounded
    upward to h_factor/w_factor, so the returned canvas always satisfies
    ``t * aligned_h * aligned_w <= max_pixels``.
    """
    minimum_pixels = t * h_factor * w_factor
    if max_pixels < minimum_pixels:
        raise ValueError(
            f"max_pixels={max_pixels} is too small. At least "
            f"{minimum_pixels} pixels are required for one aligned patch."
        )

    low, high = 1, h
    best_h, best_w = h_factor, w_factor
    while low <= high:
        content_h = (low + high) // 2
        content_w = max(1, math.floor(w * content_h / h))
        aligned_h = _ceil_to_factor(content_h, h_factor)
        aligned_w = _ceil_to_factor(content_w, w_factor)
        if t * aligned_h * aligned_w <= max_pixels:
            best_h, best_w = aligned_h, aligned_w
            low = content_h + 1
        else:
            high = content_h - 1
    return best_h, best_w


def smart_resize(
    t: int,
    h: int,
    w: int,
    t_factor: int = 1,
    h_factor: int = 28,
    w_factor: int = 28,
    min_pixels: int = 56 * 56,
    max_pixels: int = 14 * 14 * 4 * 1280,
) -> tuple[int, int]:
    """GLM-5.3-Flash ``smart_resize``: upward-aligned canvas under a
    ``t_bar * h_bar * w_bar`` pixel budget.

    Height/width always round UP to their factors (content is then padded,
    never cropped or distorted); an over-budget canvas is refit by binary
    search instead of one-shot square-root scaling. ``h_factor`` /
    ``w_factor`` carry ``patch_expand_factor`` on top of
    ``patch_size * merge_size``; ``t_factor`` is ``temporal_patch_size``. For
    a still image ``t = t_factor = temporal_patch_size`` so ``t_bar =
    temporal_patch_size``.
    """
    if min(t, h, w, t_factor, h_factor, w_factor) <= 0:
        raise ValueError("Image dimensions and alignment factors must be positive.")
    if min_pixels <= 0 or max_pixels <= 0:
        raise ValueError("min_pixels and max_pixels must be positive.")
    if min_pixels > max_pixels:
        raise ValueError("min_pixels must be less than or equal to max_pixels.")

    t_bar = max(t_factor, round(t / t_factor) * t_factor)
    h_bar = _ceil_to_factor(h, h_factor)
    w_bar = _ceil_to_factor(w, w_factor)

    if t_bar * h_bar * w_bar > max_pixels:
        h_bar, w_bar = _fit_aligned_size_within_budget(
            t=t_bar,
            h=h,
            w=w,
            h_factor=h_factor,
            w_factor=w_factor,
            max_pixels=max_pixels,
        )
    elif t_bar * h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (t * h * w))
        h_bar = _ceil_to_factor(max(1, math.ceil(h * beta)), h_factor)
        w_bar = _ceil_to_factor(max(1, math.ceil(w * beta)), w_factor)

        # Alignment can push a candidate slightly over a tight max_pixels
        # budget. Refit it when that happens.
        if t_bar * h_bar * w_bar > max_pixels:
            h_bar, w_bar = _fit_aligned_size_within_budget(
                t=t_bar,
                h=h,
                w=w,
                h_factor=h_factor,
                w_factor=w_factor,
                max_pixels=max_pixels,
            )

    return h_bar, w_bar


def _get_pad_content_size(
    image_height: int,
    image_width: int,
    canvas_height: int,
    canvas_width: int,
    allow_upscale: bool = False,
) -> tuple[int, int]:
    """Aspect-ratio-preserving content size that fits the canvas.

    Oversized images are shrunk proportionally. Small images are enlarged
    only when ``allow_upscale``. Padding is applied after the resize.
    """
    scale = min(canvas_height / image_height, canvas_width / image_width)
    if not allow_upscale:
        scale = min(1.0, scale)
    content_height = max(1, min(canvas_height, math.floor(image_height * scale)))
    content_width = max(1, min(canvas_width, math.floor(image_width * scale)))
    return content_height, content_width


def _resize_or_pad(
    stacked_images: torch.Tensor,
    target_height: int,
    target_width: int,
    resize_mode: str,
    resample: "PILImageResampling | tvF.InterpolationMode | int | None",
    resize,
    allow_upscale: bool = False,
) -> torch.Tensor:
    """Resize onto the aligned canvas, or keep the aspect ratio and
    zero-pad the right/bottom sides (``resize_mode="pad"``)."""
    height, width = stacked_images.shape[-2:]

    if resize_mode == "resize":
        return resize(
            stacked_images,
            size=SizeDict(height=target_height, width=target_width),
            resample=resample,
        )

    if resize_mode != "pad":
        raise ValueError("resize_mode must be either 'resize' or 'pad'.")

    content_height, content_width = _get_pad_content_size(
        image_height=height,
        image_width=width,
        canvas_height=target_height,
        canvas_width=target_width,
        allow_upscale=allow_upscale,
    )

    if (content_height, content_width) != (height, width):
        stacked_images = resize(
            stacked_images,
            size=SizeDict(height=content_height, width=content_width),
            resample=resample,
        )

    # torchvision padding order: [left, top, right, bottom] -> pad only the
    # right and bottom sides.
    return tvF.pad(
        stacked_images,
        padding=[0, 0, target_width - content_width, target_height - content_height],
        fill=0,
    )


def _pixel_budget(
    min_image_tokens: int | None,
    max_image_tokens: int | None,
    patch_size: int,
    merge_size: int,
    temporal_patch_size: int,
) -> tuple[int, int]:
    """(min_pixels, max_pixels) from the token bounds of
    ``processor_config.json``; one vision token covers
    ``temporal_patch_size * (patch_size * merge_size) ** 2`` pixels."""
    if min_image_tokens is None or max_image_tokens is None:
        raise ValueError(
            "min_image_tokens and max_image_tokens must be provided by "
            "processor_config.json (or per-call kwargs)."
        )
    factor = temporal_patch_size * (patch_size * merge_size) ** 2
    return min_image_tokens * factor, max_image_tokens * factor


class Glm5NextImageProcessorKwargs(ImagesKwargs, total=False):  # type: ignore[call-arg]
    patch_size: int | None
    temporal_patch_size: int | None
    merge_size: int | None
    patch_expand_factor: int | None
    resize_mode: str | None
    min_image_tokens: int | None
    max_image_tokens: int | None


class Glm5NextImageProcessor(BaseImageProcessorFast):
    """Fast torchvision image processor for GLM-5.3-Flash.

    ``patch_expand_factor`` multiplies into the ``smart_resize`` spatial
    factor ``patch_size * merge_size``. ``resize_mode`` picks the geometry:
    ``"pad"`` (default) preserves the aspect ratio and zero-pads the
    right/bottom of the upward-aligned canvas, ``"resize"`` stretches onto
    it. Defaults mirror the checkpoint's ``image_processor`` config.
    """

    do_resize = True
    resample = PILImageResampling.BICUBIC
    size = {"longest_edge": 1}  # unused: budgets come from the token bounds
    do_rescale = True
    do_normalize = True
    image_mean = OPENAI_CLIP_MEAN
    image_std = OPENAI_CLIP_STD
    do_convert_rgb = True
    patch_size = 14
    temporal_patch_size = 2
    merge_size = 2
    patch_expand_factor = 1
    resize_mode = "pad"
    min_image_tokens = 16
    max_image_tokens = 8000
    valid_kwargs = Glm5NextImageProcessorKwargs
    model_input_names = ["pixel_values", "image_grid_thw"]

    def _preprocess(
        self,
        images: list[torch.Tensor],
        do_resize: bool,
        size: SizeDict,
        resample: "PILImageResampling | tvF.InterpolationMode | int | None",
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: float | list[float] | None,
        image_std: float | list[float] | None,
        patch_size: int,
        temporal_patch_size: int,
        merge_size: int,
        patch_expand_factor: int,
        resize_mode: str | None,
        min_image_tokens: int | None,
        max_image_tokens: int | None,
        disable_grouping: bool | None,
        return_tensors: str | TensorType | None,
        **kwargs,
    ) -> BatchFeature:
        resize_mode = resize_mode if resize_mode is not None else self.resize_mode
        min_pixels, max_pixels = _pixel_budget(
            min_image_tokens if min_image_tokens is not None else self.min_image_tokens,
            max_image_tokens if max_image_tokens is not None else self.max_image_tokens,
            patch_size,
            merge_size,
            temporal_patch_size,
        )
        grouped_images, grouped_images_index = group_images_by_shape(
            images, disable_grouping=disable_grouping
        )
        resized_images_grouped = {}
        for shape, stacked_images in grouped_images.items():
            height, width = stacked_images.shape[-2:]
            if do_resize:
                resized_height, resized_width = smart_resize(
                    t=temporal_patch_size,
                    h=height,
                    w=width,
                    t_factor=temporal_patch_size,
                    h_factor=patch_size * merge_size * patch_expand_factor,
                    w_factor=patch_size * merge_size * patch_expand_factor,
                    min_pixels=min_pixels,
                    max_pixels=max_pixels,
                )
                stacked_images = _resize_or_pad(
                    stacked_images,
                    target_height=resized_height,
                    target_width=resized_width,
                    resize_mode=resize_mode,
                    resample=resample,
                    resize=self.resize,
                    allow_upscale=(temporal_patch_size * height * width < min_pixels),
                )
            resized_images_grouped[shape] = stacked_images

        resized_images = reorder_images(resized_images_grouped, grouped_images_index)

        grouped_images, grouped_images_index = group_images_by_shape(
            resized_images, disable_grouping=disable_grouping
        )
        processed_images_grouped = {}
        processed_grids = {}

        for shape, stacked_images in grouped_images.items():
            resized_height, resized_width = stacked_images.shape[-2:]

            patches = self.rescale_and_normalize(
                stacked_images,
                do_rescale,
                rescale_factor,
                do_normalize,
                image_mean,
                image_std,
            )
            if patches.ndim == 4:  # (B, C, H, W)
                patches = patches.unsqueeze(1)  # (B, T=1, C, H, W)

            if patches.shape[1] % temporal_patch_size != 0:
                repeats = patches[:, -1:].repeat(
                    1,
                    temporal_patch_size - (patches.shape[1] % temporal_patch_size),
                    1,
                    1,
                    1,
                )
                patches = torch.cat([patches, repeats], dim=1)

            batch_size, t_len, channel = patches.shape[:3]
            grid_t = t_len // temporal_patch_size
            grid_h, grid_w = resized_height // patch_size, resized_width // patch_size

            patches = patches.view(
                batch_size,
                grid_t,
                temporal_patch_size,
                channel,
                grid_h // merge_size,
                merge_size,
                patch_size,
                grid_w // merge_size,
                merge_size,
                patch_size,
            )
            # (B, grid_t, gh, gw, mh, mw, C, tp, ph, pw)
            patches = patches.permute(0, 1, 4, 7, 5, 8, 3, 2, 6, 9)

            flatten_patches = patches.reshape(
                batch_size,
                grid_t * grid_h * grid_w,
                channel * temporal_patch_size * patch_size * patch_size,
            )

            processed_images_grouped[shape] = flatten_patches
            processed_grids[shape] = [[grid_t, grid_h, grid_w]] * batch_size

        processed_images = reorder_images(
            processed_images_grouped, grouped_images_index
        )
        processed_grids = reorder_images(processed_grids, grouped_images_index)

        pixel_values = torch.cat(processed_images, dim=0)
        image_grid_thw = torch.tensor(processed_grids)

        return BatchFeature(
            data={"pixel_values": pixel_values, "image_grid_thw": image_grid_thw},
            tensor_type=return_tensors,
        )

    def preprocess(
        self, images: ImageInput, **kwargs: Unpack[Glm5NextImageProcessorKwargs]
    ) -> BatchFeature:
        return super().preprocess(images, **kwargs)

    def get_number_of_image_patches(
        self, height: int, width: int, images_kwargs: dict | None = None
    ) -> int:
        """Number of image patches (pre-merge) for a given (height, width)."""
        images_kwargs = images_kwargs or {}
        patch_size = images_kwargs.get("patch_size", self.patch_size)
        merge_size = images_kwargs.get("merge_size", self.merge_size)
        patch_expand_factor = images_kwargs.get(
            "patch_expand_factor", self.patch_expand_factor
        )
        min_pixels, max_pixels = _pixel_budget(
            images_kwargs.get("min_image_tokens", self.min_image_tokens),
            images_kwargs.get("max_image_tokens", self.max_image_tokens),
            patch_size,
            merge_size,
            self.temporal_patch_size,
        )
        resized_height, resized_width = smart_resize(
            t=self.temporal_patch_size,
            h=height,
            w=width,
            t_factor=self.temporal_patch_size,
            h_factor=patch_size * merge_size * patch_expand_factor,
            w_factor=patch_size * merge_size * patch_expand_factor,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
        grid_h, grid_w = resized_height // patch_size, resized_width // patch_size
        return grid_h * grid_w


class Glm5NextVideoProcessorKwargs(VideosKwargs, total=False):  # type: ignore[call-arg]
    fps: list[float] | float
    patch_size: int
    temporal_patch_size: int
    merge_size: int
    patch_expand_factor: int
    resize_mode: str | None
    target_fps: float | None
    max_frames: int | None
    fps_interval: int | None
    max_frame_count_dynamic: int | None
    min_image_tokens: int | None
    max_image_tokens: int | None


class Glm5NextVideoProcessor(BaseVideoProcessor):
    """Fast video processor for GLM-5.3-Flash.

    Shares ``smart_resize`` / the pad-mode geometry / the patchify with the
    image processor, and adds GLM-5.3-Flash frame sampling
    (``glm_sample_frame_indices``: ``fps_interval`` semantics with a
    temporal-patch-scaled greedy walk). Defaults mirror the checkpoint's
    ``video_processor`` config.
    """

    resample = PILImageResampling.BICUBIC
    size = {"longest_edge": 1}  # unused: budgets come from the token bounds
    image_mean = OPENAI_CLIP_MEAN
    image_std = OPENAI_CLIP_STD
    do_resize = True
    do_rescale = True
    do_normalize = True
    do_convert_rgb = True
    do_sample_frames = True
    patch_size = 14
    temporal_patch_size = 2
    patch_expand_factor = 1
    merge_size = 2
    valid_kwargs = Glm5NextVideoProcessorKwargs
    num_frames = 16
    fps = 2
    fps_interval = 2.0
    max_frame_count_dynamic = 2048
    resize_mode = "pad"
    min_image_tokens = 16
    max_image_tokens = 240000
    model_input_names = ["pixel_values_videos", "video_grid_thw"]

    def sample_frames(
        self,
        metadata: VideoMetadata,
        fps: int | float | None = None,
        **kwargs,
    ) -> np.ndarray:
        """Sample frame indices with GLM's fps-interval policy.

        ``fps`` / ``target_fps``, ``max_frames`` and ``fps_interval`` /
        ``max_frame_count_dynamic`` are the overrides described in
        :func:`glm_sample_frame_indices`.
        """
        if metadata is None or getattr(metadata, "fps", None) is None:
            raise ValueError(
                "Asked to sample frames per second but no video metadata was "
                "provided which is required when sampling in GLM-5.3-Flash. Please "
                "pass in `VideoMetadata` object or set `do_sample_frames=False`."
            )

        target_fps = fps if fps is not None else kwargs.get("target_fps")
        if target_fps is None:
            target_fps = self.fps_interval
        indices = glm_sample_frame_indices(
            metadata.total_num_frames,
            metadata.fps,
            metadata.duration or 0,
            target_fps=target_fps,
            max_frame_count=kwargs.get("max_frames") or self.max_frame_count_dynamic,
            temporal_patch_size=self.temporal_patch_size,
        )
        return np.array(indices)

    def _preprocess(
        self,
        videos: list[torch.Tensor],
        do_convert_rgb: bool = True,
        do_resize: bool = True,
        size: SizeDict | None = None,
        resample: "PILImageResampling | int | None" = PILImageResampling.BICUBIC,
        do_rescale: bool = True,
        rescale_factor: float = 1 / 255.0,
        do_normalize: bool = True,
        image_mean: float | list[float] | None = None,
        image_std: float | list[float] | None = None,
        patch_size: int | None = None,
        temporal_patch_size: int | None = None,
        patch_expand_factor: int | None = None,
        merge_size: int | None = None,
        resize_mode: str | None = None,
        min_image_tokens: int | None = None,
        max_image_tokens: int | None = None,
        return_tensors: str | TensorType | None = None,
        **kwargs,
    ) -> BatchFeature:
        patch_expand_factor = self.patch_expand_factor
        patch_size = patch_size if patch_size is not None else self.patch_size
        temporal_patch_size = (
            temporal_patch_size
            if temporal_patch_size is not None
            else self.temporal_patch_size
        )
        merge_size = merge_size if merge_size is not None else self.merge_size
        resize_mode = resize_mode if resize_mode is not None else self.resize_mode
        min_pixels, max_pixels = _pixel_budget(
            min_image_tokens if min_image_tokens is not None else self.min_image_tokens,
            max_image_tokens if max_image_tokens is not None else self.max_image_tokens,
            patch_size,
            merge_size,
            temporal_patch_size,
        )
        grouped_videos, grouped_videos_index = group_videos_by_shape(videos)
        resized_videos_grouped = {}
        for shape, stacked_videos in grouped_videos.items():
            if do_convert_rgb:
                stacked_videos = self.convert_to_rgb(stacked_videos)
            b, t_len, c, h, w = stacked_videos.shape
            num_frames, height, width = t_len, h, w
            if do_resize:
                resized_height, resized_width = smart_resize(
                    t=num_frames,
                    h=height,
                    w=width,
                    t_factor=temporal_patch_size,
                    h_factor=patch_size * merge_size * patch_expand_factor,
                    w_factor=patch_size * merge_size * patch_expand_factor,
                    min_pixels=min_pixels,
                    max_pixels=max_pixels,
                )
                stacked_videos = stacked_videos.view(b * t_len, c, h, w)
                stacked_videos = _resize_or_pad(
                    stacked_videos,
                    target_height=resized_height,
                    target_width=resized_width,
                    resize_mode=resize_mode,
                    resample=resample,
                    resize=self.resize,
                    allow_upscale=(num_frames * height * width < min_pixels),
                )
                stacked_videos = stacked_videos.view(
                    b, t_len, c, resized_height, resized_width
                )
            resized_videos_grouped[shape] = stacked_videos
        resized_videos = reorder_videos(resized_videos_grouped, grouped_videos_index)

        grouped_videos, grouped_videos_index = group_videos_by_shape(resized_videos)
        processed_videos_grouped = {}
        processed_grids = {}
        for shape, stacked_videos in grouped_videos.items():
            resized_height, resized_width = get_image_size(
                stacked_videos[0], channel_dim=ChannelDimension.FIRST
            )
            stacked_videos = self.rescale_and_normalize(
                stacked_videos,
                do_rescale,
                rescale_factor,
                do_normalize,
                image_mean,
                image_std,
            )
            patches = stacked_videos

            if pad := -patches.shape[1] % temporal_patch_size:
                repeats = patches[:, -1:].expand(-1, pad, -1, -1, -1)
                patches = torch.cat((patches, repeats), dim=1)
            batch_size, grid_t, channel = patches.shape[:3]
            grid_t = grid_t // temporal_patch_size
            grid_h, grid_w = resized_height // patch_size, resized_width // patch_size

            patches = patches.view(
                batch_size,
                grid_t,
                temporal_patch_size,
                channel,
                grid_h // merge_size,
                merge_size,
                patch_size,
                grid_w // merge_size,
                merge_size,
                patch_size,
            )
            patches = patches.permute(0, 1, 4, 7, 5, 8, 3, 2, 6, 9)
            flatten_patches = patches.reshape(
                batch_size,
                grid_t * grid_h * grid_w,
                channel * temporal_patch_size * patch_size * patch_size,
            )

            processed_videos_grouped[shape] = flatten_patches
            processed_grids[shape] = [[grid_t, grid_h, grid_w]] * batch_size

        processed_videos = reorder_videos(
            processed_videos_grouped, grouped_videos_index
        )
        processed_grids = reorder_videos(processed_grids, grouped_videos_index)
        pixel_values_videos = torch.cat(processed_videos, dim=0)
        video_grid_thw = torch.tensor(processed_grids)
        return BatchFeature(
            data={
                "pixel_values_videos": pixel_values_videos,
                "video_grid_thw": video_grid_thw,
            },
            tensor_type=return_tensors,
        )


class Glm5NextProcessorKwargs(ProcessingKwargs, total=False):  # type: ignore[call-arg]
    images_kwargs: Glm5NextImageProcessorKwargs
    videos_kwargs: Glm5NextVideoProcessorKwargs
    _defaults = {
        "text_kwargs": {
            "padding": False,
            "return_token_type_ids": False,
            "return_mm_token_type_ids": False,
        },
        "videos_kwargs": {"return_metadata": True},
    }


class Glm5NextProcessor(ProcessorMixin):
    """Wraps a GLM-5.3-Flash image processor, video processor and tokenizer.

    Token expansion per image = ``prod(image_grid_thw) // merge_size**2``; video
    frames are expanded with ``<|begin_of_image|>...<|end_of_image|>{ts} seconds``
    structure (mrope timestamps).
    """

    attributes = ["image_processor", "tokenizer", "video_processor"]
    image_processor_class = "AutoImageProcessor"
    video_processor_class = "AutoVideoProcessor"
    tokenizer_class = ("PreTrainedTokenizer", "PreTrainedTokenizerFast")

    def __init__(
        self,
        image_processor=None,
        tokenizer=None,
        video_processor=None,
        chat_template=None,
        **kwargs,
    ) -> None:
        super().__init__(
            image_processor, tokenizer, video_processor, chat_template=chat_template
        )
        self.image_token = (
            "<|image|>"
            if not hasattr(tokenizer, "image_token")
            else tokenizer.image_token
        )
        self.video_token = (
            "<|video|>"
            if not hasattr(tokenizer, "video_token")
            else tokenizer.video_token
        )
        self.image_token_id = (
            tokenizer.image_token_id
            if getattr(tokenizer, "image_token_id", None)
            else tokenizer.convert_tokens_to_ids(self.image_token)
        )
        self.video_token_id = (
            tokenizer.video_token_id
            if getattr(tokenizer, "video_token_id", None)
            else tokenizer.convert_tokens_to_ids(self.video_token)
        )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        """Build the processor directly from the checkpoint config.

        GLM-5.3-Flash stores nested image/video configs in
        ``processor_config.json`` and declares a custom processor class. This
        method reads those configs directly and caps only the video token budget.
        """
        from transformers import AutoTokenizer

        model_path = pretrained_model_name_or_path
        tokenizer = AutoTokenizer.from_pretrained(model_path, **kwargs)

        def _cap_cfg(cfg: dict, *, is_video: bool) -> dict:
            # Video keeps a serving token cap (the checkpoint's 240k-token
            # budget would starve the KV cache at startup profiling); images
            # follow the checkpoint budget verbatim so preprocessing matches
            # the HF reference exactly.
            if is_video and cfg.get("max_image_tokens") is not None:
                cfg["max_image_tokens"] = min(
                    cfg["max_image_tokens"], _MAX_VIDEO_TOKENS
                )
            return cfg

        ip_cfg = _cap_cfg(
            dict(get_image_processor_config(model_path, **kwargs)), is_video=False
        )
        image_processor = Glm5NextImageProcessor(
            **{k: v for k, v in ip_cfg.items() if k != "image_processor_type"}
        )

        processor_config = get_hf_file_to_dict(
            "processor_config.json",
            model_path,
            revision=kwargs.get("revision", "main"),
        )
        if processor_config is None:
            raise ValueError(f"Missing processor_config.json for {model_path}")
        vp_cfg = _cap_cfg(dict(processor_config["video_processor"]), is_video=True)
        video_processor = Glm5NextVideoProcessor(
            **{k: v for k, v in vp_cfg.items() if k != "video_processor_type"}
        )

        return cls(
            image_processor=image_processor,
            tokenizer=tokenizer,
            video_processor=video_processor,
        )

    def __call__(
        self,
        images: ImageInput | None = None,
        text: TextInput
        | PreTokenizedInput
        | list[TextInput]
        | list[PreTokenizedInput] = None,
        videos: VideoInput | None = None,
        **kwargs: Unpack[Glm5NextProcessorKwargs],
    ) -> BatchFeature:
        output_kwargs = self._merge_kwargs(
            Glm5NextProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )
        if images is not None:
            image_inputs = self.image_processor(
                images=images, **output_kwargs["images_kwargs"]
            )
        else:
            image_inputs = {}

        if videos is not None:
            videos_inputs = self.video_processor(
                videos=videos, **output_kwargs["videos_kwargs"]
            )
            if "return_metadata" not in kwargs:
                videos_inputs.pop("video_metadata")
        else:
            videos_inputs = {}

        if not isinstance(text, list):
            text = [text]

        # Prompt updates expand the unchanged image/video markers after this call.
        return_tensors = output_kwargs["text_kwargs"].pop("return_tensors", None)
        return_mm_token_type_ids = output_kwargs["text_kwargs"].pop(
            "return_mm_token_type_ids", False
        )
        text_inputs = self.tokenizer(text, **output_kwargs["text_kwargs"])

        if return_mm_token_type_ids:
            array_ids = np.array(text_inputs["input_ids"])
            mm_token_type_ids = np.zeros_like(text_inputs["input_ids"])
            mm_token_type_ids[array_ids == self.image_token_id] = 1
            text_inputs["mm_token_type_ids"] = mm_token_type_ids.tolist()
        return BatchFeature(
            data={**text_inputs, **image_inputs, **videos_inputs},
            tensor_type=return_tensors,
        )

    def _get_num_multimodal_tokens(self, image_sizes=None, video_sizes=None, **kwargs):
        vision_data = {}
        if image_sizes is not None:
            images_kwargs = Glm5NextProcessorKwargs._defaults.get("images_kwargs", {})
            images_kwargs.update(kwargs)
            merge_size = (
                images_kwargs.get("merge_size", None) or self.image_processor.merge_size
            )

            num_image_patches = [
                self.image_processor.get_number_of_image_patches(
                    *image_size, images_kwargs
                )
                for image_size in image_sizes
            ]
            num_image_tokens = [(n // merge_size**2) for n in num_image_patches]
            vision_data.update(
                {
                    "num_image_tokens": num_image_tokens,
                    "num_image_patches": num_image_patches,
                }
            )

        if video_sizes is not None:
            videos_kwargs = Glm5NextProcessorKwargs._defaults.get("videos_kwargs", {})
            videos_kwargs.update(kwargs)
            num_video_patches = [
                self.video_processor.get_number_of_video_patches(
                    *video_size, videos_kwargs
                )
                for video_size in video_sizes
            ]
            num_video_tokens = [(n // merge_size**2) for n in num_video_patches]
            vision_data["num_video_tokens"] = num_video_tokens

        return MultiModalData(**vision_data)

    def post_process_image_text_to_text(
        self,
        generated_outputs,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
        **kwargs,
    ):
        return self.tokenizer.batch_decode(
            generated_outputs,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            **kwargs,
        )


__all__ = [
    "Glm5NextImageProcessor",
    "Glm5NextVideoProcessor",
    "Glm5NextProcessor",
    "smart_resize",
]
