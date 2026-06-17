# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MiniMax M3 VL HuggingFace-compatible Processor / ImageProcessor /
VideoProcessor, vendored into vLLM so the model loads without
``--trust-remote-code`` (the released checkpoint only ships these classes as
remote code via ``auto_map``).

Adapted verbatim from the ``MiniMaxAI/Minimax-M3-preview`` repository files
``image_processor.py``, ``video_processor.py`` and ``processing_minimax.py``
(revision ``db01c0fe``). Both image and video processors use Qwen-style
``smart_resize`` (bound by total pixels). The original async frame-sampling
helpers are intentionally omitted: vLLM performs its own frame loading and
feeds decoded frames to the processor.
"""

import math

import regex as re
import torch
from torchvision.transforms import InterpolationMode
from transformers import AutoTokenizer, BatchFeature
from transformers.image_processing_utils_fast import (
    BaseImageProcessorFast,
    group_images_by_shape,
    reorder_images,
)
from transformers.image_utils import PILImageResampling, SizeDict
from transformers.processing_utils import (
    ImagesKwargs,
    ProcessingKwargs,
    ProcessorMixin,
    Unpack,
    VideosKwargs,
)
from transformers.utils import TensorType
from transformers.video_processing_utils import BaseVideoProcessor
from transformers.video_utils import group_videos_by_shape, reorder_videos

# Maximum allowed aspect ratio before smart_resize rejects the input.
MAX_RATIO = 200

# Fixed (non-configurable) bounds for the long-side resize logic, per the
# MiniMax-M3 size spec. ``min_short_side_pixel`` is the floor the short edge is
# enlarged to; ``*_MAX_TOTAL_PIXELS`` is the hard area cap that, once exceeded,
# aborts processing instead of downscaling.
MIN_SHORT_SIDE_PIXEL = 112
IMAGE_MAX_TOTAL_PIXELS = 12_845_056  # 3584 ** 2 (width * height)
VIDEO_MAX_TOTAL_PIXELS = 301_056_000  # width * height * frames


def round_by_factor(number: int | float, factor: int) -> int:
    return round(number / factor) * factor


def ceil_by_factor(number: int | float, factor: int) -> int:
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int | float, factor: int) -> int:
    return math.floor(number / factor) * factor


def _smart_resize_by_long_side(
    height: int,
    width: int,
    factor: int,
    max_long_side_pixel: int,
    min_short_side_pixel: int,
    max_total_pixels: int | None,
) -> tuple[int, int]:
    """Long-side based resize (MiniMax-M3 size spec).

    (a) if the long side exceeds ``max_long_side_pixel`` → shrink so the long
        side equals ``max_long_side_pixel``;
    (b) else if the short side is below ``min_short_side_pixel`` → enlarge so the
        short side equals ``min_short_side_pixel``;
    (c) if the resulting area still exceeds ``max_total_pixels`` → raise.

    (a) and (b) are mutually exclusive (they branch on the *original* long side).
    Both sides are then rounded to a multiple of ``factor``. For videos the
    ``max_total_pixels`` cap is volumetric (width * height * frames) and is
    enforced by the caller, so pass ``max_total_pixels=None`` here.
    """
    long_side = max(height, width)
    short_side = min(height, width)

    scaled_height: float = height
    scaled_width: float = width
    if long_side > max_long_side_pixel:
        beta = max_long_side_pixel / long_side
        scaled_height = height * beta
        scaled_width = width * beta
    elif short_side < min_short_side_pixel:
        beta = min_short_side_pixel / short_side
        scaled_height = height * beta
        scaled_width = width * beta

    h_bar = max(factor, round_by_factor(scaled_height, factor))
    w_bar = max(factor, round_by_factor(scaled_width, factor))

    if max_total_pixels is not None and h_bar * w_bar > max_total_pixels:
        raise ValueError(
            f"image area {h_bar * w_bar} exceeds max_total_pixels "
            f"{max_total_pixels} after resizing"
        )
    return h_bar, w_bar


def smart_resize(
    height: int,
    width: int,
    factor: int = 28,
    min_pixels: int = 4 * 28 * 28,
    max_pixels: int = 451584,
    max_long_side_pixel: int | None = None,
    min_short_side_pixel: int = MIN_SHORT_SIDE_PIXEL,
    max_total_pixels: int | None = None,
) -> tuple[int, int]:
    """Rescale (height, width) so each side is a multiple of ``factor``.

    When ``max_long_side_pixel`` is set, use the MiniMax-M3 long-side resize
    spec (see :func:`_smart_resize_by_long_side`). Otherwise fall back to the
    Qwen-VL area bound, keeping the total area within ``[min_pixels, max_pixels]``.
    """
    if max(height, width) / min(height, width) > MAX_RATIO:
        raise ValueError(
            f"absolute aspect ratio must be smaller than {MAX_RATIO}, "
            f"got {max(height, width) / min(height, width)}"
        )
    if max_long_side_pixel is not None:
        return _smart_resize_by_long_side(
            height,
            width,
            factor=factor,
            max_long_side_pixel=max_long_side_pixel,
            min_short_side_pixel=min_short_side_pixel,
            max_total_pixels=max_total_pixels,
        )
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    return h_bar, w_bar


class MiniMaxM3VLImageProcessorKwargs(ImagesKwargs, total=False):  # type: ignore[call-arg]
    patch_size: int
    temporal_patch_size: int
    merge_size: int
    max_pixels: int
    max_long_side_pixel: int


class MiniMaxM3VLImageProcessor(BaseImageProcessorFast):
    do_resize = True
    resample = PILImageResampling.BICUBIC
    # required by base-class validation, not used as the resize bound
    size = {"height": 672, "width": 672}
    default_to_square = False
    do_rescale = True
    rescale_factor = 1 / 255
    do_normalize = True
    image_mean = [0.48145466, 0.4578275, 0.40821073]
    image_std = [0.26862954, 0.26130258, 0.27577711]
    do_convert_rgb = True
    patch_size = 14
    temporal_patch_size = 2
    merge_size = 2
    max_pixels = 451584  # 672 * 672
    # Long-side resize spec (opt-in via ``max_long_side_pixel``). The latter two
    # are fixed per the spec and are not exposed as configurable kwargs.
    max_long_side_pixel = None
    min_short_side_pixel = MIN_SHORT_SIDE_PIXEL
    max_total_pixels = IMAGE_MAX_TOTAL_PIXELS
    valid_kwargs = MiniMaxM3VLImageProcessorKwargs
    model_input_names = ["pixel_values", "image_grid_thw"]

    def __init__(self, **kwargs: Unpack[MiniMaxM3VLImageProcessorKwargs]):
        super().__init__(**kwargs)

    def preprocess(
        self, images, **kwargs: Unpack[MiniMaxM3VLImageProcessorKwargs]
    ) -> BatchFeature:
        return super().preprocess(images, **kwargs)

    def _preprocess(
        self,
        images: list[torch.Tensor],
        do_resize: bool,
        size: SizeDict,
        resample: "PILImageResampling | InterpolationMode | int | None",
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: "float | list[float] | None",
        image_std: "float | list[float] | None",
        patch_size: int,
        temporal_patch_size: int,
        merge_size: int,
        max_pixels: int,
        max_long_side_pixel: "int | None",
        disable_grouping: "bool | None",
        return_tensors: "str | TensorType | None",
        **kwargs,
    ) -> BatchFeature:
        grouped_images, grouped_images_index = group_images_by_shape(
            images, disable_grouping=disable_grouping
        )
        resized_images_grouped = {}
        factor = patch_size * merge_size
        for shape, stacked_images in grouped_images.items():
            height, width = stacked_images.shape[-2:]
            if do_resize:
                resized_height, resized_width = smart_resize(
                    height,
                    width,
                    factor=factor,
                    max_pixels=max_pixels,
                    max_long_side_pixel=max_long_side_pixel,
                    min_short_side_pixel=self.min_short_side_pixel,
                    max_total_pixels=self.max_total_pixels,
                )
                stacked_images = self.resize(
                    stacked_images,
                    size=SizeDict(height=resized_height, width=resized_width),
                    resample=resample,
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
            if patches.ndim == 4:
                patches = patches.unsqueeze(1)

            if patches.shape[1] % temporal_patch_size != 0:
                repeats = patches[:, -1:].repeat(
                    1,
                    temporal_patch_size - (patches.shape[1] % temporal_patch_size),
                    1,
                    1,
                    1,
                )
                patches = torch.cat([patches, repeats], dim=1)

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

            processed_images_grouped[shape] = flatten_patches
            processed_grids[shape] = [[grid_t, grid_h, grid_w]] * batch_size

        processed_images = reorder_images(
            processed_images_grouped, grouped_images_index
        )
        processed_grids = reorder_images(processed_grids, grouped_images_index)

        pixel_values = torch.cat(processed_images, dim=0)
        image_grid_thw = torch.tensor(processed_grids, dtype=torch.long)

        return BatchFeature(
            data={"pixel_values": pixel_values, "image_grid_thw": image_grid_thw},
            tensor_type=return_tensors,
        )

    def get_number_of_image_patches(self, height: int, width: int, images_kwargs=None):
        images_kwargs = images_kwargs or {}
        patch_size = images_kwargs.get("patch_size", self.patch_size)
        merge_size = images_kwargs.get("merge_size", self.merge_size)
        max_pixels = images_kwargs.get("max_pixels", self.max_pixels)
        max_long_side_pixel = images_kwargs.get(
            "max_long_side_pixel", self.max_long_side_pixel
        )

        resized_height, resized_width = smart_resize(
            height,
            width,
            factor=patch_size * merge_size,
            max_pixels=max_pixels,
            max_long_side_pixel=max_long_side_pixel,
            min_short_side_pixel=self.min_short_side_pixel,
            max_total_pixels=self.max_total_pixels,
        )
        grid_h, grid_w = resized_height // patch_size, resized_width // patch_size
        return grid_h * grid_w


class MiniMaxM3VLVideoProcessorKwargs(VideosKwargs, total=False):  # type: ignore[call-arg]
    patch_size: int
    temporal_patch_size: int
    merge_size: int
    min_pixels: int
    max_pixels: int
    max_long_side_pixel: int
    total_pixels: int
    min_frames: int
    max_frames: int
    fps: "float | int"


class MiniMaxM3VLVideoProcessor(BaseVideoProcessor):
    do_resize = True
    resample = PILImageResampling.BICUBIC
    size = {"height": 672, "width": 672}
    default_to_square = False
    do_rescale = True
    rescale_factor = 1 / 255
    do_normalize = True
    image_mean = [0.48145466, 0.4578275, 0.40821073]
    image_std = [0.26862954, 0.26130258, 0.27577711]
    do_convert_rgb = True
    do_sample_frames = False
    patch_size = 14
    temporal_patch_size = 2
    merge_size = 2
    min_pixels = 4 * 28 * 28
    max_pixels = 768 * 28 * 28  # 602,112
    total_pixels = int(64000 * 28 * 28 * 0.9)  # ~45M, ~64k tokens budget
    # Long-side resize spec (opt-in via ``max_long_side_pixel``). The video
    # ``max_total_pixels`` cap is volumetric (width * height * frames) and is
    # enforced in ``_preprocess`` once the frame count is known.
    max_long_side_pixel = None
    min_short_side_pixel = MIN_SHORT_SIDE_PIXEL
    max_total_pixels = VIDEO_MAX_TOTAL_PIXELS
    fps = 1.0
    min_frames = 4
    max_frames = 768
    valid_kwargs = MiniMaxM3VLVideoProcessorKwargs
    model_input_names = ["pixel_values_videos", "video_grid_thw"]

    def __init__(self, **kwargs: Unpack[MiniMaxM3VLVideoProcessorKwargs]):
        super().__init__(**kwargs)

    def _preprocess(
        self,
        videos: list[torch.Tensor],
        do_convert_rgb: bool,
        do_resize: bool,
        size: SizeDict,
        resample: "PILImageResampling | InterpolationMode | int | None",
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: "float | list[float] | None",
        image_std: "float | list[float] | None",
        patch_size: int,
        temporal_patch_size: int,
        merge_size: int,
        min_pixels: int,
        max_pixels: int,
        max_long_side_pixel: "int | None" = None,
        return_tensors: "str | TensorType | None" = None,
        **kwargs,
    ) -> BatchFeature:
        grouped_videos, grouped_videos_index = group_videos_by_shape(videos)
        resized_videos_grouped = {}
        factor = patch_size * merge_size
        for shape, stacked_videos in grouped_videos.items():
            batch_size, num_frames, channels, height, width = stacked_videos.shape
            resized_height, resized_width = height, width
            if do_resize:
                resized_height, resized_width = smart_resize(
                    height,
                    width,
                    factor=factor,
                    min_pixels=min_pixels,
                    max_pixels=max_pixels,
                    max_long_side_pixel=max_long_side_pixel,
                    min_short_side_pixel=self.min_short_side_pixel,
                    # Per-frame raise disabled; the video cap is volumetric and
                    # is enforced below once num_frames is known.
                    max_total_pixels=None,
                )
                if (
                    max_long_side_pixel is not None
                    and resized_height * resized_width * num_frames
                    > self.max_total_pixels
                ):
                    raise ValueError(
                        f"video area {resized_height * resized_width * num_frames} "
                        f"(width * height * frames) exceeds max_total_pixels "
                        f"{self.max_total_pixels} after resizing"
                    )
                stacked_videos = stacked_videos.view(
                    batch_size * num_frames, channels, height, width
                )
                stacked_videos = self.resize(
                    stacked_videos,
                    size=SizeDict(height=resized_height, width=resized_width),
                    resample=resample,
                )
                stacked_videos = stacked_videos.view(
                    batch_size,
                    num_frames,
                    channels,
                    resized_height,
                    resized_width,
                )
            resized_videos_grouped[shape] = stacked_videos
        resized_videos = reorder_videos(resized_videos_grouped, grouped_videos_index)

        grouped_videos, grouped_videos_index = group_videos_by_shape(resized_videos)
        processed_videos_grouped = {}
        processed_grids = {}
        for shape, stacked_videos in grouped_videos.items():
            resized_height, resized_width = stacked_videos.shape[-2:]
            patches = self.rescale_and_normalize(
                stacked_videos,
                do_rescale,
                rescale_factor,
                do_normalize,
                image_mean,
                image_std,
            )

            if pad := -patches.shape[1] % temporal_patch_size:
                repeats = patches[:, -1:].expand(-1, pad, -1, -1, -1)
                patches = torch.cat([patches, repeats], dim=1)

            batch_size, grid_t, channels = patches.shape[:3]
            grid_t = grid_t // temporal_patch_size
            grid_h, grid_w = resized_height // patch_size, resized_width // patch_size

            patches = patches.view(
                batch_size,
                grid_t,
                temporal_patch_size,
                channels,
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
                channels * temporal_patch_size * patch_size * patch_size,
            )

            processed_videos_grouped[shape] = flatten_patches
            processed_grids[shape] = [[grid_t, grid_h, grid_w]] * batch_size

        processed_videos = reorder_videos(
            processed_videos_grouped, grouped_videos_index
        )
        processed_grids = reorder_videos(processed_grids, grouped_videos_index)
        pixel_values_videos = torch.cat(processed_videos, dim=0)
        video_grid_thw = torch.tensor(processed_grids, dtype=torch.long)

        return BatchFeature(
            data={
                "pixel_values_videos": pixel_values_videos,
                "video_grid_thw": video_grid_thw,
            },
            tensor_type=return_tensors,
        )


class MiniMaxVLProcessorKwargs(ProcessingKwargs, total=False):  # type: ignore[call-arg]
    _defaults = {
        "videos_kwargs": {
            "do_resize": False,
            "return_metadata": True,
        },
    }


class MiniMaxVLProcessor(ProcessorMixin):
    IMAGE_TOKEN = "]<]image[>["
    VIDEO_TOKEN = "]<]video[>["
    VISION_START_TOKEN = "]<]start of image[>["
    VISION_END_TOKEN = "]<]end of image[>["

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        # Bypass ProcessorMixin's dynamic module lookup, which breaks in
        # transformers >= 5.9 when image_processor_class is a string: the
        # register() API now stores classes as {"pil": cls} dicts in
        # _extra_content, but get_possibly_dynamic_module() still calls
        # .__name__ on the raw value, crashing with AttributeError on dicts.
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path, **kwargs
        )
        image_processor = MiniMaxM3VLImageProcessor.from_pretrained(
            pretrained_model_name_or_path, **kwargs
        )
        video_processor = MiniMaxM3VLVideoProcessor.from_pretrained(
            pretrained_model_name_or_path, **kwargs
        )
        return cls(
            image_processor=image_processor,
            tokenizer=tokenizer,
            video_processor=video_processor,
        )

    def __init__(
        self, image_processor=None, tokenizer=None, video_processor=None, **kwargs
    ):
        self.image_token_id = tokenizer.convert_tokens_to_ids(self.IMAGE_TOKEN)
        self.video_token_id = tokenizer.convert_tokens_to_ids(self.VIDEO_TOKEN)
        super().__init__(image_processor, tokenizer, video_processor)
        # Video expansion also uses image start/end tokens. Separate video
        # start/end tokens exist in the tokenizer, but the original MiniMax
        # serving path did not use them; keep that behavior for compatibility.
        self.vision_start_token_id = tokenizer.convert_tokens_to_ids(
            self.VISION_START_TOKEN
        )
        self.vision_end_token_id = tokenizer.convert_tokens_to_ids(
            self.VISION_END_TOKEN
        )

    def _prune_video_tokens(
        self,
        input_text: str,
        video_segments: list[int],
        video_token: str,
    ) -> str:
        """Prune video tokens by temporal_patch_size (e.g., 2:1).

        Expects the prompt to carry exactly sum(video_segments) video tokens
        — i.e. one token per *sampled* frame — then drops tokens.
        """
        # If no videos or temporal_patch_size <= 1, no pruning needed
        if not video_segments or self.video_processor.temporal_patch_size <= 1:
            return input_text

        # Split while keeping delimiters
        special_tokens = [video_token]
        pattern = "|".join(map(re.escape, special_tokens))
        parts = re.split(f"({pattern})", input_text)

        def is_timestamp(text: str) -> bool:
            """Check if text ends with timestamp format like ']<]0.0 seconds[>['"""
            return (
                text.endswith("seconds[>[")
                or text.endswith("seconds[>[ ")
                or text.endswith("seconds [>[")
                or text.endswith("seconds [>[ ")
            )

        def extract_timestamp(text: str) -> str:
            """Extract timestamp text from the end, starting from ']<]'"""
            start_index = text.rfind("]<]")
            if start_index == -1:
                raise ValueError(f"Failed to extract timestamp: {text}")
            return text[start_index:]

        # Build new text with pruned video tokens
        final_parts = []
        current_seg_idx = 0  # Which video segment we're in
        frame_in_seg = 0  # Frame index within current segment
        last_timestamp_len = 0  # Length of timestamp to potentially remove

        for part in parts:
            if part == video_token:
                if current_seg_idx < len(video_segments):
                    if frame_in_seg % self.video_processor.temporal_patch_size == 0:
                        # Keep this video token
                        final_parts.append(part)
                        frame_in_seg += 1
                        if frame_in_seg >= video_segments[current_seg_idx]:
                            current_seg_idx += 1
                            frame_in_seg = 0
                        last_timestamp_len = 0
                    else:
                        # Skip this video token
                        frame_in_seg += 1
                        if frame_in_seg >= video_segments[current_seg_idx]:
                            current_seg_idx += 1
                            frame_in_seg = 0
                        # Remove the timestamp that was already appended
                        if last_timestamp_len > 0:
                            assert len(final_parts) > 0
                            final_parts[-1] = final_parts[-1][:-last_timestamp_len]
                            last_timestamp_len = 0
                else:
                    # No more video segments, keep as is
                    final_parts.append(part)
                    last_timestamp_len = 0
            else:
                # Text part
                final_parts.append(part)
                # Check if this text ends with a timestamp
                if is_timestamp(part):
                    last_timestamp_len = len(extract_timestamp(part))
                else:
                    last_timestamp_len = 0

        return "".join(final_parts)

    def __call__(
        self,
        images=None,
        text=None,
        videos=None,
        **kwargs: Unpack[MiniMaxVLProcessorKwargs],
    ) -> BatchFeature:
        output_kwargs = self._merge_kwargs(
            MiniMaxVLProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        if images is not None:
            images_kwargs = output_kwargs["images_kwargs"]
            image_inputs = self.image_processor(images=images, **images_kwargs)
            image_grid_thw = image_inputs["image_grid_thw"]
        else:
            image_inputs = {}
            image_grid_thw = None

        if videos is not None:
            videos_kwargs = output_kwargs["videos_kwargs"]
            video_inputs = self.video_processor(videos=videos, **videos_kwargs)
            video_grid_thw = video_inputs["video_grid_thw"]
            if not kwargs.get("return_metadata"):
                video_metadata = video_inputs.pop("video_metadata")
            else:
                video_metadata = video_inputs["video_metadata"]
        else:
            video_inputs = {}
            video_grid_thw = None

        if not isinstance(text, list):
            text = [text]
        text = text.copy()

        # Expand image tokens
        if image_grid_thw is not None:
            merge_length = self.image_processor.merge_size**2
            placeholder = "]<]placeholder[>["
            index = 0
            for i in range(len(text)):
                while self.IMAGE_TOKEN in text[i]:
                    num_tokens = image_grid_thw[index].prod() // merge_length
                    text[i] = text[i].replace(
                        self.IMAGE_TOKEN,
                        self.VISION_START_TOKEN
                        + placeholder * num_tokens
                        + self.VISION_END_TOKEN,
                        1,
                    )
                    index += 1
                text[i] = text[i].replace(placeholder, self.IMAGE_TOKEN)

        # Expand video tokens
        if video_grid_thw is not None:
            merge_length = self.image_processor.merge_size**2
            placeholder = "]<]placeholder[>["
            index = 0
            for i in range(len(text)):
                while self.VIDEO_TOKEN in text[i]:
                    metadata = video_metadata[index]
                    grid_t = video_grid_thw[index][0]
                    frame_seqlen = video_grid_thw[index][1:].prod() // merge_length

                    video_placeholder = ""
                    for frame_idx in range(grid_t):
                        if (
                            metadata.fps is not None
                            and metadata.frames_indices is not None
                        ):
                            ts = (
                                metadata.frames_indices[
                                    min(
                                        frame_idx
                                        * self.video_processor.temporal_patch_size,
                                        len(metadata.frames_indices) - 1,
                                    )
                                ]
                                / metadata.fps
                            )
                            video_placeholder += f"]<]{ts:.1f} seconds[>["
                        video_placeholder += (
                            self.VISION_START_TOKEN
                            + placeholder * frame_seqlen
                            + self.VISION_END_TOKEN
                        )

                    text[i] = text[i].replace(self.VIDEO_TOKEN, video_placeholder, 1)
                    index += 1
                text[i] = text[i].replace(placeholder, self.VIDEO_TOKEN)

        # Tokenize
        return_tensors = output_kwargs["text_kwargs"].pop("return_tensors", None)
        text_inputs = self.tokenizer(text, **output_kwargs["text_kwargs"])

        return BatchFeature(
            data={**text_inputs, **image_inputs, **video_inputs},
            tensor_type=return_tensors,
        )
