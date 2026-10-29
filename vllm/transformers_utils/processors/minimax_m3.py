# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MiniMax M3.1 VL HuggingFace-compatible Processor / ImageProcessor /
VideoProcessor, vendored into vLLM so the model loads without
``--trust-remote-code`` (the released checkpoint only ships these classes as
remote code via ``auto_map``).

Both image and video processors use the MiniMax M3.1 aligned-size resize
(``compute_aligned_size``): tiny inputs are upscaled to a minimum area /
short side, then both dims are ceil-aligned to the patch factor and capped
at a per-modality long side (selectable via the OpenAI-style ``detail``
kwarg or an explicit ``max_long_side_pixel``). The original async
frame-sampling helpers are intentionally omitted: vLLM performs its own
frame loading and feeds decoded frames to the processor.
"""

import math

import torch
from torchvision.transforms import InterpolationMode
from transformers import AutoTokenizer, BatchFeature
from transformers.image_processing_backends import (
    TorchvisionBackend,
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

# Maximum allowed aspect ratio before compute_aligned_size rejects the input.
MAX_RATIO = 200

# Fixed (non-configurable) bounds of the aligned-size resize, per the
# MiniMax M3.1 size spec. ``MIN_SHORT_SIDE_PIXEL`` is the floor the short
# edge is enlarged to; ``VIDEO_MAX_TOTAL_PIXELS`` is the hard volumetric cap
# (width * height * frames) that, once exceeded, aborts video processing.
MIN_SHORT_SIDE_PIXEL = 168
MIN_PIXELS = MIN_SHORT_SIDE_PIXEL**2
VIDEO_MAX_TOTAL_PIXELS = 301_056_000

# OpenAI-style ``detail`` -> long-side pixel cap, per modality.
IMAGE_DETAIL_MAX_LONG_SIDE_PIXEL = {"low": 672, "auto": 2016, "high": 3584}
VIDEO_DETAIL_MAX_LONG_SIDE_PIXEL = {"low": 504, "auto": 672, "high": 2188}


def compute_aligned_size(
    height: int,
    width: int,
    factor: int,
    *,
    min_pixels: int = MIN_PIXELS,
    max_long_side_pixel: int,
    min_short_side: int = MIN_SHORT_SIDE_PIXEL,
) -> tuple[int, int]:
    """Compute the resize target ``(resized_height, resized_width)``.

    Three layers (mirrors sglang's M3.1 ``compute_aligned_size``):

    1. Aspect-ratio guard: reject inputs whose ratio exceeds ``MAX_RATIO``.
    2. Floor layers: upscale proportionally when the area is below
       ``min_pixels``, or the short side is below ``min_short_side``.
    3. Alignment layer: ``_align_to_factor`` ceil-aligns both dims to
       ``factor`` and caps them at ``max_long_side_pixel``.

    Args:
        height: Original height in pixels, must be > 0.
        width: Original width in pixels, must be > 0.
        factor: Alignment factor, usually ``patch_size * merge_size``.
        min_pixels: Area floor; smaller inputs are upscaled to at least
            this area before alignment.
        max_long_side_pixel: Per-dim cap applied by the alignment layer
            (used for both dims).
        min_short_side: Short-side pixel floor.

    Returns:
        ``(resized_height, resized_width)``, both multiples of ``factor``
        and at most ``max_long_side_pixel``.

    Raises:
        ValueError: If the aspect ratio exceeds ``MAX_RATIO``.

    """
    if max(height, width) / min(height, width) > MAX_RATIO:
        raise ValueError(
            f"absolute aspect ratio must be smaller than {MAX_RATIO}, "
            f"got {max(height, width) / min(height, width)}"
        )
    if height * width < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        height = math.ceil(height * beta)
        width = math.ceil(width * beta)
    if min(height, width) < min_short_side:
        beta = min_short_side / min(height, width)
        height = math.ceil(height * beta)
        width = math.ceil(width * beta)
    return _align_to_factor(
        height, width, factor, (max_long_side_pixel, max_long_side_pixel)
    )


def _align_to_factor(
    height: int,
    width: int,
    factor: int,
    max_size: tuple[int, int],
) -> tuple[int, int]:
    """Ceil-align both dims to ``factor``, then cap at per-dim ``max_size``.

    When either aligned dim exceeds its cap, scale both dims back down
    proportionally (integer arithmetic, preserving the aspect ratio) and
    ceil-align again.

    Returns:
        ``(height, width)`` with both dims multiples of ``factor`` and at
        most the corresponding ``max_size`` dim.

    """
    max_w, max_h = max_size
    # Round the caps down to factor multiples so alignment never exceeds them.
    max_w = max_w - (max_w % factor)
    max_h = max_h - (max_h % factor)
    new_w = width if width % factor == 0 else width + (factor - width % factor)
    new_h = height if height % factor == 0 else height + (factor - height % factor)
    if new_w > max_w or new_h > max_h:
        new_w_ = min((new_w * max_w) // new_w, (new_w * max_h) // new_h)
        new_h_ = min((new_h * max_w) // new_w, (new_h * max_h) // new_h)
        new_w = new_w_
        new_h = new_h_
        new_w = new_w if new_w % factor == 0 else new_w + (factor - new_w % factor)
        new_h = new_h if new_h % factor == 0 else new_h + (factor - new_h % factor)
    assert new_w <= max_w and new_h <= max_h, (
        f"_align_to_factor: result (h={new_h}, w={new_w}) exceeds max_size "
        f"{max_size}; input (h={height}, w={width}, factor={factor})"
    )
    return new_h, new_w


def resolve_max_long_side_pixel(
    detail: "str | None",
    max_long_side_pixel: "int | None",
    *,
    modality: str,
) -> int:
    """Resolve the long-side pixel cap from ``detail`` / explicit override.

    ``detail`` (default ``"auto"``) maps to a per-modality cap. An explicit
    ``max_long_side_pixel`` takes priority over a non-auto ``detail``; the
    ``"auto"`` tier value itself is treated as "unset" so the processor class
    defaults (2016 for images, 672 for videos) do not shadow ``detail``.

    Raises:
        ValueError: If ``detail`` or ``modality`` is not a known value.

    """
    if modality == "image":
        detail_map = IMAGE_DETAIL_MAX_LONG_SIDE_PIXEL
    elif modality == "video":
        detail_map = VIDEO_DETAIL_MAX_LONG_SIDE_PIXEL
    else:
        raise ValueError(f"unknown modality {modality!r}")
    detail = detail or "auto"
    if detail not in detail_map:
        raise ValueError(
            f"invalid detail {detail!r} for {modality}; "
            f"expected one of {sorted(detail_map)}"
        )
    if max_long_side_pixel is not None and max_long_side_pixel != detail_map["auto"]:
        return max_long_side_pixel
    return detail_map[detail]


class MiniMaxM3VLImageProcessorKwargs(ImagesKwargs, total=False):  # type: ignore[call-arg]
    patch_size: int
    temporal_patch_size: int
    merge_size: int
    detail: "str | None"
    max_long_side_pixel: "int | None"
    # Legacy keys from released preprocessor_config.json files; accepted for
    # backward compatibility but ignored by the aligned-size resize.
    min_pixels: int
    max_pixels: int


class MiniMaxM3VLImageProcessor(TorchvisionBackend):
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
    # Aligned-size resize bounds (M3.1). ``detail`` selects the long-side cap
    # ("low"/"auto"/"high"); an explicit ``max_long_side_pixel`` wins.
    detail = "auto"
    max_long_side_pixel = 2016
    min_short_side_pixel = MIN_SHORT_SIDE_PIXEL
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
        detail: str,
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
        max_long_side = resolve_max_long_side_pixel(
            detail, max_long_side_pixel, modality="image"
        )
        for shape, stacked_images in grouped_images.items():
            height, width = stacked_images.shape[-2:]
            if do_resize:
                resized_height, resized_width = compute_aligned_size(
                    height,
                    width,
                    factor=factor,
                    max_long_side_pixel=max_long_side,
                    min_short_side=self.min_short_side_pixel,
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
        detail = images_kwargs.get("detail", self.detail)
        max_long_side_pixel = images_kwargs.get(
            "max_long_side_pixel", self.max_long_side_pixel
        )
        max_long_side = resolve_max_long_side_pixel(
            detail, max_long_side_pixel, modality="image"
        )
        resized_height, resized_width = compute_aligned_size(
            height,
            width,
            factor=patch_size * merge_size,
            max_long_side_pixel=max_long_side,
            min_short_side=self.min_short_side_pixel,
        )
        grid_h, grid_w = resized_height // patch_size, resized_width // patch_size
        return grid_h * grid_w


class MiniMaxM3VLVideoProcessorKwargs(VideosKwargs, total=False):  # type: ignore[call-arg]
    patch_size: int
    temporal_patch_size: int
    merge_size: int
    detail: "str | None"
    max_long_side_pixel: "int | None"
    min_frames: int
    max_frames: int
    fps: "float | int"
    # Legacy keys from released preprocessor_config.json files; accepted for
    # backward compatibility but ignored by the aligned-size resize.
    min_pixels: int
    max_pixels: int
    total_pixels: int


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
    # Aligned-size resize bounds (M3.1). ``detail`` selects the long-side cap
    # ("low"/"auto"/"high"); an explicit ``max_long_side_pixel`` wins. The
    # ``max_total_pixels`` cap is volumetric (width * height * frames) and is
    # enforced in ``_preprocess`` once the frame count is known.
    detail = "auto"
    max_long_side_pixel = 672
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
        detail: str = "auto",
        max_long_side_pixel: "int | None" = None,
        return_tensors: "str | TensorType | None" = None,
        **kwargs,
    ) -> BatchFeature:
        grouped_videos, grouped_videos_index = group_videos_by_shape(videos)
        resized_videos_grouped = {}
        factor = patch_size * merge_size
        max_long_side = resolve_max_long_side_pixel(
            detail, max_long_side_pixel, modality="video"
        )
        for shape, stacked_videos in grouped_videos.items():
            batch_size, num_frames, channels, height, width = stacked_videos.shape
            resized_height, resized_width = height, width
            if do_resize:
                resized_height, resized_width = compute_aligned_size(
                    height,
                    width,
                    factor=factor,
                    max_long_side_pixel=max_long_side,
                    min_short_side=self.min_short_side_pixel,
                )
                if resized_height * resized_width * num_frames > self.max_total_pixels:
                    raise ValueError(
                        f"video area {resized_height * resized_width * num_frames} "
                        f"(width * height * frames) exceeds max_total_pixels "
                        f"{self.max_total_pixels} after resizing"
                    )
                stacked_videos = stacked_videos.view(
                    batch_size * num_frames, channels, height, width
                )
                # Resize in float32, matching the training/SFT pipeline (and
                # sglang): uint8 resize rounds intermediates and shifts pixels.
                stacked_videos = self.resize(
                    stacked_videos.float(),
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
        #
        # Reuse the tokenizer passed by vLLM instead of loading another one,
        # and keep it out of the kwargs forwarded to the sub-processor loaders.
        tokenizer = kwargs.pop("tokenizer", None)
        if tokenizer is None:
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
