# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""vLLM-native multimodal processor for GLM-5-Next.

Ports the GLM-5-Next image/video preprocessing pipeline (``smart_resize`` +
Qwen-VL-style patchify + GLM-5-Next dynamic-fps frame sampling) so vLLM no
longer depends on the transformers model classes ``GlmgaImageProcessor``,
``GlmgaVideoProcessor`` and ``Glm46VProcessor``. Only stable transformers
framework primitives (``BaseImageProcessorFast`` / ``TorchvisionBackend``,
``BaseVideoProcessor``, ``ProcessorMixin``, ``BatchFeature``, ``SizeDict`` ...)
are reused -- the GLM-5-Next-specific logic lives here.

The math is a faithful port of the training-side reference (``smart_resize``
with separate ``h_factor`` / ``w_factor`` / ``t_factor``, ``patch_expand_factor``
baked into the spatial factor, and the 10-D view->permute->reshape patchify that
yields ``(N, C*temporal_patch_size*patch_size**2) = (N, 1176)`` patches).
"""

import json
import math
import os
from typing import cast

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

logger = logging.get_logger(__name__)

# Serving cap on max pixels. The checkpoint ships an absurd ``size.longest_edge``
# (~9.6M image / ~100M video pixels) that makes vLLM's startup encoder-profiling
# reserve huge activation memory and starve the KV cache (300B weights already
# fill the GPU). ~1.25M px is a sane serving cap.
_MM_MAX_PIXELS = 1_254_400


def smart_resize(
    t: int,
    h: int,
    w: int,
    t_factor: int = 1,
    h_factor: int = 28,
    w_factor: int = 28,
    min_pixels: int = 112 * 112,
    max_pixels: int = 14 * 14 * 4 * 1280,
) -> tuple[int, int]:
    """GLM-5-Next ``smart_resize``: snap (h, w) to multiples of the spatial
    factor under a ``t_bar * h_bar * w_bar`` pixel budget.

    ``h_factor`` / ``w_factor`` carry ``patch_expand_factor`` (unlike Qwen-VL's
    single ``factor``); ``t_factor`` is ``temporal_patch_size``. For a still
    image ``t = t_factor = temporal_patch_size`` so ``t_bar = temporal_patch_size``.
    """
    if max_pixels < 0:
        h_bar = math.ceil(h / h_factor) * h_factor
        w_bar = math.ceil(w / w_factor) * w_factor
        return h_bar, w_bar

    if h == 0 or w == 0:
        raise ValueError(f"something wrong with shape, h or w is 0, got {h}, {w}")

    if max(h, w) / min(h, w) > 200:
        raise ValueError(
            f"absolute aspect ratio must be smaller than 200, "
            f"got {max(h, w) / min(h, w)}"
        )

    h_bar = round(h / h_factor) * h_factor
    w_bar = round(w / w_factor) * w_factor
    t_bar = round(t / t_factor) * t_factor

    if t_bar * h_bar * w_bar > max_pixels:
        beta = math.sqrt((t * h * w) / max_pixels)
        h_bar = math.floor(h / beta / h_factor) * h_factor
        w_bar = math.floor(w / beta / w_factor) * w_factor
    elif t_bar * h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (t * h * w))
        h_bar = math.ceil(h * beta / h_factor) * h_factor
        w_bar = math.ceil(w * beta / w_factor) * w_factor
    return h_bar, w_bar


class Glm5NextImageProcessorKwargs(ImagesKwargs, total=False):  # type: ignore[call-arg]
    patch_size: int | None
    temporal_patch_size: int | None
    merge_size: int | None
    patch_expand_factor: int | None


class Glm5NextImageProcessorFast(BaseImageProcessorFast):
    """Fast (torchvision) image processor for GLM-5-Next.

    ``patch_expand_factor`` (checkpoint ships 2) multiplies into the
    ``smart_resize`` spatial factor ``patch_size * merge_size``; dropping it
    (as GLM-4V's processor does) yields a wrong patch grid.
    """

    do_resize = True
    resample = PILImageResampling.BICUBIC
    size = {"shortest_edge": 112 * 112, "longest_edge": 28 * 28 * 15000}
    do_rescale = True
    do_normalize = True
    image_mean = OPENAI_CLIP_MEAN
    image_std = OPENAI_CLIP_STD
    do_convert_rgb = True
    patch_size = 14
    temporal_patch_size = 2
    merge_size = 2
    patch_expand_factor = 1
    valid_kwargs = Glm5NextImageProcessorKwargs
    model_input_names = ["pixel_values", "image_grid_thw"]

    def __init__(self, **kwargs: Unpack[Glm5NextImageProcessorKwargs]) -> None:
        super().__init__(**kwargs)
        if self.size is not None and (
            self.size.get("shortest_edge", None) is None
            or self.size.get("longest_edge", None) is None
        ):
            raise ValueError(
                "size must contain 'shortest_edge' and 'longest_edge' keys."
            )

    def _standardize_kwargs(self, **kwargs) -> dict:
        kwargs = super()._standardize_kwargs(**kwargs)
        size = kwargs.get("size", self.size)
        if size is not None and (not size.shortest_edge or not size.longest_edge):
            raise ValueError(
                "size must contain 'shortest_edge' and 'longest_edge' keys."
            )
        return kwargs

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
        disable_grouping: bool | None,
        return_tensors: str | TensorType | None,
        **kwargs,
    ) -> BatchFeature:
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
                    min_pixels=size.shortest_edge,
                    max_pixels=size.longest_edge,
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
        patch_size = (images_kwargs or {}).get("patch_size", self.patch_size)
        merge_size = (images_kwargs or {}).get("merge_size", self.merge_size)
        patch_expand_factor = (images_kwargs or {}).get(
            "patch_expand_factor", self.patch_expand_factor
        )
        size = (images_kwargs or {}).get(
            "size", {"shortest_edge": 112 * 112, "longest_edge": 28 * 28 * 15000}
        )
        resized_height, resized_width = smart_resize(
            t=self.temporal_patch_size,
            h=height,
            w=width,
            t_factor=self.temporal_patch_size,
            h_factor=patch_size * merge_size * patch_expand_factor,
            w_factor=patch_size * merge_size * patch_expand_factor,
            min_pixels=size["shortest_edge"],
            max_pixels=size["longest_edge"],
        )
        grid_h, grid_w = resized_height // patch_size, resized_width // patch_size
        return grid_h * grid_w


class Glm5NextVideoProcessorKwargs(VideosKwargs, total=False):  # type: ignore[call-arg]
    fps: list[float] | float
    patch_size: int
    temporal_patch_size: int
    merge_size: int
    patch_expand_factor: int
    max_duration: int
    max_image_size: dict


class Glm5NextVideoProcessor(BaseVideoProcessor):
    """Fast video processor for GLM-5-Next.

    Shares ``smart_resize`` and the patchify with the image processor, and adds
    GLM-5-Next's dynamic-fps frame sampling (``DYNAMIC_FPS_THRES``).
    """

    resample = PILImageResampling.BICUBIC
    size = {"shortest_edge": 112 * 112, "longest_edge": 28 * 28 * 2 * 30000}
    max_image_size = {"longest_edge": 28 * 28 * 2 * 30000}
    image_mean = OPENAI_CLIP_MEAN
    image_std = OPENAI_CLIP_STD
    do_resize = True
    do_rescale = True
    do_normalize = True
    do_convert_rgb = True
    do_sample_frames = True
    patch_size = 14
    temporal_patch_size = 2
    patch_expand_factor = 4
    max_duration = 300
    merge_size = 2
    valid_kwargs = Glm5NextVideoProcessorKwargs
    num_frames = 16
    fps = 2
    model_input_names = ["pixel_values_videos", "video_grid_thw"]

    def __init__(self, **kwargs: Unpack[Glm5NextVideoProcessorKwargs]) -> None:
        super().__init__(**kwargs)
        if self.size is not None and (
            self.size.get("shortest_edge", None) is None
            or self.size.get("longest_edge", None) is None
        ):
            raise ValueError(
                "size must contain 'shortest_edge' and 'longest_edge' keys."
            )

    def _standardize_kwargs(self, **kwargs) -> dict:
        kwargs = super()._standardize_kwargs(**kwargs)
        size = kwargs.get("size", self.size)
        if size is not None and (not size.shortest_edge or not size.longest_edge):
            raise ValueError(
                "size must contain 'shortest_edge' and 'longest_edge' keys."
            )
        return kwargs

    def sample_frames(
        self,
        metadata: VideoMetadata,
        fps: int | float | None = None,
        **kwargs,
    ) -> np.ndarray:
        """Sample frame indices at a duration-dependent target fps."""
        if metadata is None or getattr(metadata, "fps", None) is None:
            raise ValueError(
                "Asked to sample frames per second but no video metadata was "
                "provided which is required when sampling in GLM-5-Next. Please "
                "pass in `VideoMetadata` object or set `do_sample_frames=False`."
            )

        total_frames = metadata.total_num_frames
        max_frame_idx = total_frames - 1
        duration = metadata.duration or round(max_frame_idx / metadata.fps) + 1

        dynamic_fps_thres = {30: 3, 300: 1, 2400: 0.5}
        max_frame_count_dynamic = kwargs.get("max_frames") or 640
        max_duration = 2400
        effective_duration = min(duration, max_duration)
        target_fps = kwargs.get("target_fps")
        if not target_fps:
            if effective_duration <= 30:
                target_fps = dynamic_fps_thres[30]
            elif effective_duration <= 300:
                target_fps = dynamic_fps_thres[300]
            else:
                target_fps = dynamic_fps_thres[2400]

        extract_t = int(effective_duration * target_fps * self.temporal_patch_size)
        extract_t = min(extract_t, max_frame_count_dynamic)

        duration_per_frame = 1 / metadata.fps
        timestamps = [i * duration_per_frame for i in range(total_frames)]
        max_second = int(duration)

        if total_frames < extract_t:
            frame_indices = [
                math.floor(_i * total_frames / extract_t) for _i in range(extract_t)
            ]
        else:
            frame_indices = []
            current_second = 0.0
            inv_fps = 1 / (self.temporal_patch_size * target_fps)
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
            frame_indices = np.linspace(
                0, total_frames - 1, extract_t, dtype=int
            ).tolist()

        seen, uniq = set(), []
        for idx in frame_indices:
            if idx not in seen:
                seen.add(idx)
                uniq.append(idx)

        if len(uniq) & 1:
            uniq.append(uniq[-1])

        return np.array(uniq)

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
        if size is None:
            # Base converts the ``size`` dict to a ``SizeDict`` at init.
            size = cast(SizeDict, self.size)
        grouped_videos, grouped_videos_index = group_videos_by_shape(videos)
        resized_videos_grouped = {}
        for shape, stacked_videos in grouped_videos.items():
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
                    min_pixels=size.shortest_edge,
                    max_pixels=size.longest_edge,
                )
                stacked_videos = stacked_videos.view(b * t_len, c, h, w)
                stacked_videos = self.resize(
                    stacked_videos,
                    size=SizeDict(height=resized_height, width=resized_width),
                    resample=resample,
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

            if patches.shape[1] % temporal_patch_size != 0:
                repeats = patches[:, -1:].repeat(1, temporal_patch_size - 1, 1, 1, 1)
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
    """Wraps a GLM-5-Next image processor, video processor and tokenizer.

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

        The GLM-5-Next checkpoint stores its image/video processor configs inside
        ``processor_config.json`` (no standalone ``preprocessor_config.json``),
        and declares a custom ``processor_class`` that ``AutoProcessor`` cannot
        resolve. We read the configs here, cap ``size.longest_edge`` to a serving
        budget, and instantiate the vLLM-native sub-processors.
        """
        from transformers import AutoTokenizer

        model_path = pretrained_model_name_or_path
        tokenizer = AutoTokenizer.from_pretrained(model_path, **kwargs)

        ip_cfg = dict(get_image_processor_config(model_path))
        if isinstance(ip_cfg.get("size"), dict):
            ip_cfg["size"]["longest_edge"] = min(
                ip_cfg["size"].get("longest_edge", _MM_MAX_PIXELS), _MM_MAX_PIXELS
            )
        image_processor = Glm5NextImageProcessorFast(
            **{k: v for k, v in ip_cfg.items() if k != "image_processor_type"}
        )

        with open(os.path.join(model_path, "processor_config.json")) as f:
            vp_cfg = json.load(f)["video_processor"]
        if isinstance(vp_cfg.get("size"), dict):
            vp_cfg["size"]["longest_edge"] = min(
                vp_cfg["size"].get("longest_edge", _MM_MAX_PIXELS), _MM_MAX_PIXELS
            )
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
            image_grid_thw = image_inputs["image_grid_thw"]
        else:
            image_inputs = {}
            image_grid_thw = None

        if videos is not None:
            videos_inputs = self.video_processor(
                videos=videos, **output_kwargs["videos_kwargs"]
            )
            if "return_metadata" not in kwargs:
                video_metadata = videos_inputs.pop("video_metadata")
            else:
                video_metadata = videos_inputs["video_metadata"]
            video_grid_thw = videos_inputs["video_grid_thw"]
        else:
            videos_inputs = {}
            video_grid_thw = None

        if not isinstance(text, list):
            text = [text]

        text = text.copy()  # below lines change text in-place
        if image_grid_thw is not None:
            merge_length = self.image_processor.merge_size**2
            index = 0
            for i in range(len(text)):
                while self.image_token in text[i]:
                    num_image_tokens = image_grid_thw[index].prod() // merge_length
                    text[i] = text[i].replace(
                        self.image_token, "<|placeholder|>" * num_image_tokens, 1
                    )
                    index += 1

        if video_grid_thw is not None:
            merge_length = self.video_processor.merge_size**2
            video_index = 0
            for i in range(len(text)):
                while self.video_token in text[i]:
                    num_frames = video_grid_thw[video_index][0]
                    video_structure = ""

                    metadata = video_metadata[video_index]
                    if metadata.fps is None:
                        logger.warning_once(
                            "GLM-5-Next requires frame timestamps to construct "
                            "prompts, but the `fps` of the input video could not "
                            "be inferred. Defaulting to `fps=24`. Please provide "
                            "`video_metadata` for more accurate results."
                        )
                    metadata.fps = 24 if metadata.fps is None else metadata.fps
                    timestamps = metadata.timestamps[::2]  # mrope

                    selected_timestamps = list(timestamps[:num_frames])
                    while len(selected_timestamps) < num_frames:
                        selected_timestamps.append(
                            selected_timestamps[-1] if selected_timestamps else 0
                        )

                    for frame_idx in range(num_frames):
                        timestamp_sec = selected_timestamps[frame_idx]
                        video_structure += self.replace_frame_token_id(timestamp_sec)

                    text[i] = text[i].replace(self.video_token, video_structure, 1)
                    num_image_tokens = (
                        video_grid_thw[video_index].prod()
                        // merge_length
                        // video_grid_thw[video_index][0]
                    )
                    for frame_idx in range(num_frames):
                        if self.image_token in text[i]:
                            text[i] = text[i].replace(
                                self.image_token,
                                "<|placeholder|>" * num_image_tokens,
                                1,
                            )

                    video_index += 1

        # Restore all placeholders after both image and video blocks.
        for i in range(len(text)):
            text[i] = text[i].replace("<|placeholder|>", self.image_token)

        return_tensors = output_kwargs["text_kwargs"].pop("return_tensors", None)
        return_mm_token_type_ids = output_kwargs["text_kwargs"].pop(
            "return_mm_token_type_ids", False
        )
        text_inputs = self.tokenizer(text, **output_kwargs["text_kwargs"])
        self._check_special_mm_tokens(text, text_inputs, modalities=["image", "video"])

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

    def replace_frame_token_id(self, timestamp_sec) -> str:
        return (
            f"<|begin_of_image|>{self.image_token}<|end_of_image|>"
            f"{timestamp_sec:.1f} seconds"
        )

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
    "Glm5NextImageProcessorFast",
    "Glm5NextVideoProcessor",
    "Glm5NextProcessor",
    "smart_resize",
]
