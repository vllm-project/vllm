# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Mapping
from typing import Any

import numpy as np
import regex as re
import torch
from PIL import Image
from transformers import BaseImageProcessor, BatchFeature, TensorType
from transformers.processing_utils import ProcessorMixin

from vllm.multimodal.inputs import HfVideoItem
from vllm.tokenizers.hf import HfTokenizer
from vllm.transformers_utils.processors.kimi_k25_vision_fused import (
    navit_resize_image,
)

KIMI_K3_VIDEO_PLACEHOLDER = "<|kimi_video_placeholder|>"

# Without a checkpoint `max_num_frames_each_video`, the patch budget alone lets
# a tiny video keep ~160K frames. 768 (as in Qwen3-VL) only binds below 360p.
_MAX_FRAMES = 768


def supports_kimi_k3_video(media_proc_cfg: Mapping[str, Any]) -> bool:
    """Whether `media_proc_cfg` supports video.

    A full chunk must fit `in_patch_limit_video`, or the fused processor
    resizes frames below the planned `in_patch_limit_each_frame` size.
    """
    limit = media_proc_cfg.get("in_patch_limit_video")
    each_frame = media_proc_cfg.get("in_patch_limit_each_frame")
    chunk = media_proc_cfg.get("temporal_merge_kernel_size")
    return (
        limit is not None
        and each_frame is not None
        and chunk is not None
        and media_proc_cfg.get("sample_fps") is not None
        and media_proc_cfg.get("timestamp_mode") == "hh:mm:ss.fff"
        and limit >= chunk * each_frame
    )


def get_kimi_k3_frame_resize(
    width: int, height: int, media_proc_cfg: Mapping[str, Any]
) -> dict[str, int]:
    return navit_resize_image(
        width,
        height,
        media_proc_cfg["patch_size"],
        media_proc_cfg["merge_kernel_size"],
        media_proc_cfg["in_patch_limit_each_frame"],
        media_proc_cfg["patch_limit_on_one_side"],
        media_proc_cfg["fixed_output_tokens"],
    )


def get_kimi_k3_max_video_tokens(media_proc_cfg: Mapping[str, Any]) -> int:
    merge = media_proc_cfg["merge_kernel_size"]
    chunk = media_proc_cfg["temporal_merge_kernel_size"]
    return media_proc_cfg["in_patch_limit_video"] // (chunk * merge * merge)


def get_kimi_k3_max_video_frames(
    width: int, height: int, media_proc_cfg: Mapping[str, Any]
) -> int:
    """Most frames a video of this size keeps within the profiled video budget."""
    if not supports_kimi_k3_video(media_proc_cfg):
        raise ValueError("This Kimi-K3 checkpoint does not support video input")
    resize = get_kimi_k3_frame_resize(width, height, media_proc_cfg)
    patch = media_proc_cfg["patch_size"]
    frame_patches = ((resize["new_width"] + resize["pad_width"]) // patch) * (
        (resize["new_height"] + resize["pad_height"]) // patch
    )
    chunk = media_proc_cfg["temporal_merge_kernel_size"]
    max_chunks = get_kimi_k3_max_video_tokens(media_proc_cfg) // resize["num_tokens"]
    max_frames = min(
        media_proc_cfg["in_patch_limit_video"] // frame_patches, max_chunks * chunk
    )
    frame_cap = media_proc_cfg.get("max_num_frames_each_video")
    max_frames = min(max_frames, _MAX_FRAMES if frame_cap is None else frame_cap)
    if max_frames < 1:
        raise ValueError(f"Video frames of {width}x{height} exceed the video budget")
    return max_frames


def plan_kimi_k3_video(
    num_frames: int,
    fps: float,
    width: int,
    height: int,
    media_proc_cfg: Mapping[str, Any],
) -> list[int]:
    """Indices of the source frames Kimi-K3 samples from a video."""
    if num_frames <= 0 or not fps > 0:
        raise ValueError(f"Invalid video: {num_frames} frames at {fps} fps")
    max_frames = get_kimi_k3_max_video_frames(width, height, media_proc_cfg)
    sample_fps = min(media_proc_cfg["sample_fps"], fps)
    count = min(max(round(num_frames * sample_fps / fps), 1), max_frames)
    return np.linspace(0, num_frames - 1, count).round().astype(int).tolist()


def format_kimi_k3_timestamp(ms: int) -> str:
    hours, minutes, secs = ms // 3600000, ms // 60000 % 60, ms // 1000 % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{ms % 1000:03d}"


def _frame_to_hwc(frame: Any) -> np.ndarray:
    if isinstance(frame, Image.Image):
        return np.asarray(frame.convert("RGB"))
    if isinstance(frame, torch.Tensor):
        frame = frame.cpu().numpy()
    frame = np.asarray(frame)
    if frame.ndim != 3:
        raise ValueError(f"Expected a 3D video frame, got shape {frame.shape}")
    if frame.shape[-1] not in (1, 3, 4):
        frame = frame.transpose(1, 2, 0)
    return frame


def _frame_to_pil(frame: Any) -> Image.Image:
    frame = _frame_to_hwc(frame)
    if frame.dtype != np.uint8:
        raise ValueError(f"Expected uint8 video frames, got {frame.dtype}")
    if frame.shape[-1] == 1:
        frame = frame[..., 0]
    return Image.fromarray(frame).convert("RGB")


def encode_k3_text_prompt(
    tokenizer: HfTokenizer,
    text: str,
    image_placeholder: str,
    *,
    add_special_tokens: bool = True,
) -> list[int]:
    """Tokenize a text prompt, encoding each image or video placeholder on its own.

    The placeholders are made of ordinary tokens. Encoded together with the
    surrounding text, their first and last tokens can merge with their
    neighbours, and the placeholder is then not found in the token ids.
    """
    placeholder_ids = {
        placeholder: tokenizer.encode(placeholder, add_special_tokens=False)
        for placeholder in (image_placeholder, KIMI_K3_VIDEO_PLACEHOLDER)
    }
    pattern = "|".join(map(re.escape, placeholder_ids))
    token_ids: list[int] = []
    for segment in re.split(f"({pattern})", text):
        if segment in placeholder_ids:
            token_ids += placeholder_ids[segment]
        elif segment:
            token_ids += tokenizer.encode(
                segment, add_special_tokens=add_special_tokens
            )
    return token_ids


class KimiK3Processor(ProcessorMixin):
    """HF-style processor wrapper for the Kimi-K3 model.

    K3 exposes the standard `image` modality, so vLLM calls this processor
    with `images=[PIL, ...]`. The underlying checkpoint image processor
    (`KimiK3VisionProcessor`) works on `{"type": "image", "image": PIL}`
    media dicts, so this wrapper adapts bare PIL images into that shape before
    delegating to `preprocess`. Videos are sampled into `video_chunk` dicts.

    Text is only tokenized here; the single `<|kimi_image_placeholder|>`
    token per image is expanded into the resolution-aware media block by the
    model's `_get_prompt_updates` on the vLLM side.
    """

    attributes = ["image_processor", "tokenizer"]

    def __init__(
        self,
        image_processor: BaseImageProcessor,
        tokenizer: HfTokenizer,
    ) -> None:
        self.image_processor = image_processor
        self.tokenizer = tokenizer

    def __call__(
        self,
        text: str | list[str] | None = None,
        images: object | list[object] | None = None,
        videos: list[tuple[HfVideoItem, dict[str, Any]]] | None = None,
        return_tensors: str | TensorType | None = None,
        do_rescale: bool = True,
        do_normalize: bool = True,
        **kwargs,
    ) -> BatchFeature:
        if images is not None:
            if not isinstance(images, list):
                images = [images]
            medias: list[dict[str, Any]] = [
                {"type": "image", "image": image} for image in images
            ]
            image_inputs = self.image_processor.preprocess(
                medias,
                return_tensors=return_tensors,
                do_rescale=do_rescale,
                do_normalize=do_normalize,
            )
            image_inputs["image_sizes"] = [
                list(media["image"].size) for media in medias
            ]
        else:
            image_inputs = {}

        video_inputs = (
            self._preprocess_videos(videos, return_tensors, do_rescale, do_normalize)
            if videos
            else {}
        )

        if text is not None:
            if not isinstance(text, list):
                text = [text]
            text_inputs = self.tokenizer(text)
        else:
            text_inputs = {}

        return BatchFeature(
            data={**text_inputs, **image_inputs, **video_inputs},
            tensor_type=return_tensors,
        )

    def _preprocess_videos(
        self,
        videos: list[tuple[HfVideoItem, dict[str, Any]]],
        return_tensors: str | TensorType | None,
        do_rescale: bool,
        do_normalize: bool,
    ) -> dict[str, torch.Tensor]:
        """Sample each video into timestamped chunks of up to 4 frames."""
        cfg = self.image_processor.media_proc_cfg
        chunk_size = cfg["temporal_merge_kernel_size"]
        medias: list[dict[str, Any]] = []
        timestamps: list[int] = []
        sizes: list[list[int]] = []
        num_chunks: list[int] = []

        for frames, metadata in videos:
            num_decoded = len(frames)
            if num_decoded == 0:
                raise ValueError("Video input has no frames")
            # A missing fps fails the plan with a request error.
            fps = float(metadata.get("fps") or 0)
            indices = list(metadata.get("frames_indices") or range(num_decoded))
            if len(indices) != num_decoded:
                raise ValueError(
                    f"Video metadata lists {len(indices)} frame indices for "
                    f"{num_decoded} frames"
                )
            height, width = _frame_to_hwc(frames[0]).shape[:2]
            count = len(
                plan_kimi_k3_video(
                    int(metadata.get("total_num_frames", num_decoded)),
                    fps,
                    width,
                    height,
                    cfg,
                )
            )
            positions = (
                np.linspace(0, num_decoded - 1, count).round().astype(int).tolist()
                if num_decoded > count
                else list(range(num_decoded))
            )

            chunks = [
                positions[start : start + chunk_size]
                for start in range(0, len(positions), chunk_size)
            ]
            for chunk in chunks:
                medias.append(
                    {
                        "type": "video_chunk",
                        "video_chunk": [_frame_to_pil(frames[i]) for i in chunk],
                    }
                )
                timestamps.append(round(indices[chunk[0]] * 1_000_000 / fps) // 1000)
            sizes.append([width, height])
            num_chunks.append(len(chunks))

        outputs = self.image_processor.preprocess(
            medias,
            return_tensors=return_tensors,
            do_rescale=do_rescale,
            do_normalize=do_normalize,
        )
        return {
            "video_pixel_values": outputs["pixel_values"],
            "video_grid_thws": outputs["grid_thws"],
            "video_chunk_timestamps": torch.tensor(timestamps, dtype=torch.int64),
            "video_sizes": torch.tensor(sizes, dtype=torch.int64),
            "video_num_chunks": torch.tensor(num_chunks, dtype=torch.int64),
        }
