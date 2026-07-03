# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Optimized CPU image processor for Kimi-K2.5/K2.6 vision chunks."""

import base64
import io
import math
from typing import Any

import numpy as np
import torch
from PIL import Image
from transformers.image_processing_utils import BaseImageProcessor, BatchFeature
from transformers.utils import TensorType

from vllm.transformers_utils.repo_utils import get_hf_file_to_dict

try:
    from numba import njit, prange
except ImportError:
    njit = None
    prange = range


if njit is not None:

    @njit(parallel=True, cache=True)
    def _write_fused_patches(
        frames: np.ndarray,
        out: np.ndarray,
        out_offset: int,
        new_h: int,
        new_w: int,
        padded_h: int,
        padded_w: int,
        patch_size: int,
        normalize_lut: np.ndarray,
    ) -> None:
        # frames: [T, new_h, new_w, 3] uint8, without padding.
        # out: [total_patches, 3, patch_size, patch_size] float32.
        t_size = frames.shape[0]
        patch_h = padded_h // patch_size
        patch_w = padded_w // patch_size
        total = t_size * padded_h * padded_w * 3
        hwc = padded_h * padded_w * 3
        wc = padded_w * 3

        for linear in prange(total):
            t = linear // hwc
            rem = linear - t * hwc
            y = rem // wc
            rem = rem - y * wc
            x = rem // 3
            c = rem - x * 3

            value = frames[t, y, x, c] if y < new_h and x < new_w else 0

            patch_idx = (
                out_offset
                + t * patch_h * patch_w
                + (y // patch_size) * patch_w
                + (x // patch_size)
            )
            out[patch_idx, c, y % patch_size, x % patch_size] = normalize_lut[value, c]

else:

    def _write_fused_patches(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("numba is required for fused Kimi image preprocessing")


def _load_media_proc_cfg(model: str, revision: str | None = None) -> dict[str, Any]:
    raw = get_hf_file_to_dict("preprocessor_config.json", model, revision) or {}
    cfg = dict(raw["media_proc_cfg"])
    merge_kernel_size = cfg["merge_kernel_size"]
    if isinstance(merge_kernel_size, (list, tuple)):
        cfg["merge_kernel_size"] = int(merge_kernel_size[0])
    return cfg


def is_numba_available() -> bool:
    return njit is not None


def navit_resize_image(
    width: int,
    height: int,
    patch_size: int,
    merge_kernel_size: int,
    in_patch_limit: int,
    patch_limit_on_one_side: int,
    fixed_output_tokens: int | None,
) -> dict[str, int]:
    s1 = math.sqrt(
        in_patch_limit
        / (max(1.0, width // patch_size) * max(1.0, height // patch_size))
    )
    s2 = patch_limit_on_one_side * patch_size / width
    s3 = patch_limit_on_one_side * patch_size / height
    scale = min(1.0, s1, s2, s3)
    new_w = min(max(1, int(width * scale)), patch_limit_on_one_side * patch_size)
    new_h = min(max(1, int(height * scale)), patch_limit_on_one_side * patch_size)

    factor = merge_kernel_size * patch_size
    pad_height = (factor - new_h % factor) % factor
    pad_width = (factor - new_w % factor) % factor

    if fixed_output_tokens is not None:
        num_tokens = fixed_output_tokens
    else:
        token_height = (new_h + pad_height) // factor
        token_width = (new_w + pad_width) // factor
        num_tokens = token_height * token_width

    return {
        "num_tokens": num_tokens,
        "new_width": new_w,
        "new_height": new_h,
        "pad_width": pad_width,
        "pad_height": pad_height,
        "sampled_nframes": 1,
    }


def navit_resize_video(
    width: int,
    height: int,
    nframes: int,
    avg_fps: float,
    sample_fps: float,
    patch_size: int,
    merge_kernel_size: int,
    in_patch_limit_each_frame: int,
    patch_limit_on_one_side: int,
    in_patch_limit_total: int | None,
    max_num_frames_each_video: int | None,
    fixed_output_tokens_each_frame: int | None,
) -> dict[str, int]:
    sample_fps = min(sample_fps, avg_fps)
    sampled_nframes = max(round(nframes * sample_fps / avg_fps), 1)
    if max_num_frames_each_video is not None:
        sampled_nframes = min(sampled_nframes, max_num_frames_each_video)

    if in_patch_limit_total is not None:
        in_patch_limit_each_frame = min(
            round(in_patch_limit_total / sampled_nframes),
            in_patch_limit_each_frame,
        )

    ret = navit_resize_image(
        width,
        height,
        patch_size,
        merge_kernel_size,
        in_patch_limit_each_frame,
        patch_limit_on_one_side,
        fixed_output_tokens_each_frame,
    )
    ret["sampled_nframes"] = sampled_nframes
    return ret


def _to_pil(data: Any) -> Image.Image:
    if hasattr(data, "media") and hasattr(data, "original_bytes"):
        data = data.media
    if isinstance(data, Image.Image):
        return data if data.mode == "RGB" else data.convert("RGB")
    if isinstance(data, str):
        if data.startswith("data:"):
            raw_base64 = data.split(",", 1)[1]
            return Image.open(io.BytesIO(base64.b64decode(raw_base64))).convert("RGB")
        return Image.open(data).convert("RGB")
    if isinstance(data, bytes):
        return Image.open(io.BytesIO(data)).convert("RGB")
    raise ValueError(f"Unsupported data type: {type(data)}")


def _ensure_media_type(media: dict[str, Any]) -> dict[str, Any]:
    if media["type"] == "image":
        media["image"] = _to_pil(media["image"])
        return media
    if media["type"] == "video_chunk":
        media["video_chunk"] = [_to_pil(frame) for frame in media["video_chunk"]]
        return media
    raise ValueError(f"Unsupported media type: {media['type']}")


class KimiK25FusedVisionProcessor(BaseImageProcessor):
    model_type = "kimi_k25"

    def __init__(self, media_proc_cfg: dict[str, Any], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.media_proc_cfg = media_proc_cfg
        self.num_frames_per_chunk = media_proc_cfg["temporal_merge_kernel_size"]
        values = np.arange(256, dtype=np.float32)[:, None]
        image_mean = np.asarray(media_proc_cfg["image_mean"], dtype=np.float32)
        image_std_inv = 1.0 / np.asarray(media_proc_cfg["image_std"], dtype=np.float32)
        self.normalize_lut = (values / 255.0 - image_mean[None, :]) * image_std_inv[
            None, :
        ]

    @classmethod
    def from_model(
        cls,
        model: str,
        revision: str | None = None,
    ) -> "KimiK25FusedVisionProcessor":
        return cls(media_proc_cfg=_load_media_proc_cfg(model, revision))

    def media_tokens_calculator(self, media: dict[str, Any]) -> int:
        media = _ensure_media_type(media)
        ret = self.get_resize_config(media)
        return ret["num_tokens"]

    def get_resize_config(self, media_input: dict[str, Any]) -> dict[str, int]:
        if media_input["type"] == "image":
            width, height = media_input["image"].size
            return navit_resize_image(
                width,
                height,
                self.media_proc_cfg["patch_size"],
                self.media_proc_cfg["merge_kernel_size"],
                self.media_proc_cfg["in_patch_limit"],
                self.media_proc_cfg["patch_limit_on_one_side"],
                self.media_proc_cfg["fixed_output_tokens"],
            )

        if media_input["type"] == "video_chunk":
            frame = media_input["video_chunk"][0]
            width, height = frame.size
            num_frames = len(media_input["video_chunk"])
            in_patch_limit_each_frame = self.media_proc_cfg["in_patch_limit_each_frame"]
            if in_patch_limit_each_frame is None:
                in_patch_limit_each_frame = self.media_proc_cfg["in_patch_limit"]

            return navit_resize_video(
                width,
                height,
                num_frames,
                1.0,
                math.inf,
                self.media_proc_cfg["patch_size"],
                self.media_proc_cfg["merge_kernel_size"],
                in_patch_limit_each_frame,
                self.media_proc_cfg["patch_limit_on_one_side"],
                self.media_proc_cfg["in_patch_limit_video"],
                None,
                self.media_proc_cfg["fixed_output_tokens"],
            )

        raise ValueError(f"Unsupported type: {media_input['type']}")

    @staticmethod
    def resize_image(image: Image.Image, new_width: int, new_height: int) -> np.ndarray:
        image = image.resize((new_width, new_height), resample=Image.Resampling.BICUBIC)
        return np.asarray(image)

    def preprocess(
        self,
        medias: list[dict[str, Any]],
        return_tensors: str | TensorType | None = None,
    ) -> BatchFeature:
        if not isinstance(medias, list):
            medias = [medias]
        if not medias:
            return BatchFeature(data={}, tensor_type=return_tensors)

        if njit is None:
            raise RuntimeError("numba is required for fused Kimi image preprocessing")

        patch_size = int(self.media_proc_cfg["patch_size"])
        prepared = []
        grid_thws_np = np.empty((len(medias), 3), dtype=np.int64)
        total_patches = 0

        for idx, item in enumerate(medias):
            item = _ensure_media_type(item)
            resize_config = self.get_resize_config(item)
            new_width = resize_config["new_width"]
            new_height = resize_config["new_height"]
            pad_width = resize_config["pad_width"]
            pad_height = resize_config["pad_height"]
            padded_width = new_width + pad_width
            padded_height = new_height + pad_height

            if item["type"] == "image":
                image_np = self.resize_image(item["image"], new_width, new_height)
                frames = image_np[np.newaxis, ...]
            elif item["type"] == "video_chunk":
                frames = np.stack(
                    [
                        self.resize_image(frame, new_width, new_height)
                        for frame in item["video_chunk"]
                    ],
                    axis=0,
                )
            else:
                raise ValueError(f"Unsupported type: {item['type']}")

            t_size = frames.shape[0]
            grid_h = padded_height // patch_size
            grid_w = padded_width // patch_size
            grid_thws_np[idx, 0] = t_size
            grid_thws_np[idx, 1] = grid_h
            grid_thws_np[idx, 2] = grid_w

            num_patches = t_size * grid_h * grid_w
            prepared.append(
                (
                    frames,
                    new_height,
                    new_width,
                    padded_height,
                    padded_width,
                    num_patches,
                )
            )
            total_patches += num_patches

        pixel_values_np = np.empty(
            (total_patches, 3, patch_size, patch_size), dtype=np.float32
        )
        out_offset = 0
        for (
            frames,
            new_height,
            new_width,
            padded_height,
            padded_width,
            num_patches,
        ) in prepared:
            _write_fused_patches(
                frames,
                pixel_values_np,
                out_offset,
                new_height,
                new_width,
                padded_height,
                padded_width,
                patch_size,
                self.normalize_lut,
            )
            out_offset += num_patches

        data = {
            "pixel_values": torch.from_numpy(pixel_values_np),
            "grid_thws": torch.from_numpy(grid_thws_np),
        }
        return BatchFeature(data=data, tensor_type=return_tensors)
