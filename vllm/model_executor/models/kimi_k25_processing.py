# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Standalone multi-modal preprocessing for Kimi-K2.5.

EDITING THIS FILE
-----------------
This is intentionally a *single file* that bundles together every step of
Kimi-K2.5's vision preprocessing pipeline:

  1. Loading per-model media-processing configuration
     (`load_media_proc_cfg`, `DEFAULT_MEDIA_PROC_CFG`).
  2. Pure NumPy / PIL image and video math
     (`navit_resize_image`, `navit_resize_video`, `patchify_image`,
     `normalize_image`, `image_to_resized_padded_np`, ...).
  3. Per-media token-count and tensor builders
     (`count_media_tokens`, `process_single_media`,
     `preprocess_media_batch`).
  4. Prompt-side token expansion (`expand_media_tokens_in_input_ids`).
  5. The thin glue that plugs into vLLM's serving runtime
     (`KimiK25ProcessingInfo`, `KimiK25DummyInputsBuilder`,
     `KimiK25MultiModalProcessor`).

Most of the math here is a direct Python port of the upstream Hugging Face
`KimiK25VisionProcessor`. The intent is that algorithm engineers can change
preprocessing behaviour by modifying *only this file*, without having to
study or override vLLM's multi-modal abstractions.

The model definition in
[`kimi_k25.py`][vllm.model_executor.models.kimi_k25] imports the three glue
classes from this file and registers them via
`MULTIMODAL_REGISTRY.register_processor`; nothing else in vLLM needs to
know about Kimi-K2.5 preprocessing.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Annotated, Any, Literal

import numpy as np
import torch
from PIL import Image
from transformers import BatchFeature

from vllm.config.multimodal import BaseDummyOptions
from vllm.inputs import MultiModalDataDict
from vllm.logger import init_logger
from vllm.multimodal.inputs import MultiModalFieldConfig, MultiModalKwargsItems
from vllm.multimodal.parse import MultiModalDataItems
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder,
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    InputProcessingContext,
    PromptReplacement,
    PromptUpdate,
)
from vllm.transformers_utils.configs.kimi_k25 import KimiK25Config
from vllm.transformers_utils.repo_utils import get_hf_file_to_dict
from vllm.utils.tensor_schema import TensorSchema, TensorShape

logger = init_logger(__name__)


# ---------------------------------------------------------------------------
# Section A. Media-processing configuration
# ---------------------------------------------------------------------------
#
# `media_proc_cfg` is a flat dict of preprocessing parameters that lives in
# the checkpoint's `preprocessor_config.json`. The defaults below match the
# upstream Kimi-K2.5 release and are used as a fallback when the field is
# missing, so this module never has to fetch HF custom code at runtime.

DEFAULT_MEDIA_PROC_CFG: dict[str, Any] = {
    "patch_size": 14,
    "merge_kernel_size": 2,
    "in_patch_limit": 4096,
    "patch_limit_on_one_side": 64,
    "fixed_output_tokens": None,
    "in_patch_limit_each_frame": None,
    "in_patch_limit_video": None,
    "max_num_frames_each_video": None,
    "fixed_output_tokens_each_frame": None,
    "sample_fps": 1.0,
    "temporal_merge_kernel_size": 4,
    "timestamp_mode": "hh:mm:ss.fff",
    "image_mean": [0.5, 0.5, 0.5],
    "image_std": [0.5, 0.5, 0.5],
}


def load_media_proc_cfg(
    model: str,
    *,
    revision: str | None = None,
) -> dict[str, Any]:
    """Read `media_proc_cfg` from a model's `preprocessor_config.json`.

    This deliberately reads the JSON file directly rather than going through
    `AutoImageProcessor` / `trust_remote_code`, because the upstream Kimi-K2.5
    `preprocessor_config.json` has no `image_processor_type` field and is
    therefore invisible to transformers' auto-loader. More importantly, we
    want to *own* the preprocessing logic here rather than rely on whatever
    HF code happens to ship alongside the weights.
    """
    raw = get_hf_file_to_dict("preprocessor_config.json", model, revision) or {}
    media_proc_cfg = dict(DEFAULT_MEDIA_PROC_CFG)
    media_proc_cfg.update(raw.get("media_proc_cfg", {}))
    # `merge_kernel_size` is sometimes stored as a tuple in HF configs but
    # only the scalar tile size is used in the math below. Normalize to int.
    mk = media_proc_cfg["merge_kernel_size"]
    if isinstance(mk, (list, tuple)):
        media_proc_cfg["merge_kernel_size"] = int(mk[0])
    return media_proc_cfg


# ---------------------------------------------------------------------------
# Section B. Pure preprocessing math (no vLLM deps)
# ---------------------------------------------------------------------------


def navit_resize_image(
    width: int,
    height: int,
    patch_size: int,
    merge_kernel_size: int,
    in_patch_limit: int,
    patch_limit_on_one_side: int,
    fixed_output_tokens: int | None,
) -> dict[str, int]:
    """NaViT-style resize for a single image.

    Returns the target inner size (`new_width`/`new_height`), the padding
    needed to align with patches (`pad_width`/`pad_height`), the number of
    output tokens, and `sampled_nframes` (always 1 for an image).
    """
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


def real_sample_fps_and_max_num_frames(
    type_name: str,
    sample_fps: float,
    max_num_frames_each_video: int | None,
) -> tuple[float, int | None]:
    """For `video_chunk` the caller has already sampled the frames, so we
    short-circuit the fps math and act as a no-op."""
    if type_name == "video":
        return sample_fps, max_num_frames_each_video
    if type_name == "video_chunk":
        return math.inf, None
    return math.inf, None


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
    """NaViT-style resize that accounts for the temporal axis."""
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


def _ensure_pil(image: Any) -> Image.Image:
    """Best-effort conversion of arbitrary image data to a PIL RGB image."""
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    raise TypeError(
        f"Expected a PIL.Image.Image for Kimi-K2.5 preprocessing, "
        f"got {type(image)!r}. Decode the image upstream first."
    )


def image_to_resized_padded_np(
    image: Image.Image,
    new_width: int,
    new_height: int,
    pad_width: int,
    pad_height: int,
) -> np.ndarray:
    """Resize a PIL image with bicubic, then bottom/right-pad with zeros."""
    image = image.resize((new_width, new_height), resample=Image.Resampling.BICUBIC)
    arr = np.asarray(image)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    arr = np.pad(
        arr,
        ((0, pad_height), (0, pad_width), (0, 0)),
        mode="constant",
        constant_values=0,
    )
    return arr


def normalize_image(
    x: np.ndarray,
    mean: np.ndarray,
    std_inv: np.ndarray,
    dtype: np.dtype = np.float32,
) -> np.ndarray:
    """uint8 [0,255] -> float32 normalized."""
    x = (x / 255.0).astype(dtype)
    x -= mean
    x *= std_inv
    return x


def patchify_image(pixel_values: np.ndarray, patch_size: int) -> dict[str, np.ndarray]:
    """Reshape (T, H, W, C) into (T*Hp*Wp, C, ps, ps) plus a 3-tuple grid."""
    T, H, W, C = pixel_values.shape
    assert C == 3, "pixel_values must have 3 channels"
    patches = pixel_values.reshape(
        T, H // patch_size, patch_size, W // patch_size, patch_size, C
    )
    patches = patches.transpose(0, 1, 3, 5, 2, 4)
    patches = patches.reshape(-1, C, patch_size, patch_size)
    grid_thw = np.array([T, H // patch_size, W // patch_size], dtype=np.int64)
    return {"pixel_values": patches, "grid_thw": grid_thw}


# ---------------------------------------------------------------------------
# Section C. Per-media operations
# ---------------------------------------------------------------------------


def get_resize_config(
    media: Mapping[str, Any], cfg: Mapping[str, Any]
) -> dict[str, int]:
    """Compute the NaViT resize config for one image or video_chunk."""
    if media["type"] == "image":
        w, h = _ensure_pil(media["image"]).size
        return navit_resize_image(
            w,
            h,
            cfg["patch_size"],
            cfg["merge_kernel_size"],
            cfg["in_patch_limit"],
            cfg["patch_limit_on_one_side"],
            cfg["fixed_output_tokens"],
        )
    if media["type"] == "video_chunk":
        frames = media["video_chunk"]
        first = _ensure_pil(frames[0])
        width, height = first.size
        sample_fps, max_frames = real_sample_fps_and_max_num_frames(
            "video_chunk", cfg["sample_fps"], cfg["max_num_frames_each_video"]
        )
        in_patch_limit_each_frame = (
            cfg["in_patch_limit_each_frame"] or cfg["in_patch_limit"]
        )
        return navit_resize_video(
            width,
            height,
            nframes=len(frames),
            avg_fps=1.0,
            sample_fps=sample_fps,
            patch_size=cfg["patch_size"],
            merge_kernel_size=cfg["merge_kernel_size"],
            in_patch_limit_each_frame=in_patch_limit_each_frame,
            patch_limit_on_one_side=cfg["patch_limit_on_one_side"],
            in_patch_limit_total=cfg["in_patch_limit_video"],
            max_num_frames_each_video=max_frames,
            fixed_output_tokens_each_frame=cfg["fixed_output_tokens_each_frame"],
        )
    raise ValueError(f"Unsupported media type: {media['type']!r}")


def count_media_tokens(media: Mapping[str, Any], cfg: Mapping[str, Any]) -> int:
    """Number of `<|media_pad|>` tokens to splice into the prompt for `media`."""
    return get_resize_config(media, cfg)["num_tokens"]


def process_single_media(
    media: Mapping[str, Any], cfg: Mapping[str, Any]
) -> dict[str, np.ndarray]:
    """Run the full per-item pipeline (resize -> pad -> normalize -> patchify).

    Returns a dict with `pixel_values` (float32, NCHW patches) and `grid_thw`
    (int64, 3-tuple).
    """
    resize = get_resize_config(media, cfg)
    new_w, new_h = resize["new_width"], resize["new_height"]
    pad_w, pad_h = resize["pad_width"], resize["pad_height"]

    if media["type"] == "image":
        arr = image_to_resized_padded_np(
            _ensure_pil(media["image"]), new_w, new_h, pad_w, pad_h
        )
        thwc = arr[np.newaxis, ...]
    elif media["type"] == "video_chunk":
        frames_np = [
            image_to_resized_padded_np(_ensure_pil(f), new_w, new_h, pad_w, pad_h)
            for f in media["video_chunk"]
        ]
        thwc = np.stack(frames_np, axis=0)
    else:
        raise ValueError(f"Unsupported media type: {media['type']!r}")

    mean = np.asarray(cfg["image_mean"], dtype=np.float32)
    std_inv = 1.0 / np.asarray(cfg["image_std"], dtype=np.float32)
    thwc = normalize_image(thwc, mean, std_inv)
    return patchify_image(thwc, cfg["patch_size"])


def preprocess_media_batch(
    medias: Sequence[Mapping[str, Any]], cfg: Mapping[str, Any]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run `process_single_media` over a batch and stack the results.

    Output shapes:
      - `pixel_values`: `(sum(T*Hp*Wp), 3, patch_size, patch_size)`
      - `grid_thws`: `(num_media, 3)`
    """
    if not medias:
        ps = cfg["patch_size"]
        return (
            torch.empty(0, 3, ps, ps, dtype=torch.float32),
            torch.empty(0, 3, dtype=torch.int64),
        )

    processed = [process_single_media(m, cfg) for m in medias]
    pixel_values = torch.cat(
        [torch.from_numpy(p["pixel_values"]) for p in processed], dim=0
    )
    grid_thws = torch.stack([torch.from_numpy(p["grid_thw"]) for p in processed], dim=0)
    return pixel_values, grid_thws


# ---------------------------------------------------------------------------
# Section D. Prompt-side token expansion
# ---------------------------------------------------------------------------


def expand_media_tokens_in_input_ids(
    input_ids: list[int],
    num_tokens_per_media: list[int],
    media_token_id: int,
) -> list[int]:
    """Replace each `media_token_id` in `input_ids` with N copies of itself,
    where N is the per-media token count consumed left-to-right."""
    queue = list(num_tokens_per_media)
    out: list[int] = []
    for token in input_ids:
        if token == media_token_id and queue:
            out.extend([media_token_id] * queue.pop(0))
        else:
            out.append(token)
    if queue:
        raise ValueError(
            f"{len(queue)} media items were left unexpanded; the prompt is "
            "missing matching <|media_pad|> placeholders."
        )
    return out


# ---------------------------------------------------------------------------
# Section E. Model-facing input schema
# ---------------------------------------------------------------------------


class KimiK25MediaPixelInputs(TensorSchema):
    """Schema for the pixel inputs the Kimi-K2.5 model forward expects.

    Dimensions:
        - `np`: total number of patches (sum over all media items in the batch)
        - `ps`: patch size (e.g. 14)
        - `nm`: number of media items
    """

    type: Literal["pixel_values"] = "pixel_values"
    pixel_values: Annotated[
        torch.Tensor | list[torch.Tensor], TensorShape("np", 3, "ps", "ps")
    ]
    grid_thws: Annotated[torch.Tensor, TensorShape("nm", 3)]


# ---------------------------------------------------------------------------
# Section F. Dummy-input dimensions used for profiling.
# ---------------------------------------------------------------------------


@dataclass
class MaxImageTokenMeta:
    width: int = 3000
    height: int = 3000


# ---------------------------------------------------------------------------
# Section G. vLLM glue (intentionally thin)
# ---------------------------------------------------------------------------


class KimiK25ProcessingInfo(BaseProcessingInfo):
    """Model-level metadata vLLM needs to build the multi-modal pipeline.

    Everything here is read once at engine startup; the hot path lives in
    :class:`KimiK25MultiModalProcessor` below.
    """

    def __init__(self, ctx: InputProcessingContext) -> None:
        super().__init__(ctx)
        self.hf_config = self.get_hf_config()
        tokenizer = self.get_tokenizer()

        # Resolve <|media_pad|> token id. Prefer the tokenizer mapping, which
        # is authoritative across transformers v4/v5 retokenization changes,
        # and fall back to the value baked into config.json.
        config_token_id = self.hf_config.media_placeholder_token_id
        resolved_token_id = tokenizer.convert_tokens_to_ids("<|media_pad|>")
        is_valid_resolved = isinstance(resolved_token_id, int) and (
            tokenizer.unk_token_id is None
            or resolved_token_id != tokenizer.unk_token_id
        )
        if is_valid_resolved and resolved_token_id != config_token_id:
            logger.warning_once(
                "Kimi-K2.5 config.media_placeholder_token_id (%d) disagrees "
                "with tokenizer mapping for <|media_pad|> (%d). "
                "Using tokenizer value.",
                config_token_id,
                resolved_token_id,
            )
            self.hf_config.media_placeholder_token_id = resolved_token_id
            self.media_token_id = resolved_token_id
        else:
            self.media_token_id = config_token_id
        self.media_token = tokenizer.decode(self.media_token_id)

        # Read preprocessor_config.json directly so we don't trigger HF's
        # `trust_remote_code` path.
        self.media_proc_cfg = load_media_proc_cfg(
            ctx.model_config.model, revision=ctx.model_config.revision
        )

    def get_hf_config(self) -> KimiK25Config:
        return self.ctx.get_hf_config(KimiK25Config)

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"vision_chunk": None}

    @property
    def num_frames_per_chunk(self) -> int:
        return int(self.media_proc_cfg["temporal_merge_kernel_size"])


class KimiK25DummyInputsBuilder(BaseDummyInputsBuilder[KimiK25ProcessingInfo]):
    """Builds the dummy inputs used by vLLM during profiling."""

    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        return self.info.media_token * mm_counts.get("vision_chunk", 0)

    def _largest_dummy_item(self) -> dict[str, Any]:
        meta = MaxImageTokenMeta()
        dummy_frames = self._get_dummy_images(
            height=meta.height,
            width=meta.width,
            num_images=self.info.num_frames_per_chunk,
        )
        video_item = {"type": "video_chunk", "video_chunk": dummy_frames}
        video_tokens = count_media_tokens(video_item, self.info.media_proc_cfg)

        image_item = {
            "type": "image",
            "image": self._get_dummy_images(
                height=meta.height, width=meta.width, num_images=1
            )[0],
        }
        image_tokens = count_media_tokens(image_item, self.info.media_proc_cfg)
        return video_item if video_tokens >= image_tokens else image_item

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions],
    ) -> MultiModalDataDict:
        return {"vision_chunk": [self._largest_dummy_item()]}


class KimiK25MultiModalProcessor(BaseMultiModalProcessor[KimiK25ProcessingInfo]):
    """Multi-modal processor that *fully* bypasses HF's image processor.

    The override of `_call_hf_processor` is the single integration point
    with vLLM's runtime: it takes the prompt text and a list of media
    dicts, and returns a `BatchFeature` with the expanded `input_ids`
    plus the model-side tensors (`pixel_values`, `grid_thws`).
    """

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        # `pixel_values` is a flat concatenation of every media item's
        # patches; `grid_thws[i] = (T, Hp, Wp)` tells vLLM how many of
        # those patches belong to media item `i`.
        grid_thws = hf_inputs.get("grid_thws", torch.empty((0, 3)))
        grid_sizes = grid_thws.prod(-1)
        return dict(
            pixel_values=MultiModalFieldConfig.flat_from_sizes(
                "vision_chunk", grid_sizes
            ),
            grid_thws=MultiModalFieldConfig.batched("vision_chunk"),
        )

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        # The framework hands us `mm_data["vision_chunks"]` (plural follows
        # the modality name; see `ProcessorBatchItems.get_processor_data`).
        medias: list[Mapping[str, Any]] = list(mm_data.get("vision_chunks", []) or [])
        cfg = self.info.media_proc_cfg
        tokenizer = self.info.get_tokenizer()
        media_token_id = self.info.media_token_id
        model_dtype = self.info.ctx.model_config.dtype

        token_inputs = tokenizer(prompt)
        input_ids: list[int] = list(token_inputs["input_ids"])

        if medias:
            num_tokens_per_media = [count_media_tokens(m, cfg) for m in medias]
            input_ids = expand_media_tokens_in_input_ids(
                input_ids, num_tokens_per_media, media_token_id
            )
            pixel_values, grid_thws = preprocess_media_batch(medias, cfg)
            # `transformers.BatchFeature` would normally cast floats to the
            # model dtype for us; we mirror that here since we sidestepped
            # `info.ctx.call_hf_processor`.
            pixel_values = pixel_values.to(dtype=model_dtype)
            data: dict[str, Any] = {
                "input_ids": [input_ids],
                "pixel_values": pixel_values,
                "grid_thws": grid_thws,
            }
        else:
            data = {"input_ids": [input_ids]}

        # Mirror the behaviour of `info.ctx.call_hf_processor`, which always
        # sets `return_tensors="pt"`. This converts our list-of-lists
        # `input_ids` into a (1, L) tensor that the base class will then
        # unpack into a single prompt.
        return BatchFeature(
            data=data, tensor_type=tok_kwargs.get("return_tensors", "pt")
        )

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, Any],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        media_token_id = self.info.media_token_id
        cfg = self.info.media_proc_cfg
        # `mm_items["vision_chunk"]` is a `VisionChunkProcessorItems` wrapping
        # the raw media dicts; `.get(i)` returns the i-th dict back.
        medias = mm_items.get("vision_chunk") if "vision_chunk" in mm_items else None

        def get_replacement(item_idx: int):
            assert medias is not None
            return [media_token_id] * count_media_tokens(medias.get(item_idx), cfg)

        return [
            PromptReplacement(
                modality="vision_chunk",
                target=[media_token_id],
                replacement=get_replacement,
            ),
        ]
