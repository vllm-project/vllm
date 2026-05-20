# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Front-end multi-modal preprocessing for Kimi-K2.5.

DESIGN GOAL
-----------
Run the **entire** Kimi-K2.5 vision preprocessing pipeline *outside* of
vLLM's multi-modal registry, then hand the engine an already-rendered
`MultiModalInput` dict — the same shape that vLLM's renderer would have
produced. This mirrors the way SGLang sits in front of a vLLM gRPC
backend: the front-end owns tokenization and vision preprocessing, the
back-end is only responsible for inference.

Concretely, [`KimiK25Preprocessor.preprocess`][] takes a raw prompt
string plus a list of vision chunks and returns a dict that can be fed
straight to [`vllm.LLMEngine.add_request`][] (the gRPC-style entry
point), bypassing [`BaseRenderer._process_multimodal`][] and therefore
the entire [`MULTIMODAL_REGISTRY`][vllm.multimodal.MULTIMODAL_REGISTRY]
path.

Because nothing here inherits from `BaseProcessingInfo`,
`BaseDummyInputsBuilder`, or `BaseMultiModalProcessor`, algorithm
engineers can change Kimi-K2.5's vision preprocessing by editing
**this file only**, without learning any of vLLM's multi-modal
abstractions.

FILE LAYOUT
-----------
  - Section A — `preprocessor_config.json` loader and defaults
    (`load_media_proc_cfg`, `DEFAULT_MEDIA_PROC_CFG`).
  - Section B — Pure NumPy / PIL preprocessing math
    (`navit_resize_image`, `navit_resize_video`, `patchify_image`,
    `normalize_image`, `image_to_resized_padded_np`).
  - Section C — Per-media operations
    (`count_media_tokens`, `process_single_media`,
    `preprocess_media_batch`).
  - Section D — Prompt-side token expansion and placeholder ranges
    (`expand_media_tokens_in_input_ids`, `find_placeholder_ranges`).
  - Section E — Top-level front-end API
    (`KimiK25Preprocessor`).

Sections B–D are a direct Python port of the upstream Hugging Face
`KimiK25VisionProcessor`; modifying preprocessing behaviour usually
means editing only those.

USAGE
-----
    from vllm import LLM, SamplingParams
    from vllm.model_executor.models.kimi_k25_frontend import (
        KimiK25Preprocessor,
    )

    preprocessor = KimiK25Preprocessor.from_pretrained(
        "moonshotai/Kimi-K2.5", model_dtype="bfloat16",
    )
    llm = LLM(model="moonshotai/Kimi-K2.5", trust_remote_code=True)

    engine_input = preprocessor.preprocess(
        prompt=(
            "<|im_user|>user<|media_begin|>image<|media_content|>"
            "<|media_pad|><|media_end|>What is in this image?<|im_end|>"
            "<|im_assistant|>assistant<|im_middle|>"
        ),
        vision_chunks=[{"type": "image", "image": pil_image}],
    )
    llm.llm_engine.add_request("0", engine_input, SamplingParams(max_tokens=64))
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import torch
from PIL import Image
from transformers import BatchFeature

from vllm.inputs import MultiModalInput, mm_input
from vllm.logger import init_logger
from vllm.multimodal.hasher import MultiModalHasher
from vllm.multimodal.inputs import (
    MultiModalFieldConfig,
    MultiModalKwargsItems,
    PlaceholderRange,
)
from vllm.tokenizers import TokenizerLike, cached_tokenizer_from_config
from vllm.transformers_utils.repo_utils import get_hf_file_to_dict

if TYPE_CHECKING:
    from vllm.config import ModelConfig

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

    Reads the JSON file directly rather than going through
    `AutoImageProcessor` / `trust_remote_code`, because the upstream
    Kimi-K2.5 `preprocessor_config.json` has no `image_processor_type`
    field and is therefore invisible to transformers' auto-loader.
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
    dtype: type = np.float32,
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
        in_patch_limit_each_frame = (
            cfg["in_patch_limit_each_frame"] or cfg["in_patch_limit"]
        )
        # `video_chunk` callers have already sampled the frames they want, so
        # we short-circuit the fps math inside ``navit_resize_video`` by
        # passing ``sample_fps=inf`` and ``max_num_frames_each_video=None``.
        return navit_resize_video(
            width,
            height,
            nframes=len(frames),
            avg_fps=1.0,
            sample_fps=math.inf,
            patch_size=cfg["patch_size"],
            merge_kernel_size=cfg["merge_kernel_size"],
            in_patch_limit_each_frame=in_patch_limit_each_frame,
            patch_limit_on_one_side=cfg["patch_limit_on_one_side"],
            in_patch_limit_total=cfg["in_patch_limit_video"],
            max_num_frames_each_video=None,
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


def find_placeholder_ranges(
    input_ids: list[int],
    media_token_id: int,
    num_tokens_per_media: list[int],
) -> list[PlaceholderRange]:
    """Locate the per-item placeholder blocks in an expanded `input_ids`.

    Called right after [`expand_media_tokens_in_input_ids`][]; each
    `<|media_pad|>` block is already `num_tokens_per_media[i]` tokens long,
    so we just walk the list once and record the start offset of each block.
    """
    ranges: list[PlaceholderRange] = []
    i = 0
    while i < len(input_ids):
        if input_ids[i] == media_token_id and len(ranges) < len(num_tokens_per_media):
            n = num_tokens_per_media[len(ranges)]
            ranges.append(PlaceholderRange(offset=i, length=n, is_embed=None))
            i += n
        else:
            i += 1
    if len(ranges) != len(num_tokens_per_media):
        raise RuntimeError(
            f"Expected {len(num_tokens_per_media)} placeholder ranges, "
            f"found {len(ranges)}; expansion is out of sync with input_ids."
        )
    return ranges


# ---------------------------------------------------------------------------
# Section E. Top-level front-end API
# ---------------------------------------------------------------------------


def _build_mm_fields_config(
    grid_thws: torch.Tensor,
) -> Mapping[str, MultiModalFieldConfig]:
    """Per-item slicing rules for `pixel_values` and `grid_thws`.

    `pixel_values` is a flat concatenation of every media item's patches
    (shape `(sum(T*Hp*Wp), 3, ps, ps)`); `grid_thws[i] = (T, Hp, Wp)` lets
    vLLM compute how many of those patches belong to media item `i`.
    """
    grid_sizes = grid_thws.prod(-1)
    return dict(
        pixel_values=MultiModalFieldConfig.flat_from_sizes("vision_chunk", grid_sizes),
        grid_thws=MultiModalFieldConfig.batched("vision_chunk"),
    )


@dataclass
class KimiK25Preprocessor:
    """Stand-alone front-end that turns raw text + vision chunks into a
    fully-rendered [`MultiModalInput`][vllm.inputs.MultiModalInput].

    The returned dict is the shape vLLM's renderer would have produced
    after running `BaseMultiModalProcessor.apply()`, so it can be fed
    straight to `LLMEngine.add_request(...)` — completely bypassing the
    [`MULTIMODAL_REGISTRY`][vllm.multimodal.MULTIMODAL_REGISTRY] path.

    Construct via [`KimiK25Preprocessor.from_pretrained`][] for the common
    case, or via the dataclass constructor when you want to inject a custom
    tokenizer / media config / dtype (e.g. for testing).
    """

    model_id: str
    tokenizer: TokenizerLike
    media_token_id: int
    media_proc_cfg: Mapping[str, Any]
    model_dtype: torch.dtype = torch.bfloat16
    # Optional reference to the originating ModelConfig. Required for the
    # ``preprocess_cmpl`` batch entry point (it needs ``parse_model_prompt``);
    # left ``None`` for direct ``from_pretrained`` users that only call
    # ``preprocess``.
    model_config: "ModelConfig | None" = field(default=None, repr=False)

    @classmethod
    def from_model_config(cls, model_config: "ModelConfig") -> KimiK25Preprocessor:
        """Construct a preprocessor from a vLLM ``ModelConfig``.

        Mirrors ``from_pretrained`` but reuses the tokenizer cached by the
        running ``LLM`` instance, which avoids a second HF download on the
        ``LLM._preprocess_cmpl`` fast path.
        """
        tokenizer = cached_tokenizer_from_config(model_config)
        media_token_id = tokenizer.convert_tokens_to_ids("<|media_pad|>")
        unk_id = tokenizer.unk_token_id
        if not isinstance(media_token_id, int) or (
            unk_id is not None and media_token_id == unk_id
        ):
            raise ValueError(
                f"Could not resolve '<|media_pad|>' in tokenizer for "
                f"{model_config.model!r}; got {media_token_id!r}. Make sure "
                "the K2.5 special tokens are present in the tokenizer."
            )

        model_dtype = model_config.dtype
        if isinstance(model_dtype, str):
            model_dtype = getattr(torch, model_dtype)

        return cls(
            model_id=model_config.model,
            tokenizer=tokenizer,
            media_token_id=int(media_token_id),
            media_proc_cfg=load_media_proc_cfg(
                model_config.model, revision=model_config.revision
            ),
            model_dtype=cast(torch.dtype, model_dtype),
            model_config=model_config,
        )

    @classmethod
    def from_pretrained(
        cls,
        model: str,
        *,
        revision: str | None = None,
        tokenizer: TokenizerLike | None = None,
        trust_remote_code: bool = False,
        model_dtype: torch.dtype | str = torch.bfloat16,
    ) -> KimiK25Preprocessor:
        """Load tokenizer + `preprocessor_config.json` from a HF repo / local
        directory. The tokenizer is the source of truth for `<|media_pad|>`
        (transformers v5 may remap token IDs vs `config.json`)."""
        if tokenizer is None:
            from vllm.config import ModelConfig

            model_config = ModelConfig(
                model=model,
                revision=revision,
                trust_remote_code=trust_remote_code,
            )
            tokenizer = cached_tokenizer_from_config(model_config)

        media_token_id = tokenizer.convert_tokens_to_ids("<|media_pad|>")
        unk_id = tokenizer.unk_token_id
        if not isinstance(media_token_id, int) or (
            unk_id is not None and media_token_id == unk_id
        ):
            raise ValueError(
                f"Could not resolve '<|media_pad|>' in tokenizer for {model!r}; "
                f"got {media_token_id!r}. Make sure the K2.5 special tokens are "
                "present in the tokenizer."
            )

        if isinstance(model_dtype, str):
            model_dtype = getattr(torch, model_dtype)

        return cls(
            model_id=model,
            tokenizer=tokenizer,
            media_token_id=int(media_token_id),
            media_proc_cfg=load_media_proc_cfg(model, revision=revision),
            model_dtype=cast(torch.dtype, model_dtype),
        )

    # ----- main entry point -------------------------------------------------

    def preprocess(
        self,
        prompt: str | list[int],
        vision_chunks: Sequence[Mapping[str, Any]] | None = None,
        *,
        cache_salt: str | None = None,
    ) -> MultiModalInput:
        """End-to-end preprocessing: prompt + media -> `MultiModalInput`.

        Steps:
          1. Tokenize the prompt (lists of ints are passed through unchanged).
          2. Run the pure-Python media pipeline -- per-item token counts plus
             the flat `(pixel_values, grid_thws)` tensors, packed into
             `MultiModalKwargsItems`.
          3. Splice the expanded `<|media_pad|>` runs into `input_ids` and
             record one `PlaceholderRange` per item.
          4. Hash the raw media items (for downstream caching) and bundle
             everything into the `MultiModalInput` dict that the engine
             consumes via `LLMEngine.add_request`.

        Args:
            prompt: either a prompt string (will be tokenized here) or an
                already-tokenized list of ints with `<|media_pad|>` markers
                still in place (one per media item).
            vision_chunks: list of dicts like
                `{"type": "image", "image": PIL.Image}` or
                `{"type": "video_chunk", "video_chunk": [PIL.Image, ...]}`.
            cache_salt: optional cache-salt forwarded to the engine.
        """
        medias = list(vision_chunks or [])

        if isinstance(prompt, str):
            input_ids = list(self.tokenizer.encode(prompt))
            prompt_text: str | None = prompt
        else:
            input_ids = list(prompt)
            prompt_text = None

        mm_kwargs: MultiModalKwargsItems
        mm_placeholders: dict[str, list[PlaceholderRange]] = {}

        if medias:
            num_tokens_per_media = [
                count_media_tokens(m, self.media_proc_cfg) for m in medias
            ]
            pixel_values, grid_thws = preprocess_media_batch(
                medias, self.media_proc_cfg
            )
            # The HF processor path would cast floats to the model dtype
            # automatically; mirror that here since we bypassed it.
            pixel_values = pixel_values.to(dtype=self.model_dtype)

            hf_inputs = BatchFeature(
                data={"pixel_values": pixel_values, "grid_thws": grid_thws},
                tensor_type="pt",
            )
            mm_kwargs = MultiModalKwargsItems.from_hf_inputs(
                hf_inputs, _build_mm_fields_config(grid_thws)
            )

            input_ids = expand_media_tokens_in_input_ids(
                input_ids, num_tokens_per_media, self.media_token_id
            )
            mm_placeholders["vision_chunk"] = find_placeholder_ranges(
                input_ids, self.media_token_id, num_tokens_per_media
            )
            mm_hashes: dict[str, list[str]] = {
                "vision_chunk": [
                    MultiModalHasher.hash_kwargs(
                        model_id=self.model_id, vision_chunk=item
                    )
                    for item in medias
                ]
            }
        else:
            mm_kwargs = MultiModalKwargsItems({})
            mm_hashes = {}

        return mm_input(
            prompt_token_ids=input_ids,
            mm_kwargs=mm_kwargs,
            mm_hashes=mm_hashes,
            mm_placeholders=mm_placeholders,
            prompt=prompt_text,
            cache_salt=cache_salt,
        )

    # ----- batch entry point used by ``LLM._preprocess_cmpl`` ---------------

    def preprocess_cmpl(
        self,
        prompts: Sequence[Any],
        tokenization_kwargs: Mapping[str, Any] | None = None,
    ) -> list[MultiModalInput]:
        """Batch wrapper for ``preprocess``: ``LLM._preprocess_cmpl`` fast path.

        Accepts the same prompt formats as ``LLM.generate()`` (string,
        ``TokensPrompt``, ``TextPrompt``, ...) and returns a list of
        ``MultiModalInput`` dicts ready to be passed straight to
        ``LLMEngine.add_request``. Vision data is read from the standard
        ``multi_modal_data["vision_chunk"]`` key — the same shape that
        ``KimiK25MultiModalProcessor`` consumes via the multi-modal registry.
        """
        from vllm.renderers.inputs.preprocess import parse_model_prompt

        if self.model_config is None:
            raise RuntimeError(
                "KimiK25Preprocessor.preprocess_cmpl requires "
                "``model_config`` to be set; construct via "
                "``KimiK25Preprocessor.from_model_config(model_config)``."
            )

        tok_kwargs = dict(tokenization_kwargs or {})
        results: list[MultiModalInput] = []

        for prompt in prompts:
            parsed = parse_model_prompt(self.model_config, prompt)

            if "prompt_embeds" in parsed:
                raise NotImplementedError(
                    "Kimi-K2.5 preprocess_cmpl fast path does not support "
                    "prompt_embeds; pass text or token IDs instead."
                )

            if "prompt_token_ids" in parsed:
                prompt_for_preproc: str | list[int] = list(
                    parsed["prompt_token_ids"]
                )
            else:
                text = parsed["prompt"]
                # When tokenization kwargs are provided, tokenize here so the
                # caller's options actually take effect; otherwise delegate to
                # ``preprocess`` so it can also populate the ``prompt`` text
                # field on the resulting ``MultiModalInput``.
                if tok_kwargs:
                    prompt_for_preproc = list(
                        self.tokenizer.encode(text, **tok_kwargs)
                    )
                else:
                    prompt_for_preproc = text

            mm_data = parsed.get("multi_modal_data") or {}
            vision_chunks = mm_data.get("vision_chunk")
            cache_salt = parsed.get("cache_salt")

            results.append(
                self.preprocess(
                    prompt=prompt_for_preproc,
                    vision_chunks=vision_chunks,
                    cache_salt=cache_salt,
                )
            )

        return results
