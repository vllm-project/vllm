# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Inference-only LLaVA-OneVision-2 (OV2) model for vLLM.

Architecture notes:

  * LLM backbone is plain Qwen3-8B with 1-D position_ids (no M-RoPE).
  * Vision tower removes the CLS token (no class_embedding/class_pos_emb).
  * Vision RoPE is 3-D (T:H:W) with a 4:6:6 head_dim split and uses
    ``patch_positions`` instead of grid_thw to compute per-token freqs.
  * ``rotate_half`` is *interleaved* (``(::2, 1::2)``) rather than
    split-half.
  * Vision attention uses windowed ``cu_seqlens`` (``frame_windows_size``
    in T-dim); two backends implemented (SDPA + flash_attn varlen).
  * ``patch_positions: [total_patches, 3]`` is plumbed as a first-class
    MM kwarg alongside ``pixel_values`` / ``image_grid_thw``.
  * Video frame-backend and codec-backend both alias to the image path
    inside the HF processor, so the model implements a single visual
    code path.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
from collections.abc import Callable, Iterable, Mapping, Sequence
from functools import lru_cache
from typing import (
    Annotated,
    Any,
    Literal,
)

import numpy as np
import regex as re
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from PIL import Image
from transformers import AutoProcessor, AutoTokenizer, BatchFeature
from transformers.dynamic_module_utils import get_class_from_dynamic_module
from transformers.models.qwen2_vl import Qwen2VLImageProcessor
from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize

from vllm.compilation.decorators import (
    should_torch_compile_mm_encoder,
    support_torch_compile,
)
from vllm.config import VllmConfig
from vllm.config.multimodal import BaseDummyOptions
from vllm.distributed import parallel_state
from vllm.distributed import utils as dist_utils
from vllm.inputs import ModalityData, MultiModalDataDict
from vllm.logger import init_logger
from vllm.model_executor.layers.attention import MMEncoderAttention
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.models.interfaces import (
    MultiModalEmbeddings,
    SupportsMultiModal,
    SupportsPP,
)
from vllm.model_executor.models.module_mapping import MultiModelKeys
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    WeightsMapper,
    init_vllm_registered_model,
    maybe_prefix,
)
from vllm.model_executor.models.utils import (
    _merge_multimodal_embeddings as merge_multimodal_embeddings,
)
from vllm.model_executor.models.vision import get_vit_attn_backend
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    ImageItem,
    MultiModalFieldConfig,
    MultiModalKwargsItems,
)
from vllm.multimodal.parse import (
    DictEmbeddingItems,
    ImageSize,
    ModalityDataItems,
    MultiModalDataItems,
    MultiModalDataParser,
)
from vllm.multimodal.processing import (
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    PromptReplacement,
    PromptUpdate,
    PromptUpdateDetails,
)
from vllm.multimodal.processing.dummy_inputs import BaseDummyInputsBuilder
from vllm.multimodal.video import (
    VIDEO_LOADER_REGISTRY,
    VideoBackend,
    VideoSourceMetadata,
    VideoTargetMetadata,
)
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.processor import _merge_mm_kwargs
from vllm.transformers_utils.utils import convert_model_repo_to_path
from vllm.utils.tensor_schema import TensorSchema, TensorShape

logger = init_logger(__name__)


@lru_cache
def _load_ov2_processor(
    model: str,
    revision: str | None,
    trust_remote_code: bool,
    **kwargs: Any,
):
    # OV2's trust_remote_code processor is a bare class (not a ProcessorMixin),
    # so the shared type-checked get_hf_processor rejects it. We also can't use
    # AutoProcessor.from_pretrained: OV2's remote from_pretrained drops
    # trust_remote_code before building its nested tokenizer, which makes that
    # nested load fall back to an interactive stdin prompt that hangs in
    # non-interactive CI. Instead, assemble the processor here with
    # trust_remote_code threaded through every component explicitly.
    path = convert_model_repo_to_path(model)
    revision = revision or "main"

    processor_cls = get_class_from_dynamic_module(
        "processing_llava_onevision2.LlavaOnevision2Processor",
        path,
        revision=revision,
        trust_remote_code=trust_remote_code,
    )
    video_processor_cls = get_class_from_dynamic_module(
        "video_processing_llava_onevision2.LlavaOnevision2VideoProcessor",
        path,
        revision=revision,
        trust_remote_code=trust_remote_code,
    )

    # Slow Qwen2VLImageProcessor mirrors the remote processor (the Fast variant
    # has normalization rounding differences that change pixel_values).
    image_processor = Qwen2VLImageProcessor.from_pretrained(
        path, revision=revision, **kwargs
    )
    tokenizer = AutoTokenizer.from_pretrained(
        path, revision=revision, trust_remote_code=trust_remote_code, **kwargs
    )
    video_processor = video_processor_cls(
        image_processor=image_processor,
        min_pixels=getattr(image_processor, "min_pixels", 256 * 28 * 28),
        max_pixels=getattr(image_processor, "max_pixels", 1605632),
        patch_size=getattr(image_processor, "patch_size", 14),
        spatial_merge_size=getattr(image_processor, "merge_size", 2),
    )

    # Codec defaults live under the "codec" key of preprocessor_config.json,
    # which Qwen2VLImageProcessor does not preserve; read them directly so the
    # codec video backend keeps its configured defaults.
    codec_config: dict = {}
    try:
        config_file = os.path.join(path, "preprocessor_config.json")
        if not os.path.isfile(config_file):
            config_file = hf_hub_download(
                path, "preprocessor_config.json", revision=revision
            )
        with open(config_file, encoding="utf-8") as f:
            codec_config = json.load(f).get("codec", {}) or {}
    except Exception:
        logger.debug("OV2: no codec defaults found in preprocessor_config.json")

    return processor_cls(
        image_processor=image_processor,
        tokenizer=tokenizer,
        video_processor=video_processor,
        codec_config=codec_config,
    )


# Upper bound on frames used when profiling the worst-case video item, mirroring
# Qwen2-VL. The real frame count is decided by the HF VideoProcessor at apply
# time; this only sizes the memory-profiling estimate.
_MAX_FRAMES_PER_VIDEO = 14


def _pack_timestamps(per_video: list[list[float]]) -> torch.Tensor:
    if not per_video:
        return torch.empty((0, 0), dtype=torch.float32)
    t_max = max((len(ts) for ts in per_video), default=0)
    padded = torch.zeros((len(per_video), t_max), dtype=torch.float32)
    for i, ts in enumerate(per_video):
        padded[i, : len(ts)] = torch.tensor(ts, dtype=torch.float32)
    return padded


def _validate_video_source(path: str, model_config) -> str:
    """Confine a codec video path to ``--allowed-local-media-path``.

    The codec backend keeps the raw path string alive past vLLM's
    ``MultiModalDataParser`` and hands it to the trust-remote-code codec
    module, which opens it directly via ``cv2.VideoCapture`` / ffmpeg. That
    bypasses both ``MediaConnector``'s access controls and its redirect
    handling (``VLLM_MEDIA_URL_ALLOW_REDIRECTS``), so we restrict the codec
    backend to **local files only**: remote ``http(s)`` / ``data`` URLs are
    rejected here and must instead go through the frame backend (a registered
    ``VIDEO_LOADER_REGISTRY`` loader), which rides vLLM's connector and its
    domain/redirect gates.

    Returns the *resolved* absolute path so the codec module opens exactly the
    file that was validated, closing the validate-vs-open (symlink-retarget)
    window. Mirrors the confinement in ``MediaConnector._load_file_url``.
    """
    from pathlib import Path
    from urllib.request import url2pathname

    from urllib3.util import parse_url

    allowed_local = getattr(model_config, "allowed_local_media_path", "") or ""

    parsed = parse_url(str(path))
    scheme = (parsed.scheme or "").lower()

    if scheme in ("http", "https", "data"):
        raise ValueError(
            f"The codec video backend does not support remote {scheme!r} URLs: "
            f"its trust-remote-code decoder fetches them outside vLLM's domain "
            f"and redirect controls. Use a local file path, or the frame "
            f"backend for remote videos."
        )
    if scheme not in ("", "file"):
        raise ValueError(
            f"Unsupported codec video URL scheme {scheme!r}; only local file "
            f"paths or file:// URLs are supported."
        )

    # Local file access is opt-in: require --allowed-local-media-path and
    # confine the resolved path to that directory (connector.py:253-271).
    if not allowed_local:
        raise ValueError(
            "Local video file access is disabled. Set "
            "--allowed-local-media-path to enable reading local videos."
        )
    if scheme == "file":
        # Decode percent-encoding (mirrors MediaConnector._load_file_url),
        # including the netloc so file://host/path is handled identically.
        local = Path(url2pathname((parsed.netloc or "") + (parsed.path or "")))
    else:
        local = Path(str(path))
    # Require an absolute path: resolving a relative path against an ambiguous
    # CWD before the confinement check is brittle/unsafe.
    if not local.is_absolute():
        raise ValueError(
            f"Local video path {str(path)!r} must be absolute; "
            f"relative paths are not supported."
        )
    allowed_root = Path(allowed_local).resolve()
    resolved = local.resolve()
    if resolved != allowed_root and allowed_root not in resolved.parents:
        raise ValueError(
            f"Video path {str(path)!r} is outside the allowed local media "
            f"directory {allowed_local!r}."
        )
    return str(resolved)


def _validate_video_sources(paths, model_config) -> list[str]:
    # Return the resolved paths so the codec module opens exactly what was
    # validated (no validate-vs-open / symlink-retarget differential).
    return [_validate_video_source(path, model_config) for path in paths]


# Design note: the two video backends take deliberately different paths.
#
# * frame backend: a normal vLLM VIDEO_LOADER_REGISTRY loader
#   (``LlavaOnevision2VideoBackend`` below). Decoding to RGB
#   ``(frames, metadata)`` is exactly what loaders are for, so frame sampling
#   participates in the standard decode-stage pipeline.
#
# * codec backend: NOT a loader. OV2's codec path needs the video path string
#   to survive into ``_call_hf_processor``, where the HF processor builds the
#   codec canvas + smart_resize + patchify
#   (pixel_values/image_grid_thw/patch_positions). That transform is
#   path-level and inseparable; it cannot be reconstructed from pre-decoded
#   RGB frames, so it must run at the processor stage rather than the decode
#   stage. The small marker/parser machinery below keeps the path alive for
#   codec; ``_validate_video_source`` then confines it to a local file (codec
#   does not support remote URLs -- see its docstring).
_CODEC_VIDEO_MARKER = "ov2_codec_video"


def prepare_codec_video_input(video_path: str) -> tuple:
    """Wrap a video path for vLLM's MultiModalDataParser + OV2 codec backend.

    Returns ``(dummy_ndarray, metadata)`` where the ndarray satisfies the
    parser's 4-D shape check and the metadata carries the actual path to
    our ``_call_hf_processor``. Use as::

        multi_modal_data = {"video": prepare_codec_video_input("foo.mp4")}

    The dummy ndarray bytes encode a hash of ``video_path`` so distinct codec
    videos get distinct mm_hashes: the parser drops the metadata dict before
    hashing (only the ndarray reaches MultiModalHasher), so without this
    variance every video after the first would collide and skip the encoder.
    """
    path_str = str(video_path)
    digest = hashlib.blake2b(path_str.encode("utf-8"), digest_size=16).digest()
    dummy = np.frombuffer(digest, dtype=np.uint8).reshape(1, 1, 16, 1)
    dummy = np.broadcast_to(dummy, (1, 1, 16, 3)).copy()
    return (dummy, {_CODEC_VIDEO_MARKER: str(video_path)})


def _extract_codec_video_paths(videos: Any) -> list[str] | None:
    # vLLM's parser yields list-of-(ndarray, metadata-dict) for tuple inputs.
    # We accept either that shape or a single raw tuple (pre-parser cases).
    def _path_from(item):
        if (
            isinstance(item, tuple)
            and len(item) == 2
            and isinstance(item[1], dict)
            and _CODEC_VIDEO_MARKER in item[1]
        ):
            return item[1][_CODEC_VIDEO_MARKER]
        return None

    if isinstance(videos, list):
        paths: list[str] = []
        for item in videos:
            p = _path_from(item)
            if p is None:
                return None
            paths.append(p)
        return paths if paths else None
    p = _path_from(videos)
    return [p] if p is not None else None


_CODEC_FPS_CACHE: dict[str, float] = {}


def _codec_fps_for(video_path: str, hf_processor) -> float:
    if video_path in _CODEC_FPS_CACHE:
        return _CODEC_FPS_CACHE[video_path]
    # The codec module is shipped inside the HF transformers_modules package
    # for OV2, so an absolute import does not resolve. Locate it relative to
    # the processor module (which lives in the same package).
    proc_module_name = type(hf_processor).__module__
    pkg = proc_module_name.rsplit(".", 1)[0] if "." in proc_module_name else ""
    codec_mod = importlib.import_module(
        f"{pkg}.codec_video_processing_llava_onevision2"
        if pkg
        else "codec_video_processing_llava_onevision2"
    )
    CodecConfig = codec_mod.CodecConfig
    process_codec_video = codec_mod.process_codec_video
    cfg_defaults = dict(getattr(hf_processor, "_codec_config_defaults", {}))
    cfg = CodecConfig(**cfg_defaults)
    payload = process_codec_video(video_path, cfg)
    fps = float(payload["fps"])
    _CODEC_FPS_CACHE[video_path] = fps
    return fps


def _codec_timestamp_runs(
    patch_positions: torch.Tensor,
    fps: float,
    spatial_merge_size: int,
) -> list[tuple[float, int]]:
    # Mirrors HF's _timestamp_runs (codec_video_processing_llava_onevision2.py)
    # exactly: same column, same merge factor, same negative-t skip, same
    # zero-token-count skip. Keeping the logic local avoids importing the
    # private helper from transformers_modules.
    t_values = patch_positions[:, 0]
    unique_t, counts = torch.unique_consecutive(t_values, return_counts=True)
    merge_factor = int(spatial_merge_size) ** 2
    runs: list[tuple[float, int]] = []
    for t_val, count in zip(unique_t.tolist(), counts.tolist()):
        if int(t_val) < 0:
            continue
        token_count = int(count) // merge_factor
        if token_count <= 0:
            continue
        runs.append((float(t_val) / float(fps), token_count))
    return runs


def _create_field_factory(
    spatial_merge_size: int,
) -> Callable[[Mapping[str, torch.Tensor]], Mapping[str, MultiModalFieldConfig]]:
    """Build the per-batch field-config callback.

    OV2-specific: also exposes ``patch_positions`` as a flat-from-sizes
    field, sized by the total per-image patch count (T*H*W). The merger and
    the 3-D RoPE both consume it.
    """

    def _field_config(hf_inputs: Mapping[str, torch.Tensor]):
        image_grid_thw = hf_inputs.get("image_grid_thw", torch.empty((0, 3)))
        image_pixel_grid_sizes = image_grid_thw.prod(-1)
        image_embed_grid_sizes = (
            image_pixel_grid_sizes // spatial_merge_size // spatial_merge_size
        )

        video_grid_thw = hf_inputs.get("video_grid_thw", torch.empty((0, 3)))
        # OV2 emits one grid_thw row per frame, so vLLM's per-video sharding
        # requires explicit frame counts. video_patch_sizes sums H*W over the
        # frames that belong to each video; video_grid_thw uses the frame
        # count directly (one row per frame).
        video_num_frames = hf_inputs.get(
            "video_num_frames", torch.empty((0,), dtype=torch.long)
        )
        if video_num_frames.numel() > 0:
            per_row_patches = video_grid_thw.prod(-1)
            offsets = torch.cumsum(
                torch.cat([torch.zeros(1, dtype=torch.long), video_num_frames[:-1]]), 0
            ).tolist()
            video_patch_sizes = torch.tensor(
                [
                    int(per_row_patches[int(s) : int(s) + int(n)].sum())
                    for s, n in zip(offsets, video_num_frames.tolist())
                ],
                dtype=torch.long,
            )
        else:
            video_patch_sizes = torch.empty((0,), dtype=torch.long)

        return dict(
            pixel_values=MultiModalFieldConfig.flat_from_sizes(
                "image", image_pixel_grid_sizes
            ),
            image_embeds=MultiModalFieldConfig.flat_from_sizes(
                "image", image_embed_grid_sizes
            ),
            image_grid_thw=MultiModalFieldConfig.batched("image"),
            # OV2 first-class MM kwarg: per-patch (t,h,w)
            # positions required by the 3-D vision RoPE.
            patch_positions=MultiModalFieldConfig.flat_from_sizes(
                "image", image_pixel_grid_sizes
            ),
            pixel_values_videos=MultiModalFieldConfig.flat_from_sizes(
                "video", video_patch_sizes
            ),
            video_grid_thw=MultiModalFieldConfig.flat_from_sizes(
                "video", video_num_frames
            ),
            patch_positions_videos=MultiModalFieldConfig.flat_from_sizes(
                "video", video_patch_sizes
            ),
            video_num_frames=MultiModalFieldConfig.batched("video", keep_on_cpu=True),
            frame_timestamps=MultiModalFieldConfig.batched("video", keep_on_cpu=True),
            # Per-video flag: 0 = frame-sampling backend, 1 = codec backend.
            # Drives codec-aware ``\n`` insertion in ``get_video_replacement``.
            video_is_codec=MultiModalFieldConfig.batched("video", keep_on_cpu=True),
            # Codec backend: per-video source-frame fps. Needed at
            # replacement time to convert patch_positions t-indices into
            # the timestamp tags HF writes (``<sec seconds>``).
            codec_fps=MultiModalFieldConfig.batched("video", keep_on_cpu=True),
        )

    return _field_config


# ---------------------------------------------------------------------------
# Frame backend helpers
# ---------------------------------------------------------------------------
# The default video pathway materialises each video as a series of PIL frames
# (decoded + sampled by the registered ``LlavaOnevision2VideoBackend``) and
# feeds them to the HF processor through the *image* branch (per-frame timestamp
# marker + ``<|image_pad|>``). This mirrors the validated lmms-eval
# ``vllm_hf_chat`` adapter, and empirically scores higher on Video-MME than
# OV2's native VideoProcessor frame extractor.
_DEFAULT_TIMESTAMP_DECIMALS = 1
_DEFAULT_FPS = 1.0
_DEFAULT_MAX_FRAMES = 32
# OV2 vision tower has spatial_merge_size=2 -> temporal frame count must be
# even. The hf-chat reference pads by repeating the last frame; same here.
_TEMPORAL_MERGE_SIZE = 2

# Token sequence emitted by the OV2 vLLM dummy inputs builder for each video
# item. The frame backend expands each marker into per-frame image markers
# (timestamp + image_pad block).
_VIDEO_MARKER = "<|vision_start|><|video_pad|><|vision_end|>"
_IMAGE_MARKER = "<|vision_start|><|image_pad|><|vision_end|>"


def _frame_video_to_pil_and_timestamps(
    item: Any,
) -> tuple[list[Image.Image], list[float]]:
    """Convert a ``(frames_ndarray, metadata)`` video item into
    ``(pil_frames, timestamps_seconds)``.

    Both real ``video_url`` inputs (decoded + sampled by the registered
    ``LlavaOnevision2VideoBackend``) and dummy profiling videos arrive here as a
    ``(frames, metadata)`` tuple because the data parser runs with
    ``video_needs_metadata=True``. ``frames`` is a ``(T, H, W, C)`` uint8 array;
    ``metadata`` carries ``frames_indices`` and the source ``fps``.

    Timestamps follow the qwen_vl_utils policy: ``frame_index / original_fps``.
    The frame count is padded up to ``_TEMPORAL_MERGE_SIZE`` (repeating the last
    frame) because OV2's vision tower merges frames temporally in pairs.
    """
    if not (isinstance(item, (tuple, list)) and len(item) == 2):
        raise ValueError(
            "LlavaOnevision2 frame backend expects each video as a "
            f"(frames_ndarray, metadata) tuple; got {type(item).__name__}. "
            "Pass videos via `video_url` so the registered backend can decode "
            "and sample them."
        )
    frames, metadata = item
    if isinstance(frames, torch.Tensor):
        frames_np = frames.cpu().numpy()
    else:
        frames_np = np.asarray(frames)

    pil_frames = [Image.fromarray(f.astype(np.uint8)) for f in frames_np]

    indices = metadata.get("frames_indices") if isinstance(metadata, Mapping) else None
    if indices is None:
        indices = list(range(len(pil_frames)))
    elif not isinstance(indices, list):
        indices = list(indices)
    # Keep indices aligned with the actual frame count.
    if len(indices) != len(pil_frames):
        if len(indices) > len(pil_frames):
            indices = indices[: len(pil_frames)]
        else:
            indices = list(indices) + [indices[-1] if indices else 0] * (
                len(pil_frames) - len(indices)
            )

    fps = _DEFAULT_FPS
    if isinstance(metadata, Mapping) and metadata.get("fps"):
        fps = float(metadata["fps"])
    if fps <= 0:
        fps = _DEFAULT_FPS

    # OV2 vision tower: temporal merge=2 -> frame count must be even.
    if len(pil_frames) % _TEMPORAL_MERGE_SIZE != 0:
        pad = _TEMPORAL_MERGE_SIZE - len(pil_frames) % _TEMPORAL_MERGE_SIZE
        pil_frames = pil_frames + [pil_frames[-1]] * pad
        indices = indices + [indices[-1]] * pad

    timestamps = [idx / fps for idx in indices]
    return pil_frames, timestamps


def _expand_video_markers_in_prompt(
    prompt: str,
    per_video_timestamps: list[list[float]],
    *,
    timestamp_decimals: int,
) -> str:
    """Replace each ``<|vision_start|><|video_pad|><|vision_end|>`` with a
    sequence of ``<{t:.Nf} seconds><|vision_start|><|image_pad|><|vision_end|>``
    blocks -- one per frame -- matching ``vllm_hf_chat._build_prompt``.

    Replacement is positional: the *i*-th marker consumes
    ``per_video_timestamps[i]``.
    """
    parts: list[str] = []
    cursor = 0
    idx = 0
    pattern = re.escape(_VIDEO_MARKER)
    for m in re.finditer(pattern, prompt):
        parts.append(prompt[cursor : m.start()])
        if idx >= len(per_video_timestamps):
            raise ValueError(
                f"Prompt has more video markers than supplied timestamp "
                f"groups ({len(per_video_timestamps)})"
            )
        timestamps = per_video_timestamps[idx]
        expanded = "".join(
            f"<{t:.{timestamp_decimals}f} seconds>{_IMAGE_MARKER}" for t in timestamps
        )
        parts.append(expanded)
        cursor = m.end()
        idx += 1
    parts.append(prompt[cursor:])
    if idx != len(per_video_timestamps):
        raise ValueError(
            f"Prompt has {idx} video markers but {len(per_video_timestamps)} "
            f"timestamp groups were supplied"
        )
    return "".join(parts)


# ---------------------------------------------------------------------------
# Registered video loader backend (frame sampling)
# ---------------------------------------------------------------------------
# OV2 frame-sampling policy, expressed as a vLLM ``VideoBackend`` so videos
# entering through the standard ``video_url`` -> MediaConnector -> VideoMediaIO
# path are decoded + sampled inside vLLM (no qwen_vl_utils dependency, and the
# connector's SSRF / local-file gates apply automatically).
#
# Parity note: ``compute_frames_index_to_sample`` replicates
# ``qwen_vl_utils.smart_nframes`` (the policy the OV2 hf-chat recipe validated)
# -- frame count and index selection are byte-identical to qwen. The one
# residual difference is the source frame *count*: qwen decodes via decord
# whereas vLLM uses OpenCV/PyAV, whose ``CAP_PROP_FRAME_COUNT`` (a duration x
# fps estimate) can differ by +/-1 frame on some containers, shifting sampled
# indices by one frame on those files. Downstream metrics stay within noise.
_OV2_FRAME_FACTOR = 2
_OV2_FPS_MIN_FRAMES = 4


def _round_by_factor(n: float, factor: int) -> int:
    return round(n / factor) * factor


def _ceil_by_factor(n: float, factor: int) -> int:
    import math as _math

    return _math.ceil(n / factor) * factor


def _floor_by_factor(n: float, factor: int) -> int:
    import math as _math

    return _math.floor(n / factor) * factor


def _ov2_smart_nframes(
    total_frames: int,
    video_fps: float,
    *,
    fps: float,
    min_frames: int,
    max_frames: int,
) -> int:
    """Replicate ``qwen_vl_utils.smart_nframes`` (fps branch).

    Returns an even frame count in ``[min_frames, min(max_frames, total)]``.
    """
    min_frames = _ceil_by_factor(min_frames, _OV2_FRAME_FACTOR)
    max_frames = _floor_by_factor(max_frames, _OV2_FRAME_FACTOR)
    nframes = total_frames / video_fps * fps if video_fps > 0 else total_frames
    nframes = min(min(max(nframes, min_frames), max_frames), total_frames)
    nframes = _floor_by_factor(nframes, _OV2_FRAME_FACTOR)
    return max(int(nframes), _OV2_FRAME_FACTOR)


@VIDEO_LOADER_REGISTRY.register(
    "llava_onevision2",
    video_processor="LlavaOnevision2VideoProcessor",
)
class LlavaOnevision2VideoBackend(VideoBackend):
    """Frame-sampling backend for LLaVA-OneVision-2.

    Selected automatically for OV2 via the ``video_processor`` binding
    (``video_processor_type == "LlavaOnevision2VideoProcessor"`` in the model's
    ``video_preprocessor_config.json``). Decoding uses the inherited OpenCV /
    PyAV codecs; only the sampling index policy is overridden to match qwen.
    """

    _sampling_suffix = "_llava_onevision2"

    # OV2 hf-chat reference sampling constants (mirror the validated adapter).
    _FPS = _DEFAULT_FPS
    _MAX_FRAMES = _DEFAULT_MAX_FRAMES
    _MIN_FRAMES = _OV2_FPS_MIN_FRAMES

    @classmethod
    def compute_frames_index_to_sample(
        cls,
        source: VideoSourceMetadata,
        target: VideoTargetMetadata,
        **kwargs,
    ) -> list[int]:
        total = int(source.total_frames_num)
        if total <= 0:
            return []
        video_fps = float(source.original_fps)
        # Honor caller-provided sampling targets (via ``--media-io-kwargs`` →
        # ``VideoTargetMetadata``) so benchmarks can override the conservative
        # defaults (e.g. VSI-Bench needs max_frames=128). Fall back to the OV2
        # hf-chat reference constants when the target leaves a field unset
        # (sentinel ``<= 0``).
        target_fps = float(target.fps) if target.fps > 0 else cls._FPS
        target_max_frames = (
            int(target.num_frames) if target.num_frames > 0 else cls._MAX_FRAMES
        )
        n = _ov2_smart_nframes(
            total,
            video_fps,
            fps=target_fps,
            min_frames=cls._MIN_FRAMES,
            max_frames=target_max_frames,
        )
        # qwen uses linspace().round() (NOT the floor cast used by the base
        # uniform backend), so replicate the rounding exactly.
        idx = np.linspace(0, total - 1, n).round().astype(int).tolist()
        # smart_nframes floors to FRAME_FACTOR so ``n`` is even; guard anyway
        # since OV2's vision tower (temporal merge = 2) requires even counts.
        if len(idx) % _OV2_FRAME_FACTOR != 0:
            idx.append(idx[-1])
        return idx


class LlavaOnevision2ImagePixelInputs(TensorSchema):
    type: Literal["pixel_values"]

    pixel_values: Annotated[torch.Tensor, TensorShape("np", "cps")]
    image_grid_thw: Annotated[torch.Tensor, TensorShape("ni", 3)]
    patch_positions: Annotated[torch.Tensor, TensorShape("np", 3)]


class LlavaOnevision2ImageEmbeddingInputs(TensorSchema):
    type: Literal["image_embeds"]

    image_embeds: Annotated[torch.Tensor, TensorShape("nf", "hs")]
    image_grid_thw: Annotated[torch.Tensor, TensorShape("ni", 3)]


class LlavaOnevision2VideoPixelInputs(TensorSchema):
    type: Literal["pixel_values_videos"]

    pixel_values_videos: Annotated[torch.Tensor, TensorShape("np", "cps")]
    video_grid_thw: Annotated[torch.Tensor, TensorShape("nf", 3)]
    patch_positions_videos: Annotated[torch.Tensor, TensorShape("np", 3)]
    video_num_frames: Annotated[torch.Tensor, TensorShape("nv")]


LlavaOnevision2ImageInputs = (
    LlavaOnevision2ImagePixelInputs | LlavaOnevision2ImageEmbeddingInputs
)


class LlavaOnevision2VisionRotaryEmbedding(nn.Module):
    """3-D rotary frequency constructor with 4:6:6 (T:H:W) split.

    Mirrors ``VisionRotaryEmbedding`` in the HF reference
    (``modeling_llava_onevision2.py`` L79-L210). The three ``inv_freq_*``
    buffers are non-persistent — they are *not* in the checkpoint and must
    be reconstructed at module init time (which we do here).

    Public entry points used by the vision tower:
      * ``forward_from_positions(patch_positions)`` — per-patch (t,h,w)
        positions → per-token freqs [N, half].
    """

    def __init__(self, head_dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        assert head_dim % 2 == 0, "head_dim must be even"
        assert head_dim % 16 == 0, "head_dim must be divisible by 16 (4:6:6)"
        half = head_dim // 2
        assert half % 16 == 0, "head_dim//2 must be divisible by 16"

        self.head_dim = head_dim
        self.half = half
        self.base = float(theta)

        unit = half // 16
        self.t_size = 4 * unit
        self.h_size = 6 * unit
        self.w_size = 6 * unit
        assert self.t_size + self.h_size + self.w_size == half

        self.register_buffer(
            "inv_freq_t",
            1.0
            / (
                self.base
                ** (torch.arange(self.t_size, dtype=torch.float32) / self.t_size)
            ),
            persistent=False,
        )
        self.register_buffer(
            "inv_freq_h",
            1.0
            / (
                self.base
                ** (torch.arange(self.h_size, dtype=torch.float32) / self.h_size)
            ),
            persistent=False,
        )
        self.register_buffer(
            "inv_freq_w",
            1.0
            / (
                self.base
                ** (torch.arange(self.w_size, dtype=torch.float32) / self.w_size)
            ),
            persistent=False,
        )

    def forward_from_positions(self, patch_positions: torch.Tensor) -> torch.Tensor:
        """[N, 3] (t,h,w) int → [N, half] float frequencies."""
        device = patch_positions.device
        inv_t = self.inv_freq_t.to(device=device)
        inv_h = self.inv_freq_h.to(device=device)
        inv_w = self.inv_freq_w.to(device=device)

        t_pos = patch_positions[:, 0].float()
        h_pos = patch_positions[:, 1].float()
        w_pos = patch_positions[:, 2].float()

        ft = torch.outer(t_pos, inv_t)
        fh = torch.outer(h_pos, inv_h)
        fw = torch.outer(w_pos, inv_w)
        return torch.cat([ft, fh, fw], dim=-1)


class LlavaOnevision2VisionEmbeddings(nn.Module):
    def __init__(
        self, patch_size: int = 14, in_channels: int = 3, embed_dim: int = 1024
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.patch_embedding = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1, self.in_channels, self.patch_size, self.patch_size)
        x = self.patch_embedding(x).view(-1, self.embed_dim)
        return x


class LlavaOnevision2VisionMLP(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        bias: bool = True,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        use_data_parallel: bool = False,
    ) -> None:
        super().__init__()
        if quant_config is not None:
            raise RuntimeError("LLaVAOneVision2 does not support quantization")
        self.fc1 = ColumnParallelLinear(
            in_features,
            hidden_features,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.fc1",
            disable_tp=use_data_parallel,
        )
        self.fc2 = RowParallelLinear(
            hidden_features,
            in_features,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.fc2",
            disable_tp=use_data_parallel,
        )
        self.act_fn = F.gelu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.fc1(x)
        x = self.act_fn(x)
        x, _ = self.fc2(x)
        return x


class LlavaOnevision2VisionAttn(nn.Module):
    """Vision self-attention with windowed cu_seqlens.

    The HF checkpoint ships a *fused* qkv linear (``self_attn.qkv``), so
    we load directly into ``QKVParallelLinear`` with no stacked_params
    mapping. (Compare OV1.5, whose checkpoint had separate q/k/v.)
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        projection_size: int,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        use_data_parallel: bool = False,
    ) -> None:
        super().__init__()
        if quant_config is not None:
            raise RuntimeError("LLaVAOneVision2 does not support quantization")

        self.tp_size = (
            1
            if use_data_parallel
            else parallel_state.get_tensor_model_parallel_world_size()
        )
        self.tp_rank = parallel_state.get_tensor_model_parallel_rank()
        self.num_heads = num_heads
        self.hidden_size_per_attn_head = dist_utils.divide(projection_size, num_heads)
        self.num_attn_heads_per_partition = dist_utils.divide(num_heads, self.tp_size)

        self.qkv = QKVParallelLinear(
            hidden_size=embed_dim,
            head_size=self.hidden_size_per_attn_head,
            total_num_heads=num_heads,
            total_num_kv_heads=num_heads,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv",
            disable_tp=use_data_parallel,
        )

        self.proj = RowParallelLinear(
            input_size=projection_size,
            output_size=embed_dim,
            quant_config=quant_config,
            prefix=f"{prefix}.proj",
            disable_tp=use_data_parallel,
        )

        self.attn = MMEncoderAttention(
            num_heads=self.num_attn_heads_per_partition,
            head_size=self.hidden_size_per_attn_head,
            scale=self.hidden_size_per_attn_head**-0.5,
            prefix=f"{prefix}.attn",
        )

    @staticmethod
    def _rotate_half_interleaved(x: torch.Tensor) -> torch.Tensor:
        """OV2-specific interleaved rotate_half.

        Pairs adjacent dims: (x[::2], x[1::2]) -> (-x[1::2], x[::2]).
        NOT compatible with the split-half rotate used in OV1.5/LLaMA.
        """
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        out = torch.stack((-x_odd, x_even), dim=-1)
        return out.flatten(-2)

    def _apply_rotary_pos_embed(
        self, t: torch.Tensor, freqs: torch.Tensor
    ) -> torch.Tensor:
        # freqs is [seq_len, half]; cat([f,f],-1) pair-repeat layout
        # matches the interleaved rotate above.
        orig_dtype = t.dtype
        t = t.float()
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().unsqueeze(-2).float()
        sin = emb.sin().unsqueeze(-2).float()
        t = (t * cos) + (self._rotate_half_interleaved(t) * sin)
        return t.to(orig_dtype)

    def forward(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: torch.Tensor,
        max_seqlen: torch.Tensor | None = None,
        sequence_lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x, _ = self.qkv(x)
        seq_len = x.shape[0]
        # QKVParallelLinear packs q/k/v along the last dim as
        # [q_heads, k_heads, v_heads]; view splits the three sections, each
        # holding this partition's heads (no all-gather: MMEncoderAttention
        # runs per-partition and ``proj`` reduces across TP ranks).
        qkv = x.view(
            seq_len,
            3,
            self.num_attn_heads_per_partition,
            self.hidden_size_per_attn_head,
        )
        q, k, v = qkv.unbind(1)

        if rotary_pos_emb is not None:
            # OV2 uses interleaved RoPE on the raw freqs, applied outside
            # MMEncoderAttention (which is rotary-agnostic).
            q = self._apply_rotary_pos_embed(q, rotary_pos_emb)
            k = self._apply_rotary_pos_embed(k, rotary_pos_emb)

        # Add a leading batch dim (b=1) for MMEncoderAttention, which expects
        # (batch, seq_len, num_heads, head_size) and windows via cu_seqlens.
        output = self.attn(
            query=q.unsqueeze(0),
            key=k.unsqueeze(0),
            value=v.unsqueeze(0),
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            sequence_lengths=sequence_lengths,
        )
        output = output.reshape(seq_len, -1)

        output, _ = self.proj(output)
        return output


@support_torch_compile(
    dynamic_arg_dims={
        "x": 0,
        "cu_seqlens": 0,
        "rotary_pos_emb": 0,
        "sequence_lengths": 0,
    },
    enable_if=should_torch_compile_mm_encoder,
    is_encoder=True,
)
class LlavaOnevision2VisionTowerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_hidden_dim: int,
        norm_eps: float = 1e-6,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        use_data_parallel: bool = False,
    ) -> None:
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(dim, eps=norm_eps)
        self.layer_norm2 = nn.LayerNorm(dim, eps=norm_eps)
        self.self_attn = LlavaOnevision2VisionAttn(
            embed_dim=dim,
            num_heads=num_heads,
            projection_size=dim,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
            use_data_parallel=use_data_parallel,
        )
        self.mlp = LlavaOnevision2VisionMLP(
            dim,
            mlp_hidden_dim,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
            use_data_parallel=use_data_parallel,
        )

    def forward(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: torch.Tensor,
        max_seqlen: torch.Tensor | None = None,
        sequence_lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x + self.self_attn(
            self.layer_norm1(x),
            cu_seqlens=cu_seqlens,
            rotary_pos_emb=rotary_pos_emb,
            max_seqlen=max_seqlen,
            sequence_lengths=sequence_lengths,
        )
        x = x + self.mlp(self.layer_norm2(x))
        return x


class LlavaOnevision2PatchMerger(nn.Module):
    def __init__(
        self,
        d_model: int,
        context_dim: int,
        spatial_merge_size: int = 2,
        norm_eps: float = 1e-6,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        use_data_parallel: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_size = context_dim * (spatial_merge_size**2)
        self.ln_q = nn.LayerNorm(context_dim, eps=norm_eps)
        self.mlp = nn.ModuleList(
            [
                ColumnParallelLinear(
                    self.hidden_size,
                    self.hidden_size,
                    bias=True,
                    quant_config=quant_config,
                    prefix=f"{prefix}.mlp.0",
                    disable_tp=use_data_parallel,
                ),
                nn.GELU(),
                RowParallelLinear(
                    self.hidden_size,
                    d_model,
                    bias=True,
                    quant_config=quant_config,
                    prefix=f"{prefix}.mlp.2",
                    disable_tp=use_data_parallel,
                ),
            ]
        )

    def forward(
        self, x: torch.Tensor, patch_positions: torch.Tensor | None = None
    ) -> torch.Tensor:
        # patch_positions accepted for API symmetry with the HF impl,
        # but unused: pixel_values already arrive in spatial-merge block
        # order (Qwen2VLImageProcessor handles that across image / video-
        # frames / video-codec backends). See codec_video_processing for
        # the codec backend's ``codec_positions_for_processor`` call.
        del patch_positions
        x = self.ln_q(x)
        x = x.view(-1, self.hidden_size)
        fc1, act, fc2 = self.mlp
        x, _ = fc1(x)
        x = act(x)
        x, _ = fc2(x)
        return x


class LlavaOnevision2VisionTower(nn.Module):
    """OV2 vision tower (no CLS token, 3-D RoPE, windowed attention).

    Module attribute names mirror HF checkpoint names verbatim so the
    WeightsMapper only needs prefix rewrites (no substring rules, which would
    otherwise collide with the Qwen3 text-path ``self_attn`` modules):
      visual.embeddings.patch_embedding
      visual.layernorm_pre
      visual.encoder.layers.{i}.self_attn.{qkv,proj}
      visual.encoder.layers.{i}.layer_norm{1,2}
      visual.encoder.layers.{i}.mlp.fc{1,2}
      visual.merger.{ln_q, mlp.{0,2}}
      visual.rotary_pos_emb (non-persistent inv_freq buffers)
    """

    def __init__(
        self,
        vision_config,
        text_hidden_size: int,
        norm_eps: float = 1e-6,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        use_data_parallel: bool = False,
    ) -> None:
        super().__init__()
        if quant_config is not None:
            raise RuntimeError("LLaVAOneVision2 does not support quantization")

        patch_size = vision_config.patch_size
        spatial_merge_size = vision_config.spatial_merge_size
        in_channels = getattr(vision_config, "num_channels", 3)
        hidden_size = vision_config.hidden_size
        embed_dim = hidden_size
        depth = vision_config.num_hidden_layers
        num_heads = vision_config.num_attention_heads
        mlp_hidden_dim = vision_config.intermediate_size
        frame_windows_size = getattr(vision_config, "frame_windows_size", 4)
        rope_theta = getattr(vision_config, "rope_theta", 10000.0)

        self.spatial_merge_size = spatial_merge_size
        self.frame_windows_size = int(frame_windows_size)
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads
        self.use_data_parallel = use_data_parallel
        self.tp_size = (
            1
            if use_data_parallel
            else parallel_state.get_tensor_model_parallel_world_size()
        )

        self.embeddings = LlavaOnevision2VisionEmbeddings(
            patch_size=patch_size, in_channels=in_channels, embed_dim=embed_dim
        )
        self.layernorm_pre = nn.LayerNorm(embed_dim, eps=norm_eps)

        self.rotary_pos_emb = LlavaOnevision2VisionRotaryEmbedding(
            self.head_dim, theta=rope_theta
        )

        self.encoder = nn.Module()
        self.encoder.layers = nn.ModuleList(
            [
                LlavaOnevision2VisionTowerBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_hidden_dim=mlp_hidden_dim,
                    norm_eps=norm_eps,
                    quant_config=quant_config,
                    prefix=f"{prefix}.encoder.layers.{i}",
                    use_data_parallel=use_data_parallel,
                )
                for i in range(depth)
            ]
        )

        self.merger = LlavaOnevision2PatchMerger(
            d_model=text_hidden_size,
            context_dim=embed_dim,
            spatial_merge_size=spatial_merge_size,
            norm_eps=norm_eps,
            quant_config=quant_config,
            prefix=f"{prefix}.merger",
            use_data_parallel=use_data_parallel,
        )

        # Vision attention backend; mirrors the one MMEncoderAttention picks
        # internally so the cu_seqlens / max_seqlen metadata computed below
        # matches the kernel actually used.
        self.attn_backend = get_vit_attn_backend(
            head_size=self.head_dim, dtype=torch.get_default_dtype()
        )

    @property
    def dtype(self) -> torch.dtype:
        return self.embeddings.patch_embedding.weight.dtype

    @property
    def device(self) -> torch.device:
        return self.embeddings.patch_embedding.weight.device

    def _build_window_cu_seqlens(self, grid_thw: torch.Tensor) -> np.ndarray:
        """Build cu_seqlens that chunk each sample's T-axis into windows of
        ``frame_windows_size`` frames.

        Returns an int32 ``np.ndarray`` of shape [num_windows+1] (the
        canonical prefix-sum format). Backend-specific transforms are applied
        afterwards via ``MMEncoderAttention.maybe_recompute_cu_seqlens``.
        """
        win = self.frame_windows_size
        chunk_lengths: list[int] = []
        for row in grid_thw.tolist():
            t, h, w = int(row[0]), int(row[1]), int(row[2])
            per_frame = h * w
            t_remaining = t
            while t_remaining > 0:
                this_t = min(win, t_remaining)
                chunk_lengths.append(this_t * per_frame)
                t_remaining -= this_t
        cu = np.concatenate(
            [
                np.zeros(1, dtype=np.int32),
                np.array(chunk_lengths, dtype=np.int32).cumsum(dtype=np.int32),
            ]
        )
        return cu

    def forward(
        self,
        pixel_values: torch.Tensor,
        grid_thw: torch.Tensor,
        patch_positions: torch.Tensor,
    ) -> torch.Tensor:
        x = pixel_values.to(device=self.device, dtype=self.dtype)
        x = self.embeddings(x)
        x = self.layernorm_pre(x)

        rotary_pos_emb = self.rotary_pos_emb.forward_from_positions(
            patch_positions.to(self.device)
        )

        # Build window cu_seqlens, then derive backend-specific attention
        # metadata (passthrough for FA/SDPA; transformed for FlashInfer).
        cu_seqlens_np = self._build_window_cu_seqlens(grid_thw)
        sequence_lengths = MMEncoderAttention.maybe_compute_seq_lens(
            self.attn_backend, cu_seqlens_np, self.device
        )
        max_seqlen = torch.tensor(
            MMEncoderAttention.compute_max_seqlen(self.attn_backend, cu_seqlens_np),
            dtype=torch.int32,
        )
        cu_seqlens = MMEncoderAttention.maybe_recompute_cu_seqlens(
            self.attn_backend,
            cu_seqlens_np,
            self.embed_dim,
            self.tp_size,
            self.device,
        )

        for blk in self.encoder.layers:
            x = blk(
                x,
                cu_seqlens=cu_seqlens,
                rotary_pos_emb=rotary_pos_emb,
                max_seqlen=max_seqlen,
                sequence_lengths=sequence_lengths,
            )

        return self.merger(x, patch_positions=patch_positions)


class LlavaOnevision2ProcessingInfo(BaseProcessingInfo):
    def get_hf_config(self):
        return self.ctx.get_hf_config()

    def get_data_parser(self) -> MultiModalDataParser:
        # ``video_needs_metadata=True`` makes the parser preserve both the
        # ``(frames, metadata)`` tuples from the frame backend and the
        # ``(dummy, {marker: path})`` tuples from prepare_codec_video_input;
        # both are dispatched by metadata content in ``_call_hf_processor``.
        return LlavaOnevision2MultiModalDataParser(
            self.get_hf_config().vision_config.spatial_merge_size,
            video_needs_metadata=True,
        )

    def get_hf_processor(self, **kwargs: object):
        # OV2's trust_remote_code processor is a bare class (not a
        # ProcessorMixin), so the shared get_hf_processor cannot load it; load
        # via AutoProcessor here (as other trust_remote_code models do).
        model_config = self.ctx.model_config
        # ``_merge_mm_kwargs`` restricts caller ``mm_processor_kwargs`` to known
        # processor args and wraps values as hashable for the lru_cache.
        merged = _merge_mm_kwargs(model_config, AutoProcessor, **kwargs)
        merged.setdefault("use_fast", True)
        return _load_ov2_processor(
            model_config.model,
            model_config.revision,
            model_config.trust_remote_code,
            **merged,
        )

    def get_image_processor(self, **kwargs: object) -> Qwen2VLImageProcessor:
        return self.get_hf_processor(**kwargs).image_processor

    def get_mm_max_tokens_per_item(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> Mapping[str, int]:
        return {
            "image": self.get_max_image_tokens(),
            "video": self.get_max_video_tokens(seq_len, mm_counts),
        }

    def _get_vision_info(
        self,
        *,
        image_width: int,
        image_height: int,
        num_frames: int = 1,
        do_resize: bool = True,
        image_processor: Qwen2VLImageProcessor | None,
    ) -> tuple[ImageSize, int]:
        if image_processor is None:
            image_processor = self.get_image_processor()
        hf_config = self.get_hf_config()
        vision_config = hf_config.vision_config
        patch_size = vision_config.patch_size
        merge_size = vision_config.spatial_merge_size
        temporal_patch_size = getattr(vision_config, "temporal_patch_size", 1)
        if do_resize:
            min_pixels = getattr(image_processor, "min_pixels", None)
            max_pixels = getattr(image_processor, "max_pixels", None)
            if min_pixels is None or max_pixels is None:
                size = image_processor.size
                min_pixels = (
                    getattr(size, "shortest_edge", None) or size["shortest_edge"]
                )
                max_pixels = getattr(size, "longest_edge", None) or size["longest_edge"]
            rh, rw = smart_resize(
                height=image_height,
                width=image_width,
                factor=patch_size * merge_size,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
            )
            preprocessed = ImageSize(width=rw, height=rh)
        else:
            preprocessed = ImageSize(width=image_width, height=image_height)
        padded_frames = num_frames + num_frames % temporal_patch_size
        grid_t = max(padded_frames // temporal_patch_size, 1)
        grid_h = preprocessed.height // patch_size
        grid_w = preprocessed.width // patch_size
        num_patches = grid_t * grid_h * grid_w
        return preprocessed, num_patches // (merge_size**2)

    def get_num_image_tokens(
        self,
        *,
        image_width: int,
        image_height: int,
        image_processor: Qwen2VLImageProcessor | None,
    ) -> int:
        _, n = self._get_vision_info(
            image_width=image_width,
            image_height=image_height,
            image_processor=image_processor,
        )
        return n

    def get_image_size_with_most_features(self) -> ImageSize:
        sz, _ = self._get_vision_info(
            image_width=1800, image_height=1800, image_processor=None
        )
        return sz

    def get_max_image_tokens(self) -> int:
        w, h = self.get_image_size_with_most_features()
        return self.get_num_image_tokens(
            image_width=w, image_height=h, image_processor=None
        )

    def get_num_video_tokens(
        self,
        *,
        image_width: int,
        image_height: int,
        num_frames: int,
        image_processor: Qwen2VLImageProcessor | None = None,
    ) -> int:
        _, n = self._get_vision_info(
            image_width=image_width,
            image_height=image_height,
            num_frames=num_frames,
            image_processor=image_processor,
        )
        return n

    def _get_max_video_frames(self, max_tokens: int, start_num_frames: int = 1) -> int:
        w, h = self.get_image_size_with_most_features()
        num_frames = start_num_frames
        while True:
            next_num_frames = num_frames + 1
            next_max_tokens = self.get_num_video_tokens(
                image_width=w,
                image_height=h,
                num_frames=next_num_frames,
            )
            if next_max_tokens > max_tokens:
                break
            num_frames = next_num_frames
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
        w, h = self.get_image_size_with_most_features()
        return self.get_num_video_tokens(
            image_width=w,
            image_height=h,
            num_frames=self.get_num_frames_with_most_features(seq_len, mm_counts),
        )

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"image": None, "video": None}


class LlavaOnevision2DummyInputsBuilder(
    BaseDummyInputsBuilder[LlavaOnevision2ProcessingInfo]
):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        n_img = mm_counts.get("image", 0)
        n_vid = mm_counts.get("video", 0)
        return (
            "<|vision_start|><|image_pad|><|vision_end|>" * n_img
            + "<|vision_start|><|video_pad|><|vision_end|>" * n_vid
        )

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions] | None = None,
    ) -> MultiModalDataDict:
        n_img = mm_counts.get("image", 0)
        n_vid = mm_counts.get("video", 0)
        w, h = self.info.get_image_size_with_most_features()
        out: MultiModalDataDict = {}
        if n_img:
            out["image"] = self._get_dummy_images(width=w, height=h, num_images=n_img)
        if n_vid:
            # 4 frames per dummy video keeps profiling cheap; the real frame
            # count is decided by the HF VideoProcessor at apply time.
            out["video"] = self._get_dummy_videos(
                width=w, height=h, num_frames=4, num_videos=n_vid
            )
        return out

    def _get_dummy_videos(
        self,
        *,
        width: int,
        height: int,
        num_frames: int,
        num_videos: int,
        overrides=None,
    ):
        # ``video_needs_metadata=True`` (see ProcessingInfo.get_data_parser)
        # makes the parser require a metadata dict on every video item, so the
        # dummy profiling videos must carry one too. ``do_sample_frames=False``
        # plus ``frames_indices=range(T)`` tells the frame path to consume the
        # pre-built frames verbatim (no resampling) -- mirrors GLM-4V.
        # OV2's vision tower (temporal merge = 2) needs an even frame count.
        num_frames = max(num_frames, _OV2_FRAME_FACTOR)
        if num_frames % _OV2_FRAME_FACTOR != 0:
            num_frames += 1
        videos = super()._get_dummy_videos(
            width=width,
            height=height,
            num_frames=num_frames,
            num_videos=num_videos,
            overrides=overrides,
        )
        video_items = []
        for video in videos:
            t = video.shape[0]
            metadata = {
                "fps": 1.0,
                "duration": float(t),
                "total_num_frames": int(t),
                "frames_indices": list(range(t)),
                "video_backend": "llava_onevision2",
                "do_sample_frames": False,
            }
            video_items.append((video, metadata))
        return video_items


class LlavaOnevision2MultiModalDataParser(MultiModalDataParser):
    def __init__(self, spatial_merge_size: int, *args, **kwargs):
        self._spatial_merge_size = spatial_merge_size
        super().__init__(*args, **kwargs)

    def _parse_image_data(
        self,
        data: dict[str, torch.Tensor] | ModalityData[ImageItem],
    ) -> ModalityDataItems[Any, Any] | None:
        if isinstance(data, dict):
            return DictEmbeddingItems(
                data,
                modality="image",
                required_fields={"image_embeds", "image_grid_thw"},
                fields_factory=_create_field_factory(self._spatial_merge_size),
            )
        return super()._parse_image_data(data)


class LlavaOnevision2MultiModalProcessor(
    BaseMultiModalProcessor[LlavaOnevision2ProcessingInfo]
):
    def _get_data_parser(self) -> MultiModalDataParser:
        # Retained for symmetry; vLLM actually fetches the parser via
        # info.get_data_parser() (see ProcessingInfo override above).
        return LlavaOnevision2MultiModalDataParser(
            self.info.get_hf_config().vision_config.spatial_merge_size
        )

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        # The wrapped OV2 processor is a bare custom class without the standard
        # ProcessorMixin ``_merge_kwargs`` machinery, so vLLM's default path
        # fails; overriding this method routes the base class to call us
        # directly.
        hf_processor = self.info.get_hf_processor(**mm_kwargs)
        merged_kwargs = self.info.ctx.get_merged_mm_kwargs(
            dict(**mm_kwargs, **tok_kwargs)
        )
        merged_kwargs.setdefault("return_tensors", "pt")
        call_kwargs = {
            k: v
            for k, v in merged_kwargs.items()
            if k
            in {
                "return_tensors",
                "padding",
                "num_frames",
                "max_frames",
                "target_fps",
                "video_backend",
                "max_pixels",
                "codec_config",
            }
        }
        mm_data = dict(mm_data)
        # Explicit None + length checks: ``mm_data[...]`` may be a list, numpy
        # array, or tensor, and ``and <array>`` would raise on the ambiguous
        # truth value of a multi-element array.
        _videos = mm_data.get("videos")
        videos_present = _videos is not None and len(_videos) > 0

        codec_video_paths = (
            _extract_codec_video_paths(mm_data["videos"]) if videos_present else None
        )
        is_codec_marker = codec_video_paths is not None
        # Fallback: caller passed video_backend=codec via mm_processor_kwargs
        # without wrapping paths through prepare_codec_video_input (e.g.
        # lmms-eval's chat/vllm.py ov2_path_video=True). Recover the path
        # strings directly from mm_data["videos"] so the codec rename
        # branch still fires and video-modality fields get populated.
        is_codec_kwarg = (
            not is_codec_marker
            and videos_present
            and call_kwargs.get("video_backend") == "codec"
        )
        if is_codec_kwarg:
            raw = mm_data["videos"]
            if isinstance(raw, str):
                codec_video_paths = [raw]
            elif isinstance(raw, (list, tuple)) and all(
                isinstance(x, str) for x in raw
            ):
                codec_video_paths = list(raw)
            else:
                # Non-string payload (PIL/ndarray/etc.) - codec backend
                # cannot consume pre-decoded frames; fall through to frame
                # path.
                is_codec_kwarg = False
        is_codec = is_codec_marker or is_codec_kwarg

        if is_codec:
            # Confine codec paths to --allowed-local-media-path (local-only;
            # SSRF / local-file-read protection) and use the *resolved* paths
            # downstream so the codec module opens exactly the file that was
            # validated (no symlink-retarget / validate-vs-open gap).
            codec_video_paths = _validate_video_sources(
                codec_video_paths, self.info.ctx.model_config
            )
            # Codec backend: HF processor consumes the path string directly and
            # performs decode + canvas-packing internally. The dummy ndarray
            # we attached during prepare_codec_video_input is discarded here.
            mm_data["videos"] = (
                codec_video_paths
                if len(codec_video_paths) > 1
                else codec_video_paths[0]
            )
            # Route through the base ``_call_hf_processor`` so float-tensor
            # dtype postprocessing is applied automatically; inject
            # ``video_backend="codec"`` via mm_kwargs so the wrapped processor
            # dispatches to its codec branch.
            output = super()._call_hf_processor(
                prompt=prompt,
                mm_data=mm_data,
                mm_kwargs={**mm_kwargs, "video_backend": "codec"},
                tok_kwargs=tok_kwargs,
            )
            data = dict(output)
            return BatchFeature(
                self._rename_codec_outputs_to_video(
                    data, codec_video_paths, hf_processor
                )
            )

        # ---- Frame backend (registered LlavaOnevision2VideoBackend) ------
        # Every non-codec video reaches here as a ``(frames_ndarray, metadata)``
        # tuple (``video_needs_metadata=True``): the connector decoded + sampled
        # it through ``LlavaOnevision2VideoBackend`` for real ``video_url``
        # inputs, or the dummy-inputs builder attached synthetic metadata during
        # profiling. We materialise the frames as PIL images + per-frame
        # timestamp markers and feed them through the HF processor's *image*
        # branch (smart_resize + patchify), then re-tag the image-series outputs
        # as video-series so vLLM's ``<|video_pad|>`` placeholder replacement
        # finds them. Sampling parity with the original qwen_vl_utils policy is
        # provided by the backend's ``compute_frames_index_to_sample``; SSRF /
        # local-file gating is enforced by the connector before decoding.
        if videos_present:
            timestamp_decimals = int(
                mm_kwargs.get("timestamp_decimals", _DEFAULT_TIMESTAMP_DECIMALS)
            )

            per_video_frames: list[list[Image.Image]] = []
            per_video_timestamps: list[list[float]] = []
            for item in mm_data["videos"]:
                pil_frames, timestamps = _frame_video_to_pil_and_timestamps(item)
                per_video_frames.append(pil_frames)
                per_video_timestamps.append(timestamps)

            # Rewrite the prompt so each video marker becomes a sequence of
            # ``<{t} seconds><|vision_start|><|image_pad|><|vision_end|>``
            # blocks (matches the OV2 hf-chat reference exactly).
            new_prompt = _expand_video_markers_in_prompt(
                prompt,
                per_video_timestamps,
                timestamp_decimals=timestamp_decimals,
            )

            # Build the merged ``images`` list the wrapped HF processor will
            # consume. The processor binds the merged list *positionally* to the
            # ``<|image_pad|>`` slots in prompt order, so it must follow the
            # interleaved marker order of the prompt (not a fixed "videos first"
            # order) -- otherwise mixed image+video requests bind frames to the
            # wrong placeholder. ``row_is_video`` labels each grid row so outputs
            # can be split back into per-modality keys below.
            merged_mm_data = dict(mm_data)
            existing_images = merged_mm_data.pop("images", None)
            caller_images: list[Image.Image] = []
            if existing_images:
                if isinstance(existing_images, list):
                    caller_images.extend(existing_images)
                else:
                    caller_images.append(existing_images)

            marker_pattern = re.compile(
                "(?P<image>" + re.escape(_IMAGE_MARKER) + ")"
                "|(?P<video>" + re.escape(_VIDEO_MARKER) + ")"
            )
            flat_frames: list[Image.Image] = []
            row_is_video: list[bool] = []
            vid_idx = 0
            img_idx = 0
            for marker in marker_pattern.finditer(prompt):
                if marker.lastgroup == "video":
                    frames = per_video_frames[vid_idx]
                    vid_idx += 1
                    flat_frames.extend(frames)
                    row_is_video.extend([True] * len(frames))
                else:
                    flat_frames.append(caller_images[img_idx])
                    img_idx += 1
                    row_is_video.append(False)
            # Fallback: if the prompt carried no parseable markers (defensive --
            # should not happen for well-formed inputs), preserve the legacy
            # "video frames first, then caller images" ordering.
            if vid_idx == 0 and img_idx == 0:
                for frames in per_video_frames:
                    flat_frames.extend(frames)
                    row_is_video.extend([True] * len(frames))
                flat_frames.extend(caller_images)
                row_is_video.extend([False] * len(caller_images))

            merged_mm_data.pop("videos", None)
            merged_mm_data["images"] = flat_frames

            # Route through the base ``_call_hf_processor`` (applies float-tensor
            # dtype postprocessing automatically). The wrapped processor's image
            # branch ignores video/codec-only kwargs and does not forward extra
            # **kwargs to the image processor, so passing the full merged kwarg
            # set here is a no-op beyond return_tensors/padding.
            output = super()._call_hf_processor(
                prompt=new_prompt,
                mm_data=merged_mm_data,
                mm_kwargs=mm_kwargs,
                tok_kwargs=tok_kwargs,
            )
            data = dict(output)

            # Split the image-series processor outputs back into video rows and
            # genuine-image rows (one grid row per frame/image, in flat_frames
            # order). Video rows become video-series keys; caller-image rows
            # stay under image-series keys. Without this split a mixed
            # image+video request silently drops the image modality.
            grid = data.get("image_grid_thw")
            has_caller_images = any(not v for v in row_is_video)
            if grid is not None and has_caller_images:
                pixel_values = data.pop("pixel_values")
                patch_positions = data.pop("patch_positions")

                per_row_patches = grid.prod(-1).tolist()
                row_offsets = [0]
                for p in per_row_patches:
                    row_offsets.append(row_offsets[-1] + int(p))

                def _gather_rows(tensor, rows):
                    if not rows:
                        return tensor[:0]
                    return torch.cat(
                        [tensor[row_offsets[i] : row_offsets[i + 1]] for i in rows],
                        dim=0,
                    )

                video_rows = [i for i, v in enumerate(row_is_video) if v]
                image_rows = [i for i, v in enumerate(row_is_video) if not v]

                data["pixel_values_videos"] = _gather_rows(pixel_values, video_rows)
                data["patch_positions_videos"] = _gather_rows(
                    patch_positions, video_rows
                )
                data["video_grid_thw"] = grid[video_rows]

                data["pixel_values"] = _gather_rows(pixel_values, image_rows)
                data["patch_positions"] = _gather_rows(patch_positions, image_rows)
                data["image_grid_thw"] = grid[image_rows]
            else:
                # Video-only (no caller images): re-tag every image-series
                # output as video-series.
                if "pixel_values" in data:
                    data["pixel_values_videos"] = data.pop("pixel_values")
                if "image_grid_thw" in data:
                    data["video_grid_thw"] = data.pop("image_grid_thw")
                if "patch_positions" in data:
                    data["patch_positions_videos"] = data.pop("patch_positions")
            data["frame_timestamps"] = _pack_timestamps(per_video_timestamps)
            data["video_num_frames"] = torch.tensor(
                [len(ts) for ts in per_video_timestamps], dtype=torch.long
            )
            data["video_is_codec"] = torch.zeros(
                (len(per_video_timestamps),), dtype=torch.long
            )

            return BatchFeature(data)

        # ---- Image-only / text-only call --------------------------------
        # No videos present: delegate to the base ``_call_hf_processor``, which
        # runs the wrapped processor over the (possibly empty) image set and
        # applies float-tensor dtype postprocessing automatically.
        return super()._call_hf_processor(
            prompt=prompt,
            mm_data=mm_data,
            mm_kwargs=mm_kwargs,
            tok_kwargs=tok_kwargs,
        )

    def _rename_codec_outputs_to_video(
        self,
        data: dict,
        codec_video_paths: list[str],
        hf_processor,
    ) -> dict:
        # Codec branch emits image-series keys; vLLM expects video-series.
        # Timestamps are NOT synthesized here: the codec format packs multiple
        # source-frame timestamps into one canvas, so a per-canvas list is
        # ill-defined. We ship per-video fps and let get_video_replacement
        # reconstruct (timestamp, token_count) runs from patch_positions_videos.
        data["pixel_values_videos"] = data.pop("pixel_values")
        video_grid_thw = data.pop("image_grid_thw")
        data["video_grid_thw"] = video_grid_thw
        patch_positions = data.pop("patch_positions")
        data["patch_positions_videos"] = patch_positions

        # Read per-video fps from processor output (order matches
        # codec_video_paths); fall back to _codec_fps_for only if the processor
        # didn't populate codec_fps (older model snapshots).
        processor_fps = data.pop("codec_fps", None)
        per_video_canvas_counts: list[int] = []
        per_video_fps: list[float] = []
        grid_offset = 0
        for idx, video_path in enumerate(codec_video_paths):
            num_canvases = self._count_canvases_for_video(
                patch_positions, video_grid_thw, 0, grid_offset
            )
            per_video_canvas_counts.append(num_canvases)
            if processor_fps is not None and idx < len(processor_fps):
                per_video_fps.append(float(processor_fps[idx]))
            else:
                per_video_fps.append(_codec_fps_for(video_path, hf_processor))
            grid_offset += num_canvases

        data["video_num_frames"] = torch.tensor(
            per_video_canvas_counts, dtype=torch.long
        )
        data["video_is_codec"] = torch.ones(
            (len(per_video_canvas_counts),), dtype=torch.long
        )
        # Stored as scaled int64 (fps * 1e9 rounded) because vLLM casts
        # all float MM fields to model dtype (bfloat16), which rounds
        # 29.97003 fps to 30.0 and shifts every timestamp tag by ~0.1s.
        # Integer storage round-trips losslessly through the field-config
        # pipeline; we divide back to float at replacement time.
        data["codec_fps"] = torch.tensor(
            [int(round(f * 1_000_000_000)) for f in per_video_fps], dtype=torch.int64
        )
        # frame_timestamps is required by the field-config schema; emit an
        # empty per-video list (frames-backend populates it, codec ignores it).
        data["frame_timestamps"] = _pack_timestamps([[] for _ in codec_video_paths])
        return data

    def _count_canvases_for_video(
        self,
        patch_positions: torch.Tensor,
        video_grid_thw: torch.Tensor,
        canvas_offset: int,
        grid_offset: int,
    ) -> int:
        # Single-video case: every remaining grid row belongs to this video.
        # HF merges per-canvas rows into one [N,H,W] row per video, so the
        # canvas count lives in the t-dim, not shape[0].
        return int(video_grid_thw[grid_offset:, 0].sum().item())

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, Any],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        image_processor = self.info.get_image_processor(**hf_processor_mm_kwargs)
        tokenizer = self.info.get_tokenizer()
        vocab = tokenizer.get_vocab()
        image_pad_id = vocab["<|image_pad|>"]
        video_pad_id = vocab["<|video_pad|>"]
        vision_start_id = vocab["<|vision_start|>"]
        vision_end_id = vocab["<|vision_end|>"]
        newline_ids = tokenizer.encode("\n", add_special_tokens=False)
        merge_length = image_processor.merge_size**2
        decimals = int(hf_processor_mm_kwargs.get("timestamp_decimals", 1))

        def get_image_replacement(item_idx: int):
            out_item = out_mm_kwargs["image"][item_idx]
            grid_thw = out_item["image_grid_thw"].data
            n = int(grid_thw.prod(-1).sum()) // merge_length
            return [image_pad_id] * n

        def get_video_replacement(item_idx: int):
            out_item = out_mm_kwargs["video"][item_idx]
            grid_thw = out_item["video_grid_thw"].data
            is_codec_field = out_item.get("video_is_codec")
            is_codec = (
                bool(int(is_codec_field.data.item()))
                if is_codec_field is not None
                else False
            )
            tokens: list[int] = []

            if is_codec:
                # Codec packs multiple source-frame timestamps into one canvas,
                # so timestamps are per-run (consecutive patches sharing the
                # same source-frame t), not per-canvas. Mirrors HF's
                # rewrite_text_with_codec_positions: group patch_positions by
                # run, emit ``<sec seconds><|vision_start|><pad*N><|vision_end|>\n``.
                patch_positions = out_item["patch_positions_videos"].data
                fps_t = out_item["codec_fps"].data
                fps = float(int(fps_t.item())) / 1_000_000_000.0
                runs = _codec_timestamp_runs(
                    patch_positions, fps, image_processor.merge_size
                )
                for sec, token_count in runs:
                    tag = f"<{sec:.{decimals}f} seconds>"
                    tag_ids = tokenizer.encode(tag, add_special_tokens=False)
                    tokens.extend(tag_ids)
                    tokens.append(vision_start_id)
                    tokens.extend([image_pad_id] * token_count)
                    tokens.append(vision_end_id)
                    tokens.extend(newline_ids)
            else:
                timestamps = out_item["frame_timestamps"].data
                T_total = int(grid_thw.shape[0])
                for t in range(T_total):
                    sec = float(timestamps[t].item())
                    tag = f"<{sec:.{decimals}f} seconds>"
                    tag_ids = tokenizer.encode(tag, add_special_tokens=False)
                    n_per_frame = int(grid_thw[t].prod()) // merge_length
                    tokens.extend(tag_ids)
                    tokens.append(vision_start_id)
                    tokens.extend([image_pad_id] * n_per_frame)
                    tokens.append(vision_end_id)
            # Replacement mixes timestamp/marker tokens with image_pad
            # placeholders; only image_pad positions carry vision embeddings,
            # so a partial-embed mask is emitted via select_token_id.
            return PromptUpdateDetails.select_token_id(tokens, image_pad_id)

        return [
            PromptReplacement(
                modality="image",
                target=[image_pad_id],
                replacement=get_image_replacement,
            ),
            PromptReplacement(
                modality="video",
                target=[vision_start_id, video_pad_id, vision_end_id],
                replacement=get_video_replacement,
            ),
        ]

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return _create_field_factory(
            self.info.get_hf_config().vision_config.spatial_merge_size,
        )(hf_inputs)


@MULTIMODAL_REGISTRY.register_processor(
    LlavaOnevision2MultiModalProcessor,
    info=LlavaOnevision2ProcessingInfo,
    dummy_inputs=LlavaOnevision2DummyInputsBuilder,
)
class LlavaOnevision2ForConditionalGeneration(
    nn.Module, SupportsMultiModal, SupportsPP
):
    """vLLM-side OV2 top-level model.

    Weight name rewriting (HF checkpoint → vLLM module tree):
      Prefix rewrites only (longest match wins). Vision tower attribute names
      mirror HF names verbatim, so no substring rules are needed. Substring
      rules would otherwise collide with the Qwen3 text-path ``self_attn``
      modules and break the language-model loader.
        ``model.language_model.``        → ``language_model.model.``
        ``model.visual.``                → ``visual.``
        ``lm_head.``                     → ``language_model.lm_head.``
        ``model.`` (fallback)            → ``language_model.model.``
    """

    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "model.language_model.": "language_model.model.",
            "model.visual.": "visual.",
            "lm_head.": "language_model.lm_head.",
            "model.": "language_model.model.",
        }
    )

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("image"):
            return "<|vision_start|><|image_pad|><|vision_end|>"
        if modality.startswith("video"):
            return "<|vision_start|><|video_pad|><|vision_end|>"
        raise ValueError("Only image or video modality is supported")

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        multimodal_config = vllm_config.model_config.multimodal_config

        self.config = config
        self.multimodal_config = multimodal_config

        # Build the vision tower under the tower marker so it is shared by the
        # image, video-frame and video-codec backends (all routed through
        # ``self.visual``). When both modalities are disabled via
        # ``--limit-mm-per-prompt`` the marker turns the tower into a skipped
        # placeholder whose weights are dropped automatically by the loader.
        with self._mark_tower_model(vllm_config, {"image", "video"}):
            self.visual = LlavaOnevision2VisionTower(
                config.vision_config,
                text_hidden_size=config.text_config.hidden_size,
                norm_eps=getattr(config.vision_config, "layer_norm_eps", 1e-6),
                quant_config=None,
                prefix=maybe_prefix(prefix, "visual"),
            )

        # OV2 LLM is plain Qwen3 -- 1-D positions, no M-RoPE. The wrapper
        # LlavaOnevision2Config keeps text/vision configs nested (unlike OV1.5
        # which promoted text fields), so explicitly hand text_config to the
        # Qwen3 init path or qwen3.py will hit AttributeError on vocab_size.
        self.language_model = init_vllm_registered_model(
            vllm_config=vllm_config,
            hf_config=config.text_config,
            prefix=maybe_prefix(prefix, "language_model"),
            architectures=["Qwen3ForCausalLM"],
        )

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )

    def _validate_and_reshape_mm_tensor(
        self, mm_input: object, name: str
    ) -> torch.Tensor:
        if not isinstance(mm_input, (torch.Tensor, list)):
            raise ValueError(f"Incorrect type of {name}: {type(mm_input)}")
        if isinstance(mm_input, torch.Tensor):
            if mm_input.ndim == 2:
                return mm_input
            if mm_input.ndim != 3:
                raise ValueError(
                    f"{name} must be 2D or batched-3D, got shape={mm_input.shape}"
                )
            # Flatten the leading batch dim into the patch dim: (b, n, d) ->
            # (b*n, d), avoiding a Python list of row tensors.
            return mm_input.flatten(0, 1)
        return torch.concat(mm_input)

    def _parse_and_validate_image_input(
        self, **kwargs: object
    ) -> LlavaOnevision2ImageInputs | None:
        pixel_values = kwargs.pop("pixel_values", None)
        image_embeds = kwargs.pop("image_embeds", None)
        image_grid_thw = kwargs.pop("image_grid_thw", None)
        patch_positions = kwargs.pop("patch_positions", None)

        if pixel_values is None and image_embeds is None:
            return None

        if pixel_values is not None:
            pixel_values = self._validate_and_reshape_mm_tensor(
                pixel_values, "image pixel values"
            )
            image_grid_thw = self._validate_and_reshape_mm_tensor(
                image_grid_thw, "image grid_thw"
            )
            if patch_positions is None:
                raise ValueError(
                    "OV2 requires patch_positions alongside pixel_values; "
                    "ensure the HF processor produces it (image, video-"
                    "frames, and video-codec backends all do)."
                )
            patch_positions = self._validate_and_reshape_mm_tensor(
                patch_positions, "image patch_positions"
            )
            return LlavaOnevision2ImagePixelInputs(
                type="pixel_values",
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                patch_positions=patch_positions,
            )

        # image_embeds path
        image_embeds = self._validate_and_reshape_mm_tensor(
            image_embeds, "image embeds"
        )
        image_grid_thw = self._validate_and_reshape_mm_tensor(
            image_grid_thw, "image grid_thw"
        )
        return LlavaOnevision2ImageEmbeddingInputs(
            type="image_embeds",
            image_embeds=image_embeds,
            image_grid_thw=image_grid_thw,
        )

    def _process_image_input(
        self, image_input: LlavaOnevision2ImageInputs
    ) -> tuple[torch.Tensor, ...]:
        grid_thw = image_input["image_grid_thw"]
        assert grid_thw.ndim == 2

        if image_input["type"] == "image_embeds":
            image_embeds = image_input["image_embeds"]
        else:
            image_embeds = self.visual(
                image_input["pixel_values"],
                grid_thw=grid_thw,
                patch_positions=image_input["patch_positions"],
            )

        merge_size = self.visual.spatial_merge_size
        sizes = grid_thw.prod(-1) // merge_size // merge_size
        return image_embeds.split(sizes.tolist())

    def _parse_and_validate_video_input(
        self, **kwargs: object
    ) -> LlavaOnevision2VideoPixelInputs | None:
        pixel_values_videos = kwargs.pop("pixel_values_videos", None)
        video_grid_thw = kwargs.pop("video_grid_thw", None)
        patch_positions_videos = kwargs.pop("patch_positions_videos", None)
        video_num_frames = kwargs.pop("video_num_frames", None)
        kwargs.pop("frame_timestamps", None)
        kwargs.pop("video_is_codec", None)

        if pixel_values_videos is None:
            return None

        pixel_values_videos = self._validate_and_reshape_mm_tensor(
            pixel_values_videos, "video pixel values"
        )
        video_grid_thw = self._validate_and_reshape_mm_tensor(
            video_grid_thw, "video grid_thw"
        )
        if patch_positions_videos is None:
            raise ValueError(
                "OV2 requires patch_positions_videos alongside pixel_values_videos."
            )
        patch_positions_videos = self._validate_and_reshape_mm_tensor(
            patch_positions_videos, "video patch_positions"
        )
        if video_num_frames is None:
            raise ValueError(
                "OV2 requires video_num_frames alongside pixel_values_videos."
            )
        if isinstance(video_num_frames, list):
            video_num_frames = torch.cat([v.flatten() for v in video_num_frames])
        else:
            video_num_frames = video_num_frames.flatten()
        return LlavaOnevision2VideoPixelInputs(
            type="pixel_values_videos",
            pixel_values_videos=pixel_values_videos,
            video_grid_thw=video_grid_thw,
            patch_positions_videos=patch_positions_videos,
            video_num_frames=video_num_frames,
        )

    def _process_video_input(
        self,
        video_input: LlavaOnevision2VideoPixelInputs,
    ) -> tuple[torch.Tensor, ...]:
        # OV2 encodes each video frame independently through the same
        # vision stack as still images, then splits the resulting token
        # stream into per-video chunks using video_num_frames.
        grid_thw = video_input["video_grid_thw"]
        assert grid_thw.ndim == 2
        video_embeds = self.visual(
            video_input["pixel_values_videos"],
            grid_thw=grid_thw,
            patch_positions=video_input["patch_positions_videos"],
        )
        merge_size = self.visual.spatial_merge_size
        per_frame_tokens = grid_thw.prod(-1) // merge_size // merge_size
        # Aggregate per-frame token counts into per-video token counts.
        num_frames = video_input["video_num_frames"].tolist()
        sizes: list[int] = []
        cursor = 0
        for n in num_frames:
            n = int(n)
            sizes.append(int(per_frame_tokens[cursor : cursor + n].sum()))
            cursor += n
        return video_embeds.split(sizes)

    def _parse_and_validate_multimodal_inputs(self, **kwargs: object) -> dict:
        modalities = {}
        for key in kwargs:
            if key in ("pixel_values", "image_embeds") and "images" not in modalities:
                modalities["images"] = self._parse_and_validate_image_input(**kwargs)
            if key == "pixel_values_videos" and "videos" not in modalities:
                modalities["videos"] = self._parse_and_validate_video_input(**kwargs)
        return modalities

    def get_language_model(self) -> torch.nn.Module:
        return self.language_model

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        modalities = self._parse_and_validate_multimodal_inputs(**kwargs)
        if not modalities:
            return []
        multimodal_embeddings: tuple[torch.Tensor, ...] = ()
        for modality in modalities:
            if modality == "images":
                multimodal_embeddings += self._process_image_input(modalities["images"])
            elif modality == "videos":
                multimodal_embeddings += self._process_video_input(modalities["videos"])
        return multimodal_embeddings

    def get_input_embeddings(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: MultiModalEmbeddings | None = None,
    ) -> torch.Tensor:
        inputs_embeds = self.language_model.get_input_embeddings(input_ids)
        if multimodal_embeddings is not None and len(multimodal_embeddings) != 0:
            inputs_embeds = merge_multimodal_embeddings(
                inputs_embeds,
                multimodal_embeddings,
                input_ids == self.config.image_token_id,
            )
        return inputs_embeds

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors:
        # V1 precomputes multimodal ``inputs_embeds`` via ``embed_multimodal`` /
        # ``get_input_embeddings``, so the model forward only threads them into
        # the language model (image + video share the same embedding merge).
        if intermediate_tensors is not None:
            inputs_embeds = None

        return self.language_model.model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )

    def compute_logits(self, hidden_states: torch.Tensor):
        return self.language_model.compute_logits(hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)

    def get_mm_mapping(self) -> MultiModelKeys:
        return MultiModelKeys.from_string_field(
            language_model="language_model",
            connector="visual.merger.",
            tower_model="visual.",
        )
