# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Vendored HuggingFace-convention processors for the Inkling Titan model.

Implements the image processor (numba patchifier), the audio feature extractor
(STFT/dMel path), the composite processor, and the MM token-id constants.

Besides raw bytes / file paths, the extractors also accept the dummy inputs
vLLM generates during profiling (PIL images / numpy audio arrays).
"""

from __future__ import annotations

import io
import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from numba import njit
from transformers.feature_extraction_utils import (
    BatchFeature,
    FeatureExtractionMixin,
)
from transformers.image_processing_utils import BaseImageProcessor
from transformers.image_utils import ImageInput

# ---------------------------------------------------------------------------
# MM token-id constants
# ---------------------------------------------------------------------------

# Block-start marker token ids. These are real tokens the model was trained on
# (``<|content_image|>`` / ``<|content_audio_input|>``) that mark the start of an
# image/audio embedding block; they are kept verbatim in ``input_ids``.
IMAGE_MARKER_ID = 200005  # <|content_image|>
AUDIO_MARKER_ID = 200020  # <|content_audio_input|>

# Per-patch / per-frame placeholder ids marking where tower embeddings are
# scattered in. These are unused slots in the padded vocabulary and the
# corresponding positions are always overwritten by tower embeddings.
IMAGE_TOKEN_ID = 200054  # <|unused_200054|>
AUDIO_TOKEN_ID = 200053  # <|unused_200053|>

# ---------------------------------------------------------------------------
# Image processing
# ---------------------------------------------------------------------------

IMAGE_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
IMAGE_STD = np.array([0.26862954, 0.2613026, 0.2757771], dtype=np.float32)
PAD_RAW_VALUE = np.float32(-1.0 / 255.0)
PAD_NORM = (np.full((3,), PAD_RAW_VALUE, dtype=np.float32) - IMAGE_MEAN) / IMAGE_STD


def _validate_image_rescale(
    rescale_image_frac: float | None,
    rescale_image_max_upscaled_long_edge: int | None,
) -> None:
    if rescale_image_frac is not None and (
        not math.isfinite(rescale_image_frac) or rescale_image_frac <= 0
    ):
        raise ValueError(
            "rescale_image_frac must be positive and finite or None, "
            f"got {rescale_image_frac}"
        )
    if rescale_image_max_upscaled_long_edge is None:
        return
    if rescale_image_max_upscaled_long_edge <= 0:
        raise ValueError(
            "rescale_image_max_upscaled_long_edge must be positive or None, "
            f"got {rescale_image_max_upscaled_long_edge}"
        )
    if rescale_image_frac is None or rescale_image_frac <= 1.0:
        raise ValueError(
            "rescale_image_max_upscaled_long_edge requires rescale_image_frac > 1, "
            f"got {rescale_image_frac}"
        )


def _scaled_image_dimensions(
    width: int,
    height: int,
    rescale_image_frac: float | None,
    rescale_image_max_upscaled_long_edge: int | None,
) -> tuple[int, int]:
    """Return the long-edge-scaled ``(width, height)``."""
    if rescale_image_frac is None:
        return width, height

    long_edge = max(width, height)
    if long_edge == 0:
        return width, height

    target_long_edge = float(long_edge) * rescale_image_frac
    if rescale_image_max_upscaled_long_edge is not None:
        effective_cap = max(rescale_image_max_upscaled_long_edge, long_edge)
        target_long_edge = min(target_long_edge, float(effective_cap))

    ratio = target_long_edge / float(long_edge)
    if ratio == 1.0:
        return width, height

    def scale(value: int) -> int:
        return max(1, math.floor(float(value) * ratio + 0.5))

    return scale(width), scale(height)


def _load_image_bytes(image) -> bytes:
    """Encode a PIL image as raw PNG bytes for preprocessing.

    The HF processor is always handed ``PIL.Image`` instances by vLLM, so no
    other input types need to be supported here.
    """
    if image.mode != "RGB":
        image = image.convert("RGB")
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


@njit(cache=True)
def _fill_patches_numba(
    arr: np.ndarray,
    patch_size: int,
    patches: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    pad_norm: np.ndarray,
) -> None:
    h = arr.shape[0]
    w = arr.shape[1]
    nph = (h + patch_size - 1) // patch_size
    npw = w // patch_size + 1
    inv255 = np.float32(1.0 / 255.0)

    for k in range(nph * npw):
        i = k // npw
        j = k - i * npw
        y_base = i * patch_size
        x_base = j * patch_size

        for y in range(patch_size):
            iy = y_base + y
            for x in range(patch_size):
                ix = x_base + x
                if iy < h and ix < w:
                    for c in range(3):
                        raw = np.float32(arr[iy, ix, c]) * inv255
                        patches[k, y, x, c] = (raw - mean[c]) / std[c]
                else:
                    for c in range(3):
                        patches[k, y, x, c] = pad_norm[c]


def _encode_image_bytes(
    image_bytes: bytes,
    *,
    patch_size: int,
    rescale_image_frac: float | None,
    rescale_image_max_upscaled_long_edge: int | None,
) -> torch.Tensor:
    if patch_size <= 0:
        raise ValueError("patch_size must be greater than zero")
    _validate_image_rescale(
        rescale_image_frac,
        rescale_image_max_upscaled_long_edge,
    )

    from PIL import Image

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    scaled_size = _scaled_image_dimensions(
        image.width,
        image.height,
        rescale_image_frac=rescale_image_frac,
        rescale_image_max_upscaled_long_edge=rescale_image_max_upscaled_long_edge,
    )
    if scaled_size != image.size:
        image = image.resize(scaled_size, resample=Image.Resampling.LANCZOS)
    arr = np.array(image, dtype=np.uint8, copy=True)
    height, width, _ = arr.shape

    nph = (height + patch_size - 1) // patch_size
    npw = width // patch_size + 1
    num_patches = nph * npw

    patches = np.empty((num_patches, patch_size, patch_size, 3), dtype=np.float32)
    _fill_patches_numba(arr, patch_size, patches, IMAGE_MEAN, IMAGE_STD, PAD_NORM)

    return (
        torch.from_numpy(patches)
        .to(torch.bfloat16)
        .view(num_patches, 1, patch_size, patch_size, 3)
        .expand(num_patches, 2, patch_size, patch_size, 3)
    )


class InklingImageProcessor(BaseImageProcessor):
    r"""Turn raw images into ``vision_patches_bthwc`` for Inkling hMLP.

    ``rescale_image_frac`` scales the long edge while preserving aspect ratio.
    ``rescale_image_max_upscaled_long_edge`` optionally caps only upscaling and
    therefore requires a scale factor greater than one. The defaults, ``2.0`` and
    ``2048``, grow images toward a 2048-pixel long edge by at most 2x, while leaving
    images already at or above 2048 unchanged.
    """

    model_input_names = ["vision_patches_bthwc"]

    def __init__(
        self,
        patch_size: int = 40,
        rescale_image_frac: float | None = 2.0,
        rescale_image_max_upscaled_long_edge: int | None = 2048,
        **kwargs,
    ):
        if patch_size <= 0:
            raise ValueError("patch_size must be greater than zero")
        _validate_image_rescale(
            rescale_image_frac,
            rescale_image_max_upscaled_long_edge,
        )
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.rescale_image_frac = rescale_image_frac
        self.rescale_image_max_upscaled_long_edge = rescale_image_max_upscaled_long_edge

    def _encode_one(self, image) -> torch.Tensor:
        return _encode_image_bytes(
            _load_image_bytes(image),
            patch_size=self.patch_size,
            rescale_image_frac=self.rescale_image_frac,
            rescale_image_max_upscaled_long_edge=self.rescale_image_max_upscaled_long_edge,
        )

    def preprocess(
        self,
        images: ImageInput | list,
        return_tensors: str | None = "pt",
        **kwargs,
    ) -> BatchFeature:
        del return_tensors, kwargs
        if not isinstance(images, (list, tuple)):
            images = [images]

        per_image_patches: list[torch.Tensor] = []
        num_patches: list[int] = []
        num_tokens: list[int] = []
        for img in images:
            vp = self._encode_one(img)
            n_patches = int(vp.shape[0])
            per_image_patches.append(vp)
            num_patches.append(n_patches)
            num_tokens.append(n_patches)

        if len(per_image_patches) == 1:
            vision_patches_bthwc = per_image_patches[0]
        elif per_image_patches:
            vision_patches_bthwc = torch.cat(per_image_patches, dim=0)
        else:
            vision_patches_bthwc = torch.empty(0)

        data = {
            "vision_patches_bthwc": vision_patches_bthwc,
            "num_patches": num_patches,
            "num_tokens": num_tokens,
        }
        return BatchFeature(data=data, tensor_type=None)


# ---------------------------------------------------------------------------
# Audio feature extraction
# ---------------------------------------------------------------------------


@dataclass
class InklingAudioEncoderParams:
    """Audio preprocessing parameters used to convert raw audio into dMel bins."""

    sample_rate: int = 16_000
    window_size_multiplier: float = 2.0
    n_fft: int | None = None
    n_mels: int = 80
    num_dmel_bins: int = 16
    dmel_min_value: float = -7.0
    dmel_max_value: float = 2.0
    audio_token_duration_s: float = 0.05


def _to_exact_int(value: float, name: str, tolerance: float = 1e-6) -> int:
    rounded = round(value)
    if abs(value - rounded) > tolerance:
        raise ValueError(f"{name} must resolve to an integer sample count, got {value}")
    return int(rounded)


def _hz_to_mel(frequencies: np.ndarray) -> np.ndarray:
    """Slaney mel scale, matching the librosa/torchaudio convention."""
    frequencies = np.asarray(frequencies, dtype=np.float64)
    f_sp = 200.0 / 3.0
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = np.log(6.4) / 27.0
    linear = frequencies / f_sp
    log = (
        min_log_mel + np.log(np.maximum(frequencies, min_log_hz) / min_log_hz) / logstep
    )
    return np.where(frequencies >= min_log_hz, log, linear)


def _mel_to_hz(mels: np.ndarray) -> np.ndarray:
    mels = np.asarray(mels, dtype=np.float64)
    f_sp = 200.0 / 3.0
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = np.log(6.4) / 27.0
    linear = mels * f_sp
    log = min_log_hz * np.exp(logstep * (mels - min_log_mel))
    return np.where(mels >= min_log_mel, log, linear)


_MEL_BASIS_CACHE: dict[tuple[int, int, int], torch.Tensor] = {}


def _mel_basis(sample_rate: int, n_fft: int, n_mels: int) -> torch.Tensor:
    key = (sample_rate, n_fft, n_mels)
    cached = _MEL_BASIS_CACHE.get(key)
    if cached is not None:
        return cached

    fft_bins = n_fft // 2 + 1
    fft_freqs = np.arange(fft_bins, dtype=np.float64) * sample_rate / n_fft
    mel_edges = _mel_to_hz(
        np.linspace(
            _hz_to_mel(np.array([0.0]))[0],
            _hz_to_mel(np.array([sample_rate / 2.0]))[0],
            n_mels + 2,
            dtype=np.float64,
        )
    )
    mel_widths = np.diff(mel_edges)
    lower = (fft_freqs[None, :] - mel_edges[:-2, None]) / mel_widths[:-1, None]
    upper = (mel_edges[2:, None] - fft_freqs[None, :]) / mel_widths[1:, None]
    weights = np.maximum(0.0, np.minimum(lower, upper))

    # Slaney area normalization.
    weights *= (2.0 / (mel_edges[2:] - mel_edges[:-2]))[:, None]
    basis = torch.from_numpy(weights.astype(np.float32, copy=False)).contiguous()
    _MEL_BASIS_CACHE[key] = basis
    return basis


def _dmel_bins(audio: torch.Tensor, params: InklingAudioEncoderParams) -> torch.Tensor:
    hop_length = _to_exact_int(
        params.audio_token_duration_s * params.sample_rate,
        "audio_token_duration_s * sample_rate",
    )
    window_size = _to_exact_int(
        params.audio_token_duration_s
        * params.window_size_multiplier
        * params.sample_rate,
        "audio_token_duration_s * window_size_multiplier * sample_rate",
    )
    n_fft = params.n_fft or window_size
    if hop_length <= 0 or window_size <= 0 or n_fft <= 0:
        raise ValueError("audio hop length, window size, and n_fft must be positive")
    if audio.numel() == 0:
        return torch.empty((0, params.n_mels), dtype=torch.int32)

    right_pad = math.ceil(audio.numel() / hop_length) * hop_length - audio.numel()
    left_pad = max(n_fft - hop_length, 0)
    audio = F.pad(audio, (left_pad, right_pad))

    window = torch.hann_window(window_size, periodic=True, dtype=torch.float32)
    spec = torch.stft(
        audio.unsqueeze(0),
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=window_size,
        window=window,
        center=False,
        normalized=False,
        onesided=True,
        return_complex=True,
    )
    spec_ri = torch.view_as_real(spec)
    magnitude = (
        (spec_ri[..., 0].square() + spec_ri[..., 1].square())
        .clamp_min(1e-10)
        .sqrt()
        .squeeze(0)
    )

    mel = (
        _mel_basis(params.sample_rate, n_fft, params.n_mels)
        .matmul(magnitude)
        .clamp_min(1e-10)
        .log10()
    )
    mel = mel.to(torch.float64).clamp(
        min=params.dmel_min_value, max=params.dmel_max_value
    )
    bin_centers = torch.linspace(
        params.dmel_min_value,
        params.dmel_max_value,
        params.num_dmel_bins,
        dtype=torch.float64,
    )
    dmel_bins = (mel.unsqueeze(-1) - bin_centers).abs().argmin(dim=-1)
    return dmel_bins.to(torch.int32).T.contiguous()


class InklingAudioFeatureExtractor(FeatureExtractionMixin):
    """Convert raw audio into Inkling dMel bins in the HF feature-extractor API."""

    model_input_names = ["dmel_bins"]

    def __init__(self, params: dict | None = None, **kwargs):
        super().__init__(**kwargs)
        merged = InklingAudioEncoderParams()
        if params:
            for k, v in params.items():
                if hasattr(merged, k):
                    setattr(merged, k, v)
        # also accept flat kwargs (HF config style)
        for k in list(kwargs.keys()):
            if hasattr(merged, k):
                setattr(merged, k, kwargs[k])
        self.params = merged

    def _decode_one(self, audio) -> torch.Tensor:
        # vLLM hands the feature extractor numpy arrays (the dummy-input builder
        # during profiling, and MultiModalDataParser after resampling to the
        # target sample rate), so no other input types need to be supported.
        return torch.from_numpy(
            np.ascontiguousarray(audio.astype(np.float32, copy=False))
        ).flatten()

    def _encode_one(self, audio) -> torch.Tensor:
        return _dmel_bins(self._decode_one(audio), self.params)

    def __call__(
        self,
        audios: Sequence | None,
        return_tensors: str | None = None,
        **kwargs,
    ) -> BatchFeature:
        del return_tensors, kwargs
        if audios is None:
            audios = []
        if not isinstance(audios, (list, tuple)):
            audios = [audios]

        dmel_bins = [self._encode_one(a) for a in audios]
        data = {
            # per-clip feature: dmel bins as float32 [T, n_mels]
            "dmel_bins": [bins.to(torch.float32) for bins in dmel_bins],
            "num_audio_tokens": [int(bins.shape[0]) for bins in dmel_bins],
        }
        # return_tensors intentionally ignored: per-clip features have ragged T.
        return BatchFeature(data=data, tensor_type=None)


# ---------------------------------------------------------------------------
# Composite processor
# ---------------------------------------------------------------------------


class InklingProcessor:
    """Bundle Inkling image + audio preprocessing with the MM token ids."""

    def __init__(
        self,
        image_processor: InklingImageProcessor | None = None,
        audio_feature_extractor: InklingAudioFeatureExtractor | None = None,
        tokenizer=None,
    ):
        self.image_processor = image_processor or InklingImageProcessor()
        self.audio_feature_extractor = (
            audio_feature_extractor or InklingAudioFeatureExtractor()
        )
        self.tokenizer = tokenizer

    def process_images(self, images: list):
        """Raw images -> BatchFeature(vision_patches_bthwc, num_patches, num_tokens)."""
        return self.image_processor.preprocess(images, return_tensors="pt")

    def process_audios(self, audios: list):
        """Raw audios -> BatchFeature(dmel_bins, num_audio_tokens)."""
        return self.audio_feature_extractor(audios)


__all__ = [
    "InklingImageProcessor",
    "InklingAudioFeatureExtractor",
    "InklingAudioEncoderParams",
    "InklingProcessor",
    "IMAGE_MARKER_ID",
    "AUDIO_MARKER_ID",
    "IMAGE_TOKEN_ID",
    "AUDIO_TOKEN_ID",
]
