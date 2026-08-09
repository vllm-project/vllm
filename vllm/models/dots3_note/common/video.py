# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Train-consistent video preprocessing for Dots3Note."""

from __future__ import annotations

import hashlib
import io
import math
import os
import random
from dataclasses import dataclass
from typing import Literal

import numpy as np
from PIL import Image

_ALIGN = 28
_MIN_FRAMES = 4
_PF_FLOOR = 128
_PF_CEIL = 1024
_FPS_CAP = 1.0
_FPS_MIN = 0.2
_FRAME_OVERHEAD = 15
_BUDGET_OVERHEAD = 2240
_INTERLEAVE_MIN_SECONDS = 1.0
_AUDIO_SAMPLE_RATE = 16000
_AUDIO_SAMPLES_PER_TOKEN = 1280
_AUDIO_CHUNK_SECONDS = 30


@dataclass(frozen=True)
class Dots3NoteVideoPart:
    kind: Literal["text", "image", "audio"]
    value: str | Image.Image | np.ndarray


def _token_len(tokenizer, text: str) -> int:
    if not text:
        return 0
    if hasattr(tokenizer, "encode"):
        return len(tokenizer.encode(text, add_special_tokens=False))
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def _compute_target_size(
    orig_h: int,
    orig_w: int,
    min_pixels: int,
    max_pixels: int,
) -> tuple[int, int]:
    height = max(_ALIGN, round(orig_h / _ALIGN) * _ALIGN)
    width = max(_ALIGN, round(orig_w / _ALIGN) * _ALIGN)
    if height * width > max_pixels:
        beta = math.sqrt(orig_h * orig_w / max_pixels)
        height = max(_ALIGN, math.floor(orig_h / beta / _ALIGN) * _ALIGN)
        width = max(_ALIGN, math.floor(orig_w / beta / _ALIGN) * _ALIGN)
    elif height * width < min_pixels:
        beta = math.sqrt(min_pixels / max(1, orig_h * orig_w))
        height = math.ceil(orig_h * beta / _ALIGN) * _ALIGN
        width = math.ceil(orig_w * beta / _ALIGN) * _ALIGN
        if height * width > max_pixels:
            beta = math.sqrt(height * width / max_pixels)
            height = max(_ALIGN, math.floor(height / beta / _ALIGN) * _ALIGN)
            width = max(_ALIGN, math.floor(width / beta / _ALIGN) * _ALIGN)
    return int(height), int(width)


def _real_patches_at(
    orig_h: int,
    orig_w: int,
    patch_cap: int,
) -> int:
    target_h, target_w = _compute_target_size(
        orig_h,
        orig_w,
        _PF_FLOOR * _ALIGN * _ALIGN,
        max(_PF_FLOOR, patch_cap) * _ALIGN * _ALIGN,
    )
    return (target_h // _ALIGN) * (target_w // _ALIGN)


def _frame_hard_cap(seq_length: int) -> int:
    need = max(1, (seq_length - _BUDGET_OVERHEAD) // (_PF_FLOOR + 15))
    if need <= 1024:
        return 1024
    cap = 1
    while cap < need:
        cap <<= 1
    return cap


def _solve_degrade(
    visual_budget: int,
    duration: float,
    orig_h: int,
    orig_w: int,
    orig_fps: float,
    seq_length: int,
) -> tuple[int, int]:
    aligned_h = max(_ALIGN, round(orig_h / _ALIGN) * _ALIGN)
    aligned_w = max(_ALIGN, round(orig_w / _ALIGN) * _ALIGN)
    orig_max_pf = (aligned_h // _ALIGN) * (aligned_w // _ALIGN)
    fps_cap = min(_FPS_CAP, max(orig_fps, 1e-6))
    pf_cap = min(_PF_CEIL, max(orig_max_pf, _PF_FLOOR))
    frame_cap = _frame_hard_cap(seq_length)

    def usage(scale: float) -> tuple[int, float, int, int]:
        fps = _FPS_MIN + scale * (fps_cap - _FPS_MIN)
        patch_cap = _PF_FLOOR + scale * (pf_cap - _PF_FLOOR)
        num_frames = max(
            _MIN_FRAMES,
            min(int(round(duration * fps)), frame_cap),
        )
        patches = _real_patches_at(
            orig_h,
            orig_w,
            int(round(patch_cap)),
        )
        return (
            num_frames * (patches + _FRAME_OVERHEAD),
            fps,
            int(round(patch_cap)),
            num_frames,
        )

    if usage(1.0)[0] <= visual_budget:
        _, _, patch_cap, num_frames = usage(1.0)
        return num_frames, patch_cap

    floor_cost = _real_patches_at(orig_h, orig_w, _PF_FLOOR) + _FRAME_OVERHEAD
    if usage(0.0)[0] > visual_budget:
        return max(
            _MIN_FRAMES,
            min(visual_budget // floor_cost, frame_cap),
        ), _PF_FLOOR

    low, high = 0.0, 1.0
    for _ in range(50):
        mid = (low + high) / 2
        if usage(mid)[0] <= visual_budget:
            low = mid
        else:
            high = mid
    _, _, patch_cap, num_frames = usage(low)
    return num_frames, patch_cap


def _audio_tokens(duration: float, sample_rate: int) -> int:
    if duration <= 0:
        return 0
    total_samples = int(duration * sample_rate)
    chunk_samples = _AUDIO_CHUNK_SECONDS * sample_rate
    num_tokens = 0
    position = 0
    while position < total_samples:
        length = min(chunk_samples, total_samples - position)
        num_tokens += math.ceil(length / _AUDIO_SAMPLES_PER_TOKEN)
        position += chunk_samples
    return num_tokens + 2


def _decode_audio(
    video_bytes: bytes,
    sample_rate: int,
) -> tuple[np.ndarray | None, float]:
    from torchcodec.decoders import AudioDecoder

    try:
        decoder = AudioDecoder(
            video_bytes,
            sample_rate=sample_rate,
            num_channels=1,
        )
        samples = decoder.get_all_samples()
    except Exception:
        return None, 0.0

    waveform = samples.data
    if waveform is None or waveform.numel() == 0:
        return None, 0.0
    if waveform.ndim == 2:
        waveform = waveform.mean(dim=0) if waveform.shape[0] > 1 else waveform[0]
    array = waveform.cpu().numpy()
    pcm = (np.clip(array, -1.0, 1.0) * 32767.0).astype(np.int16)
    return pcm, float(pcm.shape[0]) / sample_rate


def _open_video(video_bytes: bytes):
    from torchcodec.decoders import VideoDecoder

    try:
        return VideoDecoder(
            video_bytes,
            dimension_order="NHWC",
            num_ffmpeg_threads=1,
            seek_mode="approximate",
        )
    except TypeError:
        return VideoDecoder(
            video_bytes,
            dimension_order="NHWC",
            num_ffmpeg_threads=1,
        )


def _jpeg_roundtrip(image: Image.Image, quality: int) -> Image.Image:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    with Image.open(buffer) as decoded:
        return decoded.convert("RGB").copy()


def _decode_frames(
    video_bytes: bytes,
    visual_budget: int,
    seq_length: int,
    jpeg_quality: int,
) -> tuple[list[tuple[float, Image.Image]], float]:
    decoder = _open_video(video_bytes)
    metadata = decoder.metadata
    duration = float(metadata.duration_seconds or 0)
    orig_h = int(metadata.height)
    orig_w = int(metadata.width)
    total_frames = int(getattr(metadata, "num_frames", 0) or 0)
    orig_fps = float(getattr(metadata, "average_fps", 0) or 0) or 25.0
    if duration <= 0 or orig_h <= 0 or orig_w <= 0:
        raise ValueError(
            "Invalid video metadata: "
            f"duration={duration}, height={orig_h}, width={orig_w}"
        )
    if total_frames <= 0:
        total_frames = max(1, int(duration * orig_fps))

    num_frames, patch_cap = _solve_degrade(
        visual_budget,
        duration,
        orig_h,
        orig_w,
        orig_fps,
        seq_length,
    )
    aligned_h = max(_ALIGN, round(orig_h / _ALIGN) * _ALIGN)
    aligned_w = max(_ALIGN, round(orig_w / _ALIGN) * _ALIGN)
    orig_max_pf = (aligned_h // _ALIGN) * (aligned_w // _ALIGN)
    max_pixels = min(patch_cap, orig_max_pf) * _ALIGN * _ALIGN
    target_h, target_w = _compute_target_size(
        orig_h,
        orig_w,
        _PF_FLOOR * _ALIGN * _ALIGN,
        max_pixels,
    )
    num_frames = max(_MIN_FRAMES, min(num_frames, total_frames))
    if num_frames == 1:
        indices = [0]
    else:
        step = (total_frames - 1) / (num_frames - 1)
        indices = [int(round(i * step)) for i in range(num_frames)]
    indices = sorted({max(0, min(index, total_frames - 1)) for index in indices})

    try:
        decoded = decoder.get_frames_at(indices=indices).data
    except (IndexError, RuntimeError):
        safe = list(indices)
        while safe and safe[-1] > 0:
            safe.pop()
            try:
                decoded = decoder.get_frames_at(indices=safe).data
                indices = safe
                break
            except (IndexError, RuntimeError):
                continue
        else:
            raise

    frames: list[tuple[float, Image.Image]] = []
    for index, frame in zip(indices, decoded):
        array = frame.cpu().numpy() if hasattr(frame, "cpu") else np.asarray(frame)
        image = Image.fromarray(array)
        if image.size != (target_w, target_h):
            image = image.resize((target_w, target_h), Image.Resampling.BICUBIC)
        frames.append(
            (round(index / orig_fps, 3), _jpeg_roundtrip(image, jpeg_quality))
        )
    return frames, duration


def _decoded_frames(video) -> tuple[np.ndarray, float]:
    metadata = None
    if isinstance(video, tuple):
        video, metadata = video
    frames = np.asarray(video)
    if frames.ndim != 4 or frames.shape[-1] not in (3, 4):
        raise TypeError(
            "Decoded NOTE video must have shape (frames, height, width, channels)"
        )
    fps = float((metadata or {}).get("fps", 1.0)) if metadata else 1.0
    return frames, max(fps, 1e-6)


def _prepare_decoded_frames(
    video,
    visual_budget: int,
    seq_length: int,
    jpeg_quality: int,
) -> tuple[list[tuple[float, Image.Image]], float]:
    frames, fps = _decoded_frames(video)
    duration = len(frames) / fps
    orig_h, orig_w = frames.shape[1:3]
    num_frames, patch_cap = _solve_degrade(
        visual_budget,
        duration,
        orig_h,
        orig_w,
        fps,
        seq_length,
    )
    num_frames = min(max(1, num_frames), len(frames))
    indices = np.linspace(0, len(frames) - 1, num_frames).round().astype(int)
    target_h, target_w = _compute_target_size(
        orig_h,
        orig_w,
        _PF_FLOOR * _ALIGN * _ALIGN,
        patch_cap * _ALIGN * _ALIGN,
    )
    output = []
    for index in sorted(set(indices.tolist())):
        image = Image.fromarray(frames[index]).convert("RGB")
        if image.size != (target_w, target_h):
            image = image.resize((target_w, target_h), Image.Resampling.BICUBIC)
        output.append((round(index / fps, 3), _jpeg_roundtrip(image, jpeg_quality)))
    return output, duration


def _format_timestamp(seconds: float) -> str:
    total_centiseconds = int(round(max(seconds, 0.0) * 100))
    hours = total_centiseconds // 360000
    minutes = (total_centiseconds // 6000) % 60
    secs = (total_centiseconds // 100) % 60
    centiseconds = total_centiseconds % 100
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{centiseconds:02d}"


def _group_bounds(
    num_frames: int,
    duration: float,
    mode: str,
    rng: random.Random,
) -> list[int]:
    if num_frames <= 1 or duration <= 0:
        return [0, num_frames]
    max_groups = min(
        num_frames,
        max(1, int(duration // _INTERLEAVE_MIN_SECONDS)),
    )
    if mode == "whole" or max_groups <= 1:
        groups = 1
    elif mode == "eval30":
        groups = round(math.sqrt(max_groups))
    elif mode == "eval_ek":
        groups = round((max_groups - 1) / math.log(max_groups))
    elif mode == "logk":
        groups = round(math.exp(rng.uniform(0.0, math.log(max_groups))))
    else:
        raise ValueError(f"Unsupported video k_mode: {mode}")
    groups = max(1, min(max_groups, groups))
    if groups == 1:
        return [0, num_frames]
    if mode == "logk":
        cuts = sorted(rng.sample(range(1, num_frames), groups - 1))
    else:
        cuts = sorted(
            {
                round(index * num_frames / groups)
                for index in range(1, groups)
                if 0 < round(index * num_frames / groups) < num_frames
            }
        )
    return [0, *cuts, num_frames]


def preprocess_dots3_note_video(
    video,
    *,
    tokenizer,
    question: str,
    seq: int,
    output_reserve: int | None = None,
    audio_cap: float = 1.0,
    audio_sample_rate: int = _AUDIO_SAMPLE_RATE,
    k_mode: str = "eval_ek",
    max_new_tokens: int = 0,
) -> list[Dots3NoteVideoPart]:
    """Expand one video into timestamped, interleaved image/audio parts."""
    if seq <= 0:
        raise ValueError(f"seq must be positive, got {seq}")
    configured_reserve = seq // 4 if output_reserve is None else output_reserve
    effective_reserve = max(configured_reserve, max_new_tokens)
    if effective_reserve >= seq:
        raise ValueError(
            "output_reserve/max_new_tokens must leave room for video input"
        )
    if audio_cap < 0:
        raise ValueError(f"audio_cap must be non-negative, got {audio_cap}")
    if audio_sample_rate <= 0:
        raise ValueError(f"audio_sample_rate must be positive, got {audio_sample_rate}")

    seq_length = seq - effective_reserve
    jpeg_quality = int(os.environ.get("XHS_VIDEO_JPEG_QUALITY", "85"))
    pcm: np.ndarray | None = None
    audio_duration = 0.0
    if isinstance(video, bytes) and audio_cap > 0:
        pcm, audio_duration = _decode_audio(video, audio_sample_rate)

    audio_token_count = (
        _audio_tokens(audio_duration, audio_sample_rate) if pcm is not None else 0
    )
    estimated_frames = max(1, int(audio_duration * _FPS_CAP))
    max_groups = min(
        estimated_frames,
        max(1, int(audio_duration // _INTERLEAVE_MIN_SECONDS)),
    )
    reserved_audio_tokens = audio_token_count + 3 * max_groups
    min_visual_tokens = _MIN_FRAMES * (_PF_FLOOR + _FRAME_OVERHEAD)
    if (
        audio_token_count > audio_cap * seq_length
        or reserved_audio_tokens + min_visual_tokens + _BUDGET_OVERHEAD > seq_length
    ):
        pcm = None
        audio_duration = 0.0
        reserved_audio_tokens = 0

    overhead = (
        _token_len(
            tokenizer,
            "<|system|>You are a helpful assistant.<|endofsystem|>\n",
        )
        + 2
        + _token_len(tokenizer, "<video_0>")
        + 64
    )
    visual_budget = max(
        _PF_FLOOR + _FRAME_OVERHEAD,
        seq_length - overhead - reserved_audio_tokens,
    )
    if isinstance(video, bytes):
        frames, video_duration = _decode_frames(
            video,
            visual_budget,
            seq_length,
            jpeg_quality,
        )
    else:
        frames, video_duration = _prepare_decoded_frames(
            video,
            visual_budget,
            seq_length,
            jpeg_quality,
        )

    if pcm is None:
        parts: list[Dots3NoteVideoPart] = []
        for timestamp, image in frames:
            parts.append(
                Dots3NoteVideoPart("text", f"<{_format_timestamp(timestamp)}>")
            )
            parts.append(Dots3NoteVideoPart("image", image))
        return parts

    video_id = hashlib.sha1(video).hexdigest()
    record_key = hashlib.sha1(f"{video_id}|{question}".encode()).hexdigest()
    seed_hex = hashlib.sha1(f"42|flatten|{record_key}".encode()).hexdigest()
    rng = random.Random(int(seed_hex[:8], 16))
    bounds = _group_bounds(len(frames), audio_duration, k_mode, rng)
    parts = []
    for group in range(len(bounds) - 1):
        start, end = bounds[group], bounds[group + 1]
        if end <= start:
            continue
        start_time = 0.0 if group == 0 else frames[start][0]
        end_time = audio_duration if group == len(bounds) - 2 else frames[end][0]
        if end_time <= start_time:
            end_time = start_time + audio_duration / max(1, len(bounds) - 1)
        for timestamp, image in frames[start:end]:
            parts.append(
                Dots3NoteVideoPart("text", f"<{_format_timestamp(timestamp)}>")
            )
            parts.append(Dots3NoteVideoPart("image", image))
        sample_start = max(0, int(round(start_time * audio_sample_rate)))
        sample_end = min(len(pcm), int(round(end_time * audio_sample_rate)))
        if sample_end > sample_start:
            segment = np.ascontiguousarray(
                pcm[sample_start:sample_end].astype(np.float32) / 32768.0
            )
            parts.append(Dots3NoteVideoPart("audio", segment))
    return parts
