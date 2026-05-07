# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# mypy: ignore-errors
"""MiMo-Omni multimodal processor for vLLM.

Ported from SGLang's MiMoV2OmniProcessor / MiMoVLProcessor implementations.
"""

import contextlib
import copy
import io
import logging
import math
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from io import BytesIO
from typing import Any, Literal

import numpy as np
import regex as re
import requests
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import BatchFeature, TensorType
from transformers.processing_utils import ProcessorMixin

try:
    from torchcodec.decoders import AudioDecoder

    _HAS_TORCHCODEC = True
except ImportError:
    AudioDecoder = None
    _HAS_TORCHCODEC = False

try:
    import torchaudio
    from torchaudio.transforms import MelSpectrogram as _MelSpectrogram

    _HAS_TORCHAUDIO = True
except ImportError:
    torchaudio = None  # type: ignore[assignment]
    _MelSpectrogram = None  # type: ignore[assignment,misc]
    _HAS_TORCHAUDIO = False

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PIXEL_MEAN = [123.675, 116.28, 103.53]
_PIXEL_STD = [58.395, 57.12, 57.375]
_mean_std_cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ImageInput:
    # PIL.Image | str (path/url/base64) | bytes | torch.Tensor (C,H,W)
    image: Any
    max_pixels: int | None = None
    min_pixels: int | None = None


@dataclass
class VideoInput:
    # tuple[frames_TCHW: torch.Tensor, timestamps_T: torch.Tensor]
    video: Any
    min_pixels: int | None = None
    max_pixels: int | None = None
    total_max_pixels: int | None = None
    fps: float | None = None
    num_frames: int | None = None
    max_frames: int | None = None
    min_frames: int | None = None
    do_include_last_frame: bool | None = False
    start_time: float | None = None
    end_time: float | None = None
    segment_type: Literal["individual", "partial"] = "individual"


@dataclass
class AudioInput:
    # str (path/url/base64) | bytes | tuple[waveform_1D, sr]
    # | np.ndarray | torch.Tensor (T,n_vq)
    audio: Any


@dataclass
class VideoAudioInput:
    video: Any  # same as VideoInput.video
    audio: Any  # same as AudioInput.audio
    min_pixels: int | None = None
    max_pixels: int | None = None
    total_max_pixels: int | None = None
    fps: float | None = None
    num_frames: int | None = None
    max_frames: int | None = None
    min_frames: int | None = None
    do_include_last_frame: bool | None = False
    start_time: float | None = None
    end_time: float | None = None
    segment_type: Literal["individual", "partial"] = "individual"


@dataclass
class Content:
    type: Literal["text", "image", "video", "audio", "video_audio"]
    content: Any
    is_target: bool | None = None


@dataclass
class MiMoVLInputSample:
    input_ids: torch.Tensor
    labels: torch.Tensor | None
    pixel_values: list[torch.Tensor]
    pixel_values_videos: list[torch.Tensor]
    image_thw_grids: list[torch.Tensor]
    video_thw_grids: list[torch.Tensor]
    audio_inputs: list[torch.Tensor]
    second_per_grid_ts: list[float] = field(default_factory=list)
    video_start_times: list[float] = field(default_factory=list)
    audio_token_lens: list[int] = field(default_factory=list)
    va_audio_inputs: list[torch.Tensor] = field(default_factory=list)
    video_audio_n_segs: list[int] = field(default_factory=list)
    video_audio_seg_lens: list[int] = field(default_factory=list)
    position_ids: torch.Tensor | None = None
    rope_deltas: torch.Tensor | None = None
    extra: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Vision utilities
# ---------------------------------------------------------------------------


def _format_timestamp(ts: float) -> str:
    return f"{int(ts // 60):02d}:{int(ts % 60):02d}"


def _smart_resize(
    h: int, w: int, factor: int, min_px: int, max_px: int
) -> tuple[int, int]:
    if min(h, w) < factor:
        if h < w:
            h, w = factor, int(w * factor / h)
        else:
            w, h = factor, int(h * factor / w)
    elif max(h, w) / min(h, w) > 200:
        raise ValueError(f"Aspect ratio > 200 not allowed: {h}x{w}")
    h_bar = round(h / factor) * factor
    w_bar = round(w / factor) * factor
    if h_bar * w_bar > max_px:
        beta = math.sqrt((h * w) / max_px)
        h_bar = math.floor(h / beta / factor) * factor
        w_bar = math.floor(w / beta / factor) * factor
    elif h_bar * w_bar < min_px:
        beta = math.sqrt(min_px / (h * w))
        h_bar = math.ceil(h * beta / factor) * factor
        w_bar = math.ceil(w * beta / factor) * factor
    return int(h_bar), int(w_bar)


def _to_rgb(img: Image.Image) -> Image.Image:
    if img.mode == "RGBA":
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[3])
        return bg
    return img.convert("RGB")


def _standardize(images: torch.Tensor) -> torch.Tensor:
    key = str(images.device)
    if key not in _mean_std_cache:
        mean = torch.tensor(_PIXEL_MEAN, device=images.device).view(1, -1, 1, 1)
        std = torch.tensor(_PIXEL_STD, device=images.device).view(1, -1, 1, 1)
        _mean_std_cache[key] = (mean, std)
    mean, std = _mean_std_cache[key]
    return (images - mean) / std


def _transform_batch(
    frames: torch.Tensor,
    factor: int,
    min_px: int,
    max_px: int,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, int, int]:
    if device is not None:
        frames = frames.to(device)
    _, _, h, w = frames.shape
    h_bar, w_bar = _smart_resize(h, w, factor, min_px, max_px)
    resized = F.interpolate(
        frames.float(), (h_bar, w_bar), mode="bilinear", align_corners=False
    )
    return _standardize(resized), w_bar, h_bar


def _transform_single(
    img: Any,
    factor: int,
    min_px: int,
    max_px: int,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, int, int]:
    if isinstance(img, torch.Tensor):
        t = img.float()
        _, h, w = t.shape
    elif isinstance(img, Image.Image):
        img = img.convert("RGB")
        w, h = img.size
        t = torch.from_numpy(np.array(img)).permute(2, 0, 1).float()
    else:
        raise TypeError(f"Expected Tensor or PIL.Image, got {type(img)}")
    if device is not None:
        t = t.to(device)
    h_bar, w_bar = _smart_resize(h, w, factor, min_px, max_px)
    out = F.interpolate(
        t.unsqueeze(0), (h_bar, w_bar), mode="bilinear", align_corners=False
    )
    return _standardize(out).squeeze(0), w_bar, h_bar


def _fetch_image(src: Any) -> Image.Image:
    if isinstance(src, Image.Image):
        return _to_rgb(src)
    if isinstance(src, bytes):
        return _to_rgb(copy.deepcopy(Image.open(BytesIO(src))))
    if isinstance(src, str):
        if src.startswith(("http://", "https://")):
            r = requests.get(src, timeout=30)
            r.raise_for_status()
            return _to_rgb(copy.deepcopy(Image.open(BytesIO(r.content))))
        if src.startswith("file://"):
            return _to_rgb(Image.open(src[7:]))
        if src.startswith("data:image"):
            import pybase64 as _b64

            _, b64 = src.split("base64,", 1)
            return _to_rgb(copy.deepcopy(Image.open(BytesIO(_b64.b64decode(b64)))))
        return _to_rgb(Image.open(src))
    raise ValueError(f"Unrecognized image source: {type(src)}")


# ---------------------------------------------------------------------------
# Core processor
# ---------------------------------------------------------------------------


class MiMoVLProcessor:
    """Core MiMo-VL multimodal processor.

    Handles image/video/audio preprocessing and token sequence construction.
    Ported from SGLang's MiMoVLProcessor.
    """

    def __init__(
        self,
        tokenizer: Any,
        patch_size: int = 14,
        merge_size: int = 2,
        temporal_patch_size: int = 2,
        temporal_compression_ratio: int = 1,
        use_video_timestamps: bool = True,
        video_audio_interleave_length: int = 0,
        audio_kernel_size: int = 3,
        audio_stride_size: int = 2,
        audio_avg_pooler: int = 2,
        audio_sampling_rate: int = 24000,
        audio_nfft: int = 960,
        audio_hop_length: int = 240,
        audio_window_size: int = 960,
        audio_fmin: float = 0.0,
        audio_fmax: float | None = None,
        audio_n_mels: int = 128,
        audio_segment_size: int = 6000,
        audio_channels: int = 8,
        audio_group_size: int = 4,
        audio_input_id_per_second: float = 25.0,
        audio_zeroemb_idx: int = 4096,
        image_min_pixels: int | None = None,
        image_max_pixels: int | None = None,
        video_min_pixels: int | None = None,
        video_max_pixels: int | None = None,
        video_total_max_pixels: int | None = None,
        fps: float | None = None,
        num_frames: int | None = None,
        max_frames: int | None = None,
        min_frames: int | None = None,
        image_token_id: int | None = None,
        video_token_id: int | None = None,
        audio_token_id: int | None = None,
        vision_start_token_id: int | None = None,
        vision_end_token_id: int | None = None,
        audio_start_token_id: int | None = None,
        audio_end_token_id: int | None = None,
        video_start_token_id: int | None = None,
        video_end_token_id: int | None = None,
        pad_token_id: int | None = None,
        rope_type: str = "rope",
        video_process_num_threads: int = 16,
        device: Any | None = None,
        **kwargs: Any,
    ) -> None:
        self.tokenizer = tokenizer
        self.video_process_num_threads = video_process_num_threads
        self.device = torch.device(device) if isinstance(device, str) else device

        self.rope_type = "rope" if rope_type == "1d" else rope_type
        assert self.rope_type in ("rope", "mrope"), (
            f"Unknown rope_type: {self.rope_type}"
        )

        # video timestamps require 1-D rope
        assert use_video_timestamps, "use_video_timestamps must be True"
        assert self.rope_type == "rope", (
            "use_video_timestamps requires rope_type='rope'"
        )
        self.use_video_timestamps = use_video_timestamps
        self.video_audio_interleave_length = video_audio_interleave_length

        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.audio_token_id = audio_token_id
        self.vision_start_token_id = vision_start_token_id
        self.vision_end_token_id = vision_end_token_id
        self.audio_start_token_id = audio_start_token_id
        self.audio_end_token_id = audio_end_token_id
        self.video_start_token_id = video_start_token_id
        self.video_end_token_id = video_end_token_id
        self.pad_token_id = pad_token_id

        self.patch_size = patch_size
        self.merge_size = merge_size
        self.temporal_patch_size = temporal_patch_size
        self.temporal_compression_ratio = temporal_compression_ratio

        self.audio_sampling_rate = audio_sampling_rate
        self.audio_nfft = audio_nfft
        self.audio_hop_length = audio_hop_length
        self.audio_window_size = audio_window_size
        self.audio_fmin = audio_fmin
        self.audio_fmax = audio_fmax
        self.audio_n_mels = audio_n_mels
        self.audio_segment_size = audio_segment_size
        self.audio_kernel_size = audio_kernel_size
        self.audio_stride_size = audio_stride_size
        self.audio_avg_pooler = audio_avg_pooler
        self.audio_channels = audio_channels
        self.audio_group_size = audio_group_size
        self.audio_input_id_per_second = audio_input_id_per_second

        self._mel_spec_kwargs = dict(
            sample_rate=audio_sampling_rate,
            n_fft=audio_nfft,
            hop_length=audio_hop_length,
            win_length=audio_window_size,
            f_min=audio_fmin,
            f_max=audio_fmax,
            n_mels=audio_n_mels,
            power=1.0,
            center=True,
        )
        self._mel_spectrogram: Any | None = None
        self._resamplers: OrderedDict = OrderedDict()
        self._resamplers_max = 16

        if isinstance(audio_zeroemb_idx, int):
            self.audio_zeroemb_idxs = torch.tensor(
                [audio_zeroemb_idx] * audio_channels, dtype=torch.int32
            )
        else:
            self.audio_zeroemb_idxs = torch.tensor(audio_zeroemb_idx, dtype=torch.int32)

        assert image_min_pixels is not None, "image_min_pixels must be set"
        assert image_max_pixels is not None, "image_max_pixels must be set"
        assert video_min_pixels is not None, "video_min_pixels must be set"
        assert video_max_pixels is not None, "video_max_pixels must be set"
        assert video_total_max_pixels is not None, "video_total_max_pixels must be set"
        assert fps is not None or num_frames is not None, (
            "fps or num_frames must be set"
        )

        self._img_kw = {"min_pixels": image_min_pixels, "max_pixels": image_max_pixels}
        self._vid_kw = {
            "min_pixels": video_min_pixels,
            "max_pixels": video_max_pixels,
            "total_max_pixels": video_total_max_pixels,
            "fps": fps,
            "num_frames": num_frames,
            "max_frames": max_frames,
            "min_frames": min_frames,
        }

    @property
    def mel_spectrogram(self) -> Any:
        if self._mel_spectrogram is None:
            if _MelSpectrogram is None:
                raise RuntimeError(
                    "torchaudio is required for audio. "
                    "Install with: pip install torchaudio"
                )
            self._mel_spectrogram = _MelSpectrogram(**self._mel_spec_kwargs)
        return self._mel_spectrogram

    def _resolve_img_kw(self, img: ImageInput) -> dict:
        return {
            "min_px": (
                img.min_pixels
                if img.min_pixels is not None
                else self._img_kw["min_pixels"]
            ),
            "max_px": (
                img.max_pixels
                if img.max_pixels is not None
                else self._img_kw["max_pixels"]
            ),
        }

    def _resolve_vid_kw(self, vid: VideoInput) -> dict:
        kw: dict = {}
        for k in ("min_pixels", "max_pixels", "total_max_pixels"):
            kw[k] = getattr(vid, k) or self._vid_kw[k]
        if vid.num_frames is not None:
            kw["num_frames"] = vid.num_frames
        elif vid.fps is not None:
            kw["fps"] = vid.fps
            if vid.max_frames is not None:
                kw["max_frames"] = vid.max_frames
            if vid.min_frames is not None:
                kw["min_frames"] = vid.min_frames
        elif self._vid_kw["num_frames"] is not None:
            kw["num_frames"] = self._vid_kw["num_frames"]
        elif self._vid_kw["fps"] is not None:
            kw["fps"] = self._vid_kw["fps"]
            if self._vid_kw["max_frames"] is not None:
                kw["max_frames"] = self._vid_kw["max_frames"]
            if self._vid_kw["min_frames"] is not None:
                kw["min_frames"] = self._vid_kw["min_frames"]
        else:
            raise ValueError(
                "No video sampling strategy specified (fps or num_frames)."
            )
        return kw

    def preprocess_audio(self, audio: Any) -> tuple[torch.Tensor, int]:
        """Decode audio bytes/path/tuple → (mel_spec (T, n_mels), token_len)."""
        if isinstance(audio, tuple):
            waveform, original_sr = audio
        else:
            if AudioDecoder is None:
                raise RuntimeError(
                    "torchcodec is required for audio. "
                    "Install with: pip install torchcodec"
                )
            if isinstance(audio, bytes):
                file_obj: Any = io.BytesIO(audio)
            elif isinstance(audio, str):
                if audio.startswith("data:"):
                    import pybase64 as _b64

                    file_obj = io.BytesIO(_b64.b64decode(audio.split(",")[1]))
                elif audio.startswith(("http://", "https://")):
                    r = requests.get(audio, timeout=30)
                    r.raise_for_status()
                    file_obj = io.BytesIO(r.content)
                else:
                    file_obj = audio
            else:
                raise ValueError(f"Unsupported audio source type: {type(audio)}")
            samples = AudioDecoder(file_obj).get_all_samples()
            waveform = samples.data
            original_sr = samples.sample_rate

        if original_sr != self.audio_sampling_rate:
            if original_sr not in self._resamplers:
                if len(self._resamplers) >= self._resamplers_max:
                    self._resamplers.popitem(last=False)
                self._resamplers[original_sr] = torchaudio.transforms.Resample(
                    orig_freq=original_sr, new_freq=self.audio_sampling_rate
                )
            self._resamplers.move_to_end(original_sr)
            waveform = self._resamplers[original_sr](waveform)

        if waveform.ndim == 2:
            waveform = waveform.mean(dim=0)
        spec = self.mel_spectrogram(waveform[None, :])
        spec = torch.log(torch.clip(spec, min=1e-7)).squeeze().transpose(0, 1)

        n = spec.shape[0]
        n = n + 3 - self.audio_kernel_size
        n = (n + 2 - self.audio_kernel_size) // self.audio_stride_size + 1
        n = n // self.audio_avg_pooler + int(n % self.audio_avg_pooler != 0)
        token_len = math.ceil(n / self.audio_group_size)
        return spec, token_len

    def process_image(self, image: ImageInput) -> torch.Tensor:
        kw = self._resolve_img_kw(image)
        src = image.image
        if isinstance(src, (str, bytes)):
            src = _fetch_image(src)
        tensor, _, _ = _transform_single(
            src,
            factor=self.patch_size * self.merge_size,
            device=self.device,
            **kw,
        )
        return tensor

    def process_video(
        self, video_input: VideoInput
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        kw = self._resolve_vid_kw(video_input)
        video = video_input.video
        if not isinstance(video, tuple):
            raise ValueError(
                f"video must be a (frames_TCHW, timestamps_T) tuple, "
                f"got {type(video)}. "
                "Decode the video before calling the processor."
            )
        frames, timestamps = video

        fps = (
            1.0
            if len(timestamps) < 2
            else float(1.0 / (float(timestamps[1]) - float(timestamps[0])))
        )
        start = (
            video_input.start_time
            if video_input.start_time is not None
            else float(timestamps[0])
        )
        end = (
            video_input.end_time
            if video_input.end_time is not None
            else float(timestamps[-1]) + 1.0 / fps
        )

        if video_input.segment_type != "individual":
            mask = (timestamps >= start) & (timestamps < end)
            idxs = torch.where(mask)[0]
            if len(idxs) == 0:
                idxs = torch.where(timestamps <= start)[0][-1:]
            frames, timestamps = frames[idxs], timestamps[idxs]

        tp = self.temporal_patch_size * self.temporal_compression_ratio
        n = frames.shape[0]
        total_px = kw["total_max_pixels"]
        max_px = max(
            kw["min_pixels"], min(total_px * tp // max(n, 1), kw["max_pixels"])
        )

        if n % tp != 0:
            pad = tp - n % tp
            frames = torch.cat(
                [frames, frames[-1:].repeat(pad, *([1] * (frames.ndim - 1)))],
                dim=0,
            )
            timestamps = torch.cat([timestamps, timestamps[-1:].repeat(pad)], dim=0)

        transformed, _, _ = _transform_batch(
            frames,
            factor=self.patch_size * self.merge_size,
            min_px=kw["min_pixels"],
            max_px=max_px,
            device=self.device,
        )
        patches, thw = self._flatten_visual(transformed, "video")
        meta = {
            "fps_sampled": fps,
            "segment_start_time": start,
            "segment_end_time": end,
        }
        return patches, thw, timestamps, meta

    def process_audio(self, audio: AudioInput) -> Any:
        src = audio.audio
        if isinstance(src, np.ndarray):
            src = (torch.from_numpy(src).float(), self.audio_sampling_rate)
        if isinstance(src, (str, bytes, tuple)):
            return self.preprocess_audio(src)
        # Pre-tokenized tensor (T, n_vq)
        assert isinstance(src, torch.Tensor) and src.ndim == 2
        T = src.shape[0]
        src = src[:, : self.audio_channels].to(torch.long)
        pad_T = (
            (T + self.audio_group_size - 1)
            // self.audio_group_size
            * self.audio_group_size
        )
        padding = (
            torch.zeros(pad_T - T, self.audio_channels, dtype=torch.long) + src[-1]
        )
        src = torch.cat([src, padding], dim=0)
        return src.reshape(
            pad_T // self.audio_group_size, self.audio_group_size, self.audio_channels
        )

    def _flatten_visual(
        self, visual: torch.Tensor, kind: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if kind == "image":
            h, w = visual.shape[-2:]
            patches = visual.unsqueeze(0).repeat(self.temporal_patch_size, 1, 1, 1)
        else:  # video / video_audio
            temporal_stride = self.temporal_compression_ratio * self.temporal_patch_size
            assert visual.shape[0] % temporal_stride == 0
            patches = visual
            h, w = patches.shape[-2:]

        C = patches.shape[1]
        grid_t = patches.shape[0] // self.temporal_patch_size
        grid_h, grid_w = h // self.patch_size, w // self.patch_size

        patches = (
            patches.contiguous()
            .view(
                grid_t,
                self.temporal_patch_size,
                C,
                grid_h // self.merge_size,
                self.merge_size,
                self.patch_size,
                grid_w // self.merge_size,
                self.merge_size,
                self.patch_size,
            )
            .permute(0, 3, 6, 4, 7, 2, 1, 5, 8)
            .contiguous()
            .view(
                grid_t * grid_h * grid_w,
                C * self.temporal_patch_size * self.patch_size * self.patch_size,
            )
        )
        thw = torch.tensor([grid_t, grid_h, grid_w], dtype=torch.int32)
        return patches, thw

    def process(
        self, contents: list[Content], verbose: bool = False
    ) -> MiMoVLInputSample:
        input_ids: list[int] = []
        labels: list[int] = []
        img_pv: list[torch.Tensor] = []
        img_grids: list[torch.Tensor] = []
        vid_pv: list[torch.Tensor] = []
        vid_grids: list[torch.Tensor] = []
        audio_inputs: list[torch.Tensor] = []
        is_audio_tokenized: list[bool] = []
        audio_token_lens: list[int] = []
        second_per_grid_ts: list[float] = []
        video_start_times: list[float] = []
        va_audio_inputs: list[torch.Tensor] = []
        video_audio_n_segs: list[int] = []
        video_audio_seg_lens: list[int] = []

        # Pre-decode videos in parallel
        vid_info = [
            (i, c.content, c.type == "video_audio")
            for i, c in enumerate(contents)
            if c.type in ("video", "video_audio")
        ]
        vid_results: dict[int, tuple] = {}
        if vid_info:
            n_t = min(self.video_process_num_threads, len(vid_info))
            if n_t > 1 and len(vid_info) > 1:
                with ThreadPoolExecutor(max_workers=n_t) as ex:
                    fut_map = {
                        ex.submit(self.process_video, vi): idx
                        for idx, vi, _ in vid_info
                    }
                    for fut in as_completed(fut_map):
                        vid_results[fut_map[fut]] = fut.result()
            else:
                for idx, vi, _ in vid_info:
                    vid_results[idx] = self.process_video(vi)

        for ci, content in enumerate(contents):
            _ids: list[int] = []
            _lbls: list[int] | None = None

            if content.type == "text":
                _ids = (
                    self.tokenizer.encode(content.content)
                    if isinstance(content.content, str)
                    else list(content.content)
                )
                if content.is_target:
                    _lbls = _ids

            elif content.type == "image":
                tensor = self.process_image(content.content)
                patches, thw = self._flatten_visual(tensor, "image")
                t, h, w = thw.tolist()
                n_tok = (t * h * w) // (self.merge_size**2)
                img_pv.append(patches)
                img_grids.append(thw)
                _ids = (
                    [self.vision_start_token_id]
                    + [self.image_token_id] * n_tok
                    + [self.vision_end_token_id]
                )

            elif content.type == "video":
                patches, thw, ts, meta = vid_results[ci]
                t, h, w = thw.tolist()
                n_per_grid = h * w // (self.merge_size**2)
                vid_pv.append(patches)
                vid_grids.append(thw)
                second_per_grid_ts.append(
                    self.temporal_patch_size / meta["fps_sampled"]
                )
                video_start_times.append(float(ts[0]))
                video_audio_n_segs.append(0)

                stride = self.temporal_patch_size * self.temporal_compression_ratio
                ts_texts = [_format_timestamp(float(x)) for x in ts[::stride]]
                ts_ids_list = [self.tokenizer.encode(s) for s in ts_texts]

                _ids = [self.video_start_token_id]
                for ts_ids in ts_ids_list:
                    _ids += (
                        ts_ids
                        + [self.vision_start_token_id]
                        + [self.video_token_id] * n_per_grid
                        + [self.vision_end_token_id]
                    )
                _ids += [self.video_end_token_id]

            elif content.type == "audio":
                processed = self.process_audio(content.content)
                if isinstance(processed, tuple):
                    is_audio_tokenized.append(False)
                    spec, tok_len = processed
                    audio_inputs.append(spec)
                else:
                    is_audio_tokenized.append(True)
                    tok_len = processed.shape[0]
                    audio_inputs.append(processed)
                audio_token_lens.append(tok_len)
                _ids = (
                    [self.audio_start_token_id]
                    + [self.audio_token_id] * tok_len
                    + [self.audio_end_token_id]
                )

            elif content.type == "video_audio":
                patches, thw, ts, meta = vid_results[ci]
                second_per_grid_ts.append(
                    self.temporal_patch_size / meta["fps_sampled"]
                )
                video_start_times.append(float(ts[0]))
                processed_audio = self.process_audio(content.content)
                tok_per_sec = self.audio_input_id_per_second / self.audio_group_size

                t, h, w = thw.tolist()
                vid_pv.append(patches)
                vid_grids.append(thw)

                if isinstance(processed_audio, tuple):
                    # Mel spec (not pre-tokenized): store in va_audio_inputs separately
                    spec, total_atok = processed_audio
                    va_audio_inputs.append(spec)
                    _va_is_tokenized = False
                else:
                    # Pre-tokenized: not expected in vLLM, but handle defensively
                    total_atok = processed_audio.shape[0]
                    _va_is_tokenized = True

                n_per_grid = h * w // (self.merge_size**2)
                stride = self.temporal_patch_size * self.temporal_compression_ratio
                grid_ts = ts[::stride]
                ts_texts = [_format_timestamp(float(x)) for x in grid_ts]
                ts_ids_list = [self.tokenizer.encode(s) for s in ts_texts]

                units: list[tuple] = []
                for i in range(len(grid_ts)):
                    a_start = int(float(grid_ts[i]) * tok_per_sec)
                    a_end = (
                        int(float(grid_ts[i + 1]) * tok_per_sec)
                        if i < len(grid_ts) - 1
                        else int(meta["segment_end_time"] * tok_per_sec)
                    )
                    seg_len = min(a_end, total_atok) - a_start
                    assert seg_len > 0, f"Zero-length audio segment at grid index {i}"
                    seg = (
                        processed_audio[a_start : a_start + seg_len]
                        if _va_is_tokenized
                        else None
                    )
                    units.append(
                        (
                            float(grid_ts[i]),
                            ts_texts[i],
                            ts_ids_list[i],
                            n_per_grid,
                            seg_len,
                            seg,
                        )
                    )

                il = self.video_audio_interleave_length
                if il == -1:
                    groups: list[list] = [list(enumerate(units))]
                elif il == 0:
                    groups = [[(i, u)] for i, u in enumerate(units)]
                else:
                    groups, cur, t_ptr = [], [], 0.0
                    for i, u in enumerate(units):
                        while u[0] >= t_ptr + il:
                            if cur:
                                groups.append(cur)
                                cur = []
                            t_ptr += il
                        cur.append((i, u))
                    if cur:
                        groups.append(cur)

                # Track n_segs (= num groups) and per-group audio token counts
                video_audio_n_segs.append(len(groups))
                for group in groups:
                    group_seg_len = sum(u[4] for _, u in group)
                    video_audio_seg_lens.append(group_seg_len)

                _ids = [self.video_start_token_id]
                for group in groups:
                    _ids += group[0][1][2]  # first-unit timestamp token ids
                    _vid_tok: list[int] = []
                    _aud_tok: list[int] = []
                    for _, u in group:
                        _, _, _, vid_n, seg_n, seg_audio = u
                        _vid_tok += (
                            [self.vision_start_token_id]
                            + [self.video_token_id] * vid_n
                            + [self.vision_end_token_id]
                        )
                        _aud_tok += [self.audio_token_id] * seg_n
                        if seg_audio is not None:
                            # Pre-tokenized per-frame segments (rare in vLLM)
                            audio_inputs.append(seg_audio)
                    _ids += (
                        _vid_tok
                        + [self.audio_start_token_id]
                        + _aud_tok
                        + [self.audio_end_token_id]
                    )
                _ids += [self.video_end_token_id]

            input_ids.extend(_ids)
            labels.extend(
                _lbls if _lbls is not None else [self.pad_token_id] * len(_ids)
            )

        ids_t = torch.tensor(input_ids)
        lbl_arr = np.roll(labels, shift=-1)
        lbl_arr[-1] = self.pad_token_id
        lbl_t = torch.tensor(lbl_arr)

        extra: dict = {}
        if is_audio_tokenized:
            assert all(is_audio_tokenized) or not any(is_audio_tokenized)
            extra["is_audio_tokenized"] = is_audio_tokenized[0]

        position_ids = torch.arange(ids_t.shape[0]).expand(3, -1)
        rope_deltas = torch.zeros((1, 1), dtype=torch.int32)

        return MiMoVLInputSample(
            input_ids=ids_t,
            labels=lbl_t,
            pixel_values=img_pv,
            pixel_values_videos=vid_pv,
            image_thw_grids=img_grids,
            video_thw_grids=vid_grids,
            audio_inputs=audio_inputs,
            second_per_grid_ts=second_per_grid_ts,
            video_start_times=video_start_times,
            audio_token_lens=audio_token_lens,
            va_audio_inputs=va_audio_inputs,
            video_audio_n_segs=video_audio_n_segs,
            video_audio_seg_lens=video_audio_seg_lens,
            position_ids=position_ids,
            rope_deltas=rope_deltas,
            extra=extra,
        )


# ---------------------------------------------------------------------------
# vLLM ProcessorMixin wrapper
# ---------------------------------------------------------------------------


class MiMoOmniProcessor(ProcessorMixin):
    """HuggingFace-compatible ProcessorMixin wrapper for MiMo-Omni.

    Accepts PIL images, pre-decoded video tuples (frames_TCHW, timestamps_T),
    and audio (file path / bytes / (waveform, sr) tuple / numpy array).
    """

    attributes = ["tokenizer"]
    tokenizer_class = "AutoTokenizer"

    # Single or multi-pad placeholders produced by the chat template / prior expansion
    _IMG_RE = re.compile(r"<\|vision_start\|>(?:<\|image_pad\|>)+<\|vision_end\|>")
    _VID_RE = re.compile(r"<\|vision_start\|>(?:<\|video_pad\|>)+<\|vision_end\|>")
    _AUD_RE = re.compile(
        r"<\|mimo_audio_start\|>(?:<\|audio_pad\|>)+<\|mimo_audio_end\|>"
    )

    _MM_RE = re.compile(
        r"(<\|vision_start\|>(?:<\|image_pad\|>)+<\|vision_end\|>"
        r"|<\|vision_start\|>(?:<\|video_pad\|>)+<\|vision_end\|>"
        r"|<\|mimo_audio_start\|>(?:<\|audio_pad\|>)+<\|mimo_audio_end\|>)"
    )

    def __init__(
        self,
        tokenizer: Any,
        *,
        patch_size: int = 14,
        merge_size: int = 2,
        temporal_patch_size: int = 2,
        temporal_compression_ratio: int = 1,
        image_min_pixels: int | None = None,
        image_max_pixels: int | None = None,
        video_min_pixels: int | None = None,
        video_max_pixels: int | None = None,
        video_total_max_pixels: int | None = None,
        fps: float = 2.0,
        num_frames: int | None = None,
        max_frames: int = 256,
        min_frames: int = 8,
        video_audio_interleave_length: int = 0,
        audio_sampling_rate: int = 24000,
        audio_nfft: int = 960,
        audio_hop_length: int = 240,
        audio_window_size: int = 960,
        audio_fmin: float = 0.0,
        audio_fmax: float | None = None,
        audio_n_mels: int = 128,
        audio_segment_size: int = 6000,
        audio_kernel_size: int = 3,
        audio_stride_size: int = 2,
        audio_avg_pooler: int = 2,
        audio_channels: int = 8,
        audio_group_size: int = 4,
        audio_input_id_per_second: float = 25.0,
        audio_zeroemb_idx: int = 4096,
        image_token_id: int | None = None,
        video_token_id: int | None = None,
        audio_token_id: int | None = None,
        vision_start_token_id: int | None = None,
        vision_end_token_id: int | None = None,
        audio_start_token_id: int | None = None,
        audio_end_token_id: int | None = None,
        video_start_token_id: int | None = None,
        video_end_token_id: int | None = None,
        rope_type: str = "rope",
    ) -> None:
        self.tokenizer = tokenizer

        unit = patch_size * merge_size
        self.mimo_processor = MiMoVLProcessor(
            tokenizer=tokenizer,
            patch_size=patch_size,
            merge_size=merge_size,
            temporal_patch_size=temporal_patch_size,
            temporal_compression_ratio=temporal_compression_ratio,
            use_video_timestamps=True,
            video_audio_interleave_length=video_audio_interleave_length,
            audio_sampling_rate=audio_sampling_rate,
            audio_nfft=audio_nfft,
            audio_hop_length=audio_hop_length,
            audio_window_size=audio_window_size,
            audio_fmin=audio_fmin,
            audio_fmax=audio_fmax,
            audio_n_mels=audio_n_mels,
            audio_segment_size=audio_segment_size,
            audio_kernel_size=audio_kernel_size,
            audio_stride_size=audio_stride_size,
            audio_avg_pooler=audio_avg_pooler,
            audio_channels=audio_channels,
            audio_group_size=audio_group_size,
            audio_input_id_per_second=audio_input_id_per_second,
            audio_zeroemb_idx=audio_zeroemb_idx,
            image_min_pixels=image_min_pixels or (4 * unit * unit),
            image_max_pixels=image_max_pixels or (4096 * unit * unit),
            video_min_pixels=video_min_pixels or (4 * unit * unit),
            video_max_pixels=video_max_pixels or (4096 * unit * unit),
            video_total_max_pixels=video_total_max_pixels or (16384 * unit * unit),
            fps=fps,
            num_frames=num_frames,
            max_frames=max_frames,
            min_frames=min_frames,
            image_token_id=image_token_id,
            video_token_id=video_token_id,
            audio_token_id=audio_token_id,
            vision_start_token_id=vision_start_token_id,
            vision_end_token_id=vision_end_token_id,
            audio_start_token_id=audio_start_token_id,
            audio_end_token_id=audio_end_token_id,
            video_start_token_id=video_start_token_id,
            video_end_token_id=video_end_token_id,
            pad_token_id=tokenizer.pad_token_id,
            rope_type=rope_type,
        )

    @classmethod
    def from_hf_config(cls, tokenizer: Any, hf_config: Any) -> "MiMoOmniProcessor":
        """Convenience factory: instantiate directly from an HF model config object."""
        vc = hf_config.vision_config
        if isinstance(vc, dict):
            patch_size = vc.get("patch_size", 14)
            merge_size = vc.get("spatial_merge_size", 2)
            temporal_patch_size = vc.get("temporal_patch_size", 2)
        else:
            patch_size = getattr(vc, "patch_size", 14)
            merge_size = getattr(vc, "spatial_merge_size", 2)
            temporal_patch_size = getattr(vc, "temporal_patch_size", 2)

        pc: dict = getattr(hf_config, "processor_config", {}) or {}
        ac = getattr(hf_config, "audio_config", None)
        audio_sr: int | None = pc.get("audio_sampling_rate")
        if audio_sr is None and ac is not None:
            if isinstance(ac, dict):
                audio_sr = ac.get("sampling_rate") or ac.get("sample_rate")
            else:
                audio_sr = getattr(ac, "sampling_rate", None) or getattr(
                    ac, "sample_rate", None
                )

        rope_type = "rope"
        rs = getattr(hf_config, "rope_scaling", None)
        if rs and rs.get("type") == "default" and rs.get("mrope_section") is not None:
            rope_type = "mrope"

        unit = patch_size * merge_size
        return cls(
            tokenizer,
            patch_size=patch_size,
            merge_size=merge_size,
            temporal_patch_size=temporal_patch_size,
            image_min_pixels=pc.get("image_min_pixels") or (4 * unit * unit),
            image_max_pixels=pc.get("image_max_pixels") or (4096 * unit * unit),
            video_min_pixels=pc.get("video_min_pixels") or (4 * unit * unit),
            video_max_pixels=pc.get("video_max_pixels") or (4096 * unit * unit),
            video_total_max_pixels=(
                pc.get("video_total_max_pixels") or (16384 * unit * unit)
            ),
            fps=pc.get("fps") or 2.0,
            num_frames=pc.get("num_frames"),
            max_frames=pc.get("max_frames") or 256,
            min_frames=pc.get("min_frames") or 8,
            video_audio_interleave_length=pc.get("video_audio_interleave_length", 0),
            audio_sampling_rate=audio_sr or 24000,
            image_token_id=pc.get("image_token_id"),
            video_token_id=pc.get("video_token_id"),
            audio_token_id=pc.get("audio_token_id"),
            vision_start_token_id=pc.get("vision_start_token_id"),
            vision_end_token_id=pc.get("vision_end_token_id"),
            audio_start_token_id=pc.get("audio_start_token_id"),
            audio_end_token_id=pc.get("audio_end_token_id"),
            video_start_token_id=pc.get("video_start_token_id"),
            video_end_token_id=pc.get("video_end_token_id"),
            rope_type=rope_type,
        )

    @property
    def image_token(self) -> str:
        """Token string used as image placeholder (for vLLM integration)."""
        return "<|image_pad|>"

    @property
    def video_token(self) -> str:
        """Token string used as video placeholder (for vLLM integration)."""
        return "<|video_pad|>"

    @property
    def image_processor(self) -> Any:
        """Minimal image-processor-like object for vLLM processing-info compat."""
        p = self.mimo_processor

        class _ImageProcessor:
            merge_size = p.merge_size
            size = {
                "shortest_edge": p._img_kw["min_pixels"],
                "longest_edge": p._img_kw["max_pixels"],
            }

        return _ImageProcessor()

    def _modality(self, token: str) -> str:
        if self._IMG_RE.fullmatch(token):
            return "image"
        if self._VID_RE.fullmatch(token):
            return "video"
        if self._AUD_RE.fullmatch(token):
            return "audio"
        return "unknown"

    def __call__(
        self,
        text: str | list[str] | None = None,
        images: Any = None,
        videos: Any = None,
        audio: Any = None,
        video_audio: Any = None,
        return_tensors: str | TensorType | None = None,
        **kwargs: Any,
    ) -> BatchFeature:
        """Process multimodal inputs into model-ready tensors.

        Args:
            text: Prompt string(s) containing multimodal placeholders
                  ``<|vision_start|><|image_pad|><|vision_end|>``,
                  ``<|vision_start|><|video_pad|><|vision_end|>``, or
                  ``<|mimo_audio_start|><|audio_pad|><|mimo_audio_end|>``.
            images: PIL.Image or list[PIL.Image].
            videos: list of ``(frames_TCHW: torch.Tensor, timestamps_T: torch.Tensor)``
                    tuples (pre-decoded).
            audio: list of ``str`` (path/url/base64), ``bytes``,
                   ``(waveform_1D, sample_rate)`` tuples, or ``np.ndarray``.
            return_tensors: Passed to :class:`BatchFeature`.

        Returns:
            :class:`BatchFeature` with keys:
            - ``input_ids``
            - ``pixel_values`` + ``image_grid_thw``
            - ``pixel_values_videos`` + ``video_grid_thw`` + ``second_per_grid_ts``
            - ``audio_features``
        """
        if isinstance(text, list):
            text = text[0] if len(text) == 1 else "\n".join(text)

        imgs: list = (
            ([images] if isinstance(images, Image.Image) else list(images))
            if images is not None
            else []
        )
        vids: list = list(videos) if videos is not None else []
        auds: list = list(audio) if audio is not None else []
        va_items: list = list(video_audio) if video_audio is not None else []

        # If audio exists but text has no audio placeholder, prepend one
        _aud_placeholder = "<|mimo_audio_start|><|audio_pad|><|mimo_audio_end|>"
        if auds and text is not None and not self._AUD_RE.search(text):
            text = _aud_placeholder + text

        # Build Content list
        contents: list[Content] = []

        if text and (imgs or vids or auds or va_items):
            parts = self._MM_RE.split(text)
            img_it = iter(imgs)
            vid_it = iter(vids)
            aud_it = iter(auds)
            va_it = iter(va_items)
            for part in parts:
                if self._MM_RE.fullmatch(part):
                    mod = self._modality(part)
                    if mod == "image":
                        with contextlib.suppress(StopIteration):
                            contents.append(
                                Content(
                                    type="image",
                                    content=ImageInput(image=next(img_it)),
                                )
                            )
                    elif mod == "video":
                        # Try regular video first, fall back to video_audio
                        vid_item = None
                        vid_type = "video"
                        with contextlib.suppress(StopIteration):
                            vid_item = next(vid_it)
                        if vid_item is None:
                            with contextlib.suppress(StopIteration):
                                vid_item = next(va_it)
                                vid_type = "video_audio"
                        if vid_item is not None:
                            if vid_type == "video":
                                contents.append(
                                    Content(
                                        type="video",
                                        content=VideoInput(video=vid_item),
                                    )
                                )
                            else:
                                contents.append(
                                    Content(
                                        type="video_audio",
                                        content=vid_item,
                                    )
                                )
                    elif mod == "audio":
                        with contextlib.suppress(StopIteration):
                            contents.append(
                                Content(
                                    type="audio",
                                    content=AudioInput(audio=next(aud_it)),
                                )
                            )
                elif part:
                    contents.append(Content(type="text", content=part))
        elif text:
            contents.append(Content(type="text", content=text))
        else:
            for img in imgs:
                contents.append(Content(type="image", content=ImageInput(image=img)))
            for vid in vids:
                contents.append(Content(type="video", content=VideoInput(video=vid)))
            for aud in auds:
                contents.append(Content(type="audio", content=AudioInput(audio=aud)))
            for va_item in va_items:
                contents.append(Content(type="video_audio", content=va_item))

        if not contents:
            ids = self.tokenizer(text or "", return_tensors=return_tensors)["input_ids"]
            return BatchFeature(data={"input_ids": ids}, tensor_type=return_tensors)

        sample = self.mimo_processor.process(contents, verbose=False)

        # vLLM expects input_ids to have a batch dimension [1, seq_len].
        data: dict = {"input_ids": sample.input_ids.unsqueeze(0)}

        if sample.pixel_values:
            data["pixel_values"] = torch.cat(sample.pixel_values, dim=0)
            data["image_grid_thw"] = torch.stack(sample.image_thw_grids)

        if sample.pixel_values_videos:
            data["pixel_values_videos"] = torch.cat(sample.pixel_values_videos, dim=0)
            data["video_grid_thw"] = torch.stack(sample.video_thw_grids)
            if sample.second_per_grid_ts:
                data["second_per_grid_ts"] = torch.tensor(
                    sample.second_per_grid_ts, dtype=torch.float32
                )
            if sample.video_start_times:
                data["video_start_times"] = torch.tensor(
                    sample.video_start_times, dtype=torch.float32
                )
            if sample.video_audio_n_segs:
                data["video_audio_n_segs"] = torch.tensor(
                    sample.video_audio_n_segs, dtype=torch.long
                )
            # video_audio_seg_lens: 2D padded tensor (num_videos, max_T).
            # Row i has the per-group audio token lengths for video i
            # (zeros for regular videos; valid values for video_audio videos).
            n_segs_list = sample.video_audio_n_segs
            max_segs = max(n_segs_list) if n_segs_list else 0
            if max_segs > 0:
                seg_lens_2d = torch.zeros(len(n_segs_list), max_segs, dtype=torch.long)
                flat_cursor = 0
                for vi, n in enumerate(n_segs_list):
                    if n > 0:
                        seg_lens_2d[vi, :n] = torch.tensor(
                            sample.video_audio_seg_lens[flat_cursor : flat_cursor + n],
                            dtype=torch.long,
                        )
                        flat_cursor += n
                data["video_audio_seg_lens"] = seg_lens_2d

        # audio_features is a list of variable-length mel-spec tensors; pop it
        # before BatchFeature conversion to avoid "batched tensors of the same
        # length" errors, then re-attach it after.
        audio_features = None
        if sample.audio_inputs:
            audio_features = sample.audio_inputs
            if "is_audio_tokenized" in sample.extra:
                data["is_audio_tokenized"] = sample.extra["is_audio_tokenized"]
            if sample.audio_token_lens:
                data["audio_token_lens"] = torch.tensor(
                    sample.audio_token_lens, dtype=torch.long
                )

        bf = BatchFeature(data=data, tensor_type=return_tensors)
        if audio_features is not None:
            bf["audio_features"] = audio_features
        # va_audio_features: list of mel-spec tensors (one per video_audio item)
        if sample.va_audio_inputs:
            bf["va_audio_features"] = sample.va_audio_inputs
        return bf
