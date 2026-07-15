# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Apertus multimodal preprocessing helpers."""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from vllm.logger import init_logger
from vllm.model_executor.models.apertus_emu35 import build_vision_tokenizer
from vllm.model_executor.models.apertus_wavetokenizer import build_audio_tokenizer

logger = init_logger(__name__)


class ApertusImageTokenizer:
    _vision_tokenizer_cache: dict[tuple[str, str, torch.dtype], Any] = {}

    def __init__(self, vision_config: dict[str, Any] | None = None) -> None:
        self.vision_config = vision_config or {}
        self.min_pixels = self.vision_config.get("min_pixels", 256 * 256)
        self.max_pixels = self.vision_config.get("max_pixels", 1400 * 1400)
        self.image_placeholder = self.vision_config.get(
            "image_placeholder", "<|image|>"
        )
        self.visual_template = self.vision_config.get(
            "visual_template", "<|visual token {token_id}|>"
        )
        self.ds_factor = self.vision_config.get("ds_factor", 16)
        self.boi_token = self.vision_config.get("boi_token", "<|img_start|>")
        self.img_token = self.vision_config.get("img_token", "<|img_token_start|>")
        self.eol_token = self.vision_config.get("eol_token", "<|img_end_of_row|>")
        self.eoi_token = self.vision_config.get("eoi_token", "<|img_end|>")


    @staticmethod
    def smart_resize(image: Image.Image, area: int, ds_factor: int) -> Image.Image:
        width, height = image.size
        aspect_ratio = width / height
        new_height = int((area / aspect_ratio) ** 0.5)
        new_width = int(new_height * aspect_ratio)
        new_height = ((new_height + ds_factor // 2) // ds_factor) * ds_factor
        new_width = ((new_width + ds_factor // 2) // ds_factor) * ds_factor
        return image.resize((new_width, new_height), Image.BICUBIC)

    @staticmethod
    def image_array_to_hwc(array: np.ndarray) -> np.ndarray:
        if array.ndim != 3:
            raise TypeError("Apertus image adapter expects 3D image arrays.")
        if array.shape[0] in (1, 3, 4) and array.shape[-1] not in (1, 3, 4):
            return np.transpose(array, (1, 2, 0))
        return array

    @staticmethod
    def image_array_to_uint8(array: np.ndarray) -> np.ndarray:
        if np.issubdtype(array.dtype, np.floating):
            if array.size and np.nanmax(array) <= 1.0:
                array = array * 255.0

        if array.dtype != np.uint8:
            array = np.clip(array, 0, 255).astype(np.uint8)

        return array

    @classmethod
    def coerce_pil_image(
        cls,
        image: Image.Image | np.ndarray | torch.Tensor,
    ) -> Image.Image:
        if isinstance(image, Image.Image):
            return image.convert("RGB")

        if isinstance(image, torch.Tensor):
            image = image.detach().cpu().numpy()

        if not isinstance(image, np.ndarray):
            raise TypeError(
                "Apertus image adapter expects PIL images, numpy arrays, or "
                "torch tensors in multi_modal_data['image'].",
            )

        array = cls.image_array_to_hwc(image)
        array = cls.image_array_to_uint8(array)
        return Image.fromarray(array).convert("RGB")

    def load_vision_tokenizer(
        self,
        model_path: str,
        device: str,
        dtype: torch.dtype,
        vision_config: dict[str, Any],
    ) -> Any:
        path = Path(model_path).expanduser()
        assert path.exists() and path.is_dir(), (
            f"Model directory {model_path} does not exist or is not a directory."
        )

        control_file_new = path / "emu35_vison_tokenizer.safetensors"
        assert control_file_new.is_file(), (
            f"Control file (either emu35_vison_tokenizer.safetensors) "
            f"must exist in {model_path}."
        )

        if device == "cpu" and dtype in (torch.float16, torch.bfloat16):
            dtype = torch.float32

        cache_key = (
            model_path,
            device,
            dtype,
        )

        if cache_key in self._vision_tokenizer_cache:
            return self._vision_tokenizer_cache[cache_key]

        logger.info(
            "[Apertus MM] loading Emu3.5 vision tokenizer on device=%r",
            device,
        )
        vision_tokenizer = build_vision_tokenizer(
            type="ibq",
            model_path=str(path.resolve()),
            device=device,
            vision_config=vision_config,
        )
        vision_tokenizer = vision_tokenizer.to(dtype=dtype)

        self._vision_tokenizer_cache[cache_key] = vision_tokenizer
        return vision_tokenizer

    def preprocess_image_to_tensor(
        self, raw_image: Image.Image | np.ndarray | torch.Tensor
    ) -> tuple[torch.Tensor, int, int]:
        """Returns Normalized Float32 Tensor [3, H, W] and Expected Grid
        Dimensions [h_tok, w_tok].
        """
        image = self.coerce_pil_image(raw_image)
        w, h = image.size
        target_area = max(min(self.max_pixels, w * h), self.min_pixels)
        resized = self.smart_resize(image, target_area, self.ds_factor)

        tensor = torch.tensor(
            (np.array(resized) / 127.5 - 1.0), dtype=torch.float32
        ).permute(2, 0, 1)
        h_tok, w_tok = (
            tensor.shape[1] // self.ds_factor,
            tensor.shape[2] // self.ds_factor,
        )
        return tensor, h_tok, w_tok


class ApertusAudioTokenizer:
    _audio_tokenizer_cache: dict[tuple[str, str, torch.dtype, bool], Any] = {}

    def __init__(self, audio_config: dict[str, Any] | None = None) -> None:
        self.audio_config = audio_config or {}
        self.audio_placeholder = self.audio_config.get("audio_placeholder", "<|audio|>")
        self.audio_tokenizer_path = self.audio_config.get(
            "audio_tokenizer_path", "novateur/WavTokenizer-large-unify-40token"
        )
        self.audio_tokenizer_type = self.audio_config.get(
            "audio_tokenizer_type", "wavtokenizer"
        )
        self.audio_tokenizer_name = self.audio_config.get(
            "audio_tokenizer_name", "WavTokenizer40"
        )
        self.audio_tokenizer_device = self.audio_config.get(
            "audio_tokenizer_device", "cuda"
        )
        self.target_sampling_rate = self.audio_config.get("target_sampling_rate", 24000)
        self.target_peak_dbfs = self.audio_config.get("target_peak_dbfs", -3.0)
        self.audio_token_offset = self.audio_config.get("token_offset", 262344)
        self.audio_start_token = self.audio_config.get(
            "audio_start_token", "<|audio_start|>"
        )
        self.audio_end_token = self.audio_config.get("audio_end_token", "<|audio_end|>")

    @staticmethod
    def to_audio_tensor(waveform: np.ndarray) -> torch.Tensor:
        audio_tensor = torch.from_numpy(waveform).float()
        if audio_tensor.dim() == 1:
            return audio_tensor.unsqueeze(0)
        return audio_tensor

    def load_audio_tokenizer(
        self,
        model_path: str,
        device: str,
        dtype: torch.dtype,
        audio_config: dict[str, Any],
    ) -> Any:
        path = Path(model_path).expanduser()
        assert path.exists() and path.is_dir(), (
            f"Model directory {model_path} does not exist or is not a directory."
        )

        control_file_new = path / "wavtokenizer_large_unify_600_24k.safetensors"
        control_file_old = path / "wavtokenizer_large_unify_600_24k.ckpt"
        assert control_file_new.is_file() or control_file_old.is_file(), (
            f"Control file (either wavtokenizer_large_unify_600_24k.safetensors or "
            f"wavtokenizer_large_unify_600_24k.ckpt) must exist in {model_path}."
        )

        tokenizer_compile = audio_config.get("apertus_audio_tokenizer_compile", False)

        cache_key = (
            model_path,
            device,
            dtype,
            tokenizer_compile,
        )
        if cache_key in self._audio_tokenizer_cache:
            return self._audio_tokenizer_cache[cache_key]

        logger.info(
            "[Apertus MM] loading Emu3.5 audio tokenizer on device=%r",
            device,
        )
        audio_tokenizer = build_audio_tokenizer(
            type="wavtokenizer",
            model_path=str(path.resolve()),
            device=device,
            audio_config=audio_config,
        )
        audio_tokenizer = audio_tokenizer.to(dtype=dtype)
        self._audio_tokenizer_cache[cache_key] = audio_tokenizer
        return audio_tokenizer

    def preprocess_audio_to_tensor(
        self, raw_audio: np.ndarray
    ) -> tuple[torch.Tensor, int]:
        """Returns Normalized Float32 Tensor [1, Length] and Expected Tokens Count."""
        if raw_audio.ndim == 0:
            raise ValueError("Audio waveform must have at least one dimension.")
        waveform = np.asarray(raw_audio, dtype=np.float32)
        tensor = self.to_audio_tensor(waveform)

        peak = tensor.abs().max().clamp(min=1e-10)
        target = 10 ** (self.target_peak_dbfs / 20.0)
        tensor = tensor * (target / peak)

        num_tok = (tensor.shape[-1] + 600 - 1) // 600
        return tensor, num_tok
