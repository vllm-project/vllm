# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Apertus multimodal preprocessing helpers."""

import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import huggingface_hub
import numpy as np
import torch
from PIL import Image

from vllm.logger import init_logger
from vllm.model_executor.models.apertus_emu35 import build_vision_tokenizer
from vllm.model_executor.models.apertus_wavetokenizer import WavTokenizer40
from vllm.multimodal.media import MediaWithBytes
from vllm.tokenizers import TokenizerLike

logger = init_logger(__name__)

_EMU35_VQ_REQUIRED_FILES = ("config.yaml", "model.ckpt")
_APERTUS_VISION_TOKENIZER_DEVICE_ENV_VAR = "VLLM_APERTUS_VISION_TOKENIZER_DEVICE"


def has_required_files(path: Path, required_files: Sequence[str]) -> bool:
    return all((path / fname).is_file() for fname in required_files)


def ensure_local_emu35_weights(
    path: str,
    hf_repo_id: str,
    *,
    required_files: Sequence[str] = _EMU35_VQ_REQUIRED_FILES,
    cache_dir: str | None = None,
) -> str:
    expanded_path = Path(path).expanduser()
    if expanded_path.exists() and expanded_path.is_dir():
        if (expanded_path / "emu35_vison_tokenizer.safetensors").is_file():
            return str(expanded_path.resolve())
        if not has_required_files(expanded_path, required_files):
            raise ValueError(
                f"Local checkpoint at {expanded_path} is missing required "
                f"files: {list(required_files)}."
            )
        return str(expanded_path.resolve())

    local_only = huggingface_hub.constants.HF_HUB_OFFLINE
    is_repo = not expanded_path.exists() and "/" in path and not os.path.isabs(path)
    repo_to_use = path if is_repo else hf_repo_id

    if local_only:
        logger.info("Using cached weights for %s (cache_dir=%s)", repo_to_use, cache_dir)
    else:
        logger.info("Downloading %s (cache_dir=%s)", repo_to_use, cache_dir)

    try:
        hf_folder = huggingface_hub.snapshot_download(
            repo_id=repo_to_use,
            allow_patterns=["config.json", "emu35_vison_tokenizer.safetensors"],
            cache_dir=cache_dir,
            local_files_only=local_only,
        )
        resolved = Path(hf_folder)
        if (resolved / "emu35_vison_tokenizer.safetensors").is_file():
            return str(resolved.resolve())
    except Exception as e:
        logger.warning("Failed downloading vision tokenizer from %s: %s", repo_to_use, e)

    if repo_to_use != hf_repo_id:
        try:
            logger.info("Downloading fallback %s (cache_dir=%s)", hf_repo_id, cache_dir)
            hf_folder = huggingface_hub.snapshot_download(
                repo_id=hf_repo_id,
                allow_patterns=["config.json", "emu35_vison_tokenizer.safetensors"],
                cache_dir=cache_dir,
                local_files_only=local_only,
            )
            resolved = Path(hf_folder)
            if (resolved / "emu35_vison_tokenizer.safetensors").is_file():
                return str(resolved.resolve())
        except Exception:
            pass

    hf_folder = huggingface_hub.snapshot_download(
        repo_id=hf_repo_id,
        allow_patterns=list(required_files),
        cache_dir=cache_dir,
        local_files_only=local_only,
    )

    resolved = Path(hf_folder)
    if not has_required_files(resolved, required_files):
        raise RuntimeError(
            f"Resolved checkpoint at {resolved} is missing required "
            f"files: {list(required_files)}."
        )

    return str(resolved.resolve())


def build_emu35_vision_tokenizer(
    *,
    vq_hub: str,
    default_repo: str,
    device: str,
    vq_type: str = "ibq",
    cache_dir: str | None = None,
) -> Any:
    local_vq_path = ensure_local_emu35_weights(
        vq_hub,
        default_repo,
        required_files=_EMU35_VQ_REQUIRED_FILES,
        cache_dir=cache_dir,
    )
    return build_vision_tokenizer(
        type=vq_type,
        model_path=local_vq_path,
        device=device,
    ).eval()


class ApertusImageTokenizer:
    DEFAULT_VQ_HUB = "BAAI/Emu3.5-VisionTokenizer"
    DEFAULT_MIN_PIXELS = 256 * 256
    DEFAULT_MAX_PIXELS = 1400 * 1400
    DEFAULT_IMAGE_PLACEHOLDER = "<|image|>"
    VISUAL_TEMPLATE = "<|visual token {token_id}|>"
    EMU35_DS_FACTOR = 16
    DEFAULT_BOI_TOKEN = "<|img_start|>"
    DEFAULT_IMG_TOKEN = "<|img_token_start|>"
    DEFAULT_EOL_TOKEN = "<|img_end_of_row|>"
    DEFAULT_EOI_TOKEN = "<|img_end|>"

    _vision_tokenizer_cache: dict[tuple[str, str, str, torch.dtype], Any] = {}

    def __init__(self) -> None:
        pass

    @staticmethod
    def coerce_int(value: object, *, default: int) -> int:
        if value is None:
            return default
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def coerce_dtype(value: object) -> torch.dtype:
        if isinstance(value, torch.dtype):
            return value
        if value is None:
            return torch.bfloat16

        mapping = {
            "float16": torch.float16,
            "fp16": torch.float16,
            "half": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }
        return mapping.get(str(value).lower().strip(), torch.bfloat16)

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
    def coerce_pil_image(image: object) -> Image.Image:
        if isinstance(image, MediaWithBytes):
            image = image.media

        if isinstance(image, Image.Image):
            return image.convert("RGB")

        if isinstance(image, torch.Tensor):
            array = image.detach().to("cpu")
            if array.ndim != 3:
                raise TypeError("Apertus image adapter expects 3D image tensors.")
            if array.shape[0] in (1, 3, 4):
                array = array.permute(1, 2, 0)
            array_np = array.numpy()
        elif isinstance(image, np.ndarray):
            array_np = image
            if array_np.ndim != 3:
                raise TypeError("Apertus image adapter expects 3D image arrays.")
            if array_np.shape[0] in (1, 3, 4) and array_np.shape[-1] not in (1, 3, 4):
                array_np = np.transpose(array_np, (1, 2, 0))
        else:
            raise TypeError(
                "Apertus image adapter expects PIL images, numpy arrays, or "
                "torch tensors in multi_modal_data['image']."
            )

        if np.issubdtype(array_np.dtype, np.floating):
            max_value = float(np.nanmax(array_np)) if array_np.size else 1.0
            if max_value <= 1.0:
                array_np = array_np * 255.0
            array_np = np.clip(array_np, 0, 255).astype(np.uint8)
        elif array_np.dtype != np.uint8:
            array_np = np.clip(array_np, 0, 255).astype(np.uint8)

        return Image.fromarray(array_np).convert("RGB")

    @staticmethod
    def apertus_special_token(
        tokenizer: TokenizerLike,
        attr_name: str,
        fallback: str,
    ) -> str:
        token = getattr(tokenizer, attr_name, None)
        return token if isinstance(token, str) and token else fallback

    @classmethod
    def placeholder_aliases(
        cls,
        tokenizer: TokenizerLike,
        mm_processor_kwargs: Mapping[str, object],
    ) -> list[str]:
        configured_placeholder = mm_processor_kwargs.get("apertus_image_placeholder")
        tokenizer_placeholder = getattr(tokenizer, "image_token", None)
        candidates = [
            configured_placeholder if isinstance(configured_placeholder, str) else "",
            tokenizer_placeholder if isinstance(tokenizer_placeholder, str) else "",
            cls.DEFAULT_IMAGE_PLACEHOLDER,
        ]

        seen: set[str] = set()
        aliases: list[str] = []
        for candidate in candidates:
            stripped = candidate.strip()
            if stripped and stripped not in seen:
                seen.add(stripped)
                aliases.append(stripped)
        return aliases

    @staticmethod
    def _resolve_vision_tokenizer_device(
        mm_processor_kwargs: Mapping[str, object],
    ) -> tuple[str, str]:
        from_kwargs = mm_processor_kwargs.get("apertus_vision_tokenizer_device")
        if isinstance(from_kwargs, str) and from_kwargs.strip():
            return from_kwargs.strip(), "mm_processor_kwargs"

        from_env = os.getenv(_APERTUS_VISION_TOKENIZER_DEVICE_ENV_VAR)
        if from_env and from_env.strip():
            return (
                from_env.strip(),
                f"env:{_APERTUS_VISION_TOKENIZER_DEVICE_ENV_VAR}",
            )

        return "cuda", "default"

    def load_vision_tokenizer(
        self,
        mm_processor_kwargs: Mapping[str, object],
    ) -> Any:
        vq_hub = str(
            mm_processor_kwargs.get(
                "apertus_vq_hub",
                mm_processor_kwargs.get("vq_hub", self.DEFAULT_VQ_HUB),
            )
        )
        vq_type = str(
            mm_processor_kwargs.get(
                "apertus_vq_type",
                mm_processor_kwargs.get("vq_type", "ibq"),
            )
        )
        vision_device, vision_device_source = self._resolve_vision_tokenizer_device(
            mm_processor_kwargs
        )
        vision_dtype = self.coerce_dtype(
            mm_processor_kwargs.get("apertus_vision_tokenizer_dtype")
        )
        if vision_device == "cpu" and vision_dtype in (torch.float16, torch.bfloat16):
            vision_dtype = torch.float32

        apertus_mm_keys = sorted(
            key for key in mm_processor_kwargs if key.startswith("apertus_")
        )
        logger.info(
            "[Apertus MM] received mm_processor_kwargs keys=%s",
            apertus_mm_keys,
        )
        logger.info(
            "[Apertus MM] resolved apertus_vision_tokenizer_device=%r source=%r",
            vision_device,
            vision_device_source,
        )
        cache_key = (
            vq_hub,
            vq_type,
            vision_device,
            vision_dtype,
        )

        if cache_key in self._vision_tokenizer_cache:
            return self._vision_tokenizer_cache[cache_key]

        cache_dir = mm_processor_kwargs.get("apertus_vq_cache_dir")
        logger.info(
            "[Apertus MM] loading Emu3.5 vision tokenizer on device=%r",
            vision_device,
        )
        vision_tokenizer = build_emu35_vision_tokenizer(
            vq_hub=vq_hub,
            default_repo=self.DEFAULT_VQ_HUB,
            device=vision_device,
            vq_type=vq_type,
            cache_dir=cache_dir if isinstance(cache_dir, str) else None,
        )
        if isinstance(vision_dtype, torch.dtype):
            vision_tokenizer = vision_tokenizer.to(dtype=vision_dtype)

        self._vision_tokenizer_cache[cache_key] = vision_tokenizer
        return vision_tokenizer

    def build_apertus_image_prompt(
        self,
        image_tokens: torch.Tensor,
        tokenizer: TokenizerLike,
    ) -> str:
        if image_tokens.ndim != 2:
            raise ValueError(
                f"Apertus image tokens must be 2D, got "
                f"shape {tuple(image_tokens.shape)}"
            )

        height, width = image_tokens.shape
        rows = [
            "".join(
                self.VISUAL_TEMPLATE.format(token_id=int(token_id)) for token_id in row
            )
            for row in image_tokens.detach().to("cpu").tolist()
        ]
        eol_token = self.apertus_special_token(
            tokenizer, "eol_token", self.DEFAULT_EOL_TOKEN
        )
        imgstr = eol_token.join(rows)

        boi_token = self.apertus_special_token(
            tokenizer, "boi_token", self.DEFAULT_BOI_TOKEN
        )
        img_token = self.apertus_special_token(
            tokenizer, "img_token", self.DEFAULT_IMG_TOKEN
        )
        eoi_token = self.apertus_special_token(
            tokenizer, "eoi_token", self.DEFAULT_EOI_TOKEN
        )

        return f"{boi_token}{height}*{width}{img_token}{imgstr}{eoi_token}"

    def encode_images(
        self,
        images: Sequence[object],
        *,
        tokenizer: TokenizerLike,
        mm_processor_kwargs: Mapping[str, object],
    ) -> list[str]:
        if not images:
            return []

        min_pixels = self.coerce_int(
            mm_processor_kwargs.get(
                "apertus_min_pixels",
                mm_processor_kwargs.get("emu_min_pixels", self.DEFAULT_MIN_PIXELS),
            ),
            default=self.DEFAULT_MIN_PIXELS,
        )
        max_pixels = self.coerce_int(
            mm_processor_kwargs.get(
                "apertus_max_pixels",
                mm_processor_kwargs.get("emu_max_pixels", self.DEFAULT_MAX_PIXELS),
            ),
            default=self.DEFAULT_MAX_PIXELS,
        )

        vision_tokenizer = self.load_vision_tokenizer(mm_processor_kwargs)
        vision_params = next(vision_tokenizer.parameters())
        vision_device = vision_params.device
        vision_dtype = vision_params.dtype

        image_prompts: list[str] = []
        for raw_image in images:
            image = self.coerce_pil_image(raw_image)
            width, height = image.size
            current_area = width * height
            target_area = max(min(max_pixels, current_area), min_pixels)
            resized_image = self.smart_resize(
                image, target_area, self.EMU35_DS_FACTOR
            )
            resized_w, resized_h = resized_image.size

            image_tensor = torch.tensor(
                (np.array(resized_image) / 127.5 - 1.0),
                device=vision_device,
                dtype=vision_dtype,
            ).permute(2, 0, 1)

            with torch.inference_mode():
                quant, emb_loss, info = vision_tokenizer.encode(image_tensor[None])
                ind = info[2]

            token_h = resized_h // self.EMU35_DS_FACTOR
            token_w = resized_w // self.EMU35_DS_FACTOR
            image_token_grid = ind.view(token_h, token_w).to(dtype=torch.int64)
            image_prompts.append(
                self.build_apertus_image_prompt(image_token_grid, tokenizer)
            )

        return image_prompts


class ApertusAudioTokenizer:
    DEFAULT_AUDIO_PLACEHOLDER = "<|audio|>"
    DEFAULT_AUDIO_TOKENIZER_PATH = "novateur/WavTokenizer-large-unify-40token"
    DEFAULT_AUDIO_TOKENIZER_TYPE = "wavtokenizer"
    DEFAULT_AUDIO_TOKENIZER_NAME = "WavTokenizer40"
    DEFAULT_AUDIO_TOKENIZER_DEVICE = "cuda"
    DEFAULT_TARGET_SAMPLING_RATE = 24000
    DEFAULT_TARGET_PEAK_DBFS = -3.0
    DEFAULT_AUDIO_TOKEN_OFFSET = 262344
    DEFAULT_AUDIO_START_TOKEN = "<|audio_start|>"
    DEFAULT_AUDIO_END_TOKEN = "<|audio_end|>"

    _audio_tokenizer_cache: dict[tuple[str, str, bool, str, str], Any] = {}

    def __init__(self) -> None:
        pass

    @staticmethod
    def coerce_int(value: object, *, default: int) -> int:
        if value is None:
            return default
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def coerce_float(value: object, *, default: float) -> float:
        if value is None:
            return default
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def coerce_bool(value: object, *, default: bool) -> bool:
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"1", "true", "t", "yes", "y", "on"}:
                return True
            if lowered in {"0", "false", "f", "no", "n", "off"}:
                return False
        return default

    @staticmethod
    def dedupe(items: Sequence[str]) -> list[str]:
        seen: set[str] = set()
        deduped: list[str] = []
        for item in items:
            if not item or item in seen:
                continue
            seen.add(item)
            deduped.append(item)
        return deduped

    @staticmethod
    def coerce_audio_waveform(audio_obj: object) -> np.ndarray:
        if isinstance(audio_obj, np.ndarray):
            audio = audio_obj
        elif torch.is_tensor(audio_obj):
            audio = audio_obj.detach().cpu().numpy()
        elif isinstance(audio_obj, list):
            audio = np.asarray(audio_obj)
        else:
            raise TypeError(f"Unsupported audio waveform type: {type(audio_obj)}")

        if audio.ndim == 0:
            raise ValueError("Audio waveform must have at least one dimension.")

        return np.asarray(audio, dtype=np.float32)

    def normalize_audio_input(
        self,
        item: object,
    ) -> np.ndarray:
        if isinstance(item, tuple) and len(item) == 2:
            return self.coerce_audio_waveform(item[0])

        if isinstance(item, Mapping):
            if "array" in item:
                return self.coerce_audio_waveform(item["array"])
            if "audio_array" in item:
                return self.coerce_audio_waveform(item["audio_array"])

            raise TypeError(
                "Unsupported mapping keys for Apertus audio input. "
                f"Got keys: {sorted(item.keys())}"
            )

        return self.coerce_audio_waveform(item)

    @staticmethod
    def to_audio_tensor(waveform: np.ndarray) -> torch.Tensor:
        audio_tensor = torch.from_numpy(waveform).float()
        if audio_tensor.dim() == 1:
            return audio_tensor.unsqueeze(0)
        return audio_tensor

    @staticmethod
    def resolve_torch_device(tokenizer: object) -> torch.device:
        if not hasattr(tokenizer, "parameters"):
            return torch.device("cpu")

        try:
            return next(tokenizer.parameters()).device
        except Exception:
            configured = getattr(tokenizer, "device", None)
            if configured is not None:
                return torch.device(configured)

        return torch.device("cpu")

    def load_special_token_id(self, tokenizer: TokenizerLike, token_str: str) -> int:
        convert = getattr(tokenizer, "convert_tokens_to_ids", None)
        if not callable(convert):
            raise AttributeError(
                "Tokenizer must expose convert_tokens_to_ids "
                "for Apertus audio prompts."
            )

        token_id = convert(token_str)
        unk_token_id = getattr(tokenizer, "unk_token_id", 1000)
        if token_id is None or (unk_token_id is not None and token_id == unk_token_id):
            token_id = 1000
            # raise ValueError(f"Token {token_str} not found in tokenizer vocabulary.")

        return int(token_id)

    def get_audio_tokenizer(
        self,
        mm_processor_kwargs: Mapping[str, object],
    ) -> Any:
        tokenizer_path = str(
            mm_processor_kwargs.get(
                "apertus_audio_tokenizer_path",
                self.DEFAULT_AUDIO_TOKENIZER_PATH,
            )
        )
        tokenizer_type = str(
            mm_processor_kwargs.get(
                "apertus_audio_tokenizer_type",
                self.DEFAULT_AUDIO_TOKENIZER_TYPE,
            )
        ).lower()
        tokenizer_name = str(
            mm_processor_kwargs.get(
                "apertus_audio_tokenizer_name",
                self.DEFAULT_AUDIO_TOKENIZER_NAME,
            )
        )
        tokenizer_device = str(
            mm_processor_kwargs.get(
                "apertus_audio_tokenizer_device",
                self.DEFAULT_AUDIO_TOKENIZER_DEVICE,
            )
        )
        tokenizer_compile = self.coerce_bool(
            mm_processor_kwargs.get("apertus_audio_tokenizer_compile"),
            default=False,
        )

        cache_key = (
            tokenizer_path,
            tokenizer_device,
            tokenizer_compile,
            tokenizer_type,
            tokenizer_name,
        )
        if cache_key in self._audio_tokenizer_cache:
            return self._audio_tokenizer_cache[cache_key]

        if tokenizer_type != "wavtokenizer" or tokenizer_name != "WavTokenizer40":
            raise ValueError(
                "Apertus audio adapter currently supports only "
                "audio_tokenizer_type=wavtokenizer and "
                "audio_tokenizer_name=WavTokenizer40."
            )

        kwargs: dict[str, object] = {
            "device": tokenizer_device,
            "torch_compile": tokenizer_compile,
        }
        if tokenizer_path:
            kwargs["checkpoint"] = tokenizer_path

        audio_tokenizer = WavTokenizer40(**kwargs)
        self._audio_tokenizer_cache[cache_key] = audio_tokenizer
        return audio_tokenizer

    def placeholder_aliases(
        self,
        mm_processor_kwargs: Mapping[str, object],
    ) -> list[str]:
        configured_placeholder = mm_processor_kwargs.get("apertus_audio_placeholder")
        return self.dedupe(
            [
                (
                    configured_placeholder
                    if isinstance(configured_placeholder, str)
                    else ""
                ),
                self.DEFAULT_AUDIO_PLACEHOLDER,
            ]
        )

    def serialize_audio_token_ids(
        self,
        token_ids: Sequence[int],
        tokenizer: TokenizerLike,
    ) -> str:
        convert_ids_to_tokens = getattr(tokenizer, "convert_ids_to_tokens", None)
        if callable(convert_ids_to_tokens):
            token_strs = convert_ids_to_tokens(list(token_ids))
            if isinstance(token_strs, str):
                token_strs = [token_strs]
            serialized = "".join(str(token_str) for token_str in token_strs)
        else:
            serialized = tokenizer.decode(list(token_ids), skip_special_tokens=False)

        return serialized

    def encode_audios(
        self,
        audios: Sequence[object],
        *,
        tokenizer: TokenizerLike,
        mm_processor_kwargs: Mapping[str, object],
    ) -> list[str]:
        if not audios:
            return []

        token_offset = self.coerce_int(
            mm_processor_kwargs.get("apertus_audio_token_offset"),
            default=self.DEFAULT_AUDIO_TOKEN_OFFSET,
        )
        target_peak_dbfs = self.coerce_float(
            mm_processor_kwargs.get("apertus_audio_target_peak_dbfs"),
            default=self.DEFAULT_TARGET_PEAK_DBFS,
        )

        audio_tokenizer = self.get_audio_tokenizer(mm_processor_kwargs)
        audio_device = self.resolve_torch_device(audio_tokenizer)
        audio_start_id = self.load_special_token_id(
            tokenizer, self.DEFAULT_AUDIO_START_TOKEN
        )
        audio_end_id = self.load_special_token_id(
            tokenizer, self.DEFAULT_AUDIO_END_TOKEN
        )

        serialized_prompts: list[str] = []
        for raw_audio in audios:
            waveform = self.normalize_audio_input(raw_audio)
            audio_tensor = self.to_audio_tensor(waveform)

            # Match benchmark script default: peak-normalize to target dBFS.
            peak = audio_tensor.abs().max().clamp(min=1e-10)
            target_peak = 10 ** (target_peak_dbfs / 20.0)
            audio_tensor = audio_tensor * (target_peak / peak)

            audio_tensor = audio_tensor.to(audio_device)
            with torch.no_grad():
                audio_codes = audio_tokenizer.encode_audio(audio_tensor)

            if audio_codes.dim() == 2:
                audio_codes = audio_codes.squeeze(0)

            shifted_codes = (
                audio_codes.detach().to("cpu", dtype=torch.int64) + token_offset
            )
            prompt_ids = [audio_start_id]
            prompt_ids.extend(shifted_codes.tolist())
            prompt_ids.append(audio_end_id)

            serialized_prompts.append(
                self.serialize_audio_token_ids(
                    prompt_ids,
                    tokenizer,
                )
            )

        return serialized_prompts
