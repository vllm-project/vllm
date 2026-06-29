# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Apertus multimodal preprocessing helpers."""

import importlib
import importlib.util
import os
import sys
from collections.abc import Mapping, Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from vllm.logger import init_logger
from vllm.multimodal.media import MediaWithBytes
from vllm.tokenizers import TokenizerLike

logger = init_logger(__name__)

_EMU35_VISION_TOKENIZER_MODULE_PREFIX = "_vllm_apertus_emu35_vision_tokenizer"
_EMU35_VQ_REQUIRED_FILES = ("config.yaml", "model.ckpt")
_APERTUS_AUDIO_TOKENIZER_REQUIRED_FILES = (
    "src/audio_tokenizers/implementations/wavtokenizer.py",
    "src/repos/wavtokenizer/encoder/utils.py",
    "src/repos/wavtokenizer/decoder/pretrained.py",
)
_APERTUS_EMU35_CODEBASE_ENV_VAR = "VLLM_APERTUS_EMU35_CODEBASE"
_APERTUS_AUDIO_TOKENIZER_CODEBASE_ENV_VAR = "VLLM_APERTUS_AUDIO_TOKENIZER_CODEBASE"
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
        if not has_required_files(expanded_path, required_files):
            raise ValueError(
                f"Local checkpoint at {expanded_path} is missing required "
                f"files: {list(required_files)}."
            )
        return str(expanded_path.resolve())

    import huggingface_hub

    local_only = huggingface_hub.constants.HF_HUB_OFFLINE
    if local_only:
        logger.info(
            "Using cached weights for %s (cache_dir=%s)", hf_repo_id, cache_dir)
    else:
        logger.info("Downloading %s (cache_dir=%s)", hf_repo_id, cache_dir)
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


def resolve_emu35_codebase(mm_processor_kwargs: Mapping[str, object]) -> Path:
    configured_codebase = mm_processor_kwargs.get("apertus_emu35_codebase")
    configured_candidate: Path | None = None
    if isinstance(configured_codebase, str) and configured_codebase.strip():
        configured_candidate = Path(
            os.path.expandvars(configured_codebase.strip())
        ).expanduser()
        module_path = (
            configured_candidate / "src" / "vision_tokenizer" / "__init__.py"
        )
        if module_path.is_file():
            return configured_candidate.resolve()

        raise FileNotFoundError(
            "Unable to locate Emu3.5 vision tokenizer code from "
            f"apertus_emu35_codebase={configured_candidate}. Expected a checkout "
            "with `src/vision_tokenizer/__init__.py`."
        )

    env_value = os.getenv(_APERTUS_EMU35_CODEBASE_ENV_VAR)
    if not env_value or not env_value.strip():
        raise FileNotFoundError(
            "Unable to locate Emu3.5 vision tokenizer code. Set "
            f"apertus_emu35_codebase in mm_processor_kwargs or "
            f"{_APERTUS_EMU35_CODEBASE_ENV_VAR}. Expected a checkout with "
            "`src/vision_tokenizer/__init__.py`."
        )

    candidate = Path(os.path.expandvars(env_value.strip())).expanduser()
    module_path = candidate / "src" / "vision_tokenizer" / "__init__.py"
    if module_path.is_file():
        return candidate.resolve()

    raise FileNotFoundError(
        "Unable to locate Emu3.5 vision tokenizer code from "
        f"{_APERTUS_EMU35_CODEBASE_ENV_VAR}={candidate}. Expected a checkout "
        "with `src/vision_tokenizer/__init__.py`."
    )


@lru_cache(maxsize=4)
def load_emu35_build_vision_tokenizer(emu35_codebase: str) -> Any:
    module_path = (
        Path(emu35_codebase).resolve() / "src" / "vision_tokenizer" / "__init__.py"
    )
    if not module_path.is_file():
        raise FileNotFoundError(
            f"Unable to locate Emu3.5 vision tokenizer module: {module_path}"
        )

    module_name = f"{_EMU35_VISION_TOKENIZER_MODULE_PREFIX}_{abs(hash(module_path))}"
    module = sys.modules.get(module_name)
    if module is None:
        spec = importlib.util.spec_from_file_location(
            module_name,
            module_path,
            submodule_search_locations=[str(module_path.parent)],
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"Failed to build import spec for {module_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

    build_vision_tokenizer = getattr(module, "build_vision_tokenizer", None)
    if build_vision_tokenizer is None:
        raise AttributeError(
            "Emu3.5 vision tokenizer module does not expose "
            "`build_vision_tokenizer`."
        )
    return build_vision_tokenizer


def build_emu35_vision_tokenizer(
    *,
    emu35_codebase: Path,
    vq_hub: str,
    default_repo: str,
    device: str,
    vq_type: str = "ibq",
    cache_dir: str | None = None,
    **kwargs: Any,
) -> Any:
    local_vq_path = ensure_local_emu35_weights(
        vq_hub,
        default_repo,
        required_files=_EMU35_VQ_REQUIRED_FILES,
        cache_dir=cache_dir,
    )
    build_vision_tokenizer = load_emu35_build_vision_tokenizer(str(emu35_codebase))
    return build_vision_tokenizer(
        type=vq_type,
        model_path=local_vq_path,
        device=device,
        **kwargs,
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

    def __init__(self) -> None:
        self._vision_tokenizer_cache: dict[
            tuple[str, str, str, torch.dtype, bool, str], Any
        ] = {}

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
    def extract_emu35_token_grid(
        encode_out: Any,
        token_height: int,
        token_width: int,
    ) -> torch.Tensor:
        def _unwrap_token_payload(payload: Any) -> Any:
            token = payload
            if isinstance(token, tuple):
                token = token[2] if len(token) >= 3 else token[-1]

            while isinstance(token, (list, tuple)):
                if not token:
                    raise ValueError(
                        "Apertus Emu3.5 encoding produced an empty token sequence."
                    )
                non_none = [item for item in token if item is not None]
                if not non_none:
                    raise ValueError(
                        "Apertus Emu3.5 encoding produced only None token entries."
                    )
                token = non_none[-1]

            if isinstance(token, Mapping):
                for key in ("token_ids", "indices", "codes", "tokens"):
                    value = token.get(key)
                    if value is not None:
                        return _unwrap_token_payload(value)
                non_none_values = [
                    value for value in token.values() if value is not None
                ]
                if not non_none_values:
                    raise ValueError(
                        "Apertus Emu3.5 encoding produced an empty token mapping."
                    )
                token = non_none_values[-1]

            return token

        token = _unwrap_token_payload(encode_out)
        if not isinstance(token, torch.Tensor):
            token = torch.tensor(token)

        while token.ndim > 2:
            token = token[0] if token.shape[0] == 1 else token[-1]

        if token.ndim == 1:
            expected = token_height * token_width
            if token.numel() != expected:
                raise ValueError(
                    "Apertus Emu3.5 token length mismatch: "
                    f"got {token.numel()}, expected {expected}."
                )
            token = token.view(token_height, token_width)
        elif token.ndim == 2:
            if token.shape == (token_height, token_width):
                pass
            elif token.numel() == token_height * token_width:
                token = token.reshape(token_height, token_width)
            else:
                raise ValueError(
                    "Apertus Emu3.5 token grid shape mismatch: "
                    f"got {tuple(token.shape)}, expected "
                    f"{(token_height, token_width)}."
                )
        else:
            raise ValueError(f"Unexpected Emu3.5 token rank: {token.ndim}.")

        return token.to(dtype=torch.int64)

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

        trust_remote_code = bool(
            mm_processor_kwargs.get("apertus_vq_trust_remote_code", True)
        )
        apertus_mm_keys = sorted(
            key for key in mm_processor_kwargs if key.startswith("apertus_")
        )
        codebase_value = mm_processor_kwargs.get("apertus_emu35_codebase")
        codebase_source = (
            "mm_processor_kwargs"
            if isinstance(codebase_value, str) and codebase_value.strip()
            else f"env:{_APERTUS_EMU35_CODEBASE_ENV_VAR}"
        )
        emu35_codebase = resolve_emu35_codebase(mm_processor_kwargs)
        logger.info(
            "[Apertus MM] received mm_processor_kwargs keys=%s",
            apertus_mm_keys,
        )
        logger.info(
            "[Apertus MM] resolved apertus_vision_tokenizer_device=%r source=%r",
            vision_device,
            vision_device_source,
        )
        logger.info(
            "[Apertus MM] resolved apertus_emu35_codebase=%r source=%r",
            str(emu35_codebase),
            codebase_source,
        )
        cache_key = (
            vq_hub,
            vq_type,
            vision_device,
            vision_dtype,
            trust_remote_code,
            str(emu35_codebase),
        )

        if cache_key in self._vision_tokenizer_cache:
            return self._vision_tokenizer_cache[cache_key]

        kwargs: dict[str, Any] = {
            "dtype": vision_dtype,
            "trust_remote_code": trust_remote_code,
        }
        cache_dir = mm_processor_kwargs.get("apertus_vq_cache_dir")
        logger.info(
            "[Apertus MM] loading Emu3.5 vision tokenizer from %r on device=%r",
            str(emu35_codebase),
            vision_device,
        )
        vision_tokenizer = build_emu35_vision_tokenizer(
            emu35_codebase=emu35_codebase,
            vq_hub=vq_hub,
            default_repo=self.DEFAULT_VQ_HUB,
            device=vision_device,
            vq_type=vq_type,
            cache_dir=cache_dir
            if isinstance(cache_dir, str)
            else None,
            **kwargs,
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
                self.VISUAL_TEMPLATE.format(token_id=int(token_id))
                for token_id in row
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
            resized_image = self.smart_resize(image, target_area, self.EMU35_DS_FACTOR)
            resized_w, resized_h = resized_image.size

            image_tensor = torch.tensor(
                (np.array(resized_image) / 127.5 - 1.0),
                device=vision_device,
                dtype=vision_dtype,
            ).permute(2, 0, 1)

            with torch.inference_mode():
                try:
                    encode_out = vision_tokenizer.encode(image_tensor[None])
                except TypeError:
                    try:
                        encode_out = vision_tokenizer.encode(
                            pixel_values=image_tensor[None]
                        )
                    except TypeError:
                        encode_out = vision_tokenizer.encode(images=image_tensor[None])

            token_h = resized_h // self.EMU35_DS_FACTOR
            token_w = resized_w // self.EMU35_DS_FACTOR
            image_token_grid = self.extract_emu35_token_grid(
                encode_out, token_h, token_w
            )
            image_prompts.append(
                self.build_apertus_image_prompt(image_token_grid, tokenizer)
            )

        return image_prompts


def resolve_apertus_audio_tokenizer_codebase(
    mm_processor_kwargs: Mapping[str, object],
) -> Path:
    def _resolve_candidate(raw_value: str, *, source: str) -> Path:
        candidate = Path(os.path.expandvars(raw_value.strip())).expanduser()
        if has_required_files(candidate, _APERTUS_AUDIO_TOKENIZER_REQUIRED_FILES):
            return candidate.resolve()
        raise FileNotFoundError(
            "Unable to locate a complete benchmark-audio-tokenizer checkout from "
            f"{source}={candidate}."
        )

    configured_codebase = mm_processor_kwargs.get("apertus_audio_tokenizer_codebase")
    if isinstance(configured_codebase, str) and configured_codebase.strip():
        return _resolve_candidate(
            configured_codebase,
            source="apertus_audio_tokenizer_codebase",
        )

    env_value = os.getenv(_APERTUS_AUDIO_TOKENIZER_CODEBASE_ENV_VAR)
    if not env_value or not env_value.strip():
        raise FileNotFoundError(
            "Unable to locate a complete benchmark-audio-tokenizer checkout "
            "for Apertus audio tokenization. Set "
            f"{_APERTUS_AUDIO_TOKENIZER_CODEBASE_ENV_VAR}."
        )

    return _resolve_candidate(
        env_value,
        source=_APERTUS_AUDIO_TOKENIZER_CODEBASE_ENV_VAR,
    )


@lru_cache(maxsize=4)
def load_wavtokenizer40_class(audio_codebase: str) -> Any:
    codebase = Path(audio_codebase).resolve()
    if str(codebase) not in sys.path:
        sys.path.insert(0, str(codebase))

    module = importlib.import_module(
        "src.audio_tokenizers.implementations.wavtokenizer"
    )
    wavtokenizer_cls = getattr(module, "WavTokenizer40", None)
    if wavtokenizer_cls is None:
        raise AttributeError(
            "benchmark-audio-tokenizer codebase does not expose WavTokenizer40."
        )
    return wavtokenizer_cls


class ApertusAudioTokenizer:
    DEFAULT_AUDIO_PLACEHOLDER = "<|audio|>"
    DEFAULT_AUDIO_TOKENIZER_PATH = (
        "/capstor/store/cscs/swissai/infra01/MLLM/wavtokenizer"
    )
    DEFAULT_AUDIO_TOKENIZER_TYPE = "wavtokenizer"
    DEFAULT_AUDIO_TOKENIZER_NAME = "WavTokenizer40"
    DEFAULT_AUDIO_TOKENIZER_DEVICE = "cuda"
    DEFAULT_TARGET_SAMPLING_RATE = 24000
    DEFAULT_TARGET_PEAK_DBFS = -3.0
    DEFAULT_AUDIO_TOKEN_OFFSET = 262344
    DEFAULT_AUDIO_START_TOKEN = "<|audio_start|>"
    DEFAULT_AUDIO_END_TOKEN = "<|audio_end|>"

    def __init__(self) -> None:
        self._audio_tokenizer_cache: dict[
            tuple[str, str, bool, str, str, str], Any
        ] = {}

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
                "Tokenizer must expose convert_tokens_to_ids for Apertus audio prompts."
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

        audio_codebase = resolve_apertus_audio_tokenizer_codebase(mm_processor_kwargs)
        cache_key = (
            tokenizer_path,
            tokenizer_device,
            tokenizer_compile,
            tokenizer_type,
            tokenizer_name,
            str(audio_codebase),
        )
        if cache_key in self._audio_tokenizer_cache:
            return self._audio_tokenizer_cache[cache_key]

        if tokenizer_type != "wavtokenizer" or tokenizer_name != "WavTokenizer40":
            raise ValueError(
                "Apertus audio adapter currently supports only "
                "audio_tokenizer_type=wavtokenizer and "
                "audio_tokenizer_name=WavTokenizer40."
            )

        wavtokenizer_cls = load_wavtokenizer40_class(str(audio_codebase))
        kwargs: dict[str, object] = {
            "device": tokenizer_device,
            "torch_compile": tokenizer_compile,
        }
        if tokenizer_path:
            kwargs["checkpoint"] = tokenizer_path

        audio_tokenizer = wavtokenizer_cls(**kwargs)
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
