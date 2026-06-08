# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GGUF utility functions."""

from functools import cache
from os import PathLike
from pathlib import Path
from typing import Any

import gguf
import regex as re
from gguf.constants import Keys, VisionProjectorType
from gguf.quants import GGMLQuantizationType
from transformers import Gemma3Config, PretrainedConfig, SiglipVisionConfig

from vllm.logger import init_logger

from .repo_utils import file_or_path_exists, hf_api, list_filtered_repo_files

logger = init_logger(__name__)


@cache
def check_gguf_file(model: str | PathLike) -> bool:
    """Check if the file is a GGUF model."""
    model = Path(model)
    if not model.is_file():
        return False
    elif model.suffix == ".gguf":
        return True

    try:
        with model.open("rb") as f:
            header = f.read(4)

        return header == b"GGUF"
    except Exception as e:
        logger.debug("Error reading file %s: %s", model, e)
        return False


@cache
def is_remote_gguf(model: str | Path) -> bool:
    """Check if the model is a remote GGUF model.

    Recognizes two forms:
    1. Standard: ``repo_id:quant_type`` where *quant_type* is a known
       GGML quantization type (e.g. ``Q4_K_M``).
    2. Non-standard: ``repo_id:quant_type`` where *quant_type* contains
       a known GGML type with extra prefixes (e.g. ``UD-Q4_K_XL``).
       A warning is logged and actual file existence is validated later
       during download.
    """
    pattern = r"^[a-zA-Z0-9][a-zA-Z0-9._-]*/[a-zA-Z0-9][a-zA-Z0-9._-]*:[A-Za-z0-9_+-]+$"
    model = str(model)
    if re.fullmatch(pattern, model):
        _, quant_type = model.rsplit(":", 1)
        if is_valid_gguf_quant_type(quant_type):
            return True
        if is_nonstandard_gguf_quant_type(quant_type):
            logger.warning(
                "Non-standard GGUF quant type '%s' detected.",
                quant_type,
            )
            return True
    return False


def is_nonstandard_gguf_quant_type(quant_type: str) -> bool:
    """Check if a non-standard quant type contains a known GGML type.

    Splits the quant type by the last ``-`` and checks whether the
    trailing part is a standard GGML type.  For example::

        UD-Q4_K_XL      → rsplit → ["UD", "Q4_K_XL"]      → Q4_K_XL valid ✓
        UD-IQ4_NL       → rsplit → ["UD", "IQ4_NL"]       → IQ4_NL  valid ✓
        Custom-UD-Q4_K  → rsplit → ["Custom-UD", "Q4_K"]  → Q4_K    valid ✓
        RANDOM          → no "-" → False
    """
    if "-" not in quant_type:
        return False
    _, remainder = quant_type.rsplit("-", 1)
    return is_valid_gguf_quant_type(remainder)


# Common suffixes used in GGUF file naming conventions
# e.g., Q4_K_M, Q3_K_S, Q5_K_L, Q2_K_XL
_GGUF_QUANT_SUFFIXES = ("_M", "_S", "_L", "_XL", "_XS", "_XXS")
_HF_CONFIG_FILES = ("config.json",)
_HF_TOKENIZER_FILES = (
    "tokenizer_config.json",
    "tokenizer.json",
    "tokenizer.model",
    "spiece.model",
    "vocab.json",
    "merges.txt",
)
_HF_REPO_ID_PATTERN = re.compile(
    r"^[a-zA-Z0-9][a-zA-Z0-9._-]*/[a-zA-Z0-9][a-zA-Z0-9._-]*$"
)
_HF_REPO_URL_PATTERN = re.compile(
    r"^https?://huggingface\.co/"
    r"([a-zA-Z0-9][a-zA-Z0-9._-]*/[a-zA-Z0-9][a-zA-Z0-9._-]*)"
)


def is_valid_gguf_quant_type(gguf_quant_type: str) -> bool:
    """Check if the quant type is a valid GGUF quant type.

    Supports both exact GGML quant types (e.g., Q4_K, IQ1_S) and
    extended naming conventions (e.g., Q4_K_M, Q3_K_S, Q5_K_L).
    """
    # Check for exact match first
    if getattr(GGMLQuantizationType, gguf_quant_type, None) is not None:
        return True

    # Check for extended naming conventions (e.g., Q4_K_M -> Q4_K)
    for suffix in _GGUF_QUANT_SUFFIXES:
        if gguf_quant_type.endswith(suffix):
            base_type = gguf_quant_type[: -len(suffix)]
            if getattr(GGMLQuantizationType, base_type, None) is not None:
                return True

    return False


def split_remote_gguf(model: str | Path) -> tuple[str, str]:
    """Split the model into repo_id and quant type."""
    model = str(model)
    if is_remote_gguf(model):
        parts = model.rsplit(":", 1)
        return (parts[0], parts[1])
    raise ValueError(
        f"Wrong GGUF model or invalid GGUF quant type: {model}.\n"
        "- It should be in repo_id:quant_type format.\n"
        f"- Valid base quant types: {GGMLQuantizationType._member_names_}\n"
        f"- Extended suffixes also supported: {_GGUF_QUANT_SUFFIXES}\n"
        "- Non-standard GGUF quant types also supported: "
        "dash-separated prefixes (e.g. UD-Q4_K_XL, Custom-Q8_0)",
    )


def _normalize_base_model_ids(base_model: Any) -> list[str]:
    if base_model is None:
        return []
    if isinstance(base_model, str):
        return [base_model] if base_model else []
    if isinstance(base_model, (list, tuple, set)):
        return [model_id for model_id in base_model if isinstance(model_id, str)]
    return []


def _get_remote_gguf_base_model_ids(
    repo_id: str,
    revision: str | None = None,
) -> list[str]:
    try:
        info = hf_api().model_info(repo_id, revision=revision)
    except Exception as e:
        logger.debug("Failed to inspect GGUF model card for %s: %s", repo_id, e)
        return []

    card_data = getattr(info, "card_data", None)
    base_model = getattr(card_data, "base_model", None)
    if base_model is None and isinstance(card_data, dict):
        base_model = card_data.get("base_model")
    return _normalize_base_model_ids(base_model)


def _normalize_hf_repo_id(value: Any) -> str | None:
    if not isinstance(value, str):
        return None

    value = value.strip()
    if value.endswith(".git"):
        value = value[:-4]

    if _HF_REPO_ID_PATTERN.fullmatch(value):
        return value

    match = _HF_REPO_URL_PATTERN.match(value)
    if match:
        repo_id = match.group(1)
        if repo_id.endswith(".git"):
            repo_id = repo_id[:-4]
        return repo_id

    return None


def _gguf_field_value(field: Any) -> Any:
    try:
        return field.contents()
    except Exception as e:
        logger.debug("Failed to read GGUF metadata field: %s", e)
        return None


@cache
def _get_local_gguf_base_model_ids(model: str | Path) -> tuple[str, ...]:
    try:
        reader = gguf.GGUFReader(str(model))
    except Exception as e:
        logger.debug("Failed to inspect GGUF metadata for %s: %s", model, e)
        return ()

    base_model_ids: list[str] = []
    for key, field in reader.fields.items():
        if not (key.startswith("general.base_model.") and key.endswith(".repo_url")):
            continue
        if repo_id := _normalize_hf_repo_id(_gguf_field_value(field)):
            base_model_ids.append(repo_id)

    # Preserve metadata order while dropping duplicates.
    return tuple(dict.fromkeys(base_model_ids))


def _source_has_any_file(
    model: str | Path,
    filenames: tuple[str, ...],
    revision: str | None = None,
) -> bool:
    return any(file_or_path_exists(model, filename, revision) for filename in filenames)


def _resolve_gguf_hf_source(
    model: str | Path,
    filenames: tuple[str, ...],
    revision: str | None = None,
) -> str | Path:
    if is_remote_gguf(model):
        source: str | Path
        source, _ = split_remote_gguf(model)
        base_model_ids = _get_remote_gguf_base_model_ids(
            source,
            revision=revision,
        )
    elif check_gguf_file(model):
        source = Path(model).parent
        base_model_ids = list(_get_local_gguf_base_model_ids(model))
    else:
        return model

    if _source_has_any_file(source, filenames, revision=revision):
        return source

    for base_model in base_model_ids:
        # GGUF repo revisions are not meaningful for the referenced HF base model.
        if _source_has_any_file(base_model, filenames, revision=None):
            return base_model

    return source


def resolve_gguf_config_source(
    model: str | Path,
    revision: str | None = None,
) -> str | Path:
    """Resolve where a GGUF model should load its HF config from.

    Remote GGUF repos often contain only quantized weights.  If their model card
    points to an original HF base model, use it only after verifying that the
    candidate actually provides ``config.json``.
    """
    if is_gguf(model):
        return _resolve_gguf_hf_source(
            model,
            _HF_CONFIG_FILES,
            revision=revision,
        )
    return model


def resolve_gguf_tokenizer_source(
    model: str | Path,
    revision: str | None = None,
) -> str | Path:
    """Resolve where a GGUF model should load tokenizer/processor files from."""
    if is_gguf(model):
        return _resolve_gguf_hf_source(
            model,
            _HF_TOKENIZER_FILES,
            revision=revision,
        )
    return model


def is_gguf(model: str | Path) -> bool:
    """Check if the model is a GGUF model.

    Args:
        model: Model name, path, or Path object to check.

    Returns:
        True if the model is a GGUF model, False otherwise.
    """
    model = str(model)

    # Check if it's a local GGUF file
    if check_gguf_file(model):
        return True

    # Check if it's a remote GGUF model (repo_id:quant_type format)
    return is_remote_gguf(model)


def detect_gguf_multimodal(model: str) -> Path | None:
    """Check if GGUF model has multimodal projector file.

    Args:
        model: Model path string

    Returns:
        Path to mmproj file if found, None otherwise
    """
    if not model.endswith(".gguf"):
        return None

    try:
        model_path = Path(model)
        if not model_path.is_file():
            return None

        model_dir = model_path.parent
        mmproj_patterns = ["mmproj.gguf", "mmproj-*.gguf", "*mmproj*.gguf"]
        for pattern in mmproj_patterns:
            mmproj_files = list(model_dir.glob(pattern))
            if mmproj_files:
                return mmproj_files[0]
        return None
    except Exception:
        return None


def extract_vision_config_from_gguf(mmproj_path: str) -> "SiglipVisionConfig | None":
    """Extract vision config parameters from mmproj.gguf metadata.

    Reads vision encoder configuration from GGUF metadata fields using
    standardized GGUF constants. Automatically detects the projector type
    (e.g., gemma3, llama4) and applies model-specific parameters accordingly.

    The function extracts standard CLIP vision parameters from GGUF metadata
    and applies projector-type-specific customizations. For unknown projector
    types, it uses safe defaults from SiglipVisionConfig.

    Args:
        mmproj_path: Path to mmproj.gguf file (str or Path)

    Returns:
        SiglipVisionConfig if extraction succeeds, None if any required
        field is missing from the GGUF metadata

    Raises:
        Exception: Exceptions from GGUF reading (file not found, corrupted
            file, etc.) propagate directly from gguf.GGUFReader
    """
    reader = gguf.GGUFReader(str(mmproj_path))

    # Detect projector type to apply model-specific parameters
    projector_type = None
    projector_type_field = reader.get_field(Keys.Clip.PROJECTOR_TYPE)
    if projector_type_field:
        try:
            projector_type = bytes(projector_type_field.parts[-1]).decode("utf-8")
        except (AttributeError, UnicodeDecodeError) as e:
            logger.warning("Failed to decode projector type from GGUF: %s", e)

    # Map GGUF field constants to SiglipVisionConfig parameters.
    # Uses official GGUF constants from gguf-py for standardization.
    # Format: {gguf_constant: (param_name, dtype)}
    VISION_CONFIG_FIELDS = {
        Keys.ClipVision.EMBEDDING_LENGTH: ("hidden_size", int),
        Keys.ClipVision.FEED_FORWARD_LENGTH: ("intermediate_size", int),
        Keys.ClipVision.BLOCK_COUNT: ("num_hidden_layers", int),
        Keys.ClipVision.Attention.HEAD_COUNT: ("num_attention_heads", int),
        Keys.ClipVision.IMAGE_SIZE: ("image_size", int),
        Keys.ClipVision.PATCH_SIZE: ("patch_size", int),
        Keys.ClipVision.Attention.LAYERNORM_EPS: ("layer_norm_eps", float),
    }

    def get_scalar(field):
        value = field.parts[-1]
        if hasattr(value, "shape") and value.shape == (1,):
            return value[0].item()
        if hasattr(value, "item"):
            return value.item()
        return value

    # Extract and validate all required fields
    config_params = {}
    for gguf_key, (param_name, dtype) in VISION_CONFIG_FIELDS.items():
        field = reader.get_field(gguf_key)
        if field is None:
            logger.warning(
                "Missing required vision config field '%s' in mmproj.gguf",
                gguf_key,
            )
            return None
        # Extract scalar value from GGUF field and convert to target type
        config_params[param_name] = dtype(get_scalar(field))

    # Apply model-specific parameters based on projector type
    if projector_type == VisionProjectorType.GEMMA3:
        # Gemma3 doesn't use the vision pooling head (multihead attention)
        # This is a vLLM-specific parameter used in SiglipVisionTransformer
        config_params["vision_use_head"] = False
        logger.info("Detected Gemma3 projector, disabling vision pooling head")
    # Add other projector-type-specific customizations here as needed
    # elif projector_type == VisionProjectorType.LLAMA4:
    #     config_params["vision_use_head"] = ...

    # Create config with extracted parameters
    # Note: num_channels and attention_dropout use SiglipVisionConfig defaults
    # (3 and 0.0 respectively) which are correct for all models
    config = SiglipVisionConfig(**config_params)

    if projector_type:
        logger.info(
            "Extracted vision config from mmproj.gguf (projector_type: %s)",
            projector_type,
        )
    else:
        logger.info("Extracted vision config from mmproj.gguf metadata")

    return config


def maybe_patch_hf_config_from_gguf(
    model: str,
    hf_config: PretrainedConfig,
) -> PretrainedConfig:
    """Patch HF config for GGUF models.

    Applies GGUF-specific patches to HuggingFace config:
    1. For multimodal models: patches architecture and vision config
    2. For all GGUF models: overrides vocab_size from embedding tensor

    This ensures compatibility with GGUF models that have extended
    vocabularies (e.g., Unsloth) where the GGUF file contains more
    tokens than the HuggingFace tokenizer config specifies.

    Args:
        model: Model path string
        hf_config: HuggingFace config to patch in-place

    Returns:
        Updated HuggingFace config
    """
    # Patch multimodal config if mmproj.gguf exists
    mmproj_path = detect_gguf_multimodal(model)
    if mmproj_path is not None:
        # Create HF config for Gemma3 multimodal
        text_config = hf_config.get_text_config()
        is_gemma3 = hf_config.model_type in ("gemma3", "gemma3_text")
        vision_config = (
            extract_vision_config_from_gguf(str(mmproj_path)) if is_gemma3 else None
        )
        if vision_config is not None:
            new_hf_config = Gemma3Config(
                text_config=text_config,
                vision_config=vision_config,
                architectures=["Gemma3ForConditionalGeneration"],
            )
            hf_config = new_hf_config

    return hf_config


def get_gguf_file_path_from_hf(
    repo_id: str | Path,
    quant_type: str,
    revision: str | None = None,
) -> str:
    """Get the GGUF file path from HuggingFace Hub based on repo_id and quant_type.

    Args:
        repo_id: The HuggingFace repository ID (e.g., "Qwen/Qwen3-0.6B")
        quant_type: The quantization type (e.g., "Q4_K_M", "F16")
        revision: Optional revision/branch name

    Returns:
        The path to the GGUF file on HuggingFace Hub (e.g., "filename.gguf"),
    """
    repo_id = str(repo_id)
    gguf_patterns = [
        f"*-{quant_type}.gguf",
        f"*-{quant_type}-*.gguf",
        f"*/*-{quant_type}.gguf",
        f"*/*-{quant_type}-*.gguf",
    ]
    matching_files = list_filtered_repo_files(
        repo_id,
        allow_patterns=gguf_patterns,
        revision=revision,
    )

    if len(matching_files) == 0:
        raise ValueError(
            "Could not find GGUF file for repo %s with quantization %s.",
            repo_id,
            quant_type,
        )

    # Sort to ensure consistent ordering (prefer non-sharded files)
    matching_files.sort(key=lambda x: (x.count("-"), x))
    gguf_filename = matching_files[0]
    return gguf_filename
