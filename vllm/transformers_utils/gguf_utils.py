# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GGUF utility functions."""

from functools import cache
from os import PathLike
from pathlib import Path

import gguf
import regex as re
from gguf.constants import Keys, VisionProjectorType
from gguf.quants import GGMLQuantizationType
from transformers import Gemma3Config, PretrainedConfig, SiglipVisionConfig

from vllm.logger import init_logger

from .repo_utils import list_filtered_repo_files

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
    """Check if the model is a remote GGUF model."""
    pattern = r"^[a-zA-Z0-9][a-zA-Z0-9._-]*/[a-zA-Z0-9][a-zA-Z0-9._-]*:[A-Za-z0-9_+-]+$"
    model = str(model)
    if re.fullmatch(pattern, model):
        _, quant_type = model.rsplit(":", 1)
        return is_valid_gguf_quant_type(quant_type)
    return False


# Common suffixes used in GGUF file naming conventions
# e.g., Q4_K_M, Q3_K_S, Q5_K_L, Q2_K_XL
_GGUF_QUANT_SUFFIXES = ("_M", "_S", "_L", "_XL", "_XS", "_XXS")


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
        f"- Extended suffixes also supported: {_GGUF_QUANT_SUFFIXES}",
    )


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
        config_params[param_name] = dtype(field.parts[-1])

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
        vision_config = extract_vision_config_from_gguf(str(mmproj_path))

        # Create HF config for Gemma3 multimodal
        text_config = hf_config.get_text_config()
        is_gemma3 = hf_config.model_type in ("gemma3", "gemma3_text")
        if vision_config is not None and is_gemma3:
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
