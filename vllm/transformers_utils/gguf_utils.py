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


def is_valid_gguf_quant_type(gguf_quant_type: str) -> bool:
    """Check if the quant type is a valid GGUF quant type."""
    return getattr(GGMLQuantizationType, gguf_quant_type, None) is not None


def split_remote_gguf(model: str | Path) -> tuple[str, str]:
    """Split the model into repo_id and quant type."""
    model = str(model)
    if is_remote_gguf(model):
        parts = model.rsplit(":", 1)
        return (parts[0], parts[1])
    raise ValueError(
        f"Wrong GGUF model or invalid GGUF quant type: {model}.\n"
        "- It should be in repo_id:quant_type format.\n"
        f"- Valid GGMLQuantizationType values: {GGMLQuantizationType._member_names_}",
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


# Mapping from GGUF architecture names to HuggingFace model_type
# GGUF metadata often uses different naming conventions than HuggingFace
# (e.g., "qwen3moe" vs "qwen3_moe", no underscores in GGUF)
GGUF_ARCH_TO_HF_MODEL_TYPE: dict[str, str] = {
    "llama": "llama",
    "phi3": "phi3",
    "phi2": "phi",
    "phi": "phi",
    "phimoe": "phimoe",
    "gemma": "gemma",
    "gemma2": "gemma2",
    "gemma3": "gemma3",
    "qwen2": "qwen2",
    "qwen2moe": "qwen2_moe",
    "qwen3": "qwen3",
    "qwen3moe": "qwen3_moe",
    "starcoder2": "starcoder2",
    "gpt2": "gpt2",
    "mistral": "mistral",
    "mixtral": "mixtral",
    "falcon": "falcon",
    "baichuan": "baichuan",
    "internlm2": "internlm2",
    "mamba": "mamba",
    "nemotron": "nemotron",
}


def extract_hf_config_from_gguf(model: str) -> dict[str, Any] | None:
    """Extract HuggingFace-compatible config dict from GGUF metadata.

    This function reads GGUF metadata and constructs a config dictionary
    that can be used to create a PretrainedConfig. Useful for GGUF repos
    that don't include config.json (e.g., bartowski repos).

    Args:
        model: Path to GGUF model file

    Returns:
        Dictionary with HF-compatible config values, or None if extraction fails

    Raises:
        Exception: Exceptions from GGUF reading propagate directly
    """
    # Use check_gguf_file to validate - it reads the header magic bytes
    # This handles both .gguf extension and HuggingFace cache blob paths
    if not check_gguf_file(model):
        return None

    try:
        model_path = Path(model)

        reader = gguf.GGUFReader(str(model_path))

        # Get architecture name
        arch_field = reader.get_field(Keys.General.ARCHITECTURE)
        if arch_field is None:
            logger.warning("No architecture field found in GGUF metadata")
            return None

        arch = bytes(arch_field.parts[-1]).decode("utf-8")
        logger.info("Extracting config from GGUF metadata (architecture: %s)", arch)

        # Map GGUF architecture to HF model_type
        model_type = GGUF_ARCH_TO_HF_MODEL_TYPE.get(arch, arch)

        config_dict: dict[str, Any] = {
            "model_type": model_type,
        }

        # Helper to extract field value
        def get_field_value(key: str, default=None):
            field = reader.get_field(key.format(arch=arch))
            if field is not None:
                val = field.parts[-1]
                # Handle arrays vs scalars
                if hasattr(val, "__len__") and len(val) == 1:
                    return val[0]
                return val
            return default

        # Extract core architecture parameters
        # Using arch-specific keys from gguf.constants.Keys

        # Context length -> max_position_embeddings
        ctx_len = get_field_value(Keys.LLM.CONTEXT_LENGTH)
        if ctx_len is not None:
            config_dict["max_position_embeddings"] = int(ctx_len)

        # Embedding length -> hidden_size
        embed_len = get_field_value(Keys.LLM.EMBEDDING_LENGTH)
        if embed_len is not None:
            config_dict["hidden_size"] = int(embed_len)

        # Feed forward length -> intermediate_size
        ff_len = get_field_value(Keys.LLM.FEED_FORWARD_LENGTH)
        if ff_len is not None:
            config_dict["intermediate_size"] = int(ff_len)

        # Block count -> num_hidden_layers
        block_count = get_field_value(Keys.LLM.BLOCK_COUNT)
        if block_count is not None:
            config_dict["num_hidden_layers"] = int(block_count)

        # Attention head count -> num_attention_heads
        head_count = get_field_value(Keys.Attention.HEAD_COUNT)
        if head_count is not None:
            config_dict["num_attention_heads"] = int(head_count)

        # KV head count -> num_key_value_heads
        kv_head_count = get_field_value(Keys.Attention.HEAD_COUNT_KV)
        if kv_head_count is not None:
            config_dict["num_key_value_heads"] = int(kv_head_count)

        # RoPE frequency base -> rope_theta
        rope_freq = get_field_value(Keys.Rope.FREQ_BASE)
        if rope_freq is not None:
            config_dict["rope_theta"] = float(rope_freq)

        # Layer norm epsilon
        rms_eps = get_field_value(Keys.Attention.LAYERNORM_RMS_EPS)
        if rms_eps is not None:
            config_dict["rms_norm_eps"] = float(rms_eps)

        # Sliding window attention
        sliding_window = get_field_value(Keys.Attention.SLIDING_WINDOW)
        if sliding_window is not None:
            config_dict["sliding_window"] = int(sliding_window)

        # Token IDs - extract first so we can use for vocab_size inference
        bos_id = get_field_value(Keys.Tokenizer.BOS_ID)
        if bos_id is not None:
            config_dict["bos_token_id"] = int(bos_id)

        eos_id = get_field_value(Keys.Tokenizer.EOS_ID)
        if eos_id is not None:
            config_dict["eos_token_id"] = int(eos_id)

        # Vocab size - priority order:
        # 1. Embedding tensor shape (most reliable - actual weights)
        # 2. LLM.VOCAB_SIZE metadata field
        # 3. Tokenizer tokens list (often incomplete in quantized files)
        vocab_size = None

        # Try to get vocab_size from embedding tensor shape
        for tensor in reader.tensors:
            if tensor.name == "token_embd.weight":
                # Shape is [hidden_size, vocab_size] for GGUF
                vocab_size = tensor.shape[-1]
                logger.info(
                    "Extracted vocab_size=%d from embedding tensor shape",
                    vocab_size
                )
                break

        # Fallback to metadata field
        if vocab_size is None:
            vocab_size = get_field_value(Keys.LLM.VOCAB_SIZE)

        # Last resort: tokenizer tokens list (often incomplete)
        if vocab_size is None:
            tokens_field = reader.get_field(Keys.Tokenizer.LIST)
            if tokens_field is not None:
                token_count = len(tokens_field.parts[-1])
                # Only use if it's plausible (more than eos_token_id)
                if eos_id is not None and token_count > int(eos_id):
                    vocab_size = token_count
                else:
                    logger.warning(
                        "Tokenizer list has %d tokens but eos_token_id=%s. "
                        "Skipping unreliable vocab_size.",
                        token_count, eos_id
                    )

        if vocab_size is not None:
            config_dict["vocab_size"] = int(vocab_size)

        # Attention softcapping (for Gemma2, etc.)
        attn_softcap = get_field_value(Keys.LLM.ATTN_LOGIT_SOFTCAPPING)
        if attn_softcap is not None:
            config_dict["attn_logit_softcapping"] = float(attn_softcap)

        final_softcap = get_field_value(Keys.LLM.FINAL_LOGIT_SOFTCAPPING)
        if final_softcap is not None:
            config_dict["final_logit_softcapping"] = float(final_softcap)

        logger.info(
            "Extracted %d config fields from GGUF metadata for %s",
            len(config_dict),
            model_type,
        )

        return config_dict

    except Exception as e:
        logger.warning("Error extracting config from GGUF: %s", e)
        return None


def extract_softcap_from_gguf(model: str) -> dict[str, float]:
    """Extract attention and final logit softcap values from GGUF metadata.

    Reads softcap parameters from GGUF metadata using arch-specific keys.
    These parameters are critical for models like Gemma2 where attention
    logit softcapping prevents numerical instability.

    Args:
        model: Path to GGUF model file

    Returns:
        Dictionary with 'attn_logit_softcapping' and/or 'final_logit_softcapping'
        keys if found in GGUF metadata, empty dict otherwise
    """
    if not model.endswith(".gguf"):
        return {}

    try:
        model_path = Path(model)
        if not model_path.is_file():
            return {}

        reader = gguf.GGUFReader(str(model_path))

        # Get architecture name to build arch-specific keys
        arch_field = reader.get_field(Keys.General.ARCHITECTURE)
        if arch_field is None:
            logger.debug("No architecture field found in GGUF metadata")
            return {}

        arch = bytes(arch_field.parts[-1]).decode("utf-8")

        result = {}

        # Extract attention logit softcapping
        attn_key = Keys.LLM.ATTN_LOGIT_SOFTCAPPING.format(arch=arch)
        attn_field = reader.get_field(attn_key)
        if attn_field is not None:
            result["attn_logit_softcapping"] = float(attn_field.parts[-1])
            logger.info(
                "Extracted attn_logit_softcapping=%.2f from GGUF metadata",
                result["attn_logit_softcapping"],
            )

        # Extract final logit softcapping
        final_key = Keys.LLM.FINAL_LOGIT_SOFTCAPPING.format(arch=arch)
        final_field = reader.get_field(final_key)
        if final_field is not None:
            result["final_logit_softcapping"] = float(final_field.parts[-1])
            logger.info(
                "Extracted final_logit_softcapping=%.2f from GGUF metadata",
                result["final_logit_softcapping"],
            )

        return result

    except Exception as e:
        logger.debug("Error extracting softcap from GGUF: %s", e)
        return {}


def extract_eos_token_id_from_gguf(model: str) -> int | None:
    """Extract EOS token ID from GGUF metadata.

    GGUF files store the EOS token ID in tokenizer.ggml.eos_token_id field.
    This may differ from HuggingFace's tokenizer config (e.g., Gemma models
    use <end_of_turn> token ID 106 as EOS in GGUF, but HF tokenizer reports
    <eos> token ID 1).

    Args:
        model: Path to GGUF model file

    Returns:
        EOS token ID from GGUF metadata, or None if not found
    """
    if not model.endswith(".gguf"):
        return None

    try:
        model_path = Path(model)
        if not model_path.is_file():
            return None

        reader = gguf.GGUFReader(str(model_path))

        eos_field = reader.get_field(Keys.Tokenizer.EOS_ID)
        if eos_field is not None:
            eos_token_id = int(eos_field.parts[-1][0])
            logger.debug(
                "Extracted eos_token_id=%d from GGUF metadata",
                eos_token_id,
            )
            return eos_token_id

        return None

    except Exception as e:
        logger.debug("Error extracting EOS token ID from GGUF: %s", e)
        return None


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
            new_hf_config = Gemma3Config.from_text_vision_configs(
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
