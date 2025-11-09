# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GGUF utility functions."""

from pathlib import Path

import gguf
from transformers import Gemma3Config, PretrainedConfig, SiglipVisionConfig

from vllm.logger import init_logger

logger = init_logger(__name__)


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
    """
    Extract vision config parameters from mmproj.gguf metadata.

    Reads vision encoder configuration from GGUF metadata fields instead of
    using hardcoded values. This makes the implementation robust across
    different Gemma3 model sizes and future variations.

    Args:
        mmproj_path: Path to mmproj.gguf file (str or Path)

    Returns:
        SiglipVisionConfig if extraction succeeds, None if required fields missing

    Raises:
        Exception: Exceptions from GGUF reading (file not found, corrupted,
            etc.) propagate directly from gguf.GGUFReader.
    """

    reader = gguf.GGUFReader(str(mmproj_path))

    # FIXME(Isotr0py, GGUF): Map from GGUF constants for standardization
    # see: https://github.com/ggml-org/llama.cpp/blob/392e09a60852d0e879d4bbedd5ace3e6852f719e/gguf-py/gguf/constants.py#L261-L281
    # Extract vision config parameters from GGUF metadata
    hidden_size = reader.get_field("clip.vision.embedding_length")
    intermediate_size = reader.get_field("clip.vision.feed_forward_length")
    num_hidden_layers = reader.get_field("clip.vision.block_count")
    num_attention_heads = reader.get_field("clip.vision.attention.head_count")
    image_size = reader.get_field("clip.vision.image_size")
    patch_size = reader.get_field("clip.vision.patch_size")
    layer_norm_eps = reader.get_field("clip.vision.attention.layer_norm_epsilon")

    # Validate all required fields are present
    if any(
        field is None
        for field in [
            hidden_size,
            intermediate_size,
            num_hidden_layers,
            num_attention_heads,
            image_size,
            patch_size,
            layer_norm_eps,
        ]
    ):
        logger.warning("Missing required vision config fields in mmproj.gguf")
        return None

    # Extract scalar values from GGUF field parts
    config = SiglipVisionConfig(
        hidden_size=int(hidden_size.parts[-1]),
        intermediate_size=int(intermediate_size.parts[-1]),
        num_hidden_layers=int(num_hidden_layers.parts[-1]),
        num_attention_heads=int(num_attention_heads.parts[-1]),
        image_size=int(image_size.parts[-1]),
        patch_size=int(patch_size.parts[-1]),
        layer_norm_eps=float(layer_norm_eps.parts[-1]),
        # Parameters not in GGUF - use safe defaults
        num_channels=3,  # Standard RGB
        attention_dropout=0.0,  # No dropout during inference
        num_image_tokens=256,  # Gemma3 uses 4x4 pooling: 4096/16=256
        vision_use_head=False,  # Gemma3 doesn't use pooling head
    )

    logger.info("Extracted vision config from mmproj.gguf metadata")
    return config


def maybe_patch_hf_config_from_gguf(
    model: str,
    hf_config: PretrainedConfig,
) -> PretrainedConfig:
    """Patch HF config for GGUF multimodal Gemma3 models.

    If model has mmproj.gguf, patches the config:
    - Forces Gemma3ForConditionalGeneration architecture
    - Extracts and sets vision_config from mmproj.gguf
    - Sets multimodal token indices

    Args:
        model: Model path string
        hf_config: HuggingFace config to patch in-place
        architectures: Original architecture list

    Returns:
        Updated architecture list (unchanged if not Gemma3 GGUF multimodal)
    """
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
            return new_hf_config

    return hf_config
