# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any

from transformers import PretrainedConfig, WhisperConfig

from vllm.logger import init_logger

logger = init_logger(__name__)


def adapt_config_dict(config_dict: dict[str, Any], **kwargs) -> PretrainedConfig:
    config_dict.update(kwargs)
    config_dict = _remap_general_mistral_args(config_dict)

    if bool(config_dict.get("quantization")):
        config_dict = _remap_mistral_quantization_args(config_dict)

    if bool(config_dict.get("moe")):
        config_dict["architectures"] = ["MixtralForCausalLM"]
    else:
        config_dict["architectures"] = ["MistralForCausalLM"]

    if bool(config_dict.get("yarn")):
        config_dict = _remap_mistral_yarn_args(config_dict)

    if bool(config_dict.get("llama_4_scaling")):
        llama_4_scaling_config_keys = ["original_max_position_embeddings", "beta"]
        assert all(
            [
                key in config_dict["llama_4_scaling"]
                for key in llama_4_scaling_config_keys
            ]
        ), (
            "llama_4_scaling config should define the keys: "
            f"{','.join(llama_4_scaling_config_keys)}"
        )

    is_vision = (config_dict.get("multimodal") or {}).get(
        "vision_encoder_args"
    ) or config_dict.get("vision_encoder")
    is_audio = bool(
        ((config_dict.get("multimodal") or {}).get("whisper_model_args") or {}).get(
            "encoder_args"
        )
    )

    assert not (is_vision and is_audio), "Vision and audio are mutually exclusive"

    if is_vision:
        config_dict = _remap_mistral_vision_args(config_dict)
    if is_audio:
        config_dict = _remap_mistral_audio_args(config_dict)

    config = PretrainedConfig.from_dict(config_dict)

    logger.debug("Initialized config %s", config)

    return config


def _remap_mistral_vision_args(config: dict) -> dict:
    if config.get("multimodal"):
        vision_config = config.pop("multimodal")
    else:
        vision_config = config.pop("vision_encoder")

    quant_config = config.get("quantization_config")
    config = {
        "model_type": "pixtral",
        "architectures": ["PixtralForConditionalGeneration"],
        "text_config": PretrainedConfig.from_dict(config),
        "vision_config": PretrainedConfig.from_dict(vision_config),
    }
    if quant_config:
        config["quantization_config"] = quant_config
    return config


def _remap_mistral_yarn_args(config: dict) -> dict:
    yarn_config_map = {
        "factor": "factor",
        "original_max_position_embeddings": "original_max_position_embeddings",
        "beta": "beta_fast",
        "alpha": "beta_slow",
        "apply_scale": "apply_yarn_scaling",
    }
    yarn_config = config.get("yarn") or {}
    config["rope_parameters"] = {
        "rope_type": "yarn",
        "mscale_all_dim": 1,
    }

    if rope_theta := config.pop("rope_theta", None):
        config["rope_parameters"]["rope_theta"] = rope_theta

    for old_name, new_name in yarn_config_map.items():
        if old_name in yarn_config:
            config["rope_parameters"][new_name] = yarn_config.pop(old_name)

    assert len(yarn_config) == 0, f"Unparsed yarn config: {yarn_config}"

    return config


def _remap_general_mistral_args(config: dict) -> dict:
    # Mistral key -> HF key
    config_mapping = {
        "dim": "hidden_size",
        "norm_eps": "rms_norm_eps",
        "n_kv_heads": "num_key_value_heads",
        "n_layers": "num_hidden_layers",
        "n_heads": "num_attention_heads",
        "hidden_dim": "intermediate_size",
    }
    # HF key -> (Mistral key, default value)
    top_level_mapping_with_default = {
        "model_type": ("model_type", "transformer"),
        "hidden_act": ("activation", "silu"),
        "tie_word_embeddings": ("tied_embeddings", False),
        "max_seq_len": ("max_seq_len", config.get("max_position_embeddings", 128_000)),
        "max_position_embeddings": ("max_position_embeddings", 128_000),
    }

    for key, new_key in config_mapping.items():
        if key in config:
            config[new_key] = config.pop(key)

    for new_key, (key, default_value) in top_level_mapping_with_default.items():
        config[new_key] = config.pop(key, default_value)

    return config


def _remap_mistral_quantization_args(config: dict) -> dict:
    quantization = config.get("quantization", {})
    if quantization.get("qformat_weight") == "fp8_e4m3":
        # This maps to the FP8 static per-tensor quantization scheme
        quantization_config = {"quant_method": "fp8", "activation_scheme": "static"}
    elif quantization.get("quant_method") == "compressed-tensors":
        # Pass through the quantization config to compressed-tensors
        quantization_config = quantization
    else:
        raise ValueError(f"Found unknown quantization='{quantization}' in config")

    config["quantization_config"] = quantization_config

    return config


def _remap_mistral_audio_args(config: dict) -> dict:
    whisper_args = config["multimodal"].pop("whisper_model_args")
    encoder_args = whisper_args["encoder_args"]
    downsample_args = whisper_args["downsample_args"]

    quant_config = config.get("quantization_config")
    config = {
        "model_type": "whixtral",
        "architectures": ["VoxtralForConditionalGeneration"],
        "text_config": PretrainedConfig.from_dict(config),
        "audio_config": WhisperConfig(
            num_mel_bins=encoder_args["audio_encoding_args"]["num_mel_bins"],
            window_size=encoder_args["audio_encoding_args"]["window_size"],
            sampling_rate=encoder_args["audio_encoding_args"]["sampling_rate"],
            hop_length=encoder_args["audio_encoding_args"]["hop_length"],
            downsample_factor=downsample_args["downsample_factor"],
            d_model=encoder_args["dim"],
            encoder_layers=encoder_args["n_layers"],
            encoder_ffn_dim=encoder_args["hidden_dim"],
            encoder_attention_heads=encoder_args["n_heads"],
            vocab_size=encoder_args["vocab_size"],
            max_source_positions=encoder_args["max_source_positions"],
            is_encoder_decoder=False,  # Override WhisperConfig default
        ),
    }
    if quant_config:
        config["quantization_config"] = quant_config
    return config
