# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any

from transformers import PretrainedConfig, WhisperConfig

from vllm.logger import init_logger

logger = init_logger(__name__)


def adapt_config_dict(
    config_dict: dict[str, Any],
    defaults: dict[str, Any],
) -> PretrainedConfig:
    config_dict = _remap_general_mistral_args(config_dict)
    config_dict = _remap_mistral_sliding_window(config_dict)

    if bool(config_dict.get("quantization")):
        config_dict = _remap_mistral_quantization_args(config_dict)

    is_moe = bool(config_dict.get("moe"))
    is_mistral_large_3 = (
        is_moe and (config_dict["moe"].get("num_shared_experts") or 0) > 0
    )
    if config_dict.get("model_type") == "mamba":
        config_dict["architectures"] = ["Mamba2ForCausalLM"]
    elif is_moe and is_mistral_large_3:
        config_dict = _remap_moe_args(config_dict)
        config_dict["model_type"] = "deepseek_v3"
        config_dict["architectures"] = ["MistralLarge3ForCausalLM"]

        assert "llama_4_scaling" in config_dict, (
            "MistralLarge3 expect llama4 scaling config."
        )
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
    elif is_moe:
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

    for k, v in defaults.items():
        config_dict.setdefault(k, v)

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


def _remap_mistral_sliding_window(config: dict) -> dict:
    # Remap sliding_window (list) -> layer_types (list) + sliding window (int)
    # for HF compatibility
    # Mistral configs may define sliding_window as list[int]. Convert it
    # to int and add the layer_types list[str] to make it HF compatible
    if sliding_window := config.get("sliding_window"):
        if isinstance(sliding_window, list):
            pattern_repeats = config["num_hidden_layers"] // len(sliding_window)
            layer_types = sliding_window * pattern_repeats
            config["layer_types"] = [
                "full_attention" if layer_type is None else "sliding_attention"
                for layer_type in layer_types
            ]
            assert len(set(sliding_window) - {None}) <= 1, sliding_window
            config["sliding_window"] = next(filter(None, sliding_window), None)
        elif isinstance(sliding_window, int) and config.get("layer_types") is None:
            config["layer_types"] = ["sliding_attention"] * config["num_hidden_layers"]
        else:
            raise ValueError(f"Unsupported sliding_window type: {sliding_window}")

    return config


def _remap_mistral_quantization_args(config: dict) -> dict:
    if config.get("quantization"):
        quantization = config.pop("quantization", {})
        if quantization.get("qformat_weight") == "fp8_e4m3":
            qscheme_act = quantization.get("qscheme_act")
            assert qscheme_act in ("NO_SCALES", "TENSOR", None), (
                "Only NO_SCALES and TENSOR (default) are supported for qscheme_act"
            )
            is_dynamic = qscheme_act == "NO_SCALES"
            config["quantization_config"] = {
                "quant_method": "fp8",
                "activation_scheme": "dynamic" if is_dynamic else "static",
            }
        else:
            raise ValueError(f"Found unknown quantization='{quantization}' in config")

    return config


def _remap_mistral_audio_args(config: dict) -> dict:
    whisper_args = config["multimodal"].pop("whisper_model_args")
    encoder_args = whisper_args["encoder_args"]
    downsample_args = whisper_args["downsample_args"]
    downsample_factor = downsample_args["downsample_factor"]

    # make sure that k/v blocks can be allocated with
    # unified k/v cache class and pool whisper k/v cache blocks
    # with downsample_factor:1 ratio
    if encoder_args.get("causal"):
        block_pool_size = downsample_factor
        config["projection_size"] = downsample_factor * encoder_args["dim"]
    else:
        block_pool_size = 1

    architecture = (
        "VoxtralRealtimeGeneration"
        if encoder_args.get("causal")
        else "VoxtralForConditionalGeneration"
    )

    quant_config = config.get("quantization_config")
    config = {
        "model_type": "voxtral",
        "architectures": [architecture],
        "text_config": PretrainedConfig.from_dict(config),
        "audio_config": WhisperConfig(
            num_mel_bins=encoder_args["audio_encoding_args"]["num_mel_bins"],
            window_size=encoder_args["audio_encoding_args"]["window_size"],
            sampling_rate=encoder_args["audio_encoding_args"]["sampling_rate"],
            hop_length=encoder_args["audio_encoding_args"]["hop_length"],
            downsample_factor=downsample_factor,
            d_model=encoder_args["dim"],
            encoder_layers=encoder_args["n_layers"],
            encoder_ffn_dim=encoder_args["hidden_dim"],
            encoder_attention_heads=encoder_args["n_heads"],
            encoder_head_dim=encoder_args["head_dim"],
            vocab_size=encoder_args["vocab_size"],
            max_source_positions=encoder_args["max_source_positions"],
            is_encoder_decoder=False,  # Override WhisperConfig default
            is_causal=encoder_args.get("causal", False),
            sliding_window=encoder_args.get("sliding_window", None),
            block_pool_size=block_pool_size,
            pos_embed=encoder_args.get("pos_embed", "sinusoidal"),
            global_log_mel_max=encoder_args["audio_encoding_args"].get(
                "global_log_mel_norm"
            ),
            # only needed for RoPE
            max_position_embeddings=block_pool_size * config["max_position_embeddings"],
        ),
    }
    if quant_config:
        config["quantization_config"] = quant_config
    return config


def _remap_moe_args(config: dict) -> dict:
    moe_config_map = {
        "route_every_n": "moe_layer_freq",
        "first_k_dense_replace": "first_k_dense_replace",
        "num_experts_per_tok": "num_experts_per_tok",
        "num_experts": "n_routed_experts",
        "expert_hidden_dim": "moe_intermediate_size",
        "routed_scale": "routed_scaling_factor",
        "num_shared_experts": "n_shared_experts",
        "num_expert_groups": "n_group",
        "num_expert_groups_per_tok": "topk_group",
    }
    moe_config = config.get("moe", {})
    for old_name, new_name in moe_config_map.items():
        if old_name in moe_config:
            value = moe_config.pop(old_name)
            config[new_name] = value

    config["topk_method"] = None
    config["norm_topk_prob"] = True
    config["scoring_func"] = "softmax"

    return config
