# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import copy
import math
from typing import Any

from transformers import PreTrainedConfig, WhisperConfig

from vllm.logger import init_logger

logger = init_logger(__name__)


def adapt_config_dict(
    config_dict: dict[str, Any],
    defaults: dict[str, Any],
) -> PreTrainedConfig:
    config_dict = _remap_general_mistral_args(config_dict)
    config_dict = _remap_mistral_sliding_window(config_dict)

    if bool(config_dict.get("quantization")):
        config_dict = _remap_mistral_quantization_args(config_dict)

    is_mla = bool(config_dict.get("qk_nope_head_dim"))
    if is_mla:
        config_dict = _remap_mistral_mla_args(config_dict)

    is_moe = bool(config_dict.get("moe"))
    is_mistral_large_3 = (
        is_moe and (config_dict["moe"].get("num_shared_experts") or 0) > 0
    )
    is_dsa = bool(config_dict.get("dsa"))
    if config_dict.get("model_type") == "mamba":
        config_dict["architectures"] = ["Mamba2ForCausalLM"]
    elif is_dsa:
        assert is_mla and is_mistral_large_3, "Mistral DSA requires MLA + MoE."
        config_dict = _remap_moe_args(config_dict)
        config_dict = _remap_mistral_large_4_args(config_dict)
        config_dict["model_type"] = "mistral_large_4"
        config_dict["architectures"] = ["MistralLarge4ForCausalLM"]
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

    config = PreTrainedConfig.from_dict(config_dict)

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
        "text_config": PreTrainedConfig.from_dict(config),
        "vision_config": PreTrainedConfig.from_dict(vision_config),
    }
    if quant_config:
        config["quantization_config"] = quant_config
    return config


def _remap_mistral_yarn_args(config: dict) -> dict:
    yarn_config_map = {
        "factor": ("factor", float),
        "original_max_position_embeddings": ("original_max_position_embeddings", int),
        "beta": ("beta_fast", float),
        "alpha": ("beta_slow", float),
    }

    yarn_config = config.get("yarn") or {}
    config["rope_parameters"] = {
        "rope_type": "yarn",
        "mscale_all_dim": 1,
    }

    if rope_theta := config.pop("rope_theta", None):
        config["rope_parameters"]["rope_theta"] = rope_theta

    for old_name, (new_name, cast) in yarn_config_map.items():
        if old_name in yarn_config:
            # Cast to remove Transformers > v5 type warnings
            config["rope_parameters"][new_name] = cast(yarn_config.pop(old_name))

    # `apply_scale: false` means no magnitude correction, which Transformers
    # spells as an explicit attention_factor of 1.
    if not yarn_config.pop("apply_scale", True):
        config["rope_parameters"]["attention_factor"] = 1.0

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
        "dtype": ("dtype", config.get("dtype")),
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
            qscheme_weight = quantization.get("qscheme_weight")
            assert qscheme_act in ("NO_SCALES", "TENSOR", "BLOCK", None), (
                "Only NO_SCALES, TENSOR (default) and BLOCK are supported for "
                "qscheme_act"
            )
            assert (qscheme_act == "BLOCK") == (qscheme_weight == "BLOCK"), (
                "BLOCK quantization must be used for both weights and activations"
            )
            is_dynamic = qscheme_act in ("NO_SCALES", "BLOCK")
            config["quantization_config"] = {
                "quant_method": "fp8",
                "activation_scheme": "dynamic" if is_dynamic else "static",
            }
            if qscheme_weight == "BLOCK":
                config["quantization_config"]["weight_block_size"] = [
                    quantization["block_size_out"],
                    quantization["block_size_in"],
                ]
        elif (
            str(quantization.get("quant_method", "")).lower().replace("_", "-")
            == "compressed-tensors"
        ):
            # Pass through compressed-tensors config, while normalizing
            # quant_method to the canonical community spelling.
            quantization["quant_method"] = "compressed-tensors"
            config["quantization_config"] = quantization
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
        "text_config": PreTrainedConfig.from_dict(config),
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
            is_encoder_decoder=False,  # Override WhisperConfig default
            is_causal=encoder_args.get("causal", False),
            sliding_window=encoder_args.get("sliding_window", None),
            block_pool_size=block_pool_size,
            pos_embed=encoder_args.get("pos_embed", "sinusoidal"),
            global_log_mel_max=encoder_args["audio_encoding_args"].get(
                "global_log_mel_max"
            ),
            # only needed for RoPE
            max_position_embeddings=block_pool_size * config["max_position_embeddings"],
        ),
    }
    # Sometimes max_source_positions is explicitly set to None in params.json but this
    # is not a valid value for WhisperConfig (or downstream code that uses it).
    if (max_source_positions := encoder_args.get("max_source_positions")) is not None:
        config["audio_config"].max_source_positions = max_source_positions
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


def _remap_mistral_mla_args(config: dict) -> dict:
    if not config.get("moe"):
        moe = {
            "num_experts": 1,
            "first_k_dense_replace": config.get("num_hidden_layers"),
            "route_every_n": 1,
            "num_shared_experts": 1,
            "expert_hidden_dim": config.get("intermediate_size"),
            "num_experts_per_tok": 1,
            "routed_scale": 1.0,
            "renorm_strategy": "WEIGHTS",
            "use_load_balancing_bias": False,
            "num_expert_groups": 1,
            "num_expert_groups_per_tok": 1,
        }
        config["moe"] = moe
    return config


def _expand_layer_pattern(pattern: list[Any], num_layers: int) -> list[Any]:
    assert pattern, "Per-layer patterns must not be empty."
    return (pattern * math.ceil(num_layers / len(pattern)))[:num_layers]


# Unprefixed `dsa` keys (written by the Mistral checkpoint converter) -> the
# prefixed spelling of earlier checkpoints.
_DSA_KEY_ALIASES = {
    "topk": "index_topk",
    "n_heads": "index_n_heads",
    "head_dim": "index_head_dim",
    "rope_head_dim": "index_rope_head_dim",
    "nope": "index_nope",
    "match_nope": "indexer_match_nope",
    "ssmax_scaling": "indexer_ssmax_scaling",
    "norm_eps": "indexer_norm_eps",
    "sharing": "index_sharing",
}


def _normalize_dsa_keys(dsa: dict) -> dict:
    """Accept both the unprefixed and the prefixed `dsa` keys."""
    dsa = dict(dsa)
    for key, prefixed in _DSA_KEY_ALIASES.items():
        if key not in dsa:
            continue
        value = dsa.pop(key)
        assert dsa.get(prefixed, value) == value, (
            f"Conflicting dsa keys: {key}={value!r}, {prefixed}={dsa[prefixed]!r}"
        )
        dsa[prefixed] = value
    return dsa


def _remap_mistral_large_4_args(config: dict) -> dict:
    """Expand Mistral per-layer patterns into explicit per-layer lists.

    Scalar features (``ssmax_scaling``, ``lerope``, ``softmax_tanh_gating``)
    are passed through unchanged.
    """
    num_layers = config["num_hidden_layers"]

    nope = config.pop("nope", None)
    config["nope_layer_flags"] = (
        _expand_layer_pattern([v == "nope" for v in nope.split(",")], num_layers)
        if nope
        else [False] * num_layers
    )

    kv_sharing = config.pop("kv_sharing", None) or {}
    assert set(kv_sharing) <= {"source_layers_str"}, (
        f"Unsupported kv_sharing config: {kv_sharing}"
    )
    if source_layers_str := kv_sharing.get("source_layers_str"):
        sources = [int(v) for v in source_layers_str.split(",")]
    else:
        sources = list(range(num_layers))
    assert len(sources) == num_layers
    for layer, source in enumerate(sources):
        assert source <= layer and sources[source] == source, (
            f"Layer {layer} must share KV with an earlier, non-sharing layer"
        )
        assert config["nope_layer_flags"][layer] == config["nope_layer_flags"][source]
    config["kv_source_layers"] = sources

    dsa = _normalize_dsa_keys(config.pop("dsa"))
    for key in (
        "index_topk",
        "index_n_heads",
        "index_head_dim",
        "indexer_norm_eps",
        "indexer_ssmax_scaling",
    ):
        config[key] = dsa.pop(key)
    assert dsa.pop("index_rope_head_dim", None) in (
        None,
        -1,
        config["qk_rope_head_dim"],
    )
    assert not dsa.pop("index_nope", False), "index_nope is not supported"
    config["indexer_nope_layer_flags"] = (
        list(config["nope_layer_flags"])
        if dsa.pop("indexer_match_nope", False)
        else [False] * num_layers
    )
    assert dsa.pop("fp8_indexer", True), "Only the FP8 indexer cache is supported"
    # Default latent KV cache dtype; --kv-cache-dtype overrides it.
    config["fp8_kv_cache"] = dsa.pop("fp8_kv_cache", False)
    index_sharing = dsa.pop("index_sharing", None)
    config["index_sharing"] = (
        _expand_layer_pattern(index_sharing.split(","), num_layers)
        if index_sharing
        else ["full"] * num_layers
    )
    assert set(config["index_sharing"]) <= {"full", "shared"}
    assert config["index_sharing"][0] == "full"
    assert not dsa, f"Unparsed dsa config: {dsa}"
    if (mtp := config.pop("mtp", None)) is not None:
        config["mtp_config"] = mtp
    return config


def _parse_dspark_target_layers(serialized: str | None) -> list[int]:
    layers = [int(v) for v in (serialized or "").split(",") if v.strip()]
    assert layers, "DSpark requires mtp.target_activation_layers"
    assert all(layer >= 0 for layer in layers), (
        "DSpark target_activation_layers must be non-negative (-1, the final "
        f"post-norm hidden state, is not supported), got {layers}"
    )
    assert layers == sorted(set(layers)), (
        f"DSpark target_activation_layers must be unique and ascending, got {layers}"
    )
    return layers


def _parse_dspark_window(serialized: str | None) -> int | None:
    """``mtp.ragged_attention``: one sliding window for all draft layers."""
    if not serialized:
        return None
    windows = {int(v) for v in serialized.split(",")}
    if len(windows) != 1:
        raise NotImplementedError(
            f"DSpark draft layers must share one sliding window, got {serialized}"
        )
    window = windows.pop()
    assert window > 0, f"DSpark attention window must be positive, got {window}"
    return window


def get_mistral_large_4_dspark_config(
    text_config: PreTrainedConfig,
) -> PreTrainedConfig:
    """Build the DSpark draft config from a Mistral Large 4 ``mtp`` block.

    Mistral DSpark checkpoints describe the drafter in ``params.json``::

        "mtp": {"n_layers": L, "depth": D, "target_activation_layers": "i,j,...",
                "ragged_attention": "W,..." | null,
                "dspark": {"markov_rank": r, "head_type": "vanilla" | "rnn",
                           "draft_input": "shared_mask" | "position_specific"}}

    and ship its BF16 weights in the target checkpoint (``mtp_dspark.*``).
    Draft layers are dense MLA blocks with the target's attention geometry,
    FFN width and RoPE (YaRN, without the target's LeRoPE alpha).
    """
    mtp = getattr(text_config, "mtp_config", None) or {}
    dspark = mtp.get("dspark")
    if dspark is None:
        raise ValueError(
            "Mistral Large 4 DSpark requires a checkpoint with an `mtp.dspark` block"
        )
    dspark = dict(dspark)
    head_type = str(dspark.pop("head_type", "vanilla"))
    draft_input = str(dspark.pop("draft_input", "shared_mask"))
    markov_rank = int(dspark.pop("markov_rank"))
    assert not dspark, f"Unparsed mtp.dspark config: {dspark}"
    if head_type not in ("vanilla", "rnn"):
        raise ValueError(f"Unsupported DSpark head type: {head_type!r}")
    if draft_input not in ("shared_mask", "position_specific"):
        raise ValueError(f"Unsupported DSpark draft input: {draft_input!r}")
    target_layers = _parse_dspark_target_layers(mtp.get("target_activation_layers"))
    n_layers = int(mtp["n_layers"])
    depth = int(mtp["depth"])
    if draft_input == "position_specific" and depth < 2:
        raise ValueError(f"Position-specific DSpark masks need depth > 1, got {depth}")
    window = _parse_dspark_window(mtp.get("ragged_attention"))

    config = copy.deepcopy(text_config)
    del config.mtp_config
    config.quantization_config = None
    vocab_size = config.vocab_size
    config.update(
        {
            "architectures": ["MistralLarge4DSparkModel"],
            "num_hidden_layers": n_layers,
            "layer_types": ["full_attention" if window is None else "sliding_attention"]
            * n_layers,
            "sliding_window": window,
            "n_predict": depth,
            "dspark_block_size": depth,
            "markov_rank": markov_rank,
            "dspark_head_type": head_type,
            "dspark_draft_input": draft_input,
            "draft_vocab_size": vocab_size,
            "target_hidden_size": config.hidden_size,
            "num_target_layers": len(target_layers),
            "sample_from_anchor": True,
            "ssmax_scaling": False,
            "dflash_config": {
                "causal": False,
                # Out-of-vocabulary sentinel replaced by the learned mask embedding.
                "mask_token_id": vocab_size,
                "target_layer_ids": target_layers,
                "use_aux_hidden_state": True,
            },
        }
    )
    return config
