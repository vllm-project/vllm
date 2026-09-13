# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Fail-closed configuration contract for the NCP-OLMo vLLM backend."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

ARCHITECTURE = "NCPOlmo3ForCausalLM"
MODEL_TYPE = "ncp_olmo3"
WEIGHT_KEY_FORMAT = "huggingface_state_dict"


class BackendContractError(ValueError):
    """Raised when an exported model cannot reconstruct NCP-OLMo exactly."""


def _required(config: Mapping[str, Any], name: str) -> Any:
    if name not in config:
        raise BackendContractError(f"missing required config field: {name}")
    return config[name]


def _positive_int(config: Mapping[str, Any], name: str) -> int:
    value = _required(config, name)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise BackendContractError(f"{name} must be a positive integer, got {value!r}")
    return value


def _require_literal(config: Mapping[str, Any], name: str, expected: Any) -> None:
    value = _required(config, name)
    if (
        isinstance(expected, bool) and not isinstance(value, bool)
    ) or value != expected:
        raise BackendContractError(f"{name} must be {expected!r}, got {value!r}")


@dataclass(frozen=True)
class NCPOlmo3BackendConfig:
    """Normalized inference-time architecture required by the vLLM backend."""

    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int
    intermediate_size: int
    vocab_size: int
    max_model_len: int
    encoder_layers: int
    decoder_layers: int
    hlm_layers: int
    chunk_size: int
    codebook_size: int
    num_codebooks: int

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any]) -> NCPOlmo3BackendConfig:
        """Validate and normalize one exported `config.json` mapping."""

        architectures = _required(config, "architectures")
        model_type = _required(config, "model_type")
        supported_identity = (
            isinstance(architectures, list)
            and model_type == MODEL_TYPE
            and ARCHITECTURE in architectures
        )
        if not supported_identity:
            raise BackendContractError(
                "unsupported NCP-ArchPreview model identity: "
                f"architectures={architectures!r}, model_type={model_type!r}; "
                f"expected ({ARCHITECTURE!r}, {MODEL_TYPE!r})"
            )
        weight_key_format = _required(config, "weight_key_format")
        if weight_key_format != WEIGHT_KEY_FORMAT:
            raise BackendContractError(
                f"weight_key_format must be {WEIGHT_KEY_FORMAT!r}, got "
                f"{weight_key_format!r}; native Megatron state dicts are not "
                "supported"
            )
        if _required(config, "conceptlm_backbone") != "olmo3":
            raise BackendContractError(
                "the NCP-ArchPreview backend only supports conceptlm_backbone='olmo3'"
            )

        hidden_size = _positive_int(config, "hidden_size")
        num_token_layers = _positive_int(config, "num_layers")
        num_attention_heads = _positive_int(config, "num_attention_heads")
        num_key_value_heads = _positive_int(config, "num_query_groups")
        intermediate_size = _positive_int(config, "ffn_hidden_size")
        vocab_size = _positive_int(config, "vocab_size")
        max_model_len = _positive_int(config, "max_sequence_length")

        encoder_layers = _positive_int(config, "conceptlm_encoder_layers")
        decoder_layers = _positive_int(config, "conceptlm_decoder_layers")
        hlm_layers = _positive_int(config, "conceptlm_special_layers")
        if encoder_layers + decoder_layers != num_token_layers:
            raise BackendContractError(
                "num_layers must equal conceptlm_encoder_layers + "
                f"conceptlm_decoder_layers, got {num_token_layers} != "
                f"{encoder_layers} + {decoder_layers}"
            )
        if hidden_size % num_attention_heads:
            raise BackendContractError(
                "hidden_size must be divisible by num_attention_heads"
            )
        if num_attention_heads % num_key_value_heads:
            raise BackendContractError(
                "num_attention_heads must be divisible by num_query_groups"
            )
        if num_key_value_heads != num_attention_heads:
            raise BackendContractError(
                "the first OLMo3 backend requires full multi-head attention: "
                "num_query_groups must equal num_attention_heads"
            )

        if _required(config, "qk_layernorm") is not True:
            raise BackendContractError("qk_layernorm must be true")
        _require_literal(config, "position_embedding_type", "rope")
        if _required(config, "rotary_interleaved") is not False:
            raise BackendContractError("rotary_interleaved must be false")

        chunk_size = _positive_int(config, "conceptlm_chunk_size")
        if _required(config, "conceptlm_shift_feature") is not True:
            raise BackendContractError("conceptlm_shift_feature must be true")
        _require_literal(
            config,
            "conceptlm_chunk_merge_method",
            "meanpooling",
        )
        _require_literal(
            config,
            "conceptlm_layer_norm_option",
            "normed_add",
        )
        _require_literal(
            config,
            "conceptlm_hlm_attention_mode",
            "backbone_window",
        )
        window_size = _required(config, "window_size")
        if (
            not isinstance(window_size, list)
            or len(window_size) != 2
            or any(
                isinstance(item, bool) or not isinstance(item, int)
                for item in window_size
            )
        ):
            raise BackendContractError(
                "window_size must be a two-integer list for backbone_window HLM"
            )
        _positive_int(config, "window_attn_skip_freq")

        if _required(config, "conceptlm_hlm_ffn_hidden_size") is not None:
            raise BackendContractError("conceptlm_hlm_ffn_hidden_size must be null")

        codebook_size = _positive_int(config, "conceptlm_v22_vq_codebook_size")
        num_codebooks = _positive_int(config, "conceptlm_v22_vq_num_codebooks")
        if hidden_size % num_codebooks:
            raise BackendContractError(
                "hidden_size must be divisible by conceptlm_v22_vq_num_codebooks"
            )
        _require_literal(config, "conceptlm_v22_vq_merge_mode", "raw_logits")
        _require_literal(config, "conceptlm_v21_dd_self_dd_mode", "dd")
        for name in (
            "conceptlm_v21_dd_encoder_self_dd",
            "conceptlm_v21_dd_concept_self_dd",
            "conceptlm_v21_dd_two_route_add",
            "conceptlm_v21_enable_concept_read_encoder",
            "conceptlm_v21_enable_decoder_read_encoder",
            "conceptlm_v21_enable_decoder_read_concept",
            "conceptlm_v21_final_read_concept_gate",
        ):
            _require_literal(config, name, True)

        normalized = cls(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            intermediate_size=intermediate_size,
            vocab_size=vocab_size,
            max_model_len=max_model_len,
            encoder_layers=encoder_layers,
            decoder_layers=decoder_layers,
            hlm_layers=hlm_layers,
            chunk_size=chunk_size,
            codebook_size=codebook_size,
            num_codebooks=num_codebooks,
        )
        return normalized

    def to_dict(self) -> dict[str, Any]:
        """Return the normalized contract as JSON-serializable data."""

        return asdict(self)
