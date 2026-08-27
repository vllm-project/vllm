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
    """Raised when an exported model cannot reconstruct ConceptLM exactly."""


def _required(config: Mapping[str, Any], name: str) -> Any:
    if name not in config:
        raise BackendContractError(f"missing required config field: {name}")
    return config[name]


def _positive_int(config: Mapping[str, Any], name: str) -> int:
    value = _required(config, name)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise BackendContractError(f"{name} must be a positive integer, got {value!r}")
    return value


def _bool(config: Mapping[str, Any], name: str) -> bool:
    value = _required(config, name)
    if not isinstance(value, bool):
        raise BackendContractError(f"{name} must be boolean, got {value!r}")
    return value


def _choice(config: Mapping[str, Any], name: str, choices: set[str]) -> str:
    value = _required(config, name)
    if value not in choices:
        expected = ", ".join(sorted(choices))
        raise BackendContractError(f"{name} must be one of {expected}, got {value!r}")
    return str(value)


def normalize_qk_norm_config(
    config: Mapping[str, Any],
    *,
    hidden_size: int,
    num_attention_heads: int,
) -> tuple[str, int]:
    """Return the validated Q/K normalization mode and parameter width."""

    qk_layernorm = config.get("qk_layernorm", True)
    if not isinstance(qk_layernorm, bool) or not qk_layernorm:
        raise BackendContractError("the OLMo3 backend requires qk_layernorm=true")
    head_dim = hidden_size // num_attention_heads
    raw_mode = config.get("qk_norm_mode")
    raw_weight_size = config.get("qk_norm_weight_size")
    if raw_mode is None:
        if raw_weight_size is None or raw_weight_size == hidden_size:
            mode = "full_hidden"
        elif raw_weight_size == head_dim:
            mode = "per_head"
        else:
            raise BackendContractError(
                "cannot infer qk_norm_mode from qk_norm_weight_size="
                f"{raw_weight_size!r}"
            )
    elif raw_mode in {"full_hidden", "per_head"}:
        mode = str(raw_mode)
    else:
        raise BackendContractError(
            f"qk_norm_mode must be one of full_hidden, per_head, got {raw_mode!r}"
        )

    expected_weight_size = hidden_size if mode == "full_hidden" else head_dim
    if raw_weight_size is None:
        weight_size = expected_weight_size
    elif (
        isinstance(raw_weight_size, bool)
        or not isinstance(raw_weight_size, int)
        or raw_weight_size <= 0
    ):
        raise BackendContractError(
            f"qk_norm_weight_size must be a positive integer, got {raw_weight_size!r}"
        )
    else:
        weight_size = raw_weight_size
    if weight_size != expected_weight_size:
        raise BackendContractError(
            f"qk_norm_mode={mode!r} requires "
            f"qk_norm_weight_size={expected_weight_size}, got {weight_size}"
        )
    return mode, weight_size


@dataclass(frozen=True)
class ConceptLMBackendConfig:
    """Normalized inference-time architecture required by the vLLM backend."""

    hidden_size: int
    num_token_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    qk_norm_mode: str
    qk_norm_weight_size: int
    intermediate_size: int
    vocab_size: int
    max_model_len: int
    encoder_layers: int
    decoder_layers: int
    hlm_layers: int
    chunk_size: int
    shift_feature: bool
    chunk_merge_method: str
    layer_norm_option: str
    hlm_attention_mode: str
    hlm_ffn_hidden_size: int | None
    codebook_size: int
    num_codebooks: int
    vq_merge_mode: str
    dd_self_mode: str
    encoder_self_dd: bool
    concept_self_dd: bool
    decoder_two_route_add: bool
    concept_read_encoder: bool
    decoder_read_encoder: bool
    decoder_read_concept: bool
    final_read_concept_gate: bool

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any]) -> ConceptLMBackendConfig:
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
                "unsupported ConceptLM model identity: "
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
                "the NCP-OLMo backend only supports conceptlm_backbone='olmo3'"
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

        qk_norm_mode, qk_norm_weight_size = normalize_qk_norm_config(
            config,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
        )

        chunk_size = _positive_int(config, "conceptlm_chunk_size")
        shift_feature = _bool(config, "conceptlm_shift_feature")
        chunk_merge_method = _choice(
            config,
            "conceptlm_chunk_merge_method",
            {"meanpooling", "first", "last"},
        )
        layer_norm_option = _choice(
            config,
            "conceptlm_layer_norm_option",
            {"rawadd", "rawadd_scalar_gate", "normed_add"},
        )
        hlm_attention_mode = _choice(
            config,
            "conceptlm_hlm_attention_mode",
            {"full", "backbone_window"},
        )
        if hlm_attention_mode == "backbone_window":
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

        raw_hlm_ffn = _required(config, "conceptlm_hlm_ffn_hidden_size")
        if raw_hlm_ffn is not None and (
            isinstance(raw_hlm_ffn, bool)
            or not isinstance(raw_hlm_ffn, int)
            or raw_hlm_ffn <= 0
        ):
            raise BackendContractError(
                "conceptlm_hlm_ffn_hidden_size must be null or a positive integer"
            )

        codebook_size = _positive_int(config, "conceptlm_v22_vq_codebook_size")
        num_codebooks = _positive_int(config, "conceptlm_v22_vq_num_codebooks")
        if hidden_size % num_codebooks:
            raise BackendContractError(
                "hidden_size must be divisible by conceptlm_v22_vq_num_codebooks"
            )

        normalized = cls(
            hidden_size=hidden_size,
            num_token_layers=num_token_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            qk_norm_mode=qk_norm_mode,
            qk_norm_weight_size=qk_norm_weight_size,
            intermediate_size=intermediate_size,
            vocab_size=vocab_size,
            max_model_len=max_model_len,
            encoder_layers=encoder_layers,
            decoder_layers=decoder_layers,
            hlm_layers=hlm_layers,
            chunk_size=chunk_size,
            shift_feature=shift_feature,
            chunk_merge_method=chunk_merge_method,
            layer_norm_option=layer_norm_option,
            hlm_attention_mode=hlm_attention_mode,
            hlm_ffn_hidden_size=raw_hlm_ffn,
            codebook_size=codebook_size,
            num_codebooks=num_codebooks,
            vq_merge_mode=_choice(
                config,
                "conceptlm_v22_vq_merge_mode",
                {"raw_logits", "softmax", "hard_top1"},
            ),
            dd_self_mode=_choice(
                config,
                "conceptlm_v21_dd_self_dd_mode",
                {"dd", "cumsum"},
            ),
            encoder_self_dd=_bool(config, "conceptlm_v21_dd_encoder_self_dd"),
            concept_self_dd=_bool(config, "conceptlm_v21_dd_concept_self_dd"),
            decoder_two_route_add=_bool(config, "conceptlm_v21_dd_two_route_add"),
            concept_read_encoder=_bool(
                config, "conceptlm_v21_enable_concept_read_encoder"
            ),
            decoder_read_encoder=_bool(
                config, "conceptlm_v21_enable_decoder_read_encoder"
            ),
            decoder_read_concept=_bool(
                config, "conceptlm_v21_enable_decoder_read_concept"
            ),
            final_read_concept_gate=_bool(
                config, "conceptlm_v21_final_read_concept_gate"
            ),
        )
        return normalized

    def to_dict(self) -> dict[str, Any]:
        """Return the normalized contract as JSON-serializable data."""

        return asdict(self)
