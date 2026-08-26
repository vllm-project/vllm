# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Configuration adapter for Qwen3-VL DFlash/DSpARK draft checkpoints."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from transformers import Qwen3Config

_VLLM_DSPARK_ARCHITECTURE = "Qwen3VLDSparkModel"


def _config_dict(value: Any, field_name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    if hasattr(value, "to_dict"):
        return dict(value.to_dict())
    raise ValueError(f"{field_name} must be an object, got {type(value).__name__}.")


def _positive_int(value: Any, field_name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer, got {value!r}.")
    return value


class Qwen3VLDFlashConfig(Qwen3Config):
    """Normalize the training-side Qwen3-VL DFlash config for DSpARK.

    Training checkpoints contain target-model geometry under ``text_config``
    and draft geometry under ``dflash_config``. vLLM executes the checkpoint as
    a standalone text-only Qwen3 DSpARK drafter, so the fields consumed by the
    draft model are exposed at the top level while the source sections remain
    available for validation and serialization.
    """

    model_type = "qwen3_vl_dflash"
    ignore_keys_at_rope_validation = {"mrope_interleaved", "mrope_section"}

    def __init__(
        self,
        text_config: Mapping[str, Any] | None = None,
        dflash_config: Mapping[str, Any] | None = None,
        vision_config: Mapping[str, Any] | None = None,
        architectures: list[str] | None = None,
        auto_map: Mapping[str, str] | None = None,
        **kwargs: Any,
    ) -> None:
        text = _config_dict(text_config, "text_config")
        dflash = _config_dict(dflash_config, "dflash_config")
        source_vision_config = _config_dict(vision_config, "vision_config")
        is_checkpoint_config = text_config is not None or dflash_config is not None
        if is_checkpoint_config and (not text or not dflash):
            raise ValueError(
                "Qwen3-VL DSpARK checkpoints require both text_config and "
                "dflash_config objects."
            )

        target_num_layers = _positive_int(
            text.get("num_hidden_layers", 32), "text_config.num_hidden_layers"
        )
        draft_num_layers = _positive_int(
            dflash.get("num_hidden_layers", 1),
            "dflash_config.num_hidden_layers",
        )
        target_layer_ids = dflash.get("target_layer_ids", [target_num_layers - 1])
        if (
            not isinstance(target_layer_ids, list)
            or not target_layer_ids
            or any(
                not isinstance(layer_id, int) or isinstance(layer_id, bool)
                for layer_id in target_layer_ids
            )
        ):
            raise ValueError(
                "dflash_config.target_layer_ids must be a non-empty integer list."
            )
        if target_layer_ids != sorted(set(target_layer_ids)):
            raise ValueError(
                "dflash_config.target_layer_ids must be unique and strictly increasing."
            )
        if target_layer_ids[0] < 0 or target_layer_ids[-1] >= target_num_layers:
            raise ValueError(
                "dflash_config.target_layer_ids must be zero-based indices within "
                f"the {target_num_layers} target text layers."
            )

        configured_target_layers = dflash.get("num_target_layers")
        if (
            configured_target_layers is not None
            and configured_target_layers != target_num_layers
        ):
            raise ValueError(
                "dflash_config.num_target_layers must match "
                f"text_config.num_hidden_layers ({target_num_layers}); got "
                f"{configured_target_layers}."
            )
        configured_feature_layers = dflash.get("num_target_feature_layers")
        if configured_feature_layers is not None and configured_feature_layers != len(
            target_layer_ids
        ):
            raise ValueError(
                "dflash_config.num_target_feature_layers must match the number of "
                f"target_layer_ids ({len(target_layer_ids)}); got "
                f"{configured_feature_layers}."
            )

        layer_types = dflash.get("layer_types")
        if layer_types is None:
            layer_types = ["full_attention"] * draft_num_layers
        if not isinstance(layer_types, list) or len(layer_types) != draft_num_layers:
            raise ValueError(
                "dflash_config.layer_types must contain one entry per draft layer "
                f"({draft_num_layers}); got {layer_types!r}."
            )

        rope_parameters = text.get("rope_parameters") or text.get("rope_scaling") or {}
        if not isinstance(rope_parameters, Mapping):
            raise ValueError("text_config.rope_scaling must be an object.")
        rope_parameters = dict(rope_parameters)
        rope_parameters.setdefault("rope_theta", text.get("rope_theta", 1_000_000.0))

        source_architectures = list(architectures or [])
        dtype = kwargs.pop("dtype", text.get("dtype"))
        transformers_version = kwargs.pop("transformers_version", None)
        block_size = _positive_int(
            dflash.get("block_size", kwargs.pop("block_size", 1)),
            "dflash_config.block_size",
        )
        markov_rank = _positive_int(
            dflash.get("markov_rank", kwargs.pop("markov_rank", 1)),
            "dflash_config.markov_rank",
        )

        super().__init__(
            transformers_version=transformers_version,
            architectures=[_VLLM_DSPARK_ARCHITECTURE],
            output_hidden_states=kwargs.pop("output_hidden_states", False),
            return_dict=kwargs.pop("return_dict", True),
            dtype=dtype,
            chunk_size_feed_forward=kwargs.pop("chunk_size_feed_forward", 0),
            is_encoder_decoder=kwargs.pop("is_encoder_decoder", False),
            id2label=kwargs.pop("id2label", None),
            label2id=kwargs.pop("label2id", None),
            problem_type=kwargs.pop("problem_type", None),
            vocab_size=_positive_int(text.get("vocab_size", 151936), "vocab_size"),
            hidden_size=_positive_int(text.get("hidden_size", 4096), "hidden_size"),
            intermediate_size=_positive_int(
                text.get("intermediate_size", 22016), "intermediate_size"
            ),
            num_hidden_layers=draft_num_layers,
            num_attention_heads=_positive_int(
                text.get("num_attention_heads", 32), "num_attention_heads"
            ),
            num_key_value_heads=_positive_int(
                text.get("num_key_value_heads", 32), "num_key_value_heads"
            ),
            head_dim=_positive_int(text.get("head_dim", 128), "head_dim"),
            hidden_act=text.get("hidden_act", "silu"),
            max_position_embeddings=_positive_int(
                text.get("max_position_embeddings", 32768),
                "max_position_embeddings",
            ),
            initializer_range=float(text.get("initializer_range", 0.02)),
            rms_norm_eps=float(text.get("rms_norm_eps", 1e-6)),
            use_cache=bool(text.get("use_cache", True)),
            tie_word_embeddings=bool(
                kwargs.pop(
                    "tie_word_embeddings", text.get("tie_word_embeddings", False)
                )
            ),
            rope_parameters=rope_parameters,
            attention_bias=bool(text.get("attention_bias", False)),
            use_sliding_window=any(
                layer_type == "sliding_attention" for layer_type in layer_types
            ),
            sliding_window=dflash.get("sliding_window"),
            max_window_layers=draft_num_layers,
            layer_types=layer_types,
            attention_dropout=float(text.get("attention_dropout", 0.0)),
            pad_token_id=kwargs.pop("pad_token_id", text.get("pad_token_id")),
            bos_token_id=kwargs.pop("bos_token_id", text.get("bos_token_id")),
            eos_token_id=kwargs.pop("eos_token_id", text.get("eos_token_id")),
        )

        # Keep the source sections without naming the target section
        # ``text_config``: PretrainedConfig.get_text_config() must return this
        # normalized, text-only draft config rather than the 28-layer target.
        self.source_architectures = source_architectures
        self.source_text_config = text
        self.vision_config = source_vision_config or None
        self.auto_map = dict(auto_map or {})

        normalized_dflash = dict(dflash)
        normalized_dflash["block_size"] = block_size
        normalized_dflash["layer_types"] = list(layer_types)
        normalized_dflash["markov_rank"] = markov_rank
        normalized_dflash["num_hidden_layers"] = draft_num_layers
        normalized_dflash["num_target_feature_layers"] = len(target_layer_ids)
        normalized_dflash["num_target_layers"] = target_num_layers
        normalized_dflash["target_layer_ids"] = list(target_layer_ids)
        normalized_dflash.setdefault("causal", False)
        normalized_dflash.setdefault("use_aux_hidden_state", True)
        self.dflash_config = normalized_dflash

        self.block_size = block_size
        self.dspark_block_size = block_size
        self.n_predict = block_size
        self.markov_rank = markov_rank
        self.markov_head_type = dflash.get("markov_head_type", "vanilla")
        self.mask_token_id = dflash.get("mask_token_id")
        self.enable_confidence_head = bool(dflash.get("enable_confidence_head", False))
        self.target_hidden_size = self.hidden_size
        self.target_layer_ids = list(target_layer_ids)
        self.dspark_target_layer_ids = list(target_layer_ids)
        self.eagle_aux_hidden_state_layer_ids = [
            layer_id + 1 for layer_id in target_layer_ids
        ]
        self.use_aux_hidden_state = True
        self.draft_vocab_size = _positive_int(
            dflash.get("draft_vocab_size", self.vocab_size), "draft_vocab_size"
        )
        self.sample_from_anchor = True
        self.dspark_bonus_anchor = False

        # Retain any harmless top-level metadata from the training config.
        for name, value in kwargs.items():
            if name != "model_type" and not hasattr(self, name):
                setattr(self, name, value)


__all__ = ["Qwen3VLDFlashConfig"]
