# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inkling model configs for the text backbone and audio/vision towers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal, cast

import torch
from transformers.configuration_utils import PretrainedConfig


class InklingModelConfig(PretrainedConfig):
    model_type = "inkling_model"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        *,
        vocab_size: int = 201024,
        hidden_size: int = 1536,
        intermediate_size: int = 768,
        dense_intermediate_size: int | None = None,
        num_hidden_layers: int = 16,
        num_attention_heads: int = 12,
        num_key_value_heads: int = 4,
        head_dim: int | None = None,
        v_head_dim: int | None = None,
        d_rel: int = 16,
        rel_extent: int = 1024,
        local_layer_ids: list[int] | None = None,
        sliding_window_size: int = 512,
        swa_num_attention_heads: int | None = None,
        swa_num_key_value_heads: int | None = None,
        swa_head_dim: int | None = None,
        swa_v_head_dim: int | None = None,
        rms_norm_eps: float = 1e-6,
        hidden_act: str = "silu",
        q_bias: bool = False,
        o_bias: bool = False,
        use_embed_norm: bool = False,
        use_sconv: bool = False,
        sconv_kernel_size: int = 4,
        dense_mlp_idx: int = 0,
        n_routed_experts: int = 0,
        n_shared_experts: int = 0,
        num_experts_per_tok: int = 1,
        route_scale: float = 1.0,
        use_gate_bias: bool = False,
        use_global_scale: bool = False,
        norm_after_topk: bool = True,
        gate_activation: Literal["sigmoid", "softmax"] = "sigmoid",
        shared_expert_sink: bool = False,
        shared_experts_size: int = 1,
        inference_moe_w13_interleaved: bool = True,
        log_scaling_n_floor: int | None = None,
        log_scaling_alpha: float = 0.1,
        unpadded_vocab_size: int | None = None,
        padded_vocab_size: int | None = None,
        logits_mup_width_multiplier: float | None = None,
        final_logit_softcapping: float | None = None,
        tie_word_embeddings: bool = False,
        num_nextn_predict_layers: int = 0,
        chain_hidden_post_norm: bool = False,
        **kwargs: Any,
    ) -> None:
        if head_dim is None:
            head_dim = hidden_size // num_attention_heads
        if v_head_dim is None:
            v_head_dim = head_dim
        if swa_num_attention_heads is None:
            swa_num_attention_heads = num_attention_heads
        if swa_num_key_value_heads is None:
            swa_num_key_value_heads = num_key_value_heads
        if swa_head_dim is None:
            swa_head_dim = head_dim
        if swa_v_head_dim is None:
            swa_v_head_dim = swa_head_dim
        if dense_intermediate_size is None:
            dense_intermediate_size = intermediate_size
        if local_layer_ids is None:
            local_layer_ids = []

        if padded_vocab_size is None:
            padded_vocab_size = vocab_size
            vocab_size = (
                unpadded_vocab_size
                if (
                    unpadded_vocab_size is not None
                    and unpadded_vocab_size < padded_vocab_size
                )
                else vocab_size
            )

        self.vocab_size = vocab_size
        self.padded_vocab_size = padded_vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.dense_intermediate_size = dense_intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.v_head_dim = v_head_dim
        self.d_rel = d_rel
        self.rel_extent = rel_extent
        self.local_layer_ids = local_layer_ids
        self.sliding_window_size = sliding_window_size
        self.swa_num_attention_heads = swa_num_attention_heads
        self.swa_num_key_value_heads = swa_num_key_value_heads
        self.swa_head_dim = swa_head_dim
        self.swa_v_head_dim = swa_v_head_dim
        self.rms_norm_eps = rms_norm_eps
        self.hidden_act = hidden_act
        self.q_bias = q_bias
        self.o_bias = o_bias
        self.use_embed_norm = use_embed_norm
        self.use_sconv = use_sconv
        self.sconv_kernel_size = sconv_kernel_size
        self.dense_mlp_idx = dense_mlp_idx
        self.n_routed_experts = n_routed_experts
        self.num_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.num_shared_experts = n_shared_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.route_scale = route_scale
        self.use_gate_bias = use_gate_bias
        self.use_global_scale = use_global_scale
        self.norm_after_topk = norm_after_topk
        self.gate_activation = gate_activation
        self.shared_expert_sink = shared_expert_sink
        self.shared_experts_size = shared_experts_size
        self.inference_moe_w13_interleaved = inference_moe_w13_interleaved
        self.log_scaling_n_floor = log_scaling_n_floor
        self.log_scaling_alpha = log_scaling_alpha
        self.unpadded_vocab_size = self.vocab_size
        self.logits_mup_width_multiplier = logits_mup_width_multiplier
        self.final_logit_softcapping = final_logit_softcapping
        # MTP (multi-token prediction) draft head: number of depth layers in the
        # checkpoint (0 if absent); chain_norm applied after each depth.
        self.num_nextn_predict_layers = num_nextn_predict_layers
        self.chain_hidden_post_norm = chain_hidden_post_norm

        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)

    @property
    def conv_layer_ids(self) -> list[int]:
        return list(range(self.num_hidden_layers))

    @property
    def linear_layer_ids(self) -> list[int]:
        return self.conv_layer_ids

    @property
    def full_attention_layer_ids(self) -> list[int]:
        return list(range(self.num_hidden_layers))

    @property
    def mamba_chunk_size(self) -> int:
        # Floor at 64: mamba_cache_chunk_size = max(mamba_chunk_size, page_size),
        # and a floor of 1 lets the radix tree adopt another request's KV at
        # tiny shared prefixes, whose different kernel-rounding perturbs decode
        # logits.
        return 64

    @property
    def mamba2_cache_params(self) -> TMLConvCacheParams | None:
        try:
            from vllm.distributed import get_tensor_model_parallel_world_size

            tp_size = get_tensor_model_parallel_world_size()
        except (AssertionError, RuntimeError):
            tp_size = 1

        def tp_local_kv_conv_dim(num_kv_heads: int, head_dim: int) -> int:
            return max(1, num_kv_heads // tp_size) * head_dim

        full_kv_conv_dim = tp_local_kv_conv_dim(self.num_key_value_heads, self.head_dim)
        local_kv_conv_dim = tp_local_kv_conv_dim(
            self.swa_num_key_value_heads, self.swa_head_dim
        )
        stream_dim = self.hidden_size
        conv_len = self.sconv_kernel_size - 1
        shape = TMLConvStateShape(
            conv=[
                (conv_len, full_kv_conv_dim),
                (conv_len, full_kv_conv_dim),
                (conv_len, local_kv_conv_dim),
                (conv_len, local_kv_conv_dim),
                (conv_len, stream_dim),
                (conv_len, stream_dim),
            ],
            temporal=(0, 0, 0),
        )
        dtype = TMLStateDType(conv=torch.bfloat16, temporal=torch.bfloat16)
        return TMLConvCacheParams(shape=shape, layers=self.conv_layer_ids, dtype=dtype)


class InklingAudioConfig(PretrainedConfig):
    model_type = "inkling_audio_model"

    def __init__(
        self,
        *,
        decoder_dmodel: int | None = None,
        n_mel_bins: int | None = None,
        mel_vocab_size: int | None = None,
        dmel_min_value: float | None = None,
        dmel_max_value: float | None = None,
        use_audio_norm: bool | None = None,
        audio_mode: Literal["dmel", "flow"] | None = None,
        **kwargs: Any,
    ) -> None:
        values = {
            "n_mel_bins": n_mel_bins,
            "mel_vocab_size": mel_vocab_size,
            "dmel_min_value": dmel_min_value,
            "dmel_max_value": dmel_max_value,
            "use_audio_norm": use_audio_norm,
            "audio_mode": audio_mode,
        }
        if decoder_dmodel is not None and (
            missing := [name for name, value in values.items() if value is None]
        ):
            raise ValueError(
                "Enabled Inkling audio tower is missing config fields: "
                + ", ".join(missing)
            )
        self.decoder_dmodel = decoder_dmodel
        self.n_mel_bins = cast(int, n_mel_bins)
        self.mel_vocab_size = cast(int, mel_vocab_size)
        self.dmel_min_value = cast(float, dmel_min_value)
        self.dmel_max_value = cast(float, dmel_max_value)
        self.use_audio_norm = cast(bool, use_audio_norm)
        self.audio_mode = cast(Literal["dmel", "flow"], audio_mode)
        super().__init__(**kwargs)


class InklingVisionConfig(PretrainedConfig):
    model_type = "inkling_vision_model"

    def __init__(
        self,
        *,
        vision_encoder_type: Literal["linear", "hmlp"] | None = None,
        decoder_dmodel: int | None = None,
        patch_size: int | None = None,
        temporal_patch_size: int | None = None,
        n_channels: int | None = None,
        n_layers: int | None = None,
        use_vision_norm: bool | None = None,
        **kwargs: Any,
    ) -> None:
        values = {
            "vision_encoder_type": vision_encoder_type,
            "patch_size": patch_size,
            "temporal_patch_size": temporal_patch_size,
            "n_channels": n_channels,
            "n_layers": n_layers,
            "use_vision_norm": use_vision_norm,
        }
        if decoder_dmodel is not None and (
            missing := [name for name, value in values.items() if value is None]
        ):
            raise ValueError(
                "Enabled Inkling vision tower is missing config fields: "
                + ", ".join(missing)
            )
        self.vision_encoder_type = cast(Literal["linear", "hmlp"], vision_encoder_type)
        self.decoder_dmodel = decoder_dmodel
        self.patch_size = cast(int, patch_size)
        self.temporal_patch_size = cast(int, temporal_patch_size)
        self.n_channels = cast(int, n_channels)
        self.n_layers = cast(int, n_layers)
        self.use_vision_norm = cast(bool, use_vision_norm)
        super().__init__(**kwargs)


class InklingMMConfig(PretrainedConfig):
    model_type = "inkling_mm_model"
    keys_to_ignore_at_inference = ["past_key_values"]
    sub_configs: ClassVar[dict[str, type[PretrainedConfig]]] = {
        "text_config": InklingModelConfig,
        "audio_config": InklingAudioConfig,
        "vision_config": InklingVisionConfig,
    }

    def __init__(
        self,
        *,
        text_config: dict[str, Any] | InklingModelConfig | None = None,
        audio_config: dict[str, Any] | InklingAudioConfig | None = None,
        vision_config: dict[str, Any] | InklingVisionConfig | None = None,
        tie_word_embeddings: bool = False,
        **kwargs: Any,
    ) -> None:
        self.text_config = (
            text_config
            if isinstance(text_config, InklingModelConfig)
            else InklingModelConfig(**(text_config or {}))
        )
        self.audio_config = (
            audio_config
            if isinstance(audio_config, InklingAudioConfig)
            else InklingAudioConfig(**(audio_config or {}))
        )
        self.vision_config = (
            vision_config
            if isinstance(vision_config, InklingVisionConfig)
            else InklingVisionConfig(**(vision_config or {}))
        )
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)

    def get_text_config(self, *args: Any, **kwargs: Any) -> InklingModelConfig:
        return self.text_config

    @property
    def vocab_size(self) -> int:
        return self.text_config.vocab_size

    @property
    def hidden_size(self) -> int:
        return self.text_config.hidden_size

    @property
    def num_hidden_layers(self) -> int:
        return self.text_config.num_hidden_layers

    @property
    def num_attention_heads(self) -> int:
        return self.text_config.num_attention_heads

    @property
    def num_key_value_heads(self) -> int:
        return self.text_config.num_key_value_heads

    @property
    def head_dim(self) -> int:
        return self.text_config.head_dim

    @property
    def full_attention_layer_ids(self) -> list[int]:
        return self.text_config.full_attention_layer_ids

    @property
    def linear_layer_ids(self) -> list[int]:
        return self.text_config.linear_layer_ids

    @property
    def conv_layer_ids(self) -> list[int]:
        return self.text_config.conv_layer_ids

    @property
    def mamba_chunk_size(self) -> int:
        return self.text_config.mamba_chunk_size

    @property
    def mamba2_cache_params(self) -> TMLConvCacheParams | None:
        return self.text_config.mamba2_cache_params


@dataclass(kw_only=True, frozen=True)
class TMLConvStateShape:
    conv: list[tuple[int, int]]
    temporal: tuple[int, int, int]


@dataclass(kw_only=True, frozen=True)
class TMLStateDType:
    conv: torch.dtype = torch.bfloat16
    temporal: torch.dtype = torch.bfloat16


@dataclass(kw_only=True, frozen=True)
class TMLConvCacheParams:
    shape: TMLConvStateShape
    layers: list[int]
    dtype: TMLStateDType = field(default_factory=TMLStateDType)
