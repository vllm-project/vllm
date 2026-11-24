# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Configuration for the recurrent Dory1.5 hybrid architecture."""

import json
from typing import Any

from huggingface_hub.dataclasses import strict
from transformers.configuration_utils import PreTrainedConfig

from vllm.exceptions import VLLMValidationError


@strict
class Dory1_5Config(PreTrainedConfig):
    model_type = "dory1_5"

    vocab_size: int = 131072
    hidden_size: int = 3072
    num_hidden_layers: int = 52
    layer_types: list[str] | None = None
    n_input_layers: int = 14
    n_recurrent_layers: int = 24
    n_output_layers: int = 14
    n_recurrent_loops: int = 4
    recurrent_kv_cache_mode: str = "per_loop"
    per_tensor_quantization: dict[str, Any] | None = None
    num_attention_heads: int = 24
    num_key_value_heads: int = 8
    head_dim: int = 128
    intermediate_size: int = 12288
    hidden_act: str = "relu2"
    linear_num_value_heads: int = 96
    linear_value_head_dim: int = 32
    linear_num_key_heads: int = 8
    linear_key_head_dim: int = 64
    linear_conv_kernel_dim: int = 4
    linear_num_householder: int = 3
    loop_conditioning_hidden_size: int = 256
    loop_conditioning_max_loops: int = 4
    loop_adapter_rank: int = 256
    loop_input_mixing: bool = True
    rms_norm_eps: float = 1e-5
    max_position_embeddings: int = 262144
    tie_word_embeddings: bool = False
    bos_token_id: int | None = 1
    eos_token_id: int | list[int] | None = 11
    pad_token_id: int | None = 11

    def __post_init__(self, **kwargs):
        if self.layer_types is None:
            pattern = "M-M-M*-M-M-M*-M-M-M*-M-M-M*-M-M-M*-M-M-M-M*-M-M-M-M-"
            kinds = {"M": "linear_attention", "*": "full_attention", "-": "mlp"}
            self.layer_types = [kinds[kind] for kind in pattern]
        super().__post_init__(**kwargs)

    def validate_layer_type(self):
        if self.layer_types is None or len(self.layer_types) != self.num_hidden_layers:
            raise ValueError("layer_types must have num_hidden_layers entries")
        if set(self.layer_types) - {"linear_attention", "full_attention", "mlp"}:
            raise ValueError(
                "Dory1.5 layers must be linear_attention, full_attention, or mlp"
            )

    def validate_architecture(self):
        partitions = (
            self.n_input_layers,
            self.n_recurrent_layers,
            self.n_output_layers,
        )
        if min(partitions) < 0 or sum(partitions) != self.num_hidden_layers:
            raise ValueError("Dory1.5 layer partitions must sum to num_hidden_layers")
        if self.n_recurrent_layers < 1 or self.n_recurrent_loops < 1:
            raise ValueError(
                "n_recurrent_layers and n_recurrent_loops must be positive"
            )
        if self.recurrent_kv_cache_mode not in ("per_loop", "last_loop"):
            raise ValueError("recurrent_kv_cache_mode must be per_loop or last_loop")
        dimensions = (
            self.hidden_size,
            self.intermediate_size,
            self.num_attention_heads,
            self.num_key_value_heads,
            self.head_dim,
            self.linear_num_value_heads,
            self.linear_value_head_dim,
            self.linear_num_key_heads,
            self.linear_key_head_dim,
            self.linear_num_householder,
            self.loop_conditioning_hidden_size,
            self.loop_conditioning_max_loops,
        )
        if min(dimensions) <= 0:
            raise ValueError("Dory1.5 dimensions must be positive")
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError(
                "num_attention_heads must be divisible by num_key_value_heads"
            )
        if self.linear_num_value_heads % self.linear_num_key_heads:
            raise ValueError(
                "linear_num_value_heads must be divisible by linear_num_key_heads"
            )
        if self.linear_conv_kernel_dim not in (2, 3, 4):
            raise ValueError("linear_conv_kernel_dim must be 2, 3, or 4")
        if self.loop_conditioning_hidden_size % 2 or self.loop_adapter_rank < 0:
            raise ValueError(
                "Loop conditioning size must be even and adapter rank nonnegative"
            )
        if self.hidden_act != "relu2" or self.rms_norm_eps <= 0:
            raise ValueError("Dory1.5 requires relu2 and positive rms_norm_eps")

    def validate_per_tensor_quantization(self):
        marker = self.per_tensor_quantization
        if marker is None:
            return
        if (
            type(marker.get("version")) is not int
            or marker["version"] != 1
            or not isinstance(marker.get("allocation"), str)
            or not marker["allocation"]
        ):
            raise ValueError(
                "per_tensor_quantization requires version 1 and an allocation filename"
            )

    def get_num_loops(self, extra_args: dict[str, Any] | None) -> int:
        num_loops = (extra_args or {}).get("num_loops", self.n_recurrent_loops)
        if type(num_loops) is not int or not 1 <= num_loops <= self.n_recurrent_loops:
            raise VLLMValidationError(
                f"num_loops must be an integer in [1, {self.n_recurrent_loops}]",
                parameter="num_loops",
            )
        return num_loops

    def request_cache_salt(
        self, extra_args: dict[str, Any] | None, cache_salt: str | None
    ) -> str | None:
        num_loops = self.get_num_loops(extra_args)
        if num_loops == self.n_recurrent_loops:
            return cache_salt
        return json.dumps((self.model_type, num_loops, cache_salt))
