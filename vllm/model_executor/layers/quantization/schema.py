# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
This file contains the Pydantic schemas for various quantization-related
parameters. When a relevant quantization technique is specified, these
parameters are loaded in the form of a JSON alongside the model weights
and augment the model with additional information needed for use of that
technique. The format of this JSON should be specified by one or more
schemas contained here.

For example, when the KV cache is quantized to FP8-E4M3 (currently only
possible on ROCm), the model can be optionally augmented with KV cache
scaling factors.
"""

from typing import Optional

from pydantic import BaseModel, ConfigDict, ValidationInfo, model_validator


class KVCacheQuantSchema(BaseModel):
    dtype: str
    # Each key is a TP rank. Each value is a dictionary mapping a TP rank's
    # layer indices to their per-tensor KV cache scaling factor.
    # TODO: Consider pulling this and its validation methods out into its
    # own schema class (tricky as its members are variable)
    scaling_factor: dict[int, dict[int, float]]

    @model_validator(mode="after")
    def check_is_fp8(self) -> "KVCacheQuantSchema":
        assert self.dtype == "float8_e4m3fn", (
            "Loaded scaling factors intended for KV cache dtype = "
            f"{self.dtype} rather than float8_e4m3fn!")
        return self

    @model_validator(mode="after")
    def check_tp_ranks(self, info: ValidationInfo) -> "KVCacheQuantSchema":
        context = info.context
        if context:
            tp_size = context["tp_size"]
            num_hidden_layers = context["num_hidden_layers"]
            assert len(self.scaling_factor) == tp_size, (
                f"Loaded dictionary has TP size {len(self.scaling_factor)} "
                f"but LLM engine is currently running with TP size {tp_size}.")
            for tp_rank, layer_maps in self.scaling_factor.items():
                assert len(layer_maps) == num_hidden_layers, (
                    f"KV cache scales map for TP rank {tp_rank} is malformed. "
                    f"Expected {num_hidden_layers} layers, got "
                    f"{len(layer_maps)}.")
            for i in range(tp_size):
                assert i in self.scaling_factor, (
                    f"KV cache scales map for TP rank {i} not found.")
        return self

    @model_validator(mode="after")
    def check_current_rank(self, info: ValidationInfo) -> "KVCacheQuantSchema":
        context = info.context
        if context:
            tp_rank = context["tp_rank"]
            num_hidden_layers = context["num_hidden_layers"]
            layer_scales_map = self.scaling_factor[tp_rank]
            for i in range(num_hidden_layers):
                assert i in layer_scales_map, (
                    f"Could not find KV cache scales for layer {i} in "
                    f"TP rank {tp_rank}.")
        return self


class QuantParamSchema(BaseModel):
    # TODO: Generalize and extend with more fields
    # (e.g. weights/activations params) once functionality is enabled
    model_config = ConfigDict(protected_namespaces=())
    model_type: Optional[str]
    kv_cache: KVCacheQuantSchema

    @model_validator(mode="after")
    def check_model_type(self, info: ValidationInfo) -> "QuantParamSchema":
        context = info.context
        if context:
            model_type = context.get("model_type", None)
            if model_type is not None:
                assert model_type == self.model_type, (
                    f"Model type is {model_type} but loaded "
                    f"scaling factors belonging to different "
                    f"model type {self.model_type}!")
        return self
