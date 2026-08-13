# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from copy import copy
from dataclasses import fields as dataclass_fields
from typing import Any

from pydantic import ConfigDict
from pydantic.dataclasses import dataclass

from vllm.logger import init_logger

logger = init_logger(__name__)


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class ModelArchitectureConfig:
    """Configuration for model architecture that required by vLLM runtime"""

    architectures: list[str]
    """List of model architecture class names (e.g., ['LlamaForCausalLM']).
       It can be None upon calling `vllm_config.with_hf_config(config.text_config)`"""

    model_type: str
    """Model type identifier (e.g., 'llama', 'gpt_oss')."""

    text_model_type: str | None
    """Text model type identifier (e.g., 'llama4_text')."""

    hidden_size: int
    """Hidden size of the model."""

    total_num_hidden_layers: int
    """Number of hidden layers in the model."""

    total_num_attention_heads: int
    """Number of attention heads in the model."""

    head_size: int
    """Head dimension of the model."""

    vocab_size: int
    """Vocabulary size of the model."""

    total_num_kv_heads: int
    """Number of key value heads in the model."""

    num_experts: int
    """Number of experts in the model."""

    num_experts_per_token: int
    """Number of routed experts selected per token."""

    quantization_config: dict[str, Any] | None
    """Quantization configuration dictionary containing quantization parameters."""

    is_deepseek_mla: bool
    """Whether the model is a DeepSeek MLA model."""

    is_mm_prefix_lm: bool
    """Whether the model uses image bidirectional attention."""

    rswa_window: int | None
    """Reference Sliding Window Attention window size (None disables R-SWA)."""

    derived_max_model_len_and_key: tuple[float, str | None]
    """Derived maximum model length and key from the hf config."""

    per_layer_overrides: "list[dict[str, Any]] | None" = None
    """Per-layer values for the fields that vary, `None` unless some field does.

    One dict per layer, holding only the fields whose value differs from the
    whole-model value above. Everything else is read from the whole-model config,
    so later edits to it are visible through `self[layer_idx]`."""

    def __getitem__(self, layer_idx: int) -> "ModelArchitectureConfig":
        """ModelArchitectureConfig for a specific layer.

        Returns `self` when no field varies by layer, so callers never need to
        branch on heterogeneity. Mirrors `PreTrainedConfig.per_layer_config[i]`.
        """
        if layer_idx < 0:
            # Homogeneous configs would ignore a negative index and heterogeneous
            # ones would wrap, so an off-by-one would only show up on the models
            # this exists for. Reject it for both.
            raise IndexError(f"layer index must not be negative, got {layer_idx}")
        if self.per_layer_overrides is None:
            return self
        layer = copy(self)
        object.__setattr__(layer, "per_layer_overrides", None)
        for name, value in self.per_layer_overrides[layer_idx].items():
            object.__setattr__(layer, name, value)
        return layer

    @classmethod
    def from_layers(
        cls, layers: "list[ModelArchitectureConfig]"
    ) -> "ModelArchitectureConfig":
        """Whole-model config for a checkpoint whose layers differ.

        Fields that agree across layers are taken as they are. Fields that differ
        are collapsed with `max`, so buffers are sized for the largest layer, and
        the differing values are kept per layer. No field is named here: which
        ones vary is whatever the checkpoint says.
        """
        if not layers:
            raise ValueError("a model must have at least one layer")

        merged: dict[str, Any] = {}
        overrides: list[dict[str, Any]] = [{} for _ in layers]
        for f in dataclass_fields(cls):
            if f.name == "per_layer_overrides":
                continue
            values = [getattr(layer, f.name) for layer in layers]
            if all(value == values[0] for value in values):
                merged[f.name] = values[0]
                continue
            # `bool` is an `int`, so an exact type check is what keeps a varying
            # flag from collapsing to `any`. `is_deepseek_mla` doing that would
            # make `use_mla` true model wide, and `get_num_kv_heads` then returns
            # 1 for every layer, discarding the overrides built here.
            if not all(type(value) in (int, float) for value in values):
                raise ValueError(
                    f"{f.name!r} varies across layers and has no whole-model "
                    f"value: {sorted(set(map(repr, values)))}. Only numeric "
                    f"fields collapse (with `max`, to size buffers for the "
                    f"largest layer); give this one an explicit rule in "
                    f"ModelArchitectureConfig.from_layers."
                )
            merged[f.name] = max(values)
            for override, value in zip(overrides, values):
                if value != merged[f.name]:
                    override[f.name] = value

        if len(layers) != merged["total_num_hidden_layers"]:
            raise ValueError(
                f"got {len(layers)} per-layer configs for a model with "
                f"{merged['total_num_hidden_layers']} layers"
            )
        # A checkpoint can be heterogeneous over attributes vLLM never reads, in
        # which case there is nothing to keep the layers apart for.
        return cls(**merged, per_layer_overrides=overrides if any(overrides) else None)
