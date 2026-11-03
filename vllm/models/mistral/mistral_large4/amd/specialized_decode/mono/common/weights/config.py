# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""The vocabulary a mono kernel is built and fed with: how a weight is stored
(``MoeMode``, ``ExpertWeight``, ``AttentionWeight``, the MXFP4 and router
layouts, ``KvCacheLayout``), what an expert format implies (``MoeFormat``), and
one layer's geometry (``LayerConfig``).

Model-free on purpose: a model names its own geometry beside its kernel and
hands it here as a ``LayerConfig``, registering it under its name when the
string form is wanted. Nothing here knows which model it is serving.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

"""Shared model geometry, arithmetic formats, and storage-layout contracts."""


class MoeMode(str, Enum):
    """Public arithmetic modes for the expert up/gate and down projections."""

    W8A8 = "w8a8"
    W8A16 = "w8a16"
    A16W4 = "a16w4"
    A8W4 = "a8w4"


class ExpertActivation(str, Enum):
    """Activation representation consumed by both expert projections."""

    FP8_BLOCK128 = "fp8_block128"
    MXFP8_BLOCK32 = "mxfp8_block32"
    BF16 = "bf16"


class ExpertWeight(str, Enum):
    """Packed expert-weight representation."""

    FP8_BLOCK128 = "fp8_block128"
    MXFP4_BLOCK32 = "mxfp4_block32"


class AttentionWeight(str, Enum):
    """Packed attention-weight representation."""

    FP8_BLOCK128 = "fp8_block128"
    FP8_PTPC = "fp8_ptpc"
    BF16 = "bf16"


class Mxfp4WeightLayout(str, Enum):
    """Physical layout of packed MXFP4 expert values."""

    NATIVE = "native"
    ATOM = "atom"


class Mxfp4ScaleLayout(str, Enum):
    """Physical layout of per-row MXFP4 E8M0 scales."""

    NATIVE = "native"
    ATOM = "atom"


class RouterWeightLayout(str, Enum):
    """Physical layout of the BF16 router matrix."""

    NATIVE = "native"
    ATOM = "atom"


class KvCacheLayout(str, Enum):
    """Physical layout of the BF16 MLA KV cache."""

    SPLIT = "split"
    ATOM = "atom"


@dataclass(frozen=True)
class MoeFormat:
    activation: ExpertActivation
    weight: ExpertWeight

    @property
    def activation_group(self) -> int | None:
        if self.activation is ExpertActivation.FP8_BLOCK128:
            return 128
        if self.activation is ExpertActivation.MXFP8_BLOCK32:
            return 32
        return None


MOE_FORMATS = {
    MoeMode.W8A8: MoeFormat(ExpertActivation.FP8_BLOCK128, ExpertWeight.FP8_BLOCK128),
    MoeMode.W8A16: MoeFormat(ExpertActivation.BF16, ExpertWeight.FP8_BLOCK128),
    MoeMode.A16W4: MoeFormat(ExpertActivation.BF16, ExpertWeight.MXFP4_BLOCK32),
    MoeMode.A8W4: MoeFormat(ExpertActivation.MXFP8_BLOCK32, ExpertWeight.MXFP4_BLOCK32),
}


def as_moe_mode(value: MoeMode | str) -> MoeMode:
    """Normalize a public mode argument and report supported values clearly."""
    if isinstance(value, MoeMode):
        return value
    try:
        return MoeMode(value)
    except ValueError as error:
        choices = ", ".join(mode.value for mode in MoeMode)
        raise ValueError(
            f"unsupported MoE mode {value!r}; expected one of: {choices}"
        ) from error


def moe_format(value: MoeMode | str) -> MoeFormat:
    """Return the independent activation and weight formats for a public mode."""
    return MOE_FORMATS[as_moe_mode(value)]


def _as_layout(value, enum_type, name):
    if isinstance(value, enum_type):
        return value
    try:
        return enum_type(value)
    except ValueError as error:
        choices = ", ".join(layout.value for layout in enum_type)
        raise ValueError(
            f"unsupported {name} {value!r}; expected one of: {choices}"
        ) from error


def as_mxfp4_weight_layout(value: Mxfp4WeightLayout | str) -> Mxfp4WeightLayout:
    return _as_layout(value, Mxfp4WeightLayout, "MXFP4 weight layout")


def as_mxfp4_scale_layout(value: Mxfp4ScaleLayout | str) -> Mxfp4ScaleLayout:
    return _as_layout(value, Mxfp4ScaleLayout, "MXFP4 scale layout")


def as_router_weight_layout(value: RouterWeightLayout | str) -> RouterWeightLayout:
    return _as_layout(value, RouterWeightLayout, "router weight layout")


def as_kv_cache_layout(value: KvCacheLayout | str) -> KvCacheLayout:
    return _as_layout(value, KvCacheLayout, "KV-cache layout")


@dataclass(frozen=True)
class LayerConfig:
    """Compile-time geometry and attention semantics for one TP decode shard."""

    name: str
    hidden: int
    q_lora: int
    kv_lora: int
    pe_dim: int
    nope_dim: int
    v_dim: int
    n_experts: int
    top_k: int
    inter: int
    route_scale: float
    local_heads: int
    attention_weight: AttentionWeight = AttentionWeight.FP8_BLOCK128
    attention_output_gate: bool = False
    attention_input_norm: bool = True
    attention_residual: bool = True
    routed_hidden: int | None = None
    shared_inter: int | None = None
    num_shared_experts: int = 1
    situ_beta: float = 1.0
    situ_linear_beta: float = 1.0
    attn_res_block_size: int | None = None

    @property
    def qkv_a_rows(self) -> int:
        rows = self.q_lora + self.kv_lora + self.pe_dim
        if self.attention_output_gate:
            rows += self.local_heads * self.v_dim
        return rows

    @property
    def moe_slots(self) -> int:
        return 1 + self.top_k

    @property
    def shared_expert(self) -> int:
        return self.n_experts

    @property
    def softmax_scale(self) -> float:
        return (self.nope_dim + self.pe_dim) ** -0.5

    @property
    def uses_latent_moe(self) -> bool:
        return self.routed_hidden is not None


EPS = 1e-5


SCALE_BM = 128


FP8_MAX = 448.0


SUPPORTED_SAMPLES = (1, 2, 4, 8)


SUPPORTED_PEERS = (1, 2, 4, 8)


MAX_LAYERS_PER_STEP = 128


# A model's named geometry, registered by the model itself. Empty here on
# purpose: this module is the vocabulary, not a roster of who speaks it.
MODEL_CONFIGS: dict[str, LayerConfig] = {}


def register_layer_config(config: LayerConfig) -> LayerConfig:
    """Make ``config`` resolvable by its name, and return it."""
    MODEL_CONFIGS[config.name] = config
    return config


def as_layer_config(value: LayerConfig | str) -> LayerConfig:
    """Normalize a public model-profile argument."""
    if isinstance(value, LayerConfig):
        return value
    try:
        return MODEL_CONFIGS[value]
    except KeyError as error:
        choices = ", ".join(MODEL_CONFIGS) or "(no model has registered one)"
        raise ValueError(
            f"unsupported model profile {value!r}; expected one of: {choices}"
        ) from error
