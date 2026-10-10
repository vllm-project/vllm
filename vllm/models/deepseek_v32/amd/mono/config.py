# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from ROCm/aiter#6173 at c39b56c36 (Apache-2.0 License),
# Copyright (c) 2025 FlyDSL Project Contributors:
# aiter/ops/flydsl/kernels/glm5_mono/config.py
# ruff: noqa: E501

"""Shared model geometry, arithmetic formats, and storage-layout contracts."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum


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
    routed_hidden: int | None = None
    num_shared_experts: int = 1

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


GLM5_CONFIG = LayerConfig(
    name="glm5",
    hidden=6144,
    q_lora=2048,
    kv_lora=512,
    pe_dim=64,
    nope_dim=192,
    v_dim=256,
    n_experts=256,
    top_k=8,
    inter=256,
    route_scale=2.5,
    local_heads=8,
)

GLM5_REFERENCE_TP = 8
GLM5_TP_SIZES = (4, GLM5_REFERENCE_TP)
GLM5_QUERY_LENGTHS = (1, 4, 5, 6)
GLM5_KERNEL_SAMPLES = (1, 2, 4, 5, 6, 8, 10, 12)
GLM5_GLOBAL_HEADS = GLM5_CONFIG.local_heads * GLM5_REFERENCE_TP
GLM5_GLOBAL_INTER = GLM5_CONFIG.inter * GLM5_REFERENCE_TP


def glm5_tp_config(tp_size: int) -> LayerConfig:
    """Return the one GLM-5 shard geometry for ``tp_size``."""
    if tp_size not in GLM5_TP_SIZES:
        raise ValueError(
            f"GLM-5 tensor parallel size must be one of {GLM5_TP_SIZES}, got {tp_size}"
        )
    return replace(
        GLM5_CONFIG,
        local_heads=GLM5_GLOBAL_HEADS // tp_size,
        inter=GLM5_GLOBAL_INTER // tp_size,
    )


def glm5_attention_heads(tp_size: int, dcp_size: int = 1) -> int:
    if dcp_size not in (1, 4) or tp_size % dcp_size:
        raise ValueError(
            f"unsupported GLM-5 TP/DCP geometry: tp={tp_size}, dcp={dcp_size}"
        )
    return GLM5_GLOBAL_HEADS // (tp_size // dcp_size)


def glm5_kernel_samples(samples: int, query_length: int) -> int:
    """Choose an LDS-safe request-aligned launch width."""
    if query_length not in GLM5_QUERY_LENGTHS or samples % query_length:
        raise ValueError(
            f"unsupported GLM-5 decode shape samples={samples}, query_length={query_length}"
        )
    for chunk in reversed(GLM5_KERNEL_SAMPLES):
        if chunk % query_length == 0 and samples % chunk == 0:
            return chunk
    raise ValueError(
        f"no GLM-5 kernel chunk for samples={samples}, query_length={query_length}"
    )


MODEL_CONFIGS = {config.name: config for config in (GLM5_CONFIG,)}


def as_layer_config(value: LayerConfig | str) -> LayerConfig:
    """Normalize a public model-profile argument."""
    if isinstance(value, LayerConfig):
        return value
    try:
        return MODEL_CONFIGS[value]
    except KeyError as error:
        choices = ", ".join(MODEL_CONFIGS)
        raise ValueError(
            f"unsupported model profile {value!r}; expected one of: {choices}"
        ) from error


# Fixed GLM-5 geometry used by its performance-specialized MonoKernel.
HIDDEN = GLM5_CONFIG.hidden
Q_LORA = GLM5_CONFIG.q_lora
KV_LORA = GLM5_CONFIG.kv_lora
PE_DIM = GLM5_CONFIG.pe_dim
NOPE_DIM = GLM5_CONFIG.nope_dim
V_DIM = GLM5_CONFIG.v_dim
QKV_A_ROWS = GLM5_CONFIG.qkv_a_rows
N_EXPERTS = GLM5_CONFIG.n_experts
EXPERT_TOP_K = GLM5_CONFIG.top_k
TOP_K = EXPERT_TOP_K
MOE_SLOTS = GLM5_CONFIG.moe_slots
SHARED_EXPERT = GLM5_CONFIG.shared_expert
INTER = GLM5_CONFIG.inter
ROUTE_SCALE = GLM5_CONFIG.route_scale
EPS = 1e-5
SCALE_BM = 128
FP8_MAX = 448.0
SOFTMAX_SCALE = GLM5_CONFIG.softmax_scale

SUPPORTED_SAMPLES = (1, 2, 4, 8)
SUPPORTED_PEERS = (1, 2, 4, 8)
MAX_LAYERS_PER_STEP = 128


def validate_shard(
    samples: int,
    heads: int,
    rank: int,
    npes: int,
    sparse_attention_topk: int,
    model_config: LayerConfig | str = GLM5_CONFIG,
    supported_samples=SUPPORTED_SAMPLES,
    expected_heads: int | None = None,
) -> None:
    """Validate one model profile before allocating GPU buffers."""
    config = as_layer_config(model_config)
    if samples not in supported_samples:
        raise ValueError(f"samples must be one of {supported_samples}, got {samples}")
    expected_heads = config.local_heads if expected_heads is None else expected_heads
    if heads != expected_heads:
        raise ValueError(
            f"{config.name} requires {expected_heads} local heads, got {heads}"
        )
    if npes not in SUPPORTED_PEERS:
        raise ValueError(f"npes must be one of {SUPPORTED_PEERS}, got {npes}")
    if not 0 <= rank < npes:
        raise ValueError(f"rank must be in [0, {npes}), got {rank}")
    if sparse_attention_topk <= 0 or sparse_attention_topk % 64:
        raise ValueError(
            "sparse_attention_topk must be a positive multiple of 64, "
            f"got {sparse_attention_topk}"
        )
