# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501
# Copyright (c) 2026 FlyDSL Project Contributors

"""ML4's own geometry, and the widths its kernel is built for.

The vocabulary these are expressed in -- `LayerConfig`, `MoeMode`, the weight
and cache layouts -- is model-free and lives in `vllm.models.mistral.mistral_large4.amd.specialized_decode.mono.common.weights.config`;
what is ML4's alone is here. The shared names are re-exported, so a kernel
source taking both from one place need not care which is which.
"""

from __future__ import annotations

from dataclasses import replace

from vllm.models.mistral.mistral_large4.amd.specialized_decode.mono.common.weights.config import (  # noqa: F401  (re-exported)
    EPS,
    FP8_MAX,
    MAX_LAYERS_PER_STEP,
    MOE_FORMATS,
    SCALE_BM,
    SUPPORTED_PEERS,
    SUPPORTED_SAMPLES,
    AttentionWeight,
    ExpertActivation,
    ExpertWeight,
    KvCacheLayout,
    LayerConfig,
    MoeFormat,
    MoeMode,
    Mxfp4ScaleLayout,
    Mxfp4WeightLayout,
    RouterWeightLayout,
    as_kv_cache_layout,
    as_layer_config,
    as_moe_mode,
    as_mxfp4_scale_layout,
    as_mxfp4_weight_layout,
    as_router_weight_layout,
    moe_format,
    register_layer_config,
)

from .variant import KV_SHARE, NOPE_LAYER  # noqa: F401  (re-exported)

ML4_CONFIG = register_layer_config(
    LayerConfig(
        name="ml4",
        hidden=6144,
        q_lora=0,  # ML4 has no q-LoRA: q = wq(RMSNorm(x)); qkv_a carries only the kv latent + rope key
        kv_lora=512,
        pe_dim=64,
        nope_dim=128
        if NOPE_LAYER
        else 64,  # NoPE layers spend the rope dims on nope (q_proj gives 128/head)
        v_dim=128,
        n_experts=128,
        top_k=4,
        inter=768,  # 6144 / TP8; ml4_tp_config(4) gives 1536
        route_scale=1.0,
        local_heads=8,
    )
)
ML4_REFERENCE_TP = 8


ML4_TP_SIZES = (4, ML4_REFERENCE_TP)


ML4_GRAPH_BATCHES = tuple(range(1, 97))


ML4_AGENTX_BATCHES = (1, 2)


ML4_QUERY_LENGTHS = (1, 4, 5, 6)


ML4_KERNEL_SAMPLES = (1, 2, 4, 5, 6, 8, 10, 12)


ML4_GLOBAL_HEADS = ML4_CONFIG.local_heads * ML4_REFERENCE_TP


ML4_GLOBAL_INTER = ML4_CONFIG.inter * ML4_REFERENCE_TP


def ml4_tp_config(tp_size: int) -> LayerConfig:
    """Return the one ML4 shard geometry for `tp_size`."""
    if tp_size not in ML4_TP_SIZES:
        raise ValueError(
            f"ML4 tensor parallel size must be one of {ML4_TP_SIZES}, got {tp_size}"
        )
    return replace(
        ML4_CONFIG,
        local_heads=ML4_GLOBAL_HEADS // tp_size,
        inter=ML4_GLOBAL_INTER // tp_size,
    )


def ml4_attention_heads(tp_size: int, dcp_size: int = 1) -> int:
    if dcp_size not in (1, 4) or tp_size % dcp_size:
        raise ValueError(
            f"unsupported ML4 TP/DCP geometry: tp={tp_size}, dcp={dcp_size}"
        )
    return ML4_GLOBAL_HEADS // (tp_size // dcp_size)


def ml4_kernel_samples(samples: int, query_length: int) -> int:
    """Choose an LDS-safe request-aligned launch width."""
    if query_length not in ML4_QUERY_LENGTHS or samples % query_length:
        raise ValueError(
            f"unsupported ML4 decode shape samples={samples}, query_length={query_length}"
        )
    for chunk in reversed(ML4_KERNEL_SAMPLES):
        if chunk % query_length == 0 and samples % chunk == 0:
            return chunk
    raise ValueError(
        f"no ML4 kernel chunk for samples={samples}, query_length={query_length}"
    )


HIDDEN = ML4_CONFIG.hidden


Q_LORA = ML4_CONFIG.q_lora


KV_LORA = ML4_CONFIG.kv_lora


PE_DIM = ML4_CONFIG.pe_dim


NOPE_DIM = ML4_CONFIG.nope_dim


V_DIM = ML4_CONFIG.v_dim


QKV_A_ROWS = ML4_CONFIG.qkv_a_rows


N_EXPERTS = ML4_CONFIG.n_experts


EXPERT_TOP_K = ML4_CONFIG.top_k


TOP_K = EXPERT_TOP_K


MOE_SLOTS = ML4_CONFIG.moe_slots


SHARED_EXPERT = ML4_CONFIG.shared_expert


INTER = ML4_CONFIG.inter


ROUTE_SCALE = ML4_CONFIG.route_scale


SOFTMAX_SCALE = 128**-0.5  # ML4: qk_head_dim**-0.5 on every layer (NoPE layers keep it)
# query rows per head that carry RoPE (0 on NoPE layers: q_pe is zero-filled; the cache keeps its 64-wide, zero, rope key)
Q_PE_DIM = 0 if NOPE_LAYER else 64
INDEX_ROPE = not NOPE_LAYER


LOCAL_HEADS = ML4_CONFIG.local_heads


def validate_shard(
    samples: int,
    heads: int,
    rank: int,
    npes: int,
    sparse_attention_topk: int,
    model_config: LayerConfig | str = ML4_CONFIG,
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


# ML4 fp32 parameter table, passed through the kernel's ``g_q`` pointer (the q-LoRA norm ML4 does not have):
# indexer-q LayerNorm weight and bias, then per-head SSMax (s, b) for the local query heads and the 32 index heads.
ML4_P_IQ_W = 0
ML4_P_IQ_B = HIDDEN_ML4 = 6144
ML4_P_SS_S = 2 * 6144
ML4_P_SS_B = ML4_P_SS_S + 64
ML4_P_ISS_S = ML4_P_SS_B + 64
ML4_P_ISS_B = ML4_P_ISS_S + 64
ML4_P_WORDS = ML4_P_ISS_B + 64
