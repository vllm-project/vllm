# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from ROCm/aiter#6173 at b02df0db8 (Apache-2.0 License),
# Copyright (c) 2025 FlyDSL Project Contributors:
# aiter/ops/flydsl/kernels/glm5_mono/layout.py

"""Shared compile-time layouts and schedules for fused model-layer kernels."""

from __future__ import annotations

from vllm.models.deepseek_v32.amd.mono.config import (
    GLM5_CONFIG,
    MAX_LAYERS_PER_STEP,
    LayerConfig,
    MoeMode,
    as_layer_config,
    moe_format,
)

BLOCKS = 256
LAYER_SLOTS = MAX_LAYERS_PER_STEP
THREADS = 512
WAVES = THREADS // 64
QKV_A_TILE = 16
Q_B_TILE = 16
UK_TILE = 128
UV_TILE = 64
ROW_TILE = 32
ROUTER_TILE = 8
UG_TILE = 16
SPLIT_KEYS = 64
NEG = -1.0e30

CM_DEV = 16
CM_SYS = 17
POLL_MAX = 12
TL_COLS = 8


def atom_mxfp4_weight_index(row_group, k_chunk, lane, step, k_size):
    row = row_group * 16 + lane % 16
    return (
        (row_group * (k_size // 64) + k_chunk * 2 + step // 2) * 128
        + (step % 2) * 64
        + (row % 16) * 4
        + lane // 16
    )


def atom_mxfp4_scale_index(row, col, cols):
    r32, a, b = row // 32, (row // 16) % 2, row % 16
    c8, d, e = col // 8, (col // 4) % 2, col % 4
    return ((((r32 * ((cols + 7) // 8) + c8) * 4 + e) * 16 + b) * 2 + d) * 2 + a


def symmetric_allreduce_nbytes(sizes: tuple[int, ...], npes: int) -> int:
    """Return bytes for two tagged-mailbox epoch slots per reduce region."""
    if not sizes or any(size <= 0 or size % 2 for size in sizes):
        raise ValueError("all BF16 reduce sizes must be positive and even")
    if npes not in {2, 4, 8}:
        raise ValueError(f"npes must be one of {{2, 4, 8}}, got {npes}")
    max_pairs = max(sizes) // 2
    slot_bytes = npes * max_pairs * 8
    return len(sizes) * 2 * slot_bytes


def dn_tile(samples: int, hidden: int = GLM5_CONFIG.hidden) -> int:
    """Return hidden rows per expert-down / FFN peer-reduce task."""
    return 32 if samples == 1 else hidden // BLOCKS


def _align(n: int, alignment: int = 256) -> int:
    return (n + alignment - 1) // alignment * alignment


def layout(
    samples: int,
    heads: int,
    npes: int,
    sparse_attention_topk: int,
    moe_mode: MoeMode | str = MoeMode.W8A8,
    model_config: LayerConfig | str = GLM5_CONFIG,
    attention_only: bool = False,
    dedicated_input_norm: bool = False,
):
    """Return scratch and double-buffered symmetric-region byte offsets."""
    config = as_layer_config(model_config)
    fmt = moe_format(moe_mode)
    quant_group = fmt.activation_group
    xq_blocks = 0 if quant_group is None else config.hidden // quant_group
    n_split = sparse_attention_topk // SPLIT_KEYS
    pair_bytes = 8
    items = [
        (
            "input_norm",
            samples * config.hidden * pair_bytes if dedicated_input_norm else 0,
        ),
        ("q_a", samples * config.q_lora * pair_bytes),
        ("kv_a", samples * (config.kv_lora + config.pe_dim) * pair_bytes),
        (
            "gate",
            (
                samples * heads * config.v_dim * pair_bytes
                if config.attention_output_gate
                else 0
            ),
        ),
        ("kvnew", samples * config.kv_lora * pair_bytes),
        ("penew", samples * config.pe_dim * pair_bytes),
        ("q_nope", samples * heads * config.nope_dim * pair_bytes),
        ("q_pe", samples * heads * config.pe_dim * pair_bytes),
        ("q_lat", samples * heads * config.kv_lora * pair_bytes),
        ("sp_acc", samples * n_split * heads * config.kv_lora * pair_bytes),
        ("sp_m", samples * n_split * heads * pair_bytes),
        ("sp_l", samples * n_split * heads * pair_bytes),
        ("o", samples * heads * config.v_dim * pair_bytes),
        ("a", samples * config.hidden * pair_bytes),
    ]
    if not attention_only:
        items += [
            ("scores", samples * config.n_experts * pair_bytes),
            (
                "xq",
                samples
                * config.hidden
                // (4 if quant_group is not None else 2)
                * pair_bytes,
            ),
            ("xqs", samples * xq_blocks * pair_bytes),
            ("sel", samples * config.moe_slots * pair_bytes),
            ("prob", samples * config.moe_slots * pair_bytes),
            ("mid", samples * config.moe_slots * config.inter * pair_bytes),
            ("ugp", BLOCKS * samples * 2 * UG_TILE * pair_bytes),
            ("xqd", samples * config.hidden * 4),
        ]
    offset, scratch = 0, {}
    for name, size in items:
        scratch[name] = offset
        offset += _align(size)
    scratch["_bytes"] = offset

    part = npes * samples * config.hidden * pair_bytes
    region = 2 * part
    symmetric = {
        "attn": 0,
        "ffn": region,
        "_part_stride": part,
        "_bytes": region if attention_only else 2 * region,
    }
    return scratch, symmetric


def stage_tasks(
    samples: int,
    heads: int,
    sparse_attention_topk: int,
    model_config: LayerConfig | str = GLM5_CONFIG,
    attention_only: bool = False,
    dedicated_input_norm: bool = False,
):
    """Return ``(stage name, task count)`` pairs in execution order."""
    config = as_layer_config(model_config)
    head_groups = (heads + WAVES - 1) // WAVES
    split_ctas_per_tile = head_groups if samples == 1 else 1
    tasks = []
    if dedicated_input_norm:
        tasks.append(("input_norm", samples))
    tasks += [
        ("qkv_a", config.qkv_a_rows // QKV_A_TILE),
        ("cache", 1),
        ("q_b", heads * (config.nope_dim + config.pe_dim) // Q_B_TILE),
        ("uk", heads * config.kv_lora // UK_TILE),
        (
            "split",
            samples * (sparse_attention_topk // SPLIT_KEYS) * split_ctas_per_tile,
        ),
        ("uv", samples * (heads * config.v_dim // UV_TILE)),
        ("o", config.hidden // ROW_TILE),
    ]
    if attention_only:
        return tasks
    tasks += [
        ("router", samples * (config.n_experts // ROUTER_TILE)),
        ("ug", BLOCKS if samples == 1 else samples * BLOCKS),
        ("down", config.hidden // dn_tile(samples, config.hidden)),
    ]
    return tasks
