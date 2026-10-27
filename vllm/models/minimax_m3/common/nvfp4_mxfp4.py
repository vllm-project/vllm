# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NVFP4→MXFP4 routed-expert layout used by the MiniMax-M3.1 AMD recipe.

The official MiniMax-M3.1 checkpoint stores routed experts as ModelOpt NVFP4
(``w1`` gate / ``w3`` up / ``w2`` down). On ROCm those experts are dequantized,
padded to the AITER MXFP4 tile, and requantized online to MXFP4
(``--quantization-config.moe mxfp4``). This module documents the layout so CPU
tests can lock scales, padding, packing, and tensor names without loading the
MoE kernels.
"""

from __future__ import annotations

NVFP4_GROUP_SIZE = 16
MXFP4_GROUP_SIZE = 32
FP4_PACK_FACTOR = 2
# Generic ROCm AITER MXFP4 (W4A16 / Triton) alignment. W4A4 uses 256 for hidden
# and 128 for intermediate; the AMD MiniMax recipe does not pin W4A4.
ROCM_AITER_MXFP4_ALIGN = 256

# Checkpoint expert names consumed by ``fused_moe_make_expert_params_mapping``.
MINIMAX_ROUTED_EXPERT_CKPT = {
    "gate": "w1",
    "down": "w2",
    "up": "w3",
}

# ModelOpt NVFP4 tensors on each routed expert. ``weight_scale_inv`` is remapped
# to ``weight_scale`` during MiniMax ``load_weights``.
MINIMAX_NVFP4_EXPERT_TENSORS = (
    "weight",
    "weight_scale",
    "weight_scale_2",
    "input_scale",
)


def round_up(value: int, alignment: int) -> int:
    """Smallest multiple of ``alignment`` that is >= ``value``."""
    if alignment <= 0:
        raise ValueError(f"alignment must be positive, got {alignment}")
    return (value + alignment - 1) // alignment * alignment


def nvfp4_packed_cols(cols: int) -> int:
    """NVFP4 packs two e2m1 values per byte along the reduction dimension."""
    if cols % FP4_PACK_FACTOR != 0:
        raise ValueError(f"NVFP4 cols must be even, got {cols}")
    return cols // FP4_PACK_FACTOR


def nvfp4_scale_cols(cols: int) -> int:
    """FP8 per-group scales: one scale per NVFP4 group of 16 elements."""
    if cols % NVFP4_GROUP_SIZE != 0:
        raise ValueError(
            f"NVFP4 cols must be divisible by {NVFP4_GROUP_SIZE}, got {cols}"
        )
    return cols // NVFP4_GROUP_SIZE


def mxfp4_packed_cols(cols: int) -> int:
    """MXFP4 packs two e2m1 values per byte along the reduction dimension."""
    if cols % FP4_PACK_FACTOR != 0:
        raise ValueError(f"MXFP4 cols must be even, got {cols}")
    return cols // FP4_PACK_FACTOR


def mxfp4_scale_cols(cols: int) -> int:
    """e8m0 per-block scales: one scale per MXFP4 group of 32 elements."""
    if cols % MXFP4_GROUP_SIZE != 0:
        raise ValueError(
            f"MXFP4 cols must be divisible by {MXFP4_GROUP_SIZE}, got {cols}"
        )
    return cols // MXFP4_GROUP_SIZE


def aiter_mxfp4_padded_dims(
    hidden_size: int,
    intermediate_size: int,
    *,
    alignment: int = ROCM_AITER_MXFP4_ALIGN,
) -> tuple[int, int]:
    """Pad hidden / intermediate to the AITER MXFP4 tile used on ROCm."""
    return round_up(hidden_size, alignment), round_up(intermediate_size, alignment)


def fused_w13_shape(
    num_experts: int,
    intermediate_size: int,
    hidden_size: int,
    *,
    num_shards: int = 2,
) -> tuple[int, int, int]:
    """Fused gate+up weight: ``[E, num_shards * I, H]`` (gate=0, up=1)."""
    return (num_experts, num_shards * intermediate_size, hidden_size)


def down_w2_shape(
    num_experts: int, hidden_size: int, intermediate_size: int
) -> tuple[int, int, int]:
    """Down-projection weight: ``[E, H, I]``."""
    return (num_experts, hidden_size, intermediate_size)


def pad_fused_expert_shapes(
    num_experts: int,
    hidden_size: int,
    intermediate_size: int,
    *,
    num_shards: int = 2,
    alignment: int = ROCM_AITER_MXFP4_ALIGN,
) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """Source and padded ``(w13, w2)`` shapes after AITER alignment."""
    padded_h, padded_i = aiter_mxfp4_padded_dims(
        hidden_size, intermediate_size, alignment=alignment
    )
    return (
        fused_w13_shape(num_experts, padded_i, padded_h, num_shards=num_shards),
        down_w2_shape(num_experts, padded_h, padded_i),
    )


def copy_padded_w13_w2(
    w13: list[list[list[float]]],
    w2: list[list[list[float]]],
    *,
    target_hidden: int,
    target_intermediate: int,
    num_shards: int = 2,
) -> tuple[list[list[list[float]]], list[list[list[float]]]]:
    """Zero-pad fused w13 / w2 like online MXFP4 requantization.

    Each gate/up shard is copied independently so NVFP4 w1/w3 scales stay
    associated with the matching half of w13 after padding.
    """
    num_experts = len(w13)
    src_intermediate = len(w13[0]) // num_shards
    src_hidden = len(w13[0][0])
    padded_w13 = [
        [[0.0] * target_hidden for _ in range(num_shards * target_intermediate)]
        for _ in range(num_experts)
    ]
    padded_w2 = [
        [[0.0] * target_intermediate for _ in range(target_hidden)]
        for _ in range(num_experts)
    ]
    for expert in range(num_experts):
        for shard in range(num_shards):
            src_start = shard * src_intermediate
            dst_start = shard * target_intermediate
            for row in range(src_intermediate):
                padded_w13[expert][dst_start + row][:src_hidden] = w13[expert][
                    src_start + row
                ]
        for row in range(src_hidden):
            padded_w2[expert][row][:src_intermediate] = w2[expert][row]
    return padded_w13, padded_w2


def split_gate_up_rows(w13_rows: int, *, num_shards: int = 2) -> tuple[range, range]:
    """Row ranges of fused w13 that belong to gate (w1) and up (w3)."""
    if w13_rows % num_shards != 0:
        raise ValueError(f"w13 rows {w13_rows} are not divisible by {num_shards}")
    intermediate = w13_rows // num_shards
    return range(0, intermediate), range(intermediate, 2 * intermediate)
