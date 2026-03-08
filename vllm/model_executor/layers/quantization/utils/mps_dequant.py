# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MPS (Metal) dequantization utilities for AWQ and GPTQ int4 models.

Uses the dequant_int4 Metal kernel package when available, with a pure
PyTorch fallback for environments where the kernel isn't installed.
"""

from typing import Any

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

_metal_dequant = None
_metal_import_attempted = False


def _get_metal_dequant():
    """Try to import Metal dequant kernel package (cached)."""
    global _metal_dequant, _metal_import_attempted
    if not _metal_import_attempted:
        _metal_import_attempted = True
        try:
            import dequant_int4

            _metal_dequant = dequant_int4
            logger.info("Using Metal dequant_int4 kernel for int4 dequantization")
        except ImportError:
            logger.info(
                "dequant_int4 Metal kernel not found, "
                "falling back to pure PyTorch dequantization"
            )
    return _metal_dequant


# ── AWQ ──

# AWQ interleaved bit shifts for extracting int4 values from packed uint32.
# Derived from: reverse_awq_order = [0,4,1,5,2,6,3,7]; shifts = order * 4
_AWQ_SHIFTS = torch.tensor([0, 16, 4, 20, 8, 24, 12, 28], dtype=torch.int32)


def _pytorch_dequant_awq(
    qweight: torch.Tensor,
    scales: torch.Tensor,
    qzeros: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    """Pure PyTorch AWQ dequantization — bitwise unpack + scale.

    Args:
        qweight: [in_features, out_features/8] packed int32
        scales: [num_groups, out_features] float16
        qzeros: [num_groups, out_features/8] packed int32
        group_size: quantization group size

    Returns:
        [in_features, out_features] float16 weight matrix
    """
    in_features = qweight.shape[0]
    out_features = scales.shape[1]

    shifts = _AWQ_SHIFTS.to(qweight.device)  # [8]

    # Unpack qweight: [in_features, out_features/8] -> [in_features, out_features]
    # Expand packed values and shift to extract each int4
    qw_expanded = qweight.unsqueeze(-1).expand(-1, -1, 8)  # [IC, OC/8, 8]
    weights = ((qw_expanded >> shifts) & 0xF).reshape(in_features, out_features)

    # Unpack qzeros: [num_groups, out_features/8] -> [num_groups, out_features]
    qz_expanded = qzeros.unsqueeze(-1).expand(-1, -1, 8)
    zeros = ((qz_expanded >> shifts) & 0xF).reshape(qzeros.shape[0], out_features)

    # Build group indices: [in_features] -> index into scales/zeros
    group_idx = torch.arange(in_features, device=qweight.device) // group_size

    # Dequantize: (weight - zero) * scale
    w_fp = weights.to(torch.float16) - zeros[group_idx].to(torch.float16)
    w_fp = w_fp * scales[group_idx]

    return w_fp


def awq_dequant_matmul(
    x: torch.Tensor,
    layer: Any,
    bias: torch.Tensor | None,
    quant_config: Any,
) -> torch.Tensor:
    """Dequantize AWQ weights and perform matmul on MPS.

    Uses Metal kernel if available, falls back to pure PyTorch.
    """
    metal = _get_metal_dequant()
    if metal is not None:
        w_fp16 = metal.dequantize_awq(
            layer.qweight,
            layer.scales,
            layer.qzeros,
            quant_config.group_size,
        )
    else:
        w_fp16 = _pytorch_dequant_awq(
            layer.qweight,
            layer.scales,
            layer.qzeros,
            quant_config.group_size,
        )

    pack_factor = quant_config.pack_factor
    out_shape = x.shape[:-1] + (layer.qweight.shape[-1] * pack_factor,)
    reshaped_x = x.reshape(-1, x.shape[-1])

    out = torch.matmul(reshaped_x, w_fp16)
    if bias is not None:
        out.add_(bias)
    return out.reshape(out_shape)


# ── GPTQ ──


def _pytorch_dequant_gptq(
    qweight: torch.Tensor,
    scales: torch.Tensor,
    qzeros: torch.Tensor,
    g_idx: torch.Tensor,
    group_size: int,
    use_v2_format: bool = False,
) -> torch.Tensor:
    """Pure PyTorch GPTQ dequantization — bitwise unpack + scale.

    Args:
        qweight: [in_features/8, out_features] packed int32
        scales: [num_groups, out_features] float16
        qzeros: [num_groups, out_features/8] packed int32
        g_idx: [in_features] int32 group index (empty if no desc_act)
        group_size: quantization group size
        use_v2_format: if True, use v2 zero-point convention (no offset).
            v1 (default): stored_zero = true_zero - 1, so add 1 back.

    Returns:
        [in_features, out_features] float16 weight matrix
    """
    out_features = qweight.shape[1]
    in_features = qweight.shape[0] * 8

    # Sequential shifts for GPTQ: nibble i at bits i*4
    shifts = torch.arange(0, 32, 4, dtype=torch.int32, device=qweight.device)  # [8]

    # Unpack qweight: [IC/8, OC] -> [IC, OC]
    # Each uint32 at [j, n] packs 8 input channels [8j..8j+7] for output n.
    # Expand shifts along dim 0, unpack, then transpose so nibbles
    # within each pack become consecutive rows before reshape.
    qw_expanded = qweight.unsqueeze(0).expand(8, -1, -1)  # [8, IC/8, OC]
    shifts_w = shifts.reshape(8, 1, 1)
    unpacked = (qw_expanded >> shifts_w) & 0xF  # [8, IC/8, OC]
    weights = unpacked.permute(1, 0, 2).reshape(in_features, out_features)

    # Unpack qzeros: [num_groups, OC/8] -> [num_groups, OC]
    zp_shifts = shifts.reshape(1, 1, 8)
    qz_expanded = qzeros.unsqueeze(-1).expand(-1, -1, 8)
    zeros = ((qz_expanded >> zp_shifts) & 0xF).reshape(qzeros.shape[0], out_features)

    # GPTQ v1 format: zeros are stored with -1 offset (stored = true - 1)
    if not use_v2_format:
        zeros = zeros + 1

    # Group indices
    has_g_idx = g_idx.numel() > 0
    if has_g_idx:
        group_idx = g_idx
    else:
        group_idx = torch.arange(in_features, device=qweight.device) // group_size

    # Dequantize: (weight - zero) * scale
    w_fp = weights.to(torch.float16) - zeros[group_idx].to(torch.float16)
    w_fp = w_fp * scales[group_idx]

    return w_fp


def gptq_dequant_matmul(
    x: torch.Tensor,
    layer: Any,
    bias: torch.Tensor | None,
    quant_config: Any,
    use_v2_format: bool = False,
) -> torch.Tensor:
    """Dequantize GPTQ weights and perform matmul on MPS.

    Uses Metal kernel if available, falls back to pure PyTorch.
    """
    metal = _get_metal_dequant()
    if metal is not None:
        # zero_adj=1 for v1 format (stored zeros offset by -1), 0 for v2
        zero_adj = 0 if use_v2_format else 1
        w_fp16 = metal.dequantize_gptq(
            layer.qweight,
            layer.scales,
            layer.qzeros,
            layer.g_idx,
            quant_config.group_size,
            zero_adj,
        )
    else:
        w_fp16 = _pytorch_dequant_gptq(
            layer.qweight,
            layer.scales,
            layer.qzeros,
            layer.g_idx,
            quant_config.group_size,
            use_v2_format,
        )

    out_shape = x.shape[:-1] + (layer.qweight.shape[-1],)
    reshaped_x = x.reshape(-1, x.shape[-1])

    out = torch.matmul(reshaped_x, w_fp16)
    if bias is not None:
        out.add_(bias)
    return out.reshape(out_shape)
