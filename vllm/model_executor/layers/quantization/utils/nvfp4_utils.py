# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm._custom_ops import (
    cutlass_scaled_mm_supports_fp4,
)
from vllm.platforms import current_platform
from vllm.utils.math_utils import round_up


def swizzle_blockscale(scale: torch.Tensor) -> torch.Tensor:
    """
    Pad and block-interleave the FP4 block-scales so that they match the data
    layout expected by the CUTLASS / FlashInfer kernels.

    Parameters
    ----------
    scale: torch.Tensor

    Returns
    -------
    torch.Tensor
        The swizzled tensor with the same logical shape as *scale*.
    """
    assert scale.dtype == torch.float8_e4m3fn, (
        "swizzle_blockscale expects the input tensor to be in "
        "torch.float8_e4m3fn format."
    )

    scale_ndim = scale.ndim
    if scale_ndim == 2:
        scale = scale.unsqueeze(0)  # (1, M, K)
    assert scale.ndim == 3, "Expected a 2-D or 3-D tensor for block scales."

    B, M, K = scale.shape

    M_padded = round_up(M, 128)
    K_padded = round_up(K, 4)

    padded = torch.zeros(
        (B, M_padded, K_padded), dtype=scale.dtype, device=scale.device
    )
    padded[:B, :M, :K] = scale

    # Reshape / permute to the layout required by the kernel.
    padded = padded.reshape(B, M_padded // 128, 4, 32, K_padded // 4, 4)
    swizzled = padded.permute(0, 1, 4, 3, 2, 5).contiguous().cuda()

    if scale_ndim == 2:
        return swizzled.reshape(M_padded, K_padded)
    return swizzled.reshape(B, M_padded, K_padded)


def cutlass_fp4_supported() -> bool:
    if not current_platform.is_cuda():
        return False
    capability_tuple = current_platform.get_device_capability()
    capability = -1 if capability_tuple is None else capability_tuple.to_int()
    return cutlass_scaled_mm_supports_fp4(capability)


def pad_nvfp4_weight_for_cutlass(
    weight: torch.Tensor,
    alignment: int = 32,
) -> tuple[torch.Tensor, int]:
    """
    Pad packed NVFP4 weights so that both N (rows) and K (columns) satisfy
    the alignment constraints required by CUTLASS / FlashInfer FP4 kernels.

    CUTLASS FP4 kernel requires both K and N matrix dimensions to be divisible
    by 32 for aligned memory access and efficient tensor core operations.
    """
    weight_current_rows = weight.shape[0]

    # Pad N dimension (rows) if not aligned
    if weight_current_rows % alignment != 0:
        total_rows = round_up(weight_current_rows, alignment)
        pad_rows = total_rows - weight_current_rows
        weight = torch.nn.functional.pad(weight, (0, 0, 0, pad_rows)).contiguous()

    # Check K dimension alignment
    # 2 FP4 items are packed per byte in the input dimension
    weight_current_col_bytes = weight.shape[1]
    weight_current_col_elements = weight_current_col_bytes * 2

    weights_padding_bytes = 0
    if weight_current_col_elements % alignment != 0:
        total_cols = round_up(weight_current_col_elements, alignment)
        pad_cols = total_cols - weight_current_col_elements
        # Convert from FP4 element count to bytes (2 FP4 values per byte)
        # pad_cols is always even since alignment=32 and current elements are even
        pad_bytes = pad_cols // 2
        weight = torch.nn.functional.pad(weight, (0, pad_bytes, 0, 0)).contiguous()
        weights_padding_bytes = pad_bytes

    return weight, weights_padding_bytes


def pad_nvfp4_activation_for_cutlass(
    x_fp4: torch.Tensor,
    weights_padding_bytes: int,
) -> torch.Tensor:
    """
    Pad packed FP4 activations to match the K-dimension padding applied to weights.
    The padding is in bytes (tensor dimension), not FP4 elements.
    """
    if weights_padding_bytes > 0:
        return torch.nn.functional.pad(x_fp4, (0, weights_padding_bytes)).contiguous()
    return x_fp4


def slice_nvfp4_output(
    out: torch.Tensor,
    output_size: int,
) -> torch.Tensor:
    """
    Slice the output tensor to remove padding in N dimension if weight was padded.
    """
    if out.shape[-1] != output_size:
        return out[..., :output_size].contiguous()
    return out


def requantize_nvfp4_moe_w13_scale_2(
    w13_weight_scale: torch.Tensor,
    w13_weight_scale_2: torch.Tensor,
) -> torch.Tensor:
    """Rescale a fused w13's block scales onto one shared per-expert scale.

    A fused w13 keeps only one *global* scale per expert for both halves
    (gate and up), even when the checkpoint quantized them independently and
    the two differ (see https://github.com/vllm-project/vllm/issues/54974).
    Using one half's global scale for the other dequantizes it as
    ``fp4 * block_scale * gs_other`` instead of ``* gs_self``, scaling that
    half's contribution by ``gs_other / gs_self``.

    Mirrors the FP8 precedent for the same fused-w13 problem
    (``requantize_with_max_scale`` / ``process_fp8_weight_tensor_strategy_moe``):
    pick one shared scale, here ``max(gs_gate, gs_up)`` per expert, and
    rescale the other half's *block* scales -- not the fp4 payload -- so both
    halves dequantize correctly under that one global scale. The rescale
    factor is always <= 1, so the e4m3 block scales only shrink and cannot
    overflow.

    Parameters
    ----------
    w13_weight_scale:
        ``[num_experts, 2 * intermediate_size, hidden_size // group_size]``
        e4m3 block scales, gate half first. Rescaled in place.
    w13_weight_scale_2:
        ``[num_experts, 2]`` per-expert global scales, gate then up.

    Returns
    -------
    torch.Tensor
        The shared per-expert global scale, shape ``[num_experts]``.
    """
    assert w13_weight_scale_2.shape[1] == 2, (
        "expected one gate and one up global scale per expert"
    )
    num_experts, fused_dim, _ = w13_weight_scale.shape
    assert fused_dim % 2 == 0, "expected gate and up halves of equal size"
    half = fused_dim // 2

    gs_gate = w13_weight_scale_2[:, 0]
    gs_up = w13_weight_scale_2[:, 1]
    gs = torch.maximum(gs_gate, gs_up)

    for half_slice, gs_half in (
        (slice(0, half), gs_gate),
        (slice(half, fused_dim), gs_up),
    ):
        block_scale = w13_weight_scale[:, half_slice, :].to(torch.float32)
        factor = (gs_half / gs).view(num_experts, 1, 1)
        w13_weight_scale[:, half_slice, :] = (block_scale * factor).to(
            torch.float8_e4m3fn
        )

    return gs.contiguous()
