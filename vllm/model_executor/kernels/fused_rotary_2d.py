# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused 2D rotary position embedding kernel for vision encoders.

Generates cos/sin embeddings directly from integer (h, w) coordinates
using trigonometric functions, eliminating the need for table lookups.
Used by Qwen3-VL (and similar models) that apply separate rotary
embeddings for height and width spatial dimensions.

Data flow (PyTorch reference):
    inv_freq: (D_half,)  -- pre-computed frequency table
    pos_ids:  (N, 2)     -- (h_pos, w_pos) integer coordinates
    cos_out:  (N, D_rot) -- [cos(h*freq), cos(w*freq)] concatenated
    sin_out:  (N, D_rot) -- [sin(h*freq), sin(w*freq)] concatenated

where D_half = rotary_dim // 2, D_rot = rotary_dim = head_dim * partial_rotary_factor
"""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _fused_rotary_2d_kernel(
    # Pointers
    pos_ids_ptr,     # (N, 2) int tensor: [h_pos, w_pos]
    inv_freq_ptr,    # (D_half,) float tensor: pre-computed frequencies
    cos_out_ptr,     # (N, D_rot) output: cos embeddings
    sin_out_ptr,     # (N, D_rot) output: sin embeddings
    # Dimensions
    N: tl.constexpr,          # number of positions
    D_half: tl.constexpr,     # rotary_dim // 2
    D_rot: tl.constexpr,      # rotary_dim = 2 * D_half
    # Block size
    BLOCK_D: tl.constexpr,    # tile size for the D_half dimension
):
    """Compute 2D rotary embeddings from (h, w) integer coordinates.

    Each program handles one spatial position (one row of pos_ids).
    It reads the (h, w) pair and computes cos/sin for all D_rot dimensions.

    Output layout (matching PyTorch flatten(1)):
        cos_out[n, 0:D_half]        = cos(h * inv_freq[0:D_half])
        cos_out[n, D_half:D_rot]    = cos(w * inv_freq[0:D_half])
        sin_out[n, 0:D_half]        = sin(h * inv_freq[0:D_half])
        sin_out[n, D_half:D_rot]    = sin(w * inv_freq[0:D_half])
    """
    pid = tl.program_id(0)  # position index

    # Load the (h, w) coordinates for this position
    h_pos = tl.load(pos_ids_ptr + pid * 2).to(tl.float32)
    w_pos = tl.load(pos_ids_ptr + pid * 2 + 1).to(tl.float32)

    # Process D_half dimensions in tiles
    for d_start in tl.static_range(0, D_half, BLOCK_D):
        d_offsets = d_start + tl.arange(0, BLOCK_D)
        mask = d_offsets < D_half

        # Load inverse frequencies for this tile
        freq = tl.load(inv_freq_ptr + d_offsets, mask=mask, other=0.0)

        # Compute angles
        h_angle = h_pos * freq
        w_angle = w_pos * freq

        # Compute cos/sin
        h_cos = tl.cos(h_angle)
        h_sin = tl.sin(h_angle)
        w_cos = tl.cos(w_angle)
        w_sin = tl.sin(w_angle)

        # Store h-component in first half: [0, D_half)
        out_offset_h = pid * D_rot + d_offsets
        tl.store(cos_out_ptr + out_offset_h, h_cos, mask=mask)
        tl.store(sin_out_ptr + out_offset_h, h_sin, mask=mask)

        # Store w-component in second half: [D_half, D_rot)
        out_offset_w = pid * D_rot + D_half + d_offsets
        tl.store(cos_out_ptr + out_offset_w, w_cos, mask=mask)
        tl.store(sin_out_ptr + out_offset_w, w_sin, mask=mask)


def fused_rotary_2d(
    pos_ids: torch.Tensor,
    inv_freq: torch.Tensor,
    dtype: torch.dtype = torch.float16,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute 2D rotary position embeddings using a fused Triton kernel.

    Args:
        pos_ids: (N, 2) integer tensor of (h_pos, w_pos) coordinates.
        inv_freq: (D_half,) float tensor of inverse frequencies, computed as
            1.0 / (base ** (arange(0, rotary_dim, 2) / rotary_dim)).
        dtype: Output dtype (default: float16).

    Returns:
        cos_out: (N, D_rot) tensor where D_rot = 2 * D_half.
        sin_out: (N, D_rot) tensor.
    """
    assert pos_ids.ndim == 2 and pos_ids.shape[1] == 2, \
        f"pos_ids must be (N, 2), got {pos_ids.shape}"
    assert inv_freq.ndim == 1, \
        f"inv_freq must be 1-D, got {inv_freq.shape}"

    N = pos_ids.shape[0]
    D_half = inv_freq.shape[0]
    D_rot = 2 * D_half

    # Ensure inputs are contiguous and on the same device
    pos_ids = pos_ids.contiguous()
    inv_freq = inv_freq.contiguous().to(torch.float32)

    # Allocate outputs
    cos_out = torch.empty(N, D_rot, device=pos_ids.device, dtype=dtype)
    sin_out = torch.empty(N, D_rot, device=pos_ids.device, dtype=dtype)

    # Choose block size: next power of 2 of D_half, clamped to [16, 128]
    BLOCK_D = max(16, min(128, triton.next_power_of_2(D_half)))

    # Launch kernel: one program per position
    grid = (N,)
    _fused_rotary_2d_kernel[grid](
        pos_ids,
        inv_freq,
        cos_out,
        sin_out,
        N=N,
        D_half=D_half,
        D_rot=D_rot,
        BLOCK_D=BLOCK_D,
    )

    return cos_out, sin_out


def pytorch_rotary_2d_reference(
    pos_ids: torch.Tensor,
    inv_freq: torch.Tensor,
    dtype: torch.dtype = torch.float16,
) -> tuple[torch.Tensor, torch.Tensor]:
    """PyTorch reference implementation matching the table-lookup approach.

    This mimics what rot_pos_emb() does:
        cos_table[pos_ids].flatten(1), sin_table[pos_ids].flatten(1)

    but computes it from scratch using inv_freq directly.

    Args:
        pos_ids: (N, 2) integer tensor of (h_pos, w_pos) coordinates.
        inv_freq: (D_half,) float tensor of inverse frequencies.
        dtype: Output dtype.

    Returns:
        cos_out: (N, D_rot) tensor.
        sin_out: (N, D_rot) tensor.
    """
    # Convert to float for computation
    inv_freq_f = inv_freq.float()
    h_pos = pos_ids[:, 0].float()  # (N,)
    w_pos = pos_ids[:, 1].float()  # (N,)

    # Compute angles: (N,) x (D_half,) -> (N, D_half)
    h_angles = h_pos.unsqueeze(1) * inv_freq_f.unsqueeze(0)  # (N, D_half)
    w_angles = w_pos.unsqueeze(1) * inv_freq_f.unsqueeze(0)  # (N, D_half)

    # Concatenate h and w components: (N, 2*D_half) = (N, D_rot)
    cos_out = torch.cat([h_angles.cos(), w_angles.cos()], dim=1).to(dtype)
    sin_out = torch.cat([h_angles.sin(), w_angles.sin()], dim=1).to(dtype)

    return cos_out, sin_out
