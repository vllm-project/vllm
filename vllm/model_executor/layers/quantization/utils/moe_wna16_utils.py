# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch


def repack_int4_to_int32(w: torch.Tensor) -> torch.Tensor:
    """Repack [E, N, K//2] uint8 → [E, K, N//8] int32.

    Input: K-packed uint8 (2 int4 per byte, low nibble first).
    Output: N-packed int32 (8 int4 per int32, GPTQ sequential shifts
            [0,4,...,28]).
    """
    E, N, K_half = w.shape
    K = K_half * 2
    lo = (w & 0xF).to(torch.int32)
    hi = ((w >> 4) & 0xF).to(torch.int32)
    unpacked = torch.stack([lo, hi], dim=-1).reshape(E, N, K)
    transposed = unpacked.permute(0, 2, 1).contiguous()
    N8 = N // 8
    shifts = torch.arange(8, device=w.device, dtype=torch.int32) * 4
    packed = (transposed.view(E, K, N8, 8) << shifts).sum(dim=-1, dtype=torch.int32)
    return packed.contiguous()


def unpack_zp_int4_to_fp16(zp: torch.Tensor) -> torch.Tensor:
    """Unpack [E, N//2, K_groups] uint8 → [E, K_groups, N] fp16."""
    E, N_half, K_groups = zp.shape
    lo = (zp & 0xF).to(torch.int32)
    hi = ((zp >> 4) & 0xF).to(torch.int32)
    unpacked = torch.stack([lo, hi], dim=2).reshape(E, N_half * 2, K_groups)
    return unpacked.permute(0, 2, 1).contiguous().to(torch.float16)
