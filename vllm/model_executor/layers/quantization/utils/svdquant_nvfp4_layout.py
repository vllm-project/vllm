# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""nunchaku NVFP4 SVDQuant fragment-layout adapters.

Bridge between the canonical row-major SVDQuant NVFP4 on-disk format
and nunchaku's PTX-MMA-tile fragment layout. Bit-preserving pure
view+permute chain — no quant/dequant.

Used in two directions:

* Checkpoint conversion (vllm-omni converter): a nunchaku-published
  checkpoint is unpacked to canonical row-major for writing to disk.
* Load-time pack (vLLM `SVDQuantLinearMethod.process_weights_after_loading`):
  for the nunchaku kernel backend, repack the row-major on-disk tensors
  into fragment layout before the kernel sees them.

Verified against `svdq_gemm_w4a4_cuda(fp4=True)`: round-trip is
bit-exact, and half-swap via unpack→swap→pack reproduces the permuted
nunchaku output bit-exactly. Workbench source:
SVDQuant kernel `baseline/kernels/_nvfp4.py`.

Pair semantics:
  * `unpack_nunchaku_wscales_fp4(s_nun)`     `[K/16, N] fragment → row-major`
  * `pack_nunchaku_wscales_fp4(s_row)`       `[K/16, N] row-major → fragment`
  * `unpack_nunchaku_qweight_fp4(q_nun)`     `[N, K/2] fragment → row-major uint8 nibble bytes`
  * `pack_nunchaku_qweight_fp4(nibs_row)`    `[N, K] nibbles → [N, K/2] fragment int8`

These plus `nunchaku.lora.flux.nunchaku_converter.{pack,unpack}_lowrank_weight`
cover every fragment-layout param needed for SVDQuant W4A4 NVFP4
half-swap (qweight, wscales, proj_up).

Constants assume `NunchakuWeightPacker(bits=4, warp_n=128)`:
  wscales: s_pack_size=4, num_s_lanes=32, num_s_packs=1, insn_k/group=4
  qweight: num_n_packs=8, n_pack_size=2, num_n_lanes=8, reg_n=1,
           num_k_packs=1, k_pack_size=2, num_k_lanes=4, reg_k=8
"""
from __future__ import annotations

import torch

_WARP_N = 128
_INSN_K = 64
_GROUP = 16


def _pack_nibbles(nibs: torch.Tensor) -> torch.Tensor:
    """`[*, K] uint8 nibbles → [*, K/2] uint8`. Low nibble = even k."""
    assert nibs.shape[-1] % 2 == 0
    lo = nibs[..., 0::2]
    hi = nibs[..., 1::2]
    return (lo | (hi << 4)).to(torch.uint8)


def _unpack_nibbles(packed: torch.Tensor) -> torch.Tensor:
    """`[*, K/2] uint8 → [*, K] uint8 nibbles`. Inverse of `_pack_nibbles`."""
    lo = packed & 0x0F
    hi = (packed >> 4) & 0x0F
    out = torch.stack([lo, hi], dim=-1)
    return out.view(*packed.shape[:-1], packed.shape[-1] * 2)


def _wscale_view_shape(N: int, K: int) -> tuple[int, ...]:
    assert N % _WARP_N == 0, f"N ({N}) must be multiple of {_WARP_N}"
    assert K % _INSN_K == 0, f"K ({K}) must be multiple of {_INSN_K}"
    return (N // _WARP_N, 1, 4, 4, 8, K // _INSN_K, 4)


def pack_nunchaku_wscales_fp4(scales_row: torch.Tensor) -> torch.Tensor:
    """Row-major `[K/16, N]` fp8 → nunchaku fragment `[K/16, N]` fp8."""
    KG, N = scales_row.shape
    K = KG * _GROUP
    s = scales_row.transpose(0, 1).contiguous()
    s = s.view(*_wscale_view_shape(N, K))
    s = s.permute(0, 5, 1, 4, 3, 2, 6).contiguous()
    return s.view(-1, N)


def unpack_nunchaku_wscales_fp4(scales_nun: torch.Tensor) -> torch.Tensor:
    """nunchaku fragment `[K/16, N]` fp8 → row-major `[K/16, N]` fp8."""
    KG, N = scales_nun.shape
    K = KG * _GROUP
    s = scales_nun.view(N // _WARP_N, K // _INSN_K, 1, 8, 4, 4, 4)
    # Inverse of permute (0, 5, 1, 4, 3, 2, 6) is (0, 2, 5, 4, 3, 1, 6).
    s = s.permute(0, 2, 5, 4, 3, 1, 6).contiguous()
    s = s.view(N, K // _GROUP)
    return s.transpose(0, 1).contiguous()


def pack_nunchaku_qweight_fp4(nibs_row: torch.Tensor) -> torch.Tensor:
    """`[N, K] uint8 nibbles → [N, K/2] nunchaku fragment int8`."""
    N, K = nibs_row.shape
    assert N % _WARP_N == 0, f"N ({N}) must be multiple of {_WARP_N}"
    assert K % _INSN_K == 0, f"K ({K}) must be multiple of {_INSN_K}"
    n_tiles, k_tiles = N // _WARP_N, K // _INSN_K
    w = nibs_row.to(torch.int32)
    w = w.reshape(n_tiles, 8, 2, 8, 1, k_tiles, 1, 2, 4, 8)
    w = w.permute(0, 5, 6, 1, 3, 8, 2, 7, 4, 9).contiguous()
    w = w & 0xF
    shift = torch.arange(0, 32, 4, dtype=torch.int32, device=w.device)
    w = (w << shift).sum(dim=-1, dtype=torch.int32)
    return w.view(dtype=torch.int8).view(N, -1).contiguous()


def unpack_nunchaku_qweight_fp4(q_nun: torch.Tensor) -> torch.Tensor:
    """`[N, K/2] nunchaku fragment int8 → [N, K/2] uint8` (low nibble = even k)."""
    N, K2 = q_nun.shape
    K = K2 * 2
    assert N % _WARP_N == 0
    assert K % _INSN_K == 0
    n_tiles, k_tiles = N // _WARP_N, K // _INSN_K
    q_int = q_nun.contiguous().view(dtype=torch.int32)
    q_int = q_int.reshape(n_tiles, k_tiles, 1, 8, 8, 4, 2, 2, 1)
    shifts = torch.arange(0, 32, 4, dtype=torch.int32, device=q_int.device)
    nibs = ((q_int.unsqueeze(-1) >> shifts) & 0xF).to(torch.uint8)
    # Inverse of permute (0, 5, 6, 1, 3, 8, 2, 7, 4, 9) is (0, 3, 6, 4, 8, 1, 2, 7, 5, 9).
    nibs = nibs.permute(0, 3, 6, 4, 8, 1, 2, 7, 5, 9).contiguous()
    nibs = nibs.view(N, K)
    return _pack_nibbles(nibs)


__all__ = [
    "pack_nunchaku_qweight_fp4",
    "unpack_nunchaku_qweight_fp4",
    "pack_nunchaku_wscales_fp4",
    "unpack_nunchaku_wscales_fp4",
]
