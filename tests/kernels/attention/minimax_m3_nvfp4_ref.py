# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Torch reference of MiniMax-M3's NVFP4 K/V and index-K pages on ROCm.

A 128-token page is ``[K side | V side]``; a side holds every KV head's e2m1
data (8192 B per head), then every head's e4m3 block scales (1024 B per head).
Within a head: K data ``[token][64 B]``, K scales in the FlyDSL prefill's lane
order, V data ``[token octet][dim octet][8 dims][4 B]`` (a byte holds two
consecutive tokens of one dim), V scales ``[group][token]``. An index page is
``[128 x 64 B data][128 x 8 B scales]``.
"""

import torch

BLOCK = 128
D = 128
E2M1_GRID = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def quantize_nvfp4(x: torch.Tensor, global_scale: float):
    """E2M1 codes ``[T, 128]`` and e4m3 scale bytes ``[T, 8]`` of fp32 rows.

    The CUDA encoder's scale math and a nearest-grid E2M1 rounding with ties to
    the even code, written independently of the kernels' bit arithmetic.
    """
    groups = x.view(-1, 8, 16)
    one_sixth = torch.tensor(1 / 6, dtype=torch.float32)
    # fp32 throughout, like the kernels: 1 / 0.002 is not 500 in fp32.
    gs = torch.tensor(global_scale, dtype=torch.float32, device=x.device)
    scale = (groups.abs().amax(-1) * one_sixth * (1.0 / gs)).clamp(max=448)
    scale_e4m3 = scale.to(torch.float8_e4m3fn)
    scale_value = scale_e4m3.float()
    quant = torch.where(scale_value != 0, 1.0 / (scale_value * gs), 0.0)
    y = groups * quant[..., None]
    grid = torch.tensor(E2M1_GRID, device=x.device)
    dist = (y.abs()[..., None] - grid).abs()
    is_min = dist == dist.amin(-1, keepdim=True)
    ids = torch.arange(8, device=x.device)
    # Neighbouring codes alternate parity, so every tie has an even candidate.
    even_min = torch.where(is_min & (ids % 2 == 0), ids, 8).amin(-1)
    mag = torch.where(even_min < 8, even_min, dist.argmin(-1))
    codes = mag | (torch.signbit(y).long() << 3)
    return codes.view(-1, D), scale_e4m3.view(torch.uint8).view(-1, 8)


def dequant(data: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Token-major NVFP4 rows -> fp32 e2m1 x block scale, ``[..., 128]``."""
    codes = torch.stack((data & 15, data >> 4), -1).flatten(-2).long()
    grid = torch.tensor(E2M1_GRID, device=data.device)
    mag = grid[codes & 7] * torch.where(codes >= 8, -1.0, 1.0)
    s = scales.view(torch.float8_e4m3fn).float()
    return (mag.unflatten(-1, (8, 16)) * s[..., None]).flatten(-2)


def head_offsets(side: int, device):
    """Byte offsets inside one head's regions of side 0 (K) or 1 (V).

    Returns the data byte and nibble shift of each (token, dim) ``[128, 128]``
    and the scale byte of each (token, group) ``[128, 8]``.
    """
    t = torch.arange(BLOCK, device=device)[:, None]
    d = torch.arange(D, device=device)[None, :]
    g = torch.arange(8, device=device)[None, :]
    if side == 0:
        r = t % 32
        n = 4 * (r // 8) + r % 4
        data = t * 64 + d // 2
        shift = (d % 2) * 4 + 0 * t
        scale = ((((t // 32) * 4 + g // 2) * 16 + n) * 2 + (r % 8) // 4) * 2 + g % 2
    else:
        data = ((t // 8 * 16 + d // 8) * 8 + d % 8) * 4 + (t % 8) // 2
        shift = (t % 2) * 4 + 0 * d
        scale = g * BLOCK + t
    return data, shift, scale


def _region_bases(side: int, heads: int, device):
    """``[H, 1, 1]`` page offsets of each head's data and scale regions."""
    h = torch.arange(heads, device=device)[:, None, None]
    side0 = side * heads * BLOCK * 72
    return side0 + h * BLOCK * 64, side0 + heads * BLOCK * 64 + h * BLOCK * 8


def _flat(cache: torch.Tensor) -> torch.Tensor:
    """Writable 1-D view of a cache whose block stride may exceed one page."""
    n = (cache.shape[0] - 1) * cache.stride(0) + cache[0].numel()
    return cache.as_strided((n,), (1,))


def _insert_side(cache, rows, slots, side, global_scale):
    tokens, heads, _ = rows.shape
    codes, scales = quantize_nvfp4(rows.reshape(-1, D).float(), global_scale)
    slots = slots.long()
    keep = slots >= 0
    codes = codes.view(tokens, heads, D)[keep].to(torch.uint8)
    scales = scales.view(tokens, heads, 8)[keep]
    blk, pos = slots[keep] // BLOCK, slots[keep] % BLOCK
    data_off, shift, scale_off = head_offsets(side, rows.device)
    data_base, scale_base = _region_bases(side, heads, rows.device)
    page = (blk * cache.stride(0))[:, None, None]
    byte = page + data_base[:, 0] + data_off[pos][:, None]  # [n, H, 128]
    sh = shift[pos][:, None].expand_as(byte)
    flat = _flat(cache)
    for s in (0, 4):  # each byte takes one nibble per pass
        m = sh == s
        flat[byte[m]] = (flat[byte[m]] & (0xF0 >> s)) | (codes[m] << s)
    flat[page + scale_base[:, 0] + scale_off[pos][:, None]] = scales


def insert_nvfp4_kv(k, v, kv_cache, slots, k_scale, v_scale):
    """Quantize K / V ``[T, Hkv, 128]`` into their pages; slots < 0 skip."""
    _insert_side(kv_cache, k, slots, 0, float(k_scale))
    _insert_side(kv_cache, v, slots, 1, float(v_scale))


def insert_nvfp4_index(index_cache, keys, slots):
    """Quantize index keys ``[T, 128]`` into their pages (no global scale)."""
    codes, scales = quantize_nvfp4(keys.float(), 1.0)
    slots = slots.long()
    keep = slots >= 0
    codes, scales = codes[keep].to(torch.uint8), scales[keep]
    blk, pos = slots[keep] // BLOCK, slots[keep] % BLOCK
    base = (blk * index_cache.stride(0))[:, None]
    flat = _flat(index_cache)
    flat[base + pos[:, None] * 64 + torch.arange(64, device=keys.device)] = codes[
        :, 0::2
    ] | (codes[:, 1::2] << 4)
    flat[base + BLOCK * 64 + pos[:, None] * 8 + torch.arange(8, device=keys.device)] = (
        scales
    )


def nvfp4_kv_cache_token_major(kv_cache: torch.Tensor):
    """``(k_data, k_scale, v_data, v_scale)`` of a main cache, token-major.

    Data ``[blocks, Hkv, 128, 64]`` (the low nibble the even dim), scales
    ``[blocks, Hkv, 128, 8]``, all uint8.
    """
    blocks, heads = kv_cache.shape[0], kv_cache.shape[1] // 2
    pages = kv_cache.reshape(blocks, -1)
    out = []
    for side in (0, 1):
        data_off, shift, scale_off = head_offsets(side, kv_cache.device)
        data_base, scale_base = _region_bases(side, heads, kv_cache.device)
        codes = (pages[:, data_base + data_off] >> shift.to(torch.uint8)) & 0xF
        out.append(codes[..., 0::2] | (codes[..., 1::2] << 4))
        out.append(pages[:, scale_base + scale_off])
    return tuple(out)
