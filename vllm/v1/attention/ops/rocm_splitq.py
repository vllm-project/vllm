# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SplitQ KV-cache format: layout, rotations and a PyTorch reference.

Each (token, KV head) is stored in one byte slot. The head is split at the
rotary boundary:

* RoPE dims ``[0, R)`` of K: symmetric int8 with a per-token-head scale.
* NoPE dims ``[R, D)`` of K and all ``D`` dims of V: random sign flips plus
  a block Walsh-Hadamard transform, then a uniform midrise quantizer with
  ``2**bits`` levels. The rotated coordinates are close to Gaussian, so the
  step is fixed relative to the vector RMS and refined per vector by least
  squares. Everything is computed on the fly; there is no calibration.

Scores are computed in the rotated space (the query NoPE part is rotated the
same way) and the attention output is rotated back once per query head.

Slot layout (bytes)::

    [k_rope int8 R][k_nope codes][v codes][fp16 scales][pad to 8]

4-bit codes: each little-endian uint32 covers 8 dims; byte j holds dim j in
its low nibble and dim j + 4 in its high nibble, so ``w & 0x0F0F0F0F`` and
``(w >> 4) & 0x0F0F0F0F`` give dims 0..3 and 4..7 as bytes.

3-bit codes: a 2-bit plane (uint32 per 16 dims; ``(w >> 2i) & 0x03030303``
gives dims 4i..4i+3) followed by a 1-bit plane (uint32 per 32 dims;
``(w >> i) & 0x01010101`` gives dims 4i..4i+3). code = lo + 4 * hi.

Scales (fp16): k_rope, one per 64-dim NoPE block, v.
"""

import functools
import math
from dataclasses import dataclass

import torch

HADAMARD_BLOCK = 64
_SIGN_SEED = 0x5E17
# MSE-optimal midrise step for N(0, 1) inputs.
_GAUSSIAN_STEP = {3: 0.5865, 4: 0.3355}

_rope_dim_by_head_size: dict[int, int] = {}


def register_rope_dim(head_size: int, rope_dim: int) -> None:
    prev = _rope_dim_by_head_size.setdefault(head_size, rope_dim)
    if prev != rope_dim:
        raise ValueError(
            f"SplitQ: conflicting rotary dims {prev} and {rope_dim} for "
            f"head_size={head_size}"
        )


def rope_dim_from_hf_config(hf_text_config, head_size: int) -> int:
    factor = getattr(hf_text_config, "partial_rotary_factor", None)
    if factor is None:
        rope_params = getattr(hf_text_config, "rope_parameters", None) or {}
        factor = rope_params.get("partial_rotary_factor", 1.0)
    return int(head_size * factor)


def registered_rope_dim(head_size: int) -> int:
    if head_size not in _rope_dim_by_head_size:
        from vllm.config import get_current_vllm_config_or_none

        vllm_config = get_current_vllm_config_or_none()
        if vllm_config is None or vllm_config.model_config is None:
            raise RuntimeError(
                "SplitQ needs the model's rotary dim before sizing the KV cache"
            )
        register_rope_dim(
            head_size,
            rope_dim_from_hf_config(
                vllm_config.model_config.hf_text_config, head_size
            ),
        )
    return _rope_dim_by_head_size[head_size]


@dataclass(frozen=True)
class SplitQFormat:
    head_size: int
    rope_dim: int
    bits: int

    def __post_init__(self):
        d, r = self.head_size, self.rope_dim
        if self.bits not in (3, 4):
            raise ValueError(f"SplitQ supports 3 or 4 bits, got {self.bits}")
        if r % 16 or (d - r) % HADAMARD_BLOCK or d % HADAMARD_BLOCK:
            raise ValueError(
                f"SplitQ needs rope_dim % 16 == 0 and the NoPE part and "
                f"head_size in multiples of {HADAMARD_BLOCK}; got "
                f"head_size={d}, rope_dim={r}"
            )

    @classmethod
    def from_cache_dtype(cls, cache_dtype: str, head_size: int, rope_dim: int):
        bits = {"splitq_k4v4": 4, "splitq_k3v3": 3}[cache_dtype]
        return cls(head_size, rope_dim, bits)

    @property
    def nope_dim(self) -> int:
        return self.head_size - self.rope_dim

    @property
    def levels(self) -> int:
        return 1 << self.bits

    @property
    def num_nope_blocks(self) -> int:
        return self.nope_dim // HADAMARD_BLOCK

    @property
    def off_k_nope(self) -> int:
        return self.rope_dim

    @property
    def off_v(self) -> int:
        return self.off_k_nope + self.nope_dim * self.bits // 8

    @property
    def off_scales(self) -> int:
        return self.off_v + self.head_size * self.bits // 8

    @property
    def num_scales(self) -> int:
        return 2 + self.num_nope_blocks

    @property
    def slot_bytes(self) -> int:
        raw = self.off_scales + 2 * self.num_scales
        return (raw + 7) // 8 * 8

    @property
    def step(self) -> float:
        return _GAUSSIAN_STEP[self.bits]


@functools.cache
def _signs_cpu(n: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(_SIGN_SEED)
    return torch.randint(0, 2, (n,), generator=g, dtype=torch.int64) * -2 + 1


@functools.cache
def signs(n: int, device_str: str) -> torch.Tensor:
    """Fixed ±1 pattern applied before the Hadamard transform."""
    return _signs_cpu(n).to(device=device_str, dtype=torch.float32)


def sign_bits(n: int) -> torch.Tensor:
    """The sign pattern as int32 words, bit i set where the sign is -1."""
    neg = (_signs_cpu(n) < 0).to(torch.int64).view(-1, 32)
    words = (neg << torch.arange(32)).sum(-1)
    return torch.where(words >= 2**31, words - 2**32, words).to(torch.int32)


def fwht_blocks(x: torch.Tensor, block: int = HADAMARD_BLOCK) -> torch.Tensor:
    """Orthonormal Walsh-Hadamard transform over consecutive blocks of the
    last dim."""
    shape = x.shape
    y = x.reshape(-1, shape[-1] // block, block).float()
    h = 1
    while h < block:
        y = y.view(*y.shape[:-1], block // (2 * h), 2, h)
        a, b = y[..., 0, :], y[..., 1, :]
        y = torch.stack((a + b, a - b), dim=-2).view(*y.shape[:-3], block)
        h *= 2
    return (y * (1.0 / math.sqrt(block))).view(shape)


def rotate(
    x: torch.Tensor, sgn: torch.Tensor, block: int = HADAMARD_BLOCK
) -> torch.Tensor:
    return fwht_blocks(x.float() * sgn, block)


def unrotate(
    y: torch.Tensor, sgn: torch.Tensor, block: int = HADAMARD_BLOCK
) -> torch.Tensor:
    return fwht_blocks(y.float(), block) * sgn


def v_block(head_size: int) -> int:
    """V is rotated as one block when head_size is a power of two."""
    if head_size & (head_size - 1) == 0:
        return head_size
    return HADAMARD_BLOCK


def _quantize_uniform(x: torch.Tensor, fmt: SplitQFormat):
    """Midrise quantizer with a Gaussian step and a least-squares scale.

    Returns (codes uint8 in [0, levels), scale) with x ≈ (codes - c0) * scale.
    """
    levels = fmt.levels
    c0 = (levels - 1) / 2
    rms = x.pow(2).mean(-1, keepdim=True).sqrt().clamp_min(1e-12)
    step = rms * fmt.step
    codes = torch.floor(x / step + levels / 2).clamp(0, levels - 1)
    centered = codes - c0
    denom = centered.pow(2).sum(-1, keepdim=True).clamp_min(1e-12)
    scale = (x * centered).sum(-1, keepdim=True) / denom
    return codes.to(torch.uint8), scale


def _pack(codes: torch.Tensor, bits: int) -> torch.Tensor:
    """codes: (..., n) uint8 -> (..., n * bits / 8) uint8, layout above."""
    c = codes.to(torch.int64)
    if bits == 4:
        g = c.view(*c.shape[:-1], -1, 2, 4)  # (word, half, byte)
        b = g[..., 0, :] | (g[..., 1, :] << 4)
        return b.reshape(*c.shape[:-1], -1).to(torch.uint8)
    lo = (c & 3).view(*c.shape[:-1], -1, 4, 4)  # (word, i, byte)
    lo_b = (lo << (2 * torch.arange(4, device=c.device)).view(4, 1)).sum(-2)
    hi = (c >> 2).view(*c.shape[:-1], -1, 8, 4)
    hi_b = (hi << torch.arange(8, device=c.device).view(8, 1)).sum(-2)
    return torch.cat(
        (lo_b.reshape(*c.shape[:-1], -1), hi_b.reshape(*c.shape[:-1], -1)), -1
    ).to(torch.uint8)


def _unpack(packed: torch.Tensor, n: int, bits: int) -> torch.Tensor:
    p = packed.to(torch.int64)
    if bits == 4:
        g = p.view(*p.shape[:-1], -1, 4)
        c = torch.stack((g & 15, g >> 4), dim=-2)
        return c.reshape(*p.shape[:-1], n)
    lo_bytes = n // 4
    lo = p[..., :lo_bytes].view(*p.shape[:-1], -1, 1, 4)
    lo = (lo >> (2 * torch.arange(4, device=p.device)).view(4, 1)) & 3
    hi = p[..., lo_bytes:].view(*p.shape[:-1], -1, 1, 4)
    hi = (hi >> torch.arange(8, device=p.device).view(8, 1)) & 1
    return lo.reshape(*p.shape[:-1], n) + 4 * hi.reshape(*p.shape[:-1], n)


def reference_quantize(
    key: torch.Tensor, value: torch.Tensor, fmt: SplitQFormat
) -> torch.Tensor:
    """(T, H, D) K and V -> (T, H, slot_bytes) uint8 slots."""
    t, h, d = key.shape
    r, dev = fmt.rope_dim, key.device
    out = torch.zeros(t, h, fmt.slot_bytes, dtype=torch.uint8, device=dev)
    scales = torch.empty(t, h, fmt.num_scales, dtype=torch.float32, device=dev)

    k = key.float()
    k_rope = k[..., :r]
    kr_scale = k_rope.abs().amax(-1, keepdim=True).clamp_min(1e-12) / 127.0
    kr_q = torch.round(k_rope / kr_scale).clamp(-127, 127).to(torch.int8)
    out[..., :r] = kr_q.view(torch.uint8)
    scales[..., 0] = kr_scale[..., 0]

    kn = rotate(k[..., r:], signs(fmt.nope_dim, str(dev)))
    kn = kn.view(t, h, fmt.num_nope_blocks, HADAMARD_BLOCK)
    kn_codes, kn_scale = _quantize_uniform(kn, fmt)
    out[..., fmt.off_k_nope : fmt.off_v] = _pack(
        kn_codes.view(t, h, fmt.nope_dim), fmt.bits
    )
    scales[..., 1 : 1 + fmt.num_nope_blocks] = kn_scale[..., 0]

    vr = rotate(value, signs(d, str(dev)), v_block(d))
    v_codes, v_scale = _quantize_uniform(vr, fmt)
    out[..., fmt.off_v : fmt.off_scales] = _pack(v_codes, fmt.bits)
    scales[..., -1] = v_scale[..., 0]

    sc = scales.to(torch.float16).view(torch.uint8).view(t, h, -1)
    out[..., fmt.off_scales : fmt.off_scales + sc.shape[-1]] = sc
    return out


def reference_dequantize(
    slots: torch.Tensor, fmt: SplitQFormat, rotated: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    """(T, H, slot) slots -> fp32 K and V of shape (T, H, D).

    With ``rotated=True`` the NoPE part of K and all of V stay in the
    rotated space, which is what the kernels consume.
    """
    t, h, _ = slots.shape
    r, d, dev = fmt.rope_dim, fmt.head_size, slots.device
    c0 = (fmt.levels - 1) / 2
    sc = (
        slots[..., fmt.off_scales : fmt.off_scales + 2 * fmt.num_scales]
        .contiguous()
        .view(torch.float16)
        .float()
    )
    k_rope = slots[..., :r].contiguous().view(torch.int8).float() * sc[..., :1]

    kn = _unpack(slots[..., fmt.off_k_nope : fmt.off_v], fmt.nope_dim, fmt.bits)
    kn = (kn.float() - c0).view(t, h, fmt.num_nope_blocks, HADAMARD_BLOCK)
    kn = (kn * sc[..., 1 : 1 + fmt.num_nope_blocks, None]).view(t, h, -1)

    v = _unpack(slots[..., fmt.off_v : fmt.off_scales], d, fmt.bits)
    v = (v.float() - c0) * sc[..., -1:]
    if not rotated:
        kn = unrotate(kn, signs(fmt.nope_dim, str(dev)))
        v = unrotate(v, signs(d, str(dev)), v_block(d))
    return torch.cat((k_rope, kn), -1), v


def reference_attention(
    query: torch.Tensor,
    cache: torch.Tensor,
    block_table: torch.Tensor,
    q_to_req: torch.Tensor,
    q_to_klen: torch.Tensor,
    fmt: SplitQFormat,
    scale: float,
) -> torch.Tensor:
    """Per-query-token attention over a SplitQ paged cache.

    query: (Q, Hq, D). cache: (num_blocks, Hkv, block_size, slot) uint8.
    Query token i attends to the first ``q_to_klen[i]`` tokens of request
    ``q_to_req[i]``.
    """
    num_q, hq, d = query.shape
    _, hkv, block_size, _ = cache.shape
    group = hq // hkv
    out = torch.zeros(num_q, hq, d, dtype=torch.float32, device=query.device)
    for i in range(num_q):
        klen = int(q_to_klen[i])
        if klen <= 0:
            continue
        req = int(q_to_req[i])
        pos = torch.arange(klen, device=cache.device)
        blocks = block_table[req, pos // block_size].long()
        slots = cache[blocks, :, pos % block_size]  # (klen, Hkv, slot)
        k, v = reference_dequantize(slots, fmt)
        k = k.repeat_interleave(group, dim=1)
        v = v.repeat_interleave(group, dim=1)
        s = torch.einsum("hd,thd->ht", query[i].float(), k) * scale
        p = torch.softmax(s, dim=-1)
        out[i] = torch.einsum("ht,thd->hd", p, v)
    return out
