# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SplitQ KV-cache format: layout, rotations and a PyTorch reference.

Each (token, KV head) is stored in one byte slot. V is always quantized
whole; K has two layouts:

* Block K (``splitq_k3v4``, ``splitq_k3v3``): random sign flips plus a
  Walsh-Hadamard transform over each 64-dim block. The RoPE blocks get 4-bit
  codes, the NoPE blocks 3-bit codes, and every block has its own scale.
* Compact K (``splitq_k3v3_compact``): sign flips plus one Hadamard transform
  over all dims, 3-bit codes, one scale. Spreading the RoPE dims over the
  whole head lets them share the 3-bit budget.
* V: sign flips plus one Hadamard transform over all dims, ``v_bits`` codes.

The rotated coordinates are close to Gaussian, so each code indexes a
Lloyd-Max codebook for N(0, 1), stored as int8 (``LUT``): a code ``c`` of a
block with scale ``s`` stands for ``LUT[c] * s / LUT_ONE``. The scale makes
``x . x_hat = |x|^2``: a least-squares fit would shrink every reconstructed
vector, which flattens the softmax and scales down the attention output.
Everything is computed on the fly; there is no calibration.

Scores are computed in the rotated space (the query is rotated like K) and
the attention output is rotated back once per query head.

Slot layout (bytes)::

    [k_rope codes][k_nope codes][v codes][fp16 scales][pad to 8]

Compact K has no 4-bit part and pads to 4 bytes.

4-bit codes: each little-endian uint32 covers 8 dims; byte j holds dim j in
its low nibble and dim j + 4 in its high nibble, so ``w & 0x0F0F0F0F`` and
``(w >> 4) & 0x0F0F0F0F`` give dims 0..3 and 4..7 as bytes.

3-bit codes: a 2-bit plane (uint32 per 16 dims; ``(w >> 2i) & 0x03030303``
gives dims 4i..4i+3) followed by a 1-bit plane (uint32 per 32 dims;
``(w >> i) & 0x01010101`` gives dims 4i..4i+3). code = lo + 4 * hi.

Scales (fp16): one per 64-dim K block (one in total for compact K), then V.
"""

import functools
import math
from dataclasses import dataclass

import torch

HADAMARD_BLOCK = 64
_SIGN_SEED = 0x5E17
ROPE_BITS = 4
NOPE_BITS = 3
# Lloyd-Max codebooks for N(0, 1) as int8: LUT[bits][c] / LUT_MULT[bits]
# approximates centroid c (the kernels map codes through the same tables).
LUT = {
    3: (-127, -79, -45, -14, 14, 45, 79, 127),
    4: (-126, -95, -74, -58, -43, -30, -18, -6, 6, 18, 30, 43, 58, 74, 95, 126),
}
LUT_MULT = {3: 59.0, 4: 46.0}
# Fixed point of the stored scales: a code stands for LUT[c] * scale / LUT_ONE,
# which keeps small blocks' scales out of the fp16 subnormal range.
LUT_ONE = 64.0

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
            rope_dim_from_hf_config(vllm_config.model_config.hf_text_config, head_size),
        )
    return _rope_dim_by_head_size[head_size]


@dataclass(frozen=True)
class SplitQFormat:
    head_size: int
    rope_dim: int
    v_bits: int
    compact: bool = False

    def __post_init__(self):
        d, r = self.head_size, self.rope_dim
        if self.v_bits not in (3, 4):
            raise ValueError(f"SplitQ supports 3- or 4-bit V, got {self.v_bits}")
        if self.compact and d & (d - 1):
            raise ValueError(
                f"Compact SplitQ K needs a power-of-two head_size, got {d}"
            )
        if r % HADAMARD_BLOCK or d % HADAMARD_BLOCK:
            raise ValueError(
                f"SplitQ needs head_size and rope_dim in multiples of "
                f"{HADAMARD_BLOCK}; got head_size={d}, rope_dim={r}"
            )

    @classmethod
    def from_cache_dtype(cls, cache_dtype: str, head_size: int, rope_dim: int):
        v_bits, compact = {
            "splitq_k3v4": (4, False),
            "splitq_k3v3": (3, False),
            "splitq_k3v3_compact": (3, True),
        }[cache_dtype]
        return cls(head_size, rope_dim, v_bits, compact)

    @property
    def kernel_code(self) -> int:
        """Format id the kernels take: V bits, plus 16 for compact K."""
        return self.v_bits + (16 if self.compact else 0)

    @property
    def k_rotation_block(self) -> int:
        return self.head_size if self.compact else HADAMARD_BLOCK

    @property
    def k_wide_dim(self) -> int:
        """K dims stored with ROPE_BITS codes (the rest use NOPE_BITS)."""
        return 0 if self.compact else self.rope_dim

    @property
    def nope_dim(self) -> int:
        return self.head_size - self.rope_dim

    @property
    def num_k_blocks(self) -> int:
        return self.head_size // HADAMARD_BLOCK

    @property
    def num_rope_blocks(self) -> int:
        return self.rope_dim // HADAMARD_BLOCK

    @property
    def off_k_nope(self) -> int:
        return self.k_wide_dim * ROPE_BITS // 8

    @property
    def off_v(self) -> int:
        return self.off_k_nope + (self.head_size - self.k_wide_dim) * NOPE_BITS // 8

    @property
    def off_scales(self) -> int:
        return self.off_v + self.head_size * self.v_bits // 8

    @property
    def num_k_scales(self) -> int:
        return 1 if self.compact else self.num_k_blocks

    @property
    def num_scales(self) -> int:
        return self.num_k_scales + 1

    @property
    def slot_bytes(self) -> int:
        raw = self.off_scales + 2 * self.num_scales
        align = 4 if self.compact else 8
        return (raw + align - 1) // align * align


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


@functools.cache
def _lut(bits: int, device_str: str) -> torch.Tensor:
    return torch.tensor(LUT[bits], dtype=torch.float32, device=device_str)


def _quantize_lut(x: torch.Tensor, bits: int):
    """Nearest Lloyd-Max code on the block RMS; the scale keeps
    ``x . x_hat = |x|^2``.

    Returns (codes uint8 in [0, 2**bits), scale) with
    x ≈ LUT[codes] * scale / LUT_ONE.
    """
    lut = _lut(bits, str(x.device))
    ss = x.pow(2).sum(-1, keepdim=True)
    rms = (ss / x.shape[-1]).sqrt().clamp_min(1e-12)
    y = x / rms * LUT_MULT[bits]
    codes = torch.bucketize(y, (lut[1:] + lut[:-1]) / 2)
    dot = (x * lut[codes]).sum(-1, keepdim=True)
    scale = torch.where(dot > 0, ss / dot.clamp_min(1e-30), 0.0) * LUT_ONE
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
    nkb = fmt.num_k_blocks

    if fmt.compact:
        k = rotate(key, signs(d, str(dev)), d)
        k_codes, k_scale = _quantize_lut(k, NOPE_BITS)
        out[..., : fmt.off_v] = _pack(k_codes, NOPE_BITS)
    else:
        k = rotate(key, signs(d, str(dev))).view(t, h, nkb, HADAMARD_BLOCK)
        nr = fmt.num_rope_blocks
        kr_codes, kr_scale = _quantize_lut(k[:, :, :nr], ROPE_BITS)
        kn_codes, kn_scale = _quantize_lut(k[:, :, nr:], NOPE_BITS)
        out[..., : fmt.off_k_nope] = _pack(kr_codes.view(t, h, r), ROPE_BITS)
        out[..., fmt.off_k_nope : fmt.off_v] = _pack(
            kn_codes.view(t, h, fmt.nope_dim), NOPE_BITS
        )
        k_scale = torch.cat((kr_scale[..., 0], kn_scale[..., 0]), -1)

    vr = rotate(value, signs(d, str(dev)), v_block(d))
    v_codes, v_scale = _quantize_lut(vr, fmt.v_bits)
    out[..., fmt.off_v : fmt.off_scales] = _pack(v_codes, fmt.v_bits)

    scales = torch.cat((k_scale, v_scale), -1)
    sc = scales.to(torch.float16).view(torch.uint8).view(t, h, -1)
    out[..., fmt.off_scales : fmt.off_scales + sc.shape[-1]] = sc
    return out


def reference_dequantize(
    slots: torch.Tensor, fmt: SplitQFormat, rotated: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    """(T, H, slot) slots -> fp32 K and V of shape (T, H, D).

    With ``rotated=True`` K and V stay in the rotated space, which is what
    the kernels consume.
    """
    t, h, _ = slots.shape
    r, d, dev = fmt.rope_dim, fmt.head_size, slots.device
    sc = (
        slots[..., fmt.off_scales : fmt.off_scales + 2 * fmt.num_scales]
        .contiguous()
        .view(torch.float16)
        .float()
        / LUT_ONE
    )
    if fmt.compact:
        k = _unpack(slots[..., : fmt.off_v], d, NOPE_BITS)
        k = _lut(NOPE_BITS, str(dev))[k] * sc[..., :1]
    else:
        kr = _unpack(slots[..., : fmt.off_k_nope], r, ROPE_BITS)
        kn = _unpack(slots[..., fmt.off_k_nope : fmt.off_v], fmt.nope_dim, NOPE_BITS)
        k = torch.cat(
            (_lut(ROPE_BITS, str(dev))[kr], _lut(NOPE_BITS, str(dev))[kn]), -1
        ).view(t, h, fmt.num_k_blocks, HADAMARD_BLOCK)
        k = (k * sc[..., : fmt.num_k_blocks, None]).view(t, h, d)

    v = _unpack(slots[..., fmt.off_v : fmt.off_scales], d, fmt.v_bits)
    v = _lut(fmt.v_bits, str(dev))[v] * sc[..., -1:]
    if not rotated:
        k = unrotate(k, signs(d, str(dev)), fmt.k_rotation_block)
        v = unrotate(v, signs(d, str(dev)), v_block(d))
    return k, v


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
