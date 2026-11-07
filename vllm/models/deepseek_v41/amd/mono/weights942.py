# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host-side weight preparation for the gfx942 (MI300X, MI325X) mono kernels.

On gfx942 vLLM dequantizes the dense MXFP8 linears to bf16 when it loads them
and keeps their E8M0 scales. ``ocp_from_dequant`` rebuilds the original e4m3
bytes from the two, exactly. ``dense_copy`` turns OCP e4m3 bytes into the
copy the gfx942 GEMV reads (``attention.gemv._step_load_942``):

* FNUZ bytes. A byte keeps its bit pattern, which stands for half the OCP
  value in FNUZ, and its scale code is raised by one to compensate. OCP -0
  (0x80) is the FNUZ NaN and becomes 0x00.
* K padded with zero bytes to a multiple of 128.
* Tile-major: for each tile of 16 rows and each 128-byte K step, the 16
  rows' bytes of the step stored together (2 KB). Lane 16 j + r of the
  GEMV takes row r's four 8-byte chunks of lane group j (K 32 b + 8 j .. + 8,
  b = 0 .. 3) as two 16-byte halves, blocks 0 and 1, then blocks 2 and 3.
  The 2 KB hold the first halves of the 64 lanes in lane order, then the
  second halves, so each of the two 16-byte loads of a lane reads 1 KB that
  is contiguous across the wave.
* ``chunk_q``: within each 8-byte chunk the even K first, then the odd K.
  The MoE stages quantize their activations in that order, to match the
  bytes the FP4 conversion of the routed experts produces
  (``common.gfx942.fp4x8_fnuz``).
"""

import torch
import torch.nn.functional as F

EVEN_ODD = [0, 2, 4, 6, 1, 3, 5, 7]


def ocp_from_dequant(w: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """The OCP e4m3 bytes [N, K] behind a bf16 weight vLLM dequantized at load
    (``w`` = e4m3 value x 2^(code - 127), ``scale`` the codes in 32 x 32
    blocks [N / 32, K / 32]). Refuses a weight the bytes do not give back
    exactly."""
    n, k = w.shape
    assert scale.shape == (n // 32, k // 32), (w.shape, scale.shape)
    mul = torch.exp2(127.0 - scale.float())
    mul = mul.repeat_interleave(32, 0).repeat_interleave(32, 1)
    q = (w.float() * mul).to(torch.float8_e4m3fn)
    back = (q.float() / mul).to(w.dtype)
    if not torch.equal(back, w):
        bad = (back != w).sum().item()
        raise ValueError(f"{bad} of {w.numel()} values are not e4m3 x scale")
    return q.view(torch.uint8)


def dense_copy(w8: torch.Tensor, scale: torch.Tensor, chunk_q: bool = False):
    """OCP e4m3 bytes [N, K] and E8M0 codes [N / 32, K / 32] -> the gfx942
    copy: (bytes [N, Kp], FNUZ codes [N / 32, Kp / 32]), Kp = K rounded up
    to 128. A padded block holds zero bytes and code 128."""
    assert w8.dtype == torch.uint8 and scale.dtype == torch.uint8
    n, k = w8.shape
    assert n % 32 == 0 and k % 32 == 0, w8.shape
    assert scale.shape == (n // 32, k // 32), (w8.shape, scale.shape)
    assert int(scale.max()) < 250, "a scale code this large overflows the factor"
    kp = -(-k // 128) * 128
    b = torch.where(w8 == 0x80, torch.zeros_like(w8), w8)
    codes = scale.to(torch.int32) + 1
    if kp != k:
        b = F.pad(b, (0, kp - k))
        codes = F.pad(codes, (0, (kp - k) // 32), value=128)
    v = b.view(n, kp // 128, 4, 4, 8)  # [row, step, block, lane group, byte]
    if chunk_q:
        v = v[..., EVEN_ODD]
    # [tile, r, step, block half, block in the half, lane group j, byte]
    v = v.reshape(n // 16, 16, kp // 128, 2, 2, 4, 8)
    # -> [tile, step, block half, j, r, block in the half, byte]
    v = v.permute(0, 2, 3, 5, 1, 4, 6).contiguous().view(n, kp)
    return v, codes.to(torch.uint8).contiguous()


def linear_copy(w: torch.Tensor, scale: torch.Tensor):
    """A vLLM-loaded MXFP8 linear on gfx942 (bf16 dequantized, or e4m3 bytes)
    with its E8M0 codes [N / 32, K / 32] -> ``dense_copy`` with natural chunks
    (the attention and shared expert activations are not reordered)."""
    scale = scale if scale.dtype == torch.uint8 else scale.view(torch.uint8)
    n, k = w.shape
    if scale.shape == (n, k // 32):
        # vLLM on gfx942 repeats each 32 x 32 block's code over the block's
        # 32 rows when it loads the checkpoint. Fold them back to one code
        # a block, and refuse a block whose rows do not agree.
        rows = scale.view(n // 32, 32, k // 32)
        if not torch.equal(rows, rows[:, :1].expand_as(rows)):
            raise ValueError(f"row scales {tuple(scale.shape)} are not 32 x 32 blocks")
        scale = rows[:, 0]
    if w.dtype == torch.bfloat16:
        w8 = ocp_from_dequant(w, scale)
    elif w.dtype == torch.float8_e4m3fn:
        w8 = w.view(torch.uint8)
    else:
        raise TypeError(f"no gfx942 copy of a {w.dtype} linear")
    return dense_copy(w8.contiguous(), scale.contiguous(), chunk_q=False)


def moe_copies(w13, w13_s, w2, w2_s, inter: int):
    """VLLM's plain MXFP4 expert tensors before its kernel conversion (w13
    [E, 2 Np, K / 2], w2 [E, H, Np / 2] and codes [E, rows, K / 32], with
    w13's gate rows, then its up rows, each padded with zero rows from
    ``inter`` to Np) -> the gfx942 mono copies: w13 [E, 2 inter, K / 2] with
    gate row i at 2 i and up row i at 2 i + 1, w2 [E, H, inter / 2], their
    codes alike, no -0 nibbles. The weight bytes are in ``fp4_tile_major``
    order. The w13 codes are in ``codes_tile_rounds`` order, and the w2 codes
    stay row-major because a lane group reads 4 whole rows of them. Refuses
    padding rows that are not zero."""
    e, two_np, kh = w13.shape
    np_ = two_np // 2
    assert inter <= np_ and inter % 32 == 0, (w13.shape, inter)
    assert w13_s.shape == (e, two_np, 2 * kh // 32), (w13.shape, w13_s.shape)
    assert w2.shape[2] * 2 == np_ and w2_s.shape[2] * 32 == np_, (w2.shape, w2_s.shape)
    if inter < np_:
        pad = (
            w13[:, inter:np_].any(),
            w13[:, np_ + inter :].any(),
            w2[:, :, inter // 2 :].any(),
        )
        assert not any(bool(p) for p in pad), "the expert padding is not zero"
    out = []
    for t, kdim in ((w13, kh), (w13_s, 2 * kh // 32)):
        gate, up = t[:, :inter], t[:, np_ : np_ + inter]
        out.append(
            torch.stack([gate, up], dim=2).reshape(e, 2 * inter, kdim).contiguous()
        )
    out.append(w2[:, :, : inter // 2].contiguous())
    out.append(w2_s[:, :, : inter // 32].contiguous())
    fp4_drop_negative_zero_(out[0])
    fp4_drop_negative_zero_(out[2])
    out[0] = fp4_tile_major(out[0])
    out[1] = codes_tile_rounds(out[1])
    out[2] = fp4_tile_major(out[2])
    return tuple(out)


def fp4_tile_major(w: torch.Tensor) -> torch.Tensor:
    """Packed e2m1 rows [..., rows, K / 2] (rows a multiple of 16) -> the byte
    order the gfx942 routed GEMVs load. For each tile of 16 rows and each
    64-byte step (four K blocks), the copy stores the 64 lanes' 16 bytes in
    lane order. Lane 16 j + r holds the 16 bytes that ``fp4_lane_major``
    gives lane group j of row r. One 16-byte load a lane then reads 1 KB that
    is contiguous, 8 whole 128-byte lines. In the ``fp4_lane_major`` order
    the same load would read 64 bytes from each of 16 rows that are K / 2
    bytes apart, so it would take 16 memory requests instead of 8 and fetch
    each line with two loads. On MI325X a CTA's ug GEMV takes 23.2 us in this
    order and 32.5 us in the ``fp4_lane_major`` order (6 rows, all 4 waves),
    with the same results bit for bit.
    A last 32-byte step (two K blocks, the 576-wide w2 rows) is stored the
    same way with 8 bytes a lane, 512 contiguous bytes."""
    kb = w.shape[-1]
    full = kb // 64 * 64
    assert kb - full in (0, 32) and w.shape[-2] % 16 == 0, w.shape
    lm = fp4_lane_major(w).reshape(-1, 16, kb)  # [tile, r, byte]
    head = lm[..., :full].reshape(-1, 16, full // 64, 4, 16)  # [tile, r, step, j, byte]
    parts = [head.permute(0, 2, 3, 1, 4).reshape(-1, 16 * full)]
    if kb > full:
        tail = lm[..., full:].reshape(-1, 16, 4, 8)  # [tile, r, j, byte]
        parts.append(tail.permute(0, 2, 1, 3).reshape(-1, 16 * 32))
    return torch.cat(parts, dim=-1).contiguous().view(w.shape)


def codes_tile_rounds(codes: torch.Tensor, blocks: int = 8) -> torch.Tensor:
    """e8m0 codes [..., rows, K / 32] (rows a multiple of 16) -> the order the
    gfx942 ug GEMV loads. For each tile of 16 rows and each round of
    ``blocks`` K blocks (``moe.UG_ROUND_942``), the copy stores the 16 rows'
    codes of the round row after row, 16 x ``blocks`` bytes together. Lane
    group j then reads the codes of its rows 4 j .. 4 j + 3 with two 16-byte
    loads from one line. Row-major codes would take four 8-byte loads from
    four lines."""
    kc = codes.shape[-1]
    assert kc % blocks == 0 and codes.shape[-2] % 16 == 0, codes.shape
    v = codes.reshape(-1, 16, kc // blocks, blocks)  # [tile, r, round, block]
    return v.permute(0, 2, 1, 3).contiguous().view(codes.shape)


def fp4_lane_major(w: torch.Tensor) -> torch.Tensor:
    """Packed e2m1 rows [..., K / 2] -> each row's bytes in lane order, for
    ``fp4_tile_major``. A lane of the 16 x 16 x 32 MFMA takes the dword of K
    32 i + 8 j .. + 8 of a row in each 32-wide K block i, with j = lane / 16.
    Within each 64-byte step (four K blocks) the copy stores lane group j's
    four dwords together at bytes 16 j .. 16 j + 15, so one 16-byte load a
    lane reads them, and the 4 lane groups of a row read 64 contiguous bytes.
    A last 32-byte step (two K blocks, the 576-wide w2 rows) stores lane
    group j's two dwords at bytes 8 j .. 8 j + 7. The scale codes keep their
    order, because each K block keeps its code."""
    *lead, kb = w.shape
    full = kb // 64 * 64
    head = w[..., :full].reshape(*lead, kb // 64, 4, 4, 4).transpose(-3, -2)
    parts = [head.reshape(*lead, full)]
    if kb > full:
        assert kb - full == 32, w.shape
        tail = w[..., full:].reshape(*lead, 2, 4, 4).transpose(-3, -2)
        parts.append(tail.reshape(*lead, 32))
    return torch.cat(parts, dim=-1).contiguous()


def fp4_drop_negative_zero_(w: torch.Tensor) -> int:
    """Rewrite every -0 nibble (0x8) of a packed e2m1 tensor to +0, in place,
    chunk by chunk. -0 and +0 give the same products, so vLLM's own MoE
    kernels compute the same results afterwards. gfx942's FP4 conversion
    (``common.gfx942.fp4x8_fnuz``) would turn -0 into the FNUZ NaN. Returns
    how many nibbles changed."""
    assert w.dtype == torch.uint8
    flat = w.view(-1)
    changed = 0
    step = 1 << 28
    for i in range(0, flat.numel(), step):
        part = flat[i : i + step]
        lo = (part & 0x0F) == 0x08
        hi = (part & 0xF0) == 0x80
        changed += int(lo.sum()) + int(hi.sum())
        part.copy_(torch.where(lo, part & 0xF0, part))
        part.copy_(torch.where(hi, part & 0x0F, part))
    return changed
