# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PyTorch definitions of what the Ampere kernels compute.

Each function here is the specification one CuTe DSL kernel of this package
implements: the same formats, the same rounding, the same order of FP32
operations where it decides a bit. The tests compare every kernel with its
function.

Formats (all little-endian bytes, as on SM100):

* E2M1 (FP4): a sign bit and a 3-bit code of magnitudes
  {0, 0.5, 1, 1.5, 2, 3, 4, 6}; two codes per byte, the even element in the
  low nibble.
* E4M3 (FP8, `float8_e4m3fn`) and UE8M0 (a power of two `2^(b - 127)`;
  the byte 0 decodes to 0, as the kernels build the FP32 exponent field from
  it).
* MXFP4: E2M1 values with one UE8M0 scale per 32 elements, the scale being
  `2^ceil(log2(amax / 6))`.
* NVFP4 (the 288-byte compressed KV record): E2M1 values with one E4M3 scale
  per 16 elements, the scale being `amax / 6` rounded to E4M3.
* MXFP8 (the 528-byte sliding-window KV record): E4M3 values with one UE8M0
  scale per 32 elements.

Quantizers round to nearest, ties to even, and saturate, as the SM100
conversion instructions do.
"""

import torch

E2M1_MAGNITUDES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
# float32(1 / 6), the factor every quantizer here applies to the block amax.
ONE_SIXTH = torch.tensor(1.0 / 6.0, dtype=torch.float32).item()


def ue8m0_to_f32(codes: torch.Tensor) -> torch.Tensor:
    """UE8M0 bytes to FP32: the byte is the FP32 exponent field."""
    return (codes.to(torch.int32) << 23).view(torch.float32)


def e2m1_to_f32(codes: torch.Tensor) -> torch.Tensor:
    """4-bit E2M1 codes (an int tensor) to FP32."""
    codes = codes.to(torch.int64)
    table = torch.tensor(E2M1_MAGNITUDES, dtype=torch.float32, device=codes.device)
    magnitude = table[codes & 7]
    return torch.where((codes & 8) != 0, -magnitude, magnitude)


def e4m3_to_f32(codes: torch.Tensor) -> torch.Tensor:
    """E4M3 bytes to FP32."""
    return codes.contiguous().view(torch.float8_e4m3fn).to(torch.float32)


def f32_to_e2m1(x: torch.Tensor) -> torch.Tensor:
    """FP32 to 4-bit E2M1 codes (int32), nearest-even and saturating.

    The magnitude code counts the rounding thresholds below `|x|`; at a
    threshold exactly, the even code wins (0.75, 1.75, 3.5 round up; 0.25,
    1.25, 2.5, 5 round down). The sign bit follows `x`'s, zero included.
    """
    a = x.abs()
    code = (
        (a > 0.25).int()
        + (a >= 0.75).int()
        + (a > 1.25).int()
        + (a >= 1.75).int()
        + (a > 2.5).int()
        + (a >= 3.5).int()
        + (a > 5.0).int()
    )
    return code | (torch.signbit(x).int() << 3)


def f32_to_e4m3(x: torch.Tensor) -> torch.Tensor:
    """FP32 within the E4M3 range to E4M3 bytes, nearest-even."""
    return x.to(torch.float8_e4m3fn).view(torch.uint8)


def ceil_log2(x: torch.Tensor) -> torch.Tensor:
    """`ceil(log2(x))` of positive normal FP32 values, exactly (int32)."""
    bits = x.contiguous().view(torch.int32)
    exponent = ((bits >> 23) & 0xFF) - 127
    return exponent + ((bits & 0x7FFFFF) != 0).int()


def mxfp4_quantize(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """FP32 `[..., D]` to MXFP4: packed values `[..., D / 2]` and UE8M0
    scales `[..., D / 32]` (uint8)."""
    blocks = x.float().unflatten(-1, (-1, 32))
    amax = blocks.abs().amax(-1).clamp_min(6.0 * 2.0**-126)
    exponent = ceil_log2(amax * ONE_SIXTH).clamp(-127, 127)
    inverse = ue8m0_to_f32(127 - exponent)
    codes = f32_to_e2m1(blocks * inverse.unsqueeze(-1)).flatten(-2)
    packed = codes[..., 0::2] | (codes[..., 1::2] << 4)
    return packed.to(torch.uint8), (exponent + 127).to(torch.uint8)


def mxfp4_dequantize(packed: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """MXFP4 values `[..., D / 2]` and scales `[..., D / 32]` to FP32."""
    codes = torch.stack((packed & 0xF, packed >> 4), dim=-1).flatten(-2)
    values = e2m1_to_f32(codes).unflatten(-1, (-1, 32))
    return (values * ue8m0_to_f32(scales).unsqueeze(-1)).flatten(-2)


def mxfp8_weight_dequantize(codes: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """An MXFP8 weight `[N, K]` (E4M3 bytes) with one UE8M0 scale per
    32 x 32 block (`[N / 32, K / 32]`, `2^(s - 127)` for every byte
    `s`) as BF16: the product in FP32, which is exact, rounded to BF16, as
    the load-time dequantization computes it."""
    factors = torch.exp2(scales.float() - 127.0)
    factors = factors.repeat_interleave(32, 0).repeat_interleave(32, 1)
    return (e4m3_to_f32(codes) * factors).to(torch.bfloat16)


def mxfp8_linear(
    x: torch.Tensor, codes: torch.Tensor, scales: torch.Tensor
) -> torch.Tensor:
    """`x [M, K] . dequant(W)^T` in FP64 (`mxfp8_weight_dequantize`); the
    kernel's result is this sum in FP32, in its own order, rounded to BF16."""
    weight = mxfp8_weight_dequantize(codes, scales)
    return x.double() @ weight.double().t()


def rope_pairs(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, inverse: bool = False
) -> torch.Tensor:
    """GPT-J (interleaved) RoPE of FP32 `x [..., 2 P]` with `cos`/`sin`
    `[..., P]`: each product rounded, then the difference or sum."""
    even, odd = x[..., 0::2], x[..., 1::2]
    if inverse:
        new_even = even * cos + odd * sin
        new_odd = odd * cos - even * sin
    else:
        new_even = even * cos - odd * sin
        new_odd = odd * cos + even * sin
    return torch.stack((new_even, new_odd), dim=-1).flatten(-2)


def _cos_sin(cos_sin_cache: torch.Tensor, positions: torch.Tensor):
    """`cos`/`sin` FP32 rows `[T, 32]` of a [max_pos, 64] cache."""
    rows = cos_sin_cache[positions.long()].float()
    return rows[:, :32], rows[:, 32:64]


def indexer_q_rope_quant(
    positions: torch.Tensor,
    q: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    weights: torch.Tensor,
    weights_scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Indexer queries `q [T, H, 128]` (BF16; RoPE on dims 64-127) to MXFP4.

    Returns values `[T, H, 64]`, scales `[T, H, 4]` and the FP32 head
    weights `weights * weights_scale` (the softmax and head scales).
    """
    x = q.float()
    cos, sin = _cos_sin(cos_sin_cache, positions)
    rope = rope_pairs(x[..., 64:], cos[:, None], sin[:, None])
    x = torch.cat((x[..., :64], rope.bfloat16().float()), dim=-1)
    values, scales = mxfp4_quantize(x)
    return values, scales, weights.float() * weights_scale


def indexer_k_norm_rope(
    k_pre: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    compress_ratio: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Index keys `[T, 128]` (BF16): RMSNorm (BF16 rounded), RoPE at the
    group's first position (BF16 rounded), MXFP4 (values `[T, 64]`, scales
    `[T, 4]`). Rows that are not group boundaries are computed too."""
    k = k_pre.float()
    variance = (k * k).sum(-1, keepdim=True) / k.shape[-1]
    k = (k * torch.rsqrt(variance + eps) * norm_weight.float()).bfloat16().float()
    first = positions.long() // compress_ratio * compress_ratio
    cos, sin = _cos_sin(cos_sin_cache, first)
    rope = rope_pairs(k[:, 64:], cos, sin).bfloat16().float()
    return mxfp4_quantize(torch.cat((k[:, :64], rope), dim=-1))


def index_key_pages(k_cache: torch.Tensor) -> torch.Tensor:
    """The pages `[blocks, page_bytes]` of an indexer key cache whose first
    two dims are `[blocks, block_size]` (its 3-D allocation or the 4-D view
    the logits kernels take). Each page holds its block's 64-byte values, then
    their 4-byte scales; pages may be padded."""
    return torch.as_strided(
        k_cache, (k_cache.shape[0], k_cache.stride(0)), (k_cache.stride(0), 1)
    )


def store_index_keys(
    values: torch.Tensor,
    scales: torch.Tensor,
    positions: torch.Tensor,
    slot_mapping: torch.Tensor,
    compress_ratio: int,
    k_cache: torch.Tensor,
) -> None:
    """Store MXFP4 index keys of group-boundary rows into the paged cache
    `[blocks, block_size, 68]` (uint8; each page holds its block's values,
    then their scales)."""
    block_size = k_cache.shape[1]
    pages = index_key_pages(k_cache)
    n = slot_mapping.numel()
    slots = slot_mapping.long()
    keep = (slots >= 0) & ((positions[:n].long() + 1) % compress_ratio == 0)
    slots, values, scales = slots[keep], values[:n][keep], scales[:n][keep]
    page, row = slots // block_size, slots % block_size
    cols = torch.arange(64, device=slots.device)
    pages[page[:, None], (row * 64)[:, None] + cols] = values
    cols = torch.arange(4, device=slots.device)
    pages[page[:, None], (block_size * 64 + row * 4)[:, None] + cols] = scales


def nvfp4_kv_record(
    latent: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    compress_ratio: int,
) -> torch.Tensor:
    """Compressed KV latents `[T, 512]` (BF16) to 288-byte NVFP4 records:
    RoPE (FP32, not rounded) on dims 448-511 at the group's first position,
    then 256 bytes of E2M1 pairs and 32 E4M3 scales of 16 dims."""
    x = latent.float()
    first = positions.long() // compress_ratio * compress_ratio
    cos, sin = _cos_sin(cos_sin_cache, first)
    x = torch.cat((x[:, :448], rope_pairs(x[:, 448:], cos, sin)), dim=-1)
    tiles = x.unflatten(-1, (32, 16))
    amax = tiles.abs().amax(-1)
    scale_codes = f32_to_e4m3((amax * ONE_SIXTH).clamp(2.0**-9, 448.0))
    scale = e4m3_to_f32(scale_codes)
    codes = f32_to_e2m1(tiles / scale.unsqueeze(-1)).flatten(-2)
    packed = (codes[:, 0::2] | (codes[:, 1::2] << 4)).to(torch.uint8)
    return torch.cat((packed, scale_codes), dim=-1)


def decode_mxfp8_record(records: torch.Tensor) -> torch.Tensor:
    """528-byte MXFP8 KV records `[..., 528]` to FP32 `[..., 512]`."""
    values = e4m3_to_f32(records[..., :512]).unflatten(-1, (16, 32))
    scales = ue8m0_to_f32(records[..., 512:528])
    return (values * scales.unsqueeze(-1)).flatten(-2)


def decode_nvfp4_record(records: torch.Tensor) -> torch.Tensor:
    """288-byte NVFP4 KV records `[..., 288]` to FP32 `[..., 512]`."""
    packed = records[..., :256]
    codes = torch.stack((packed & 0xF, packed >> 4), dim=-1).flatten(-2)
    values = e2m1_to_f32(codes).unflatten(-1, (32, 16))
    scales = e4m3_to_f32(records[..., 256:288])
    return (values * scales.unsqueeze(-1)).flatten(-2)


def store_v4_records(cache: torch.Tensor, slots: torch.Tensor, x: torch.Tensor) -> None:
    """Store FP32 rows `x [n, 512]` at `slots` of a V4-record cache
    `[blocks, block_size, 584]`: dims 0-447 as E4M3 with one UE8M0 scale per
    64 (`2^ceil(log2(amax / 448))`), dims 448-511 as BF16. Each page holds
    its tokens' 576 value bytes, then their 8 scale bytes (the last zero)."""
    block_size = cache.shape[1]
    pages = index_key_pages(cache)
    blocks = x[:, :448].unflatten(-1, (7, 64))
    amax = blocks.abs().amax(-1).clamp_min(2.0**-100)
    exponent = ceil_log2(amax / 448).clamp(-127, 127)
    fp8 = f32_to_e4m3(blocks / ue8m0_to_f32(exponent + 127).unsqueeze(-1))
    rope = x[:, 448:].bfloat16().view(torch.uint8)
    values = torch.cat((fp8.flatten(-2), rope), dim=-1)
    scales = torch.cat(
        ((exponent + 127).to(torch.uint8), exponent.new_zeros(len(x), 1).byte()), -1
    )
    page, row = slots.long() // block_size, slots.long() % block_size
    cols = torch.arange(576, device=x.device)
    pages[page[:, None], (row * 576)[:, None] + cols] = values
    cols = torch.arange(8, device=x.device)
    pages[page[:, None], (block_size * 576 + row * 8)[:, None] + cols] = scales


def decode_v4_records(cache: torch.Tensor, slots: torch.Tensor) -> torch.Tensor:
    """The V4 records at `slots` (any shape) of `cache` as FP32
    `[..., 512]`."""
    block_size = cache.shape[1]
    pages = index_key_pages(cache)
    page, row = slots.long() // block_size, slots.long() % block_size
    cols = torch.arange(576, device=cache.device)
    values = pages[page[..., None], (row * 576)[..., None] + cols]
    cols = torch.arange(7, device=cache.device)
    scales = pages[page[..., None], (block_size * 576 + row * 8)[..., None] + cols]
    fp8 = e4m3_to_f32(values[..., :448]).unflatten(-1, (7, 64))
    fp8 = (fp8 * ue8m0_to_f32(scales).unsqueeze(-1)).flatten(-2)
    rope = values[..., 448:].contiguous().view(torch.bfloat16).float()
    return torch.cat((fp8, rope), dim=-1)


def _gather_records(cache: torch.Tensor, slots: torch.Tensor) -> torch.Tensor:
    """Records of global `slots` from a paged cache [blocks, block_size, W]."""
    block_size = cache.shape[1]
    slots = slots.long()
    return cache[slots // block_size, slots % block_size]


def sparse_mla(
    q: torch.Tensor,
    swa_cache: torch.Tensor,
    swa_indices: torch.Tensor,
    swa_lens: torch.Tensor,
    compressed_cache: torch.Tensor | None,
    compressed_indices: torch.Tensor | None,
    compressed_lens: torch.Tensor | None,
    sink: torch.Tensor,
    scale: float,
    positions: torch.Tensor | None,
    cos_sin_cache: torch.Tensor | None,
    chunk: int = 64,
    v4_records: bool = False,
) -> torch.Tensor:
    """Sparse MLA of BF16 queries `q [T, H, 512]` over each token's window
    keys (528-byte records at `swa_indices[t, :swa_lens[t]]`) and compressed
    keys (288-byte records at `compressed_indices[t, :compressed_lens[t]]`);
    a negative slot in either list names no key. `v4_records` reads both
    caches as 584-byte V4 records instead.

    Keys double as values. Per head: `softmax` with the sink logit
    `sink[h]` in the denominator, FP32 throughout; the output is rounded to
    BF16, inverse-RoPE'd on dims 448-511 at the token's position and rounded
    to BF16 again (the input `wo_a` takes). Without `positions` and
    `cos_sin_cache` the RoPE dims stay rotated. Tokens without keys output
    0.
    """

    def decode(cache: torch.Tensor, slots: torch.Tensor, nvfp4: bool):
        if v4_records:
            return decode_v4_records(cache, slots)
        records = _gather_records(cache, slots)
        return decode_nvfp4_record(records) if nvfp4 else decode_mxfp8_record(records)

    out = torch.empty_like(q)
    for start in range(0, q.shape[0], chunk):
        rows = slice(start, min(start + chunk, q.shape[0]))
        keys, valid = [], []
        width = swa_indices.shape[-1]
        lanes = torch.arange(width, device=q.device)
        valid.append((lanes < swa_lens[rows, None]) & (swa_indices[rows] >= 0))
        slots = torch.where(valid[-1], swa_indices[rows], 0)
        keys.append(decode(swa_cache, slots, False))
        if compressed_cache is not None:
            assert compressed_indices is not None and compressed_lens is not None
            width = compressed_indices.shape[-1]
            lanes = torch.arange(width, device=q.device)
            valid.append(
                (lanes < compressed_lens[rows, None]) & (compressed_indices[rows] >= 0)
            )
            slots = torch.where(valid[-1], compressed_indices[rows], 0)
            keys.append(decode(compressed_cache, slots, True))
        k = torch.cat(keys, dim=1)  # [t, L, 512], BF16-exact values
        mask = torch.cat(valid, dim=1)  # [t, L]
        scores = torch.einsum("thd,tld->thl", q[rows].float(), k) * scale
        scores = scores.masked_fill(~mask[:, None, :], float("-inf"))
        top = torch.maximum(scores.amax(-1), sink.float())
        p = torch.exp(scores - top[..., None])
        denominator = p.sum(-1) + torch.exp(sink.float() - top)
        o = torch.einsum("thl,tld->thd", p, k) / denominator[..., None]
        o = o.bfloat16().float()
        if cos_sin_cache is not None:
            assert positions is not None
            cos, sin = _cos_sin(cos_sin_cache, positions[rows])
            o[..., 448:] = rope_pairs(
                o[..., 448:], cos[:, None], sin[:, None], inverse=True
            )
        o = torch.where(mask.any(-1)[:, None, None], o, 0.0)
        out[rows] = o.bfloat16()
    return out


def mqa_logits(
    q_values: torch.Tensor,
    q_scales: torch.Tensor,
    k_values: torch.Tensor,
    k_scales: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Indexer logits of MXFP4 queries `[M, H, 64]` (+ scales `[M, H, 4]`)
    against MXFP4 keys `[N, 64]` (+ `[N, 4]`): `sum_h weights[m, h] *
    relu(q[m, h] . k[n])`, FP32 `[M, N]`."""
    q = mxfp4_dequantize(q_values, q_scales)
    k = mxfp4_dequantize(k_values, k_scales)
    scores = torch.einsum("mhd,nd->mhn", q, k).relu()
    return (scores * weights.float().unsqueeze(-1)).sum(1)


def gather_index_keys(
    k_cache: torch.Tensor, block_table: torch.Tensor, length: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """The first `length` MXFP4 keys of one request from the paged indexer
    cache: values `[length, 64]` and scales `[length, 4]`."""
    block_size = k_cache.shape[1]
    pages = index_key_pages(k_cache)
    position = torch.arange(length, device=k_cache.device)
    page = block_table[position // block_size].long()
    row = position % block_size
    values = pages[
        page[:, None], (row * 64)[:, None] + torch.arange(64, device=row.device)
    ]
    scales = pages[
        page[:, None],
        (block_size * 64 + row * 4)[:, None] + torch.arange(4, device=row.device),
    ]
    return values, scales


def candidate_mask(
    logits: torch.Tensor,
    lens: torch.Tensor,
    candidates: torch.Tensor,
    block: int,
) -> torch.Tensor:
    """Decode logits `[rows, width]` with every key of row `m` below
    `lens[m]` whose block of `block` keys is not among the row's
    `candidates` (`-1` padded) set to `-inf`."""
    out = logits.clone()
    for row in range(logits.shape[0]):
        n = int(lens[row])
        keep = torch.zeros(n, dtype=torch.bool, device=logits.device)
        blocks = candidates[row][candidates[row] >= 0].long()
        keys = blocks[:, None] * block + torch.arange(block, device=blocks.device)
        keys = keys.flatten()
        keep[keys[keys < n]] = True
        out[row, :n] = torch.where(keep, logits[row, :n], float("-inf"))
    return out


def decode_topk(
    logits: torch.Tensor, lens: torch.Tensor, k: int, group: int = 1
) -> torch.Tensor:
    """The `k` best of each row's first `lens[row]` scores, or of its
    blocks of `group` scores (a block scores its maximum; the newest block
    is pinned first): int32 `[rows, k]` in ascending order, `-1` padded;
    ties at the threshold go to the lowest indices."""
    rows, width = logits.shape
    out = torch.full((rows, k), -1, dtype=torch.int32, device=logits.device)
    for row in range(rows):
        n = max(0, min(int(lens[row]), width))
        scores = logits[row, :n]
        if group > 1:
            blocks = (n + group - 1) // group
            padded = scores.new_full((blocks * group,), float("-inf"))
            padded[:n] = scores
            scores = padded.view(blocks, group).amax(-1)
            if blocks:
                scores[blocks - 1] = float("inf")
            n = blocks
        if n <= k:
            out[row, :n] = torch.arange(n, dtype=torch.int32, device=logits.device)
            continue
        best = torch.sort(-scores, stable=True).indices[:k]
        out[row] = torch.sort(best).values.int()
    return out


def mhc_post(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_mix: torch.Tensor,
    comb_mix: torch.Tensor,
) -> torch.Tensor:
    """The mHC post-mix: `r'[j] = post[j] x + sum_k comb[k, j] r[k]` of the
    `[rows, 4, hidden]` BF16 streams, rounded to BF16."""
    mixed = torch.einsum("nkj,nkh->njh", comb_mix.double(), residual.double())
    mixed = mixed + post_mix.double().view(-1, 4, 1) * x.double()[:, None, :]
    return mixed.bfloat16()


def mhc_post_collapse(
    x: torch.Tensor | None,
    residual: torch.Tensor,
    post_mix: torch.Tensor | None,
    comb_mix: torch.Tensor | None,
    pre_mix: torch.Tensor | None,
    norm_weight: torch.Tensor,
    eps: float,
    write_aux: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """A sublayer boundary's critical path: the streams (post-mixed with
    `x`, else `residual` itself), the next layer input -- the RMSNorm of
    the streams collapsed with the carried pre-mix (stream zero without one),
    the collapse rounded to BF16 before the norm -- and, with `write_aux`,
    the BF16 stream mean."""
    streams = residual if x is None else mhc_post(x, residual, post_mix, comb_mix)
    if pre_mix is None:
        pre_mix = torch.zeros(residual.shape[0], 4, device=residual.device)
        pre_mix[:, 0] = 1
    collapsed = (pre_mix.double()[:, :, None] * streams.double()).sum(1)
    collapsed = collapsed.bfloat16().double()
    inverse_rms = torch.rsqrt(collapsed.square().mean(-1, keepdim=True) + eps)
    layer_input = (collapsed * inverse_rms * norm_weight.double()).bfloat16()
    aux = streams.double().mean(1).bfloat16()
    return streams, layer_input, aux if write_aux else aux[:0]


def mhc_coefficients(
    x: torch.Tensor | None,
    residual: torch.Tensor,
    post_mix: torch.Tensor | None,
    comb_mix: torch.Tensor | None,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    pre_eps: float,
    sinkhorn_eps: float,
    post_mult: float,
    sinkhorn_repeat: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The next boundary's post-, comb- and pre-mix (FP32) from the BF16
    streams (`mhc_post` of `x` when given, else `residual`, which may
    also be one `[rows, hidden]` row): 24 projections scaled by the
    streams' inverse RMS, sigmoids, and a softmax followed by alternating
    column and row normalizations (`sinkhorn_repeat` column steps)."""
    streams = residual if x is None else mhc_post(x, residual, post_mix, comb_mix)
    a = streams.reshape(streams.shape[0], -1).double()
    mixes = a @ fn.double().t()
    mixes = mixes * torch.rsqrt(a.square().mean(-1, keepdim=True) + rms_eps)
    scale, base = hc_scale.double(), hc_base.double()
    pre = torch.sigmoid(mixes[:, :4] * scale[0] + base[:4]) + pre_eps
    post = torch.sigmoid(mixes[:, 4:8] * scale[1] + base[4:8]) * post_mult
    comb = mixes[:, 8:].view(-1, 4, 4) * scale[2] + base[8:].view(4, 4)
    comb = torch.softmax(comb, dim=-1) + sinkhorn_eps
    comb = comb / (comb.sum(-2, keepdim=True) + sinkhorn_eps)
    for _ in range(sinkhorn_repeat - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + sinkhorn_eps)
        comb = comb / (comb.sum(-2, keepdim=True) + sinkhorn_eps)
    return post.float(), comb.float(), pre.float()
