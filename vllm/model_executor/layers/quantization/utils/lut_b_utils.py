# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""LUT-B weight format: 3-bit indices into per-tile E4M3 codebooks.

Rubin's `tcgen05.mma ... decompress::lut::b` instructions decode each weight
from a three-bit index into an eight-entry E4M3 codebook shared by eight
output channels. This module implements a calibration-free format matching
that contract. Each `8 (N) x 64 (K)` tile of a `[N, K]` weight stores
192 bytes of packed three-bit indices (row-major over the tile) and an
eight-entry E4M3 codebook. Separately, every output channel carries one UE8M0
scale per 32 K (`[N, K / 32]`), which is exactly the hardware's block32 B
scale, so the 16 scales of a tile may differ. They keep small-magnitude rows
and blocks out of the E4M3 subnormal range. In total the format uses 3.375
bits per weight.
"""

import torch

from vllm.platforms import current_platform
from vllm.triton_utils import HAS_TRITON

LUT_B_BLOCK_N = 8
LUT_B_BLOCK_K = 64
LUT_B_SCALE_BLOCK_K = 32
LUT_B_CODEBOOK_SIZE = 8
LUT_B_PACKED_TILE_BYTES = LUT_B_BLOCK_N * LUT_B_BLOCK_K * 3 // 8
LUT_B_SCALE_ROUNDS = 1
LUT_B_MAX_TILES_PER_CHUNK = 1024

_E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max
_UE8M0_BIAS = 127
_FP32_TINY = torch.finfo(torch.float32).tiny


def pack_lut_b_indices(indices: torch.Tensor) -> torch.Tensor:
    """Pack groups of eight 3-bit indices into three little-endian bytes."""
    if indices.shape[-1] % 8 != 0:
        raise ValueError("The number of LUT indices must be divisible by 8")

    groups = indices.reshape(*indices.shape[:-1], -1, 8).to(torch.int32)
    words = torch.zeros(groups.shape[:-1], dtype=torch.int32, device=indices.device)
    for i in range(8):
        words |= groups[..., i] << (3 * i)
    packed = torch.stack((words & 0xFF, (words >> 8) & 0xFF, (words >> 16) & 0xFF), -1)
    return packed.to(torch.uint8).flatten(start_dim=-2)


def unpack_lut_b_indices(packed: torch.Tensor) -> torch.Tensor:
    """Inverse of `pack_lut_b_indices`."""
    if packed.shape[-1] % 3 != 0:
        raise ValueError("The number of packed bytes must be divisible by 3")

    groups = packed.reshape(*packed.shape[:-1], -1, 3).to(torch.int32)
    words = groups[..., 0] | (groups[..., 1] << 8) | (groups[..., 2] << 16)
    indices = torch.stack([(words >> (3 * i)) & 0x7 for i in range(8)], -1)
    return indices.to(torch.uint8).flatten(start_dim=-2)


def _snap_to_e4m3(values: torch.Tensor) -> torch.Tensor:
    values = values.clamp(-_E4M3_MAX, _E4M3_MAX)
    return values.to(torch.float8_e4m3fn).to(torch.float32)


def _assign(values: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
    boundaries = ((centers[:, :-1] + centers[:, 1:]) * 0.5).contiguous()
    return torch.searchsorted(boundaries, values.contiguous())


def _normalize_centers(
    centers: torch.Tensor, preserve_zero: torch.Tensor
) -> torch.Tensor:
    """Snap to E4M3, keep an exact zero where needed, and sort."""
    centers = _snap_to_e4m3(centers)
    zero_index = centers.abs().argmin(dim=1)
    rows = preserve_zero.nonzero(as_tuple=True)[0]
    centers[rows, zero_index[rows]] = 0
    return centers.sort(dim=1).values


def _circle_exponents(blocks: torch.Tensor) -> torch.Tensor:
    """Block exponents that put every block amax in `[2**cut, 2**(cut + 1))`.

    `cut` starts the largest gap between the fractional parts of the block
    log2 amaxes around the unit circle, so blocks whose amaxes differ only by
    noise share an alignment instead of straddling a power of two.
    """
    amax = blocks.abs().amax(dim=-1)
    valid = amax > 0
    log_amax = torch.log2(amax.clamp_min(_FP32_TINY))
    phases = torch.where(valid, log_amax - torch.floor(log_amax), torch.inf)
    phases = phases.sort(dim=1).values
    count = valid.sum(dim=1, keepdim=True)
    safe_count = count.clamp_min(1)
    position = torch.arange(phases.shape[1], device=blocks.device)
    following = phases.gather(1, (position + 1) % safe_count)
    gaps = torch.where(position == count - 1, following + 1, following) - phases
    gaps = torch.where(position < count, gaps, -1.0)
    cut = phases.gather(1, (gaps.argmax(dim=1, keepdim=True) + 1) % safe_count)
    cut = torch.where(count > 0, cut, 0.0)
    exponent = torch.where(valid, torch.floor(log_amax - cut), 0.0)
    return exponent.clamp(-_UE8M0_BIAS, _UE8M0_BIAS)


def _segment_cost(
    prefixes: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    start: torch.Tensor,
    end: torch.Tensor,
) -> torch.Tensor:
    """Weighted squared deviation of sorted values `[start, end)` from their mean."""
    mass, delta, spread = (p.gather(1, end) - p.gather(1, start) for p in prefixes)
    cost = spread - delta.square() / mass.clamp_min(_FP32_TINY)
    return torch.where(mass > 0, cost, 0.0)


def _dp_stage(
    prefixes: tuple[torch.Tensor, ...], previous: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """One more segment: the best cost and last split for every prefix.

    The best split is monotone in the prefix length, so prefixes are solved
    coarse to fine: each lies between the splits of its two neighbors.
    """
    tiles, size = previous.shape[0], previous.shape[1] - 1
    current = torch.full_like(previous, torch.inf)
    split = torch.zeros_like(previous, dtype=torch.long)
    half = size // 2
    while half:
        nodes = torch.arange(half, size, 2 * half, device=previous.device)
        low = split[:, nodes - half]
        right = nodes + half
        high = torch.where(right < size, split[:, right.clamp(max=size - 1)], size - 1)
        high = torch.minimum(high, nodes - 1)
        candidates = low[..., None] + torch.arange(
            int((high - low).max()) + 1, device=previous.device
        )
        valid = candidates <= high[..., None]
        candidates = torch.minimum(candidates, high[..., None]).flatten(1)
        ends = nodes[:, None].expand(-1, candidates.shape[1] // nodes.numel())
        cost = previous.gather(1, candidates) + _segment_cost(
            prefixes, candidates, ends.flatten().expand(tiles, -1)
        )
        cost = cost.view(valid.shape).masked_fill(~valid, torch.inf)
        best, index = cost.min(dim=-1)
        current[:, nodes] = best
        split[:, nodes] = low + index
        half //= 2
    return current, split


def _fit_codebooks(
    values: torch.Tensor, weights: torch.Tensor, preserve_zero: torch.Tensor
) -> torch.Tensor:
    """Exact weighted k-means codebook for each row of `values`.

    The optimal clusters are runs of the sorted values, so dynamic programming
    over the split points finds the global optimum before E4M3 snapping.
    """
    values, order = values.sort(dim=1)
    weights = weights.gather(1, order)
    tiles, size = values.shape
    mean = (weights * values).sum(1, keepdim=True) / weights.sum(
        1, keepdim=True
    ).clamp_min(_FP32_TINY)
    centered = values - mean
    zero = values.new_zeros(tiles, 1)
    prefixes = tuple(
        torch.cat((zero, (weights * centered**power).cumsum(1)), dim=1)
        for power in range(3)
    )

    ends = torch.arange(size + 1, device=values.device).expand(tiles, -1)
    cost = _segment_cost(prefixes, torch.zeros_like(ends), ends)
    cost[:, 0] = torch.inf
    splits = []
    for _ in range(2, LUT_B_CODEBOOK_SIZE):
        cost, split = _dp_stage(prefixes, cost)
        splits.append(split)

    starts = torch.arange(size, device=values.device).expand(tiles, -1)
    last = cost[:, :size] + _segment_cost(
        prefixes, starts, torch.full_like(starts, size)
    )
    start = last.argmin(dim=1, keepdim=True)
    end = torch.full_like(start, size)
    centers = values.new_empty(tiles, LUT_B_CODEBOOK_SIZE)
    for stage in range(LUT_B_CODEBOOK_SIZE, 0, -1):
        if stage == 1:
            start = torch.zeros_like(end)
        elif stage < LUT_B_CODEBOOK_SIZE:
            start = splits[stage - 2].gather(1, end)
        mass, delta = (p.gather(1, end) - p.gather(1, start) for p in prefixes[:2])
        center = torch.where(mass > 0, delta / mass.clamp_min(_FP32_TINY), 0.0)
        centers[:, stage - 1] = (mean + center).squeeze(1)
        end = start
    return _normalize_centers(centers, preserve_zero)


def _fit_tiles(
    blocks: torch.Tensor, exponent: torch.Tensor, preserve_zero: torch.Tensor
) -> torch.Tensor:
    """Codebooks for `blocks` divided by their block scales.

    Each value is weighed by its scale squared, so the fit minimizes the error
    of the original weights.
    """
    values = (blocks * torch.exp2(-exponent).unsqueeze(-1)).flatten(1)
    weights = torch.exp2(2 * (exponent - exponent.amax(dim=1, keepdim=True)))
    weights = weights.unsqueeze(-1).expand_as(blocks).flatten(1)
    return _fit_codebooks(values, weights, preserve_zero)


def _block_errors(
    blocks: torch.Tensor,
    exponent: torch.Tensor,
    tile_exponent: torch.Tensor,
    centers: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Squared error of each K32 block, in units of the tile scale."""
    values = blocks * torch.exp2(-exponent).unsqueeze(-1)
    indices = _assign(values.flatten(1), centers)
    decoded = torch.gather(centers, 1, indices).view_as(blocks)
    relative_scale = torch.exp2(exponent - tile_exponent).unsqueeze(-1)
    return ((decoded - values) * relative_scale).square().sum(-1), indices


def _quantize_tiles(
    blocks: torch.Tensor, num_scale_rounds: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fit `[tiles, 16, 32]` K32 blocks; returns indices, codebooks, exponents."""
    preserve_zero = (blocks == 0).flatten(1).any(dim=1)
    exponent = _circle_exponents(blocks)
    tile_exponent = exponent.amax(dim=1, keepdim=True)
    centers = _fit_tiles(blocks, exponent, preserve_zero)
    for round in range(num_scale_rounds):
        error, _ = _block_errors(blocks, exponent, tile_exponent, centers)
        base = exponent
        radius = 2 if round == 0 else 1
        for delta in range(-radius, radius + 1):
            if delta == 0:
                continue
            trial = (base + delta).clamp(-_UE8M0_BIAS, _UE8M0_BIAS)
            trial_error, _ = _block_errors(blocks, trial, tile_exponent, centers)
            better = trial_error < error
            exponent = torch.where(better, trial, exponent)
            error = torch.where(better, trial_error, error)
        refit = _fit_tiles(blocks, exponent, preserve_zero)
        refit_error, _ = _block_errors(blocks, exponent, tile_exponent, refit)
        keep = refit_error.sum(dim=1) <= error.sum(dim=1)
        centers = torch.where(keep[:, None], refit, centers)
    _, indices = _block_errors(blocks, exponent, tile_exponent, centers)
    return indices, centers, exponent


@torch.no_grad()
def quantize_lut_b(
    weight: torch.Tensor,
    *,
    num_scale_rounds: int = LUT_B_SCALE_ROUNDS,
    max_tiles_per_chunk: int = LUT_B_MAX_TILES_PER_CHUNK,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize a `[N, K]` or stacked `[E, N, K]` weight to LUT-B.

    Returns `(packed, codebooks, scales)` with shapes
    `[E, N/8, K/64, 192]` (uint8), `[E, N/8, K/64, 8]` (float8_e4m3fn)
    and `[E, N, K/32]` (uint8 UE8M0). The expert dim is dropped for 2D
    inputs.

    Block exponents start from the circular cut of the block amax phases, and
    each tile's codebook is the exact weighted k-means fit, snapped to E4M3.
    Each of the `num_scale_rounds` rounds then moves every block exponent
    within +-2 (later rounds +-1) against the fixed codebook and refits the
    codebook, keeping it only if the reconstruction error does not grow.
    """
    squeeze = weight.ndim == 2
    if squeeze:
        weight = weight.unsqueeze(0)
    if weight.ndim != 3:
        raise ValueError(f"LUT-B expects a 2D or 3D weight, got {weight.shape}")
    num_experts, n, k = weight.shape
    if n % LUT_B_BLOCK_N or k % LUT_B_BLOCK_K:
        raise ValueError(
            f"LUT-B requires dims divisible by ({LUT_B_BLOCK_N}, {LUT_B_BLOCK_K}),"
            f" got ({n}, {k})"
        )

    if weight.is_cuda and HAS_TRITON and current_platform.is_cuda():
        from vllm.kernels.triton.lut_b_quantize import quantize_lut_b_triton

        packed, codebooks, scales = quantize_lut_b_triton(weight, num_scale_rounds)
        if squeeze:
            return packed[0], codebooks[0], scales[0]
        return packed, codebooks, scales

    n_tiles, k_tiles = n // LUT_B_BLOCK_N, k // LUT_B_BLOCK_K
    tile_shape = (num_experts * n_tiles, k_tiles)
    packed = weight.new_empty(*tile_shape, LUT_B_PACKED_TILE_BYTES, dtype=torch.uint8)
    codebooks = weight.new_empty(
        *tile_shape, LUT_B_CODEBOOK_SIZE, dtype=torch.float8_e4m3fn
    )
    blocks_per_row = LUT_B_BLOCK_K // LUT_B_SCALE_BLOCK_K
    scales = weight.new_empty(
        num_experts * n_tiles,
        LUT_B_BLOCK_N,
        k_tiles * blocks_per_row,
        dtype=torch.uint8,
    )

    rows = weight.reshape(num_experts * n_tiles, LUT_B_BLOCK_N, k)
    rows_per_chunk = max(1, max_tiles_per_chunk // k_tiles)
    for start in range(0, rows.shape[0], rows_per_chunk):
        end = min(start + rows_per_chunk, rows.shape[0])
        # [tiles, 16 K32 blocks (row-major over 8 rows x 2), 32]
        blocks = (
            rows[start:end]
            .to(torch.float32)
            .reshape(end - start, LUT_B_BLOCK_N, k_tiles, LUT_B_BLOCK_K)
            .transpose(1, 2)
            .reshape(-1, LUT_B_BLOCK_N * blocks_per_row, LUT_B_SCALE_BLOCK_K)
        )
        indices, centers, exponent = _quantize_tiles(blocks, num_scale_rounds)

        packed[start:end] = pack_lut_b_indices(indices).view(end - start, k_tiles, -1)
        codebooks[start:end] = centers.to(torch.float8_e4m3fn).view(
            end - start, k_tiles, -1
        )
        scales[start:end] = (
            (exponent + _UE8M0_BIAS)
            .to(torch.uint8)
            .view(end - start, k_tiles, LUT_B_BLOCK_N, blocks_per_row)
            .transpose(1, 2)
            .reshape(end - start, LUT_B_BLOCK_N, -1)
        )

    out_shape = (num_experts, n_tiles, k_tiles)
    packed = packed.view(*out_shape, -1)
    codebooks = codebooks.view(*out_shape, -1)
    scales = scales.view(num_experts, n, k // LUT_B_SCALE_BLOCK_K)
    if squeeze:
        return packed[0], codebooks[0], scales[0]
    return packed, codebooks, scales


def _check_scale_shape(scales: torch.Tensor, num_experts: int, n: int, k: int):
    expected = (num_experts, n, k // LUT_B_SCALE_BLOCK_K)
    if scales.shape != expected:
        raise ValueError(f"Scale shape {tuple(scales.shape)} != {expected}")


def dequantize_lut_b(
    packed: torch.Tensor,
    codebooks: torch.Tensor,
    scales: torch.Tensor,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Reconstruct the `[N, K]` or `[E, N, K]` weight from LUT-B tiles."""
    squeeze = packed.ndim == 3
    if squeeze:
        packed, codebooks, scales = packed[None], codebooks[None], scales[None]
    if packed.ndim != 4 or packed.shape[-1] != LUT_B_PACKED_TILE_BYTES:
        raise ValueError(f"Unexpected packed LUT-B shape {packed.shape}")
    if codebooks.shape != (*packed.shape[:3], LUT_B_CODEBOOK_SIZE):
        raise ValueError(f"Codebook shape {codebooks.shape} != {packed.shape}")
    num_experts, n_tiles, k_tiles = packed.shape[:3]
    n, k = n_tiles * LUT_B_BLOCK_N, k_tiles * LUT_B_BLOCK_K
    _check_scale_shape(scales, num_experts, n, k)

    books = codebooks.to(torch.float32)
    indices = unpack_lut_b_indices(packed).long()
    values = torch.gather(books, -1, indices).view(
        num_experts, n_tiles, k_tiles, LUT_B_BLOCK_N, LUT_B_BLOCK_K
    )
    weight = values.permute(0, 1, 3, 2, 4).reshape(num_experts, n, k)
    # Codebook entries times a power of two are exact in FP32.
    block_scales = torch.exp2(scales.to(torch.float32) - _UE8M0_BIAS)
    weight = weight * block_scales.repeat_interleave(LUT_B_SCALE_BLOCK_K, dim=-1)
    weight = weight.to(out_dtype)
    return weight[0] if squeeze else weight


# Rubin (SM107) kernel layout: one contiguous shared-memory image per
# (expert, 128 N, 128 K) tile. See csrc/moe/lut_b_moe_sm107.cu.
LUT_B_SM107_TILE_N = 128
LUT_B_SM107_TILE_K = 128
LUT_B_SM107_TILE_BYTES = 6144 + 512 + 512


def lut_b_to_sm107_tiles(
    packed: torch.Tensor, codebooks: torch.Tensor, scales: torch.Tensor
) -> torch.Tensor:
    """Rearrange stacked LUT-B weights into the SM107 MoE kernel layout.

    Returns `[E, N / 128, K / 128, 7168]` uint8. Each tile holds:

    * 6144 bytes of 3-bit indices, 48 bytes (K128) per output channel, in the
      no-swizzle K-major core-matrix layout (8 rows x 16 bytes),
    * 512 bytes of UE8M0 B scales: byte `q` of word `[lane][col]` is the
      scale of channel `col * 32 + lane` for K32 block `q`,
    * 512 bytes of codebooks: lane `g` holds the two eight-entry books (one
      per K64 half) of channel group `g`; lanes 16-31 are zero padding.
    """
    num_experts, n_tiles, k_tiles = packed.shape[:3]
    n, k = n_tiles * LUT_B_BLOCK_N, k_tiles * LUT_B_BLOCK_K
    if n % LUT_B_SM107_TILE_N or k % LUT_B_SM107_TILE_K:
        raise ValueError(f"SM107 LUT-B tiles need N, K divisible by 128, got {n}, {k}")
    nt, kt = n // LUT_B_SM107_TILE_N, k // LUT_B_SM107_TILE_K
    _check_scale_shape(scales, num_experts, n, k)

    # Each logical tile row is 64 indices = 24 bytes; two K64 tiles form the
    # 48-byte K128 row of one channel.
    rows = packed.view(num_experts, n_tiles, k_tiles, LUT_B_BLOCK_N, 24)
    rows = rows.permute(0, 1, 3, 2, 4).reshape(num_experts, n, kt, 48)
    rows = rows.view(num_experts, nt, 16, 8, kt, 3, 16)
    index_bytes = rows.permute(0, 1, 4, 2, 5, 3, 6).reshape(num_experts, nt, kt, 6144)

    block_scales = scales.view(num_experts, nt, 4, 32, kt, 4)
    scale_bytes = block_scales.permute(0, 1, 4, 3, 2, 5).reshape(
        num_experts, nt, kt, 512
    )

    books = codebooks.view(torch.uint8).view(num_experts, nt, 16, kt, 2, 8)
    books = books.permute(0, 1, 3, 2, 4, 5).reshape(num_experts, nt, kt, 256)
    book_bytes = torch.nn.functional.pad(books, (0, 256))

    return torch.cat((index_bytes, scale_bytes, book_bytes), dim=-1).contiguous()
