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

LUT_B_BLOCK_N = 8
LUT_B_BLOCK_K = 64
LUT_B_SCALE_BLOCK_K = 32
LUT_B_CODEBOOK_SIZE = 8
LUT_B_PACKED_TILE_BYTES = LUT_B_BLOCK_N * LUT_B_BLOCK_K * 3 // 8
LUT_B_LLOYD_ITERATIONS = 16
LUT_B_SCALE_ROUNDS = 4
LUT_B_SCALE_ROUND_ITERATIONS = 4
LUT_B_MAX_TILES_PER_CHUNK = 4096

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


def _init_farthest(values: torch.Tensor) -> torch.Tensor:
    centers = values.new_empty(values.shape[0], LUT_B_CODEBOOK_SIZE)
    centers[:, 0] = values.mean(dim=1)
    distance = (values - centers[:, :1]).square()
    for i in range(1, LUT_B_CODEBOOK_SIZE):
        center = torch.gather(values, 1, distance.argmax(dim=1, keepdim=True))
        centers[:, i] = center.squeeze(1)
        distance = torch.minimum(distance, (values - center).square())
    return centers


def _init_quantile(values: torch.Tensor) -> torch.Tensor:
    ranks = (torch.arange(LUT_B_CODEBOOK_SIZE, device=values.device) + 0.5) * (
        values.shape[1] / LUT_B_CODEBOOK_SIZE
    )
    return values.sort(dim=1).values[:, ranks.long()]


def _init_uniform(values: torch.Tensor) -> torch.Tensor:
    fractions = torch.linspace(0, 1, LUT_B_CODEBOOK_SIZE, device=values.device)
    low = values.amin(dim=1, keepdim=True)
    high = values.amax(dim=1, keepdim=True)
    return low + (high - low) * fractions


def _lloyd(
    values: torch.Tensor,
    weights: torch.Tensor,
    centers: torch.Tensor,
    preserve_zero: torch.Tensor,
    num_iterations: int,
) -> torch.Tensor:
    for _ in range(num_iterations):
        indices = _assign(values, centers)
        sums = torch.zeros_like(centers).scatter_add_(1, indices, values * weights)
        counts = torch.zeros_like(centers).scatter_add_(1, indices, weights)
        centers = torch.where(counts > 0, sums / counts.clamp_min(_FP32_TINY), centers)
        centers = _normalize_centers(centers, preserve_zero)
    return centers


def _fit_tiles(
    values: torch.Tensor, weights: torch.Tensor, num_iterations: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fit one E4M3 codebook per row of `values` with weighted Lloyd iterations.

    Three initializations are tried and the lowest-error fit per tile is kept.
    Returns `(indices, centers, loss)`, where `loss` is the weighted squared
    error per tile.
    """
    preserve_zero = (values == 0).any(dim=1)
    best_loss = values.new_full((values.shape[0],), torch.inf)
    best_indices = torch.empty_like(values, dtype=torch.uint8)
    best_centers = values.new_empty(values.shape[0], LUT_B_CODEBOOK_SIZE)

    for init in (_init_farthest, _init_quantile, _init_uniform):
        centers = _normalize_centers(init(values), preserve_zero)
        centers = _lloyd(values, weights, centers, preserve_zero, num_iterations)
        indices = _assign(values, centers)
        error = (torch.gather(centers, 1, indices) - values).square()
        loss = (error * weights).sum(dim=1)
        better = loss < best_loss
        best_loss = torch.where(better, loss, best_loss)
        best_indices[better] = indices[better].to(torch.uint8)
        best_centers[better] = centers[better]

    return best_indices, best_centers, best_loss


def _ue8m0_exponent(amax: torch.Tensor) -> torch.Tensor:
    """Power of two that puts `amax` in (E4M3_MAX / 2, E4M3_MAX], as MXFP8."""
    exponent = torch.ceil(torch.log2(amax.clamp_min(_FP32_TINY) / _E4M3_MAX))
    return exponent.clamp(-_UE8M0_BIAS, _UE8M0_BIAS)


def _rescale(
    blocks: torch.Tensor, exponent: torch.Tensor, tile_exponent: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-tile values divided by their block scales, and Lloyd weights.

    Each value is weighed by its scale squared, relative to the tile scale, so
    a codebook fit minimizes the error of the rescaled weights.
    """
    values = (blocks * torch.exp2(-exponent).unsqueeze(-1)).flatten(1)
    weights = torch.exp2(2 * (exponent - tile_exponent)).unsqueeze(-1)
    return values, weights.expand_as(blocks).flatten(1)


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


def _refine_scales(
    blocks: torch.Tensor,
    exponent: torch.Tensor,
    tile_exponent: torch.Tensor,
    centers: torch.Tensor,
    num_rounds: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Coordinate descent over block exponents and the tile codebook.

    Each round moves every block exponent by -1, 0 or +1, whichever decodes
    that block best with the current codebook, then refits the codebook for
    the new exponents. Each step is kept only where it lowers the error.
    """
    preserve_zero = (blocks == 0).flatten(1).any(dim=1)
    error, _ = _block_errors(blocks, exponent, tile_exponent, centers)
    for _ in range(num_rounds):
        base = exponent
        for delta in (-1, 1):
            trial = (base + delta).clamp(-_UE8M0_BIAS, _UE8M0_BIAS)
            trial_error, _ = _block_errors(blocks, trial, tile_exponent, centers)
            better = trial_error < error
            exponent = torch.where(better, trial, exponent)
            error = torch.where(better, trial_error, error)

        values, weights = _rescale(blocks, exponent, tile_exponent)
        trial_centers = _lloyd(
            values, weights, centers, preserve_zero, LUT_B_SCALE_ROUND_ITERATIONS
        )
        trial_error, _ = _block_errors(blocks, exponent, tile_exponent, trial_centers)
        better = trial_error.sum((1, 2)) < error.sum((1, 2))
        centers = torch.where(better[:, None], trial_centers, centers)
        error = torch.where(better[:, None, None], trial_error, error)

    _, indices = _block_errors(blocks, exponent, tile_exponent, centers)
    return indices, centers, exponent


@torch.no_grad()
def quantize_lut_b(
    weight: torch.Tensor,
    *,
    num_iterations: int = LUT_B_LLOYD_ITERATIONS,
    num_scale_rounds: int = LUT_B_SCALE_ROUNDS,
    max_tiles_per_chunk: int = LUT_B_MAX_TILES_PER_CHUNK,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize a `[N, K]` or stacked `[E, N, K]` weight to LUT-B.

    Returns `(packed, codebooks, scales)` with shapes
    `[E, N/8, K/64, 192]` (uint8), `[E, N/8, K/64, 8]` (float8_e4m3fn)
    and `[E, N, K/32]` (uint8 UE8M0). The expert dim is dropped for 2D
    inputs.

    Each tile is fitted twice: once with its 16 block scales tied to one
    exponent for the whole tile, and once with each block's exponent taken
    from its own amax. Starting from the better fit, `num_scale_rounds` rounds
    of coordinate descent then adjust the block exponents and the codebook
    together. Every step only lowers the reconstruction error, so the result
    is never worse than a single scale per tile.
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
        # [tiles, 8 rows, 2 K32 blocks, 32]
        blocks = (
            rows[start:end]
            .to(torch.float32)
            .reshape(end - start, LUT_B_BLOCK_N, k_tiles, LUT_B_BLOCK_K)
            .transpose(1, 2)
            .reshape(-1, LUT_B_BLOCK_N, blocks_per_row, LUT_B_SCALE_BLOCK_K)
        )
        block_exponent = _ue8m0_exponent(blocks.abs().amax(dim=-1))
        tile_exponent = block_exponent.amax(dim=(1, 2), keepdim=True)

        best: tuple[torch.Tensor, ...] | None = None
        for exponent in (tile_exponent.expand_as(block_exponent), block_exponent):
            fit = _fit_tiles(*_rescale(blocks, exponent, tile_exponent), num_iterations)
            if best is None:
                best = (*fit, exponent.clone())
                continue
            better = fit[2] < best[2]
            for kept, new in zip(best, (*fit, exponent)):
                kept[better] = new[better]
        assert best is not None
        _, centers, _, exponent = best
        indices, centers, exponent = _refine_scales(
            blocks, exponent, tile_exponent, centers, num_scale_rounds
        )

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
