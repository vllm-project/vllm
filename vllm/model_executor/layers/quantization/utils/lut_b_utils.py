# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Logical LUT-B packing and calibration-free codebook fitting."""

import torch

LUT_B_BLOCK_N = 8
LUT_B_BLOCK_K = 64
LUT_B_CODEBOOK_SIZE = 8
LUT_B_PACKED_TILE_BYTES = LUT_B_BLOCK_N * LUT_B_BLOCK_K * 3 // 8
LUT_B_LLOYD_ITERATIONS = 16
LUT_B_MAX_TILES_PER_CHUNK = 4096


def pack_lut_b_indices(indices: torch.Tensor) -> torch.Tensor:
    """Pack eight 3-bit LUT indices into three bytes."""
    if indices.shape[-1] % 8 != 0:
        raise ValueError("The number of LUT indices must be divisible by 8")

    index_groups = indices.reshape(*indices.shape[:-1], -1, 8).to(torch.int32)
    words = torch.zeros(
        index_groups.shape[:-1],
        dtype=torch.int32,
        device=indices.device,
    )
    for index in range(8):
        words |= index_groups[..., index] << (3 * index)

    return (
        torch.stack(
            (
                words & 0xFF,
                (words >> 8) & 0xFF,
                (words >> 16) & 0xFF,
            ),
            dim=-1,
        )
        .to(torch.uint8)
        .flatten(start_dim=-2)
    )


def unpack_lut_b_indices(packed: torch.Tensor) -> torch.Tensor:
    """Unpack three-byte groups into eight 3-bit LUT indices."""
    if packed.shape[-1] % 3 != 0:
        raise ValueError("The number of packed bytes must be divisible by 3")

    byte_groups = packed.reshape(*packed.shape[:-1], -1, 3).to(torch.int32)
    words = (
        byte_groups[..., 0] | (byte_groups[..., 1] << 8) | (byte_groups[..., 2] << 16)
    )
    return (
        torch.stack(
            tuple((words >> (3 * index)) & 0x7 for index in range(8)),
            dim=-1,
        )
        .to(torch.uint8)
        .flatten(start_dim=-2)
    )


def _snap_to_e4m3(values: torch.Tensor) -> torch.Tensor:
    finfo = torch.finfo(torch.float8_e4m3fn)
    return values.clamp(min=finfo.min, max=finfo.max).to(torch.float8_e4m3fn)


def _assign_lut_indices(values: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
    boundaries = ((centers[:, :-1] + centers[:, 1:]) * 0.5).contiguous()
    return torch.searchsorted(boundaries, values.contiguous()).to(torch.int64)


def _initialize_farthest_centers(values: torch.Tensor) -> torch.Tensor:
    centers = torch.empty(
        values.shape[0],
        LUT_B_CODEBOOK_SIZE,
        dtype=torch.float32,
        device=values.device,
    )
    centers[:, 0] = values.mean(dim=1)
    minimum_distance = (values - centers[:, :1]).square()
    for center_index in range(1, LUT_B_CODEBOOK_SIZE):
        farthest = minimum_distance.argmax(dim=1, keepdim=True)
        new_center = torch.gather(values, 1, farthest).squeeze(1)
        centers[:, center_index] = new_center
        minimum_distance = torch.minimum(
            minimum_distance,
            (values - new_center[:, None]).square(),
        )
    return centers


def _initialize_quantile_centers(values: torch.Tensor) -> torch.Tensor:
    sorted_values = values.sort(dim=1).values
    ranks = (
        (torch.arange(LUT_B_CODEBOOK_SIZE, device=values.device) + 0.5)
        * values.shape[1]
        / LUT_B_CODEBOOK_SIZE
    ).to(torch.int64)
    return sorted_values[:, ranks.clamp_max(values.shape[1] - 1)]


def _initialize_uniform_centers(values: torch.Tensor) -> torch.Tensor:
    fractions = torch.linspace(
        0,
        1,
        LUT_B_CODEBOOK_SIZE,
        dtype=torch.float32,
        device=values.device,
    )
    minimum = values.amin(dim=1, keepdim=True)
    maximum = values.amax(dim=1, keepdim=True)
    return minimum + (maximum - minimum) * fractions


def _fit_from_centers(
    values: torch.Tensor,
    centers: torch.Tensor,
    num_iterations: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    centers = _snap_to_e4m3(centers).to(torch.float32).sort(dim=1).values
    for _ in range(num_iterations):
        indices = _assign_lut_indices(values, centers)
        sums = torch.zeros_like(centers)
        counts = torch.zeros_like(centers)
        sums.scatter_add_(1, indices, values)
        counts.scatter_add_(1, indices, torch.ones_like(values))
        updated = sums / counts.clamp_min(1)
        centers = torch.where(counts > 0, updated, centers)
        centers = _snap_to_e4m3(centers).to(torch.float32).sort(dim=1).values

    indices = _assign_lut_indices(values, centers)
    return indices.to(torch.uint8), centers.to(torch.float8_e4m3fn)


def _fit_lut_b_tiles(
    values: torch.Tensor,
    num_iterations: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    values = values.to(torch.float32)
    initializers = (
        _initialize_farthest_centers,
        _initialize_quantile_centers,
        _initialize_uniform_centers,
    )
    best_loss = torch.full(
        (values.shape[0],),
        torch.inf,
        dtype=torch.float32,
        device=values.device,
    )
    best_indices = torch.empty_like(values, dtype=torch.uint8)
    best_centers = torch.empty(
        values.shape[0],
        LUT_B_CODEBOOK_SIZE,
        dtype=torch.float8_e4m3fn,
        device=values.device,
    )

    for initializer in initializers:
        indices, centers = _fit_from_centers(
            values,
            initializer(values),
            num_iterations,
        )
        reconstructed = torch.gather(
            centers.to(torch.float32),
            1,
            indices.to(torch.int64),
        )
        loss = (reconstructed - values).square().sum(dim=1)
        selected = loss < best_loss
        best_loss[selected] = loss[selected]
        best_indices[selected] = indices[selected]
        best_centers[selected] = centers[selected]

    return best_indices, best_centers


@torch.no_grad()
def quantize_lut_b(
    weight: torch.Tensor,
    *,
    max_tiles_per_chunk: int = LUT_B_MAX_TILES_PER_CHUNK,
    num_iterations: int = LUT_B_LLOYD_ITERATIONS,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fit LUT-B tiles for a ``[N, K]`` or stacked ``[E, N, K]`` weight."""
    squeeze_expert = weight.ndim == 2
    if squeeze_expert:
        weight = weight.unsqueeze(0)
    if weight.ndim != 3:
        raise ValueError(f"LUT-B expects a 2D or 3D weight, got {weight.shape}")

    num_experts, n, k = weight.shape
    if n % LUT_B_BLOCK_N != 0 or k % LUT_B_BLOCK_K != 0:
        raise ValueError(
            "LUT-B requires weight dimensions divisible by "
            f"({LUT_B_BLOCK_N}, {LUT_B_BLOCK_K}), got ({n}, {k})"
        )

    n_tiles = n // LUT_B_BLOCK_N
    k_tiles = k // LUT_B_BLOCK_K
    packed = torch.empty(
        num_experts,
        n_tiles,
        k_tiles,
        LUT_B_PACKED_TILE_BYTES,
        dtype=torch.uint8,
        device=weight.device,
    )
    codebooks = torch.empty(
        num_experts,
        n_tiles,
        k_tiles,
        LUT_B_CODEBOOK_SIZE,
        dtype=torch.float8_e4m3fn,
        device=weight.device,
    )

    outer_rows = weight.reshape(num_experts * n_tiles, LUT_B_BLOCK_N, k)
    packed_rows = packed.reshape(num_experts * n_tiles, k_tiles, -1)
    codebook_rows = codebooks.reshape(num_experts * n_tiles, k_tiles, -1)
    rows_per_chunk = max(1, max_tiles_per_chunk // k_tiles)
    for start in range(0, outer_rows.shape[0], rows_per_chunk):
        end = min(start + rows_per_chunk, outer_rows.shape[0])
        tiles = (
            outer_rows[start:end]
            .reshape(end - start, LUT_B_BLOCK_N, k_tiles, LUT_B_BLOCK_K)
            .permute(0, 2, 1, 3)
            .reshape(-1, LUT_B_BLOCK_N * LUT_B_BLOCK_K)
        )
        indices, fitted_codebooks = _fit_lut_b_tiles(tiles, num_iterations)
        packed_rows[start:end] = pack_lut_b_indices(indices).reshape(
            end - start,
            k_tiles,
            LUT_B_PACKED_TILE_BYTES,
        )
        codebook_rows[start:end] = fitted_codebooks.reshape(
            end - start,
            k_tiles,
            LUT_B_CODEBOOK_SIZE,
        )

    if squeeze_expert:
        return packed.squeeze(0), codebooks.squeeze(0)
    return packed, codebooks


def dequantize_lut_b(
    packed: torch.Tensor,
    codebooks: torch.Tensor,
    *,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """Reconstruct a ``[N, K]`` or stacked ``[E, N, K]`` LUT-B weight."""
    squeeze_expert = packed.ndim == 3
    if squeeze_expert:
        packed = packed.unsqueeze(0)
        codebooks = codebooks.unsqueeze(0)
    if packed.ndim != 4 or packed.shape[-1] != LUT_B_PACKED_TILE_BYTES:
        raise ValueError(f"Unexpected packed LUT-B shape {packed.shape}")
    if codebooks.shape != (*packed.shape[:3], LUT_B_CODEBOOK_SIZE):
        raise ValueError(
            f"Codebook shape {codebooks.shape} does not match {packed.shape}"
        )

    num_experts, n_tiles, k_tiles = packed.shape[:3]
    indices = unpack_lut_b_indices(packed).reshape(-1, LUT_B_BLOCK_N * LUT_B_BLOCK_K)
    values = torch.gather(
        codebooks.reshape(-1, LUT_B_CODEBOOK_SIZE).to(out_dtype),
        1,
        indices.to(torch.int64),
    ).reshape(
        num_experts,
        n_tiles,
        k_tiles,
        LUT_B_BLOCK_N,
        LUT_B_BLOCK_K,
    )
    weight = values.permute(0, 1, 3, 2, 4).reshape(
        num_experts,
        n_tiles * LUT_B_BLOCK_N,
        k_tiles * LUT_B_BLOCK_K,
    )
    return weight.squeeze(0) if squeeze_expert else weight
