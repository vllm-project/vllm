# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
import torch.nn.functional as F

from vllm.model_executor.layers.quantization.utils.nvfp4_emulation_utils import (
    dequantize_to_dtype,
)
from vllm.model_executor.utils import replace_parameter

from .base import NvFp4LinearKernel, NvFp4LinearLayerConfig

LUT_B_BLOCK_N = 8
LUT_B_BLOCK_K = 64
LUT_B_CODEBOOK_SIZE = 8
LUT_B_PACKED_TILE_BYTES = LUT_B_BLOCK_N * LUT_B_BLOCK_K * 3 // 8
LUT_B_LLOYD_ITERATIONS = 4
LUT_B_MULTISTART_ITERATIONS = 16
LUT_B_MAX_TILES_PER_CHUNK = 4096
_LUT_B_CALIBRATION_FREE_ALGORITHMS: dict[str, tuple[bool, int]] = {
    "multistart": (False, 0),
    "scaled": (True, 0),
    "residual_1": (False, 1),
    "residual_2": (False, 2),
    "residual_4": (False, 4),
    "residual_8": (False, 8),
    "scaled_residual_1": (True, 1),
    "scaled_residual_2": (True, 2),
    "scaled_residual_4": (True, 4),
    "scaled_residual_8": (True, 8),
}


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
    values = values.clamp(min=finfo.min, max=finfo.max)
    return values.to(torch.float8_e4m3fn).to(torch.float32)


def _assign_lut_indices(values: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
    boundaries = ((centers[:, :-1] + centers[:, 1:]) * 0.5).contiguous()
    return torch.searchsorted(boundaries, values.contiguous()).to(torch.int64)


def _initialize_lut_centers(values: torch.Tensor) -> torch.Tensor:
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
        distance = (values - new_center[:, None]).square()
        minimum_distance = torch.minimum(minimum_distance, distance)
    return _snap_to_e4m3(centers).sort(dim=1).values


def _initialize_quantile_lut_centers(values: torch.Tensor) -> torch.Tensor:
    sorted_values = values.sort(dim=1).values
    ranks = (
        (torch.arange(LUT_B_CODEBOOK_SIZE, device=values.device) + 0.5)
        * values.shape[1]
        / LUT_B_CODEBOOK_SIZE
    ).to(torch.int64)
    ranks.clamp_max_(values.shape[1] - 1)
    return sorted_values[:, ranks]


def _initialize_uniform_lut_centers(values: torch.Tensor) -> torch.Tensor:
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


def _fit_lut_b_tiles_from_centers(
    values: torch.Tensor,
    centers: torch.Tensor,
    num_iterations: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    centers = _snap_to_e4m3(centers).sort(dim=1).values

    for _ in range(num_iterations):
        indices = _assign_lut_indices(values, centers)
        sums = torch.zeros_like(centers)
        counts = torch.zeros_like(centers)
        sums.scatter_add_(1, indices, values)
        counts.scatter_add_(1, indices, torch.ones_like(values))
        updated = sums / counts.clamp_min(1)
        centers = torch.where(counts > 0, updated, centers)
        centers = _snap_to_e4m3(centers).sort(dim=1).values

    indices = _assign_lut_indices(values, centers)
    return indices.to(torch.uint8), centers.to(torch.float8_e4m3fn)


def _fit_lut_b_tiles(
    values: torch.Tensor,
    num_iterations: int = LUT_B_LLOYD_ITERATIONS,
) -> tuple[torch.Tensor, torch.Tensor]:
    values = values.to(torch.float32)
    centers = _initialize_lut_centers(values)
    return _fit_lut_b_tiles_from_centers(values, centers, num_iterations)


def _fit_lut_b_tiles_multistart(
    values: torch.Tensor,
    num_iterations: int = LUT_B_MULTISTART_ITERATIONS,
) -> tuple[torch.Tensor, torch.Tensor]:
    values = values.to(torch.float32)
    initializers = (
        _initialize_lut_centers,
        _initialize_quantile_lut_centers,
        _initialize_uniform_lut_centers,
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
        indices, centers = _fit_lut_b_tiles_from_centers(
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


def _weight_to_lut_b_tiles(weight: torch.Tensor) -> torch.Tensor:
    n, k = weight.shape
    return (
        weight.reshape(
            n // LUT_B_BLOCK_N,
            LUT_B_BLOCK_N,
            k // LUT_B_BLOCK_K,
            LUT_B_BLOCK_K,
        )
        .permute(0, 2, 1, 3)
        .reshape(-1, LUT_B_BLOCK_N * LUT_B_BLOCK_K)
    )


@torch.no_grad()
def quantize_lut_b(
    weight: torch.Tensor,
    *,
    max_tiles_per_chunk: int = LUT_B_MAX_TILES_PER_CHUNK,
    num_iterations: int = LUT_B_LLOYD_ITERATIONS,
    multistart: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fit the canonical LUT-B representation for a logical ``[N, K]`` weight."""
    if weight.ndim != 2:
        raise ValueError(f"LUT-B expects a 2D weight, got shape {weight.shape}")
    n, k = weight.shape
    if n % LUT_B_BLOCK_N != 0 or k % LUT_B_BLOCK_K != 0:
        raise ValueError(
            "LUT-B requires weight dimensions divisible by "
            f"({LUT_B_BLOCK_N}, {LUT_B_BLOCK_K}), got ({n}, {k})"
        )

    n_tiles = n // LUT_B_BLOCK_N
    k_tiles = k // LUT_B_BLOCK_K
    packed = torch.empty(
        n_tiles,
        k_tiles,
        LUT_B_PACKED_TILE_BYTES,
        dtype=torch.uint8,
        device=weight.device,
    )
    codebooks = torch.empty(
        n_tiles,
        k_tiles,
        LUT_B_CODEBOOK_SIZE,
        dtype=torch.float8_e4m3fn,
        device=weight.device,
    )

    n_tiles_per_chunk = max(1, max_tiles_per_chunk // k_tiles)
    for n_tile_start in range(0, n_tiles, n_tiles_per_chunk):
        n_tile_end = min(n_tile_start + n_tiles_per_chunk, n_tiles)
        row_start = n_tile_start * LUT_B_BLOCK_N
        row_end = n_tile_end * LUT_B_BLOCK_N
        tiles = (
            weight[row_start:row_end]
            .reshape(
                n_tile_end - n_tile_start,
                LUT_B_BLOCK_N,
                k_tiles,
                LUT_B_BLOCK_K,
            )
            .permute(0, 2, 1, 3)
            .reshape(-1, LUT_B_BLOCK_N * LUT_B_BLOCK_K)
        )
        fit_tiles = _fit_lut_b_tiles_multistart if multistart else _fit_lut_b_tiles
        indices, fitted_codebooks = fit_tiles(tiles, num_iterations=num_iterations)
        packed[n_tile_start:n_tile_end] = pack_lut_b_indices(indices).reshape(
            n_tile_end - n_tile_start,
            k_tiles,
            LUT_B_PACKED_TILE_BYTES,
        )
        codebooks[n_tile_start:n_tile_end] = fitted_codebooks.reshape(
            n_tile_end - n_tile_start,
            k_tiles,
            LUT_B_CODEBOOK_SIZE,
        )

    return packed, codebooks


@torch.no_grad()
def quantize_lut_b_calibration_free(
    weight: torch.Tensor,
    *,
    algorithm: str,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
]:
    """Quantize a weight with a selectable calibration-free LUT-B algorithm."""
    try:
        use_output_scale, residual_count = _LUT_B_CALIBRATION_FREE_ALGORITHMS[algorithm]
    except KeyError:
        raise ValueError(
            f"Unknown LUT-B algorithm {algorithm!r}; "
            f"expected one of {sorted(_LUT_B_CALIBRATION_FREE_ALGORITHMS)}"
        ) from None

    output_scale = None
    normalized_weight = weight
    if use_output_scale:
        output_scale = (
            weight.to(torch.float32).square().mean(dim=1).sqrt() / 32.0
        ).clamp_min_(torch.finfo(torch.float32).tiny)
        normalized_weight = weight.to(torch.float32) / output_scale[:, None]

    packed, codebooks = quantize_lut_b(
        normalized_weight,
        num_iterations=LUT_B_MULTISTART_ITERATIONS,
        multistart=True,
    )
    if residual_count == 0:
        return packed, codebooks, output_scale, None, None

    reconstructed = dequantize_lut_b(
        packed,
        codebooks,
        out_dtype=torch.float32,
        output_scale=output_scale,
    )
    errors = _weight_to_lut_b_tiles(weight.to(torch.float32)) - (
        _weight_to_lut_b_tiles(reconstructed)
    )
    residual_position = errors.abs().topk(residual_count, dim=1).indices.to(torch.int16)
    residual_value = torch.gather(
        errors,
        1,
        residual_position.to(torch.int64),
    )
    n_tiles, k_tiles = packed.shape[:2]
    residual_shape: tuple[int, ...] = (n_tiles, k_tiles)
    if residual_count > 1:
        residual_shape = (*residual_shape, residual_count)
    return (
        packed,
        codebooks,
        output_scale,
        residual_position.reshape(residual_shape),
        residual_value.to(torch.bfloat16).reshape(residual_shape),
    )


def dequantize_lut_b(
    packed: torch.Tensor,
    codebooks: torch.Tensor,
    *,
    out_dtype: torch.dtype,
    output_scale: torch.Tensor | None = None,
    residual_position: torch.Tensor | None = None,
    residual_value: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fully reconstruct a logical ``[N, K]`` weight from LUT-B tiles."""
    if packed.ndim != 3 or packed.shape[-1] != LUT_B_PACKED_TILE_BYTES:
        raise ValueError(f"Unexpected packed LUT-B shape {packed.shape}")
    if codebooks.shape != (*packed.shape[:2], LUT_B_CODEBOOK_SIZE):
        raise ValueError(
            f"Codebook shape {codebooks.shape} does not match {packed.shape}"
        )

    n_tiles, k_tiles = packed.shape[:2]
    indices = unpack_lut_b_indices(packed).reshape(-1, LUT_B_BLOCK_N * LUT_B_BLOCK_K)
    values = torch.gather(
        codebooks.reshape(-1, LUT_B_CODEBOOK_SIZE).to(out_dtype),
        1,
        indices.to(torch.int64),
    ).reshape(
        n_tiles,
        k_tiles,
        LUT_B_BLOCK_N,
        LUT_B_BLOCK_K,
    )
    if output_scale is not None:
        if output_scale.shape != (n_tiles * LUT_B_BLOCK_N,):
            raise ValueError(
                f"Unexpected LUT-B output scale shape {output_scale.shape}"
            )
        values *= output_scale.to(out_dtype).reshape(
            n_tiles,
            1,
            LUT_B_BLOCK_N,
            1,
        )

    if (residual_position is None) != (residual_value is None):
        raise ValueError("LUT-B residual position and value must be provided together")
    if residual_position is not None and residual_value is not None:
        expected_prefix = (n_tiles, k_tiles)
        if residual_position.shape[:2] != expected_prefix:
            raise ValueError(
                f"Unexpected LUT-B residual position shape {residual_position.shape}"
            )
        if residual_value.shape != residual_position.shape:
            raise ValueError(
                f"Unexpected LUT-B residual value shape {residual_value.shape}"
            )
        if residual_position.ndim == 2:
            residual_position = residual_position.unsqueeze(-1)
            residual_value = residual_value.unsqueeze(-1)
        elif residual_position.ndim != 3:
            raise ValueError(f"Unexpected LUT-B residual rank {residual_position.ndim}")
        values = values.reshape(n_tiles, k_tiles, -1)
        values.scatter_add_(
            2,
            residual_position.to(torch.int64),
            residual_value.to(out_dtype),
        )
        values = values.reshape(
            n_tiles,
            k_tiles,
            LUT_B_BLOCK_N,
            LUT_B_BLOCK_K,
        )

    return values.permute(0, 2, 1, 3).reshape(
        n_tiles * LUT_B_BLOCK_N, k_tiles * LUT_B_BLOCK_K
    )


def dequantize_nvfp4_weight(
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_global_scale: torch.Tensor,
    logical_widths: list[int],
    *,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """Decode standardized NVFP4 tensors while preserving fused-layer scales."""
    global_scales = weight_global_scale.flatten()
    if global_scales.numel() == 1:
        global_scales = global_scales.expand(len(logical_widths))
    if global_scales.numel() != len(logical_widths):
        raise ValueError(
            f"Expected {len(logical_widths)} NVFP4 global scales, "
            f"got {global_scales.numel()}"
        )

    decoded = []
    row_start = 0
    for width, global_scale in zip(logical_widths, global_scales):
        row_end = row_start + width
        decoded.append(
            dequantize_to_dtype(
                weight[row_start:row_end],
                weight_scale[row_start:row_end],
                global_scale,
                out_dtype,
                block_size=16,
                swizzle=False,
            )
        )
        row_start = row_end
    if row_start != weight.shape[0]:
        raise ValueError(
            f"Logical widths cover {row_start} rows, but the weight has "
            f"{weight.shape[0]}"
        )
    return torch.cat(decoded, dim=0)


class LutBNvFp4LinearKernel(NvFp4LinearKernel):
    """Calibration-free LUT-B repacking with an unfused reference forward."""

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        return True, None

    @classmethod
    def can_implement(cls, config: NvFp4LinearLayerConfig) -> tuple[bool, str | None]:
        return True, None

    def supports_per_partition_weight_global_scale(self) -> bool:
        return True

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        source_weight = dequantize_nvfp4_weight(
            layer.weight,
            layer.weight_scale,
            layer.weight_global_scale,
            layer.logical_widths,
            out_dtype=torch.float32,
        )
        packed, codebooks = quantize_lut_b(source_weight)
        replace_parameter(layer, "weight", packed)
        replace_parameter(layer, "weight_codebook", codebooks)

        for name in (
            "weight_scale",
            "weight_global_scale",
            "weight_scale_2",
            "input_scale",
            "input_scale_2",
            "input_global_scale",
            "input_global_scale_inv",
            "alpha",
        ):
            if hasattr(layer, name):
                delattr(layer, name)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        weight = dequantize_lut_b(
            layer.weight,
            layer.weight_codebook,
            out_dtype=x.dtype,
        )
        return F.linear(x, weight, bias)
