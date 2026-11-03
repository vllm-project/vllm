# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import functools
from collections.abc import Callable, Iterable

import numpy as np
import torch

from vllm._custom_ops import (
    cutlass_scaled_mm_supports_fp4,
)
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils.math_utils import round_up

logger = init_logger(__name__)


def swizzle_blockscale(scale: torch.Tensor) -> torch.Tensor:
    """Pad and block-interleave the FP4 block-scales so that they match the data
    layout expected by the CUTLASS / FlashInfer kernels.

    Args:
        scale: The FP4 block-scale tensor to swizzle.

    Returns:
        The swizzled tensor with the same logical shape as `scale`.

    """
    assert scale.dtype == torch.float8_e4m3fn, (
        "swizzle_blockscale expects the input tensor to be in "
        "torch.float8_e4m3fn format."
    )

    scale_ndim = scale.ndim
    if scale_ndim == 2:
        scale = scale.unsqueeze(0)  # (1, M, K)
    assert scale.ndim == 3, "Expected a 2-D or 3-D tensor for block scales."

    B, M, K = scale.shape

    M_padded = round_up(M, 128)
    K_padded = round_up(K, 4)

    padded = torch.zeros(
        (B, M_padded, K_padded), dtype=scale.dtype, device=scale.device
    )
    padded[:B, :M, :K] = scale

    # Reshape / permute to the layout required by the kernel.
    padded = padded.reshape(B, M_padded // 128, 4, 32, K_padded // 4, 4)
    swizzled = padded.permute(0, 1, 4, 3, 2, 5).contiguous().cuda()

    if scale_ndim == 2:
        return swizzled.reshape(M_padded, K_padded)
    return swizzled.reshape(B, M_padded, K_padded)


def cutlass_fp4_supported() -> bool:
    if not current_platform.is_cuda():
        return False
    capability_tuple = current_platform.get_device_capability()
    capability = -1 if capability_tuple is None else capability_tuple.to_int()
    return cutlass_scaled_mm_supports_fp4(capability)


def pad_nvfp4_weight_for_cutlass(
    weight: torch.Tensor,
    alignment: int = 32,
) -> tuple[torch.Tensor, int]:
    """Pad packed NVFP4 weights so that both N (rows) and K (columns) satisfy
    the alignment constraints required by CUTLASS / FlashInfer FP4 kernels.

    CUTLASS FP4 kernel requires both K and N matrix dimensions to be divisible
    by 32 for aligned memory access and efficient tensor core operations.
    """
    weight_current_rows = weight.shape[0]

    # Pad N dimension (rows) if not aligned
    if weight_current_rows % alignment != 0:
        total_rows = round_up(weight_current_rows, alignment)
        pad_rows = total_rows - weight_current_rows
        weight = torch.nn.functional.pad(weight, (0, 0, 0, pad_rows)).contiguous()

    # Check K dimension alignment
    # 2 FP4 items are packed per byte in the input dimension
    weight_current_col_bytes = weight.shape[1]
    weight_current_col_elements = weight_current_col_bytes * 2

    weights_padding_bytes = 0
    if weight_current_col_elements % alignment != 0:
        total_cols = round_up(weight_current_col_elements, alignment)
        pad_cols = total_cols - weight_current_col_elements
        # Convert from FP4 element count to bytes (2 FP4 values per byte)
        # pad_cols is always even since alignment=32 and current elements are even
        pad_bytes = pad_cols // 2
        weight = torch.nn.functional.pad(weight, (0, pad_bytes, 0, 0)).contiguous()
        weights_padding_bytes = pad_bytes

    return weight, weights_padding_bytes


def nvfp4_weight_padding_bytes(layer: torch.nn.Module) -> int:
    """Compute K-dim padding (packed bytes) of an NVFP4 weight,
    Aligned with `pad_nvfp4_weight_for_cutlass` above.
    """
    return layer.weight.shape[1] - layer.input_size_per_partition // 2


def pad_nvfp4_activation_for_cutlass(
    x_fp4: torch.Tensor,
    weights_padding_bytes: int,
) -> torch.Tensor:
    """Pad packed FP4 activations to match the K-dimension padding applied to weights.
    The padding is in bytes (tensor dimension), not FP4 elements.
    """
    if weights_padding_bytes > 0:
        return torch.nn.functional.pad(x_fp4, (0, weights_padding_bytes)).contiguous()
    return x_fp4


def slice_nvfp4_output(
    out: torch.Tensor,
    output_size: int,
) -> torch.Tensor:
    """Slice the output tensor to remove padding in N dimension if weight was padded."""
    if out.shape[-1] != output_size:
        return out[..., :output_size].contiguous()
    return out


# Decoders for CSF-compressed NVFP4 block scales.
#
# ModelOpt checkpoints declare ``weight_scale_encoding`` per layer and store
# each E4M3 block-scale matrix as several ``<name>.<stream>`` tensors. Every
# decoder rebuilds the exact scale bytes of the uncompressed checkpoint.
#
# - ``csf``: ``nvfp4_csf_fixed`` (uint8, ``[rows // 16, 16 * (1 + cols // 2)]``)
#   holds, per 16-row slab, one base byte per row followed by the rows' 4-bit
#   codes, two per byte, low nibble first; a scale byte is ``base + code``.
#   ``nvfp4_csf_exceptions`` (uint32) holds sorted words whose low 24 bits are
#   a flat ``row * cols + col`` position and whose high 8 bits replace that
#   byte.
# - ``csf_32x2_v1``: ``csf32_bases`` (uint8, one per row) and ``csf32_codes``
#   (uint8, 4-bit codes) cover 32-row x 2-column blocks, grouped into 128 x 4
#   tiles, 32 tiles to a page. ``csf32_exceptions`` (uint16) holds the in-block
#   position in the low byte and the scale byte in the high byte;
#   ``csf32_page_bases`` (uint32, byte offset of each page's first exception)
#   and ``csf32_offsets`` (uint16, per page a 0 then each block's exception
#   end) index them.

# numpy keeps these small ops off torch's intra-op thread pool, whose per-op
# sync dominates when weight loading runs with many threads.


def _decode_csf(fixed: torch.Tensor, exceptions: torch.Tensor) -> np.ndarray:
    if fixed.dtype != torch.uint8 or fixed.dim() != 2 or fixed.shape[1] % 16:
        raise ValueError(f"Invalid CSF fixed stream {fixed.dtype} {fixed.shape}")
    if exceptions.dtype != torch.uint32:
        raise ValueError(f"Invalid CSF exceptions dtype {exceptions.dtype}")
    rows = fixed.shape[0] * 16
    cols = (fixed.shape[1] // 16 - 1) * 2
    if rows * cols > 1 << 24:
        raise ValueError("CSF scale exceeds 24-bit exception positions")
    raw = fixed.numpy()
    bases = raw[:, :16].reshape(rows, 1)
    if (bases > 240).any():
        raise ValueError("CSF row bases must be in 0..240")
    packed = raw[:, 16:].reshape(rows, cols // 2)
    codes = np.stack((packed & 0xF, packed >> 4), axis=-1).reshape(rows, cols)
    out = (bases + codes).reshape(-1)
    words = exceptions.reshape(-1).numpy()
    positions = words & 0xFFFFFF
    if positions.size and (
        positions[-1] >= rows * cols or (positions[1:] <= positions[:-1]).any()
    ):
        raise ValueError("CSF exception positions must be increasing and in range")
    out[positions] = words >> 24
    return out.reshape(rows, cols)


@functools.cache
def _csf32_blocks(rows: int, cols: int):
    """Origin, width and page of each 32x2 block in storage order, and the flat
    ``[rows, cols]`` position of each code."""
    tile_r, tile_c, pair, block = np.meshgrid(
        np.arange(-(-rows // 128)),
        np.arange(-(-cols // 4)),
        np.arange(2),
        np.arange(4),
        indexing="ij",
    )
    r0 = (128 * tile_r + 32 * block).ravel()
    c0 = (4 * tile_c + 2 * pair).ravel()
    page = (tile_r * tile_c.shape[1] + tile_c).ravel() // 32
    keep = (r0 < rows) & (c0 < cols)
    r0, c0, page = r0[keep], c0[keep], page[keep]
    width = np.minimum(cols - c0, 2)
    i, j = np.meshgrid(np.arange(32), np.arange(2), indexing="ij")
    cells = (r0[:, None, None] + i) * cols + c0[:, None, None] + j
    return r0, c0, width, page, cells[j < width[:, None, None]]


def _decode_csf_32x2(
    bases: torch.Tensor,
    codes: torch.Tensor,
    page_bases: torch.Tensor,
    offsets: torch.Tensor,
    exceptions: torch.Tensor,
) -> np.ndarray:
    dtypes = (torch.uint8, torch.uint8, torch.uint32, torch.uint16, torch.uint16)
    streams = (bases, codes, page_bases, offsets, exceptions)
    rows = bases.numel()
    cols = 2 * codes.numel() // max(rows, 1)
    if (
        any(t.dtype != d for t, d in zip(streams, dtypes))
        or rows % 32
        or rows * cols != 2 * codes.numel()
        or cols == 0
    ):
        raise ValueError("Invalid csf_32x2_v1 streams")
    r0, c0, width, page, cells = _csf32_blocks(rows, cols)
    num_pages = int(page[-1]) + 1
    base, packed, origins, index, words = (t.reshape(-1).numpy() for t in streams)
    if origins.size != num_pages or index.size != r0.size + num_pages:
        raise ValueError("Invalid csf_32x2_v1 index sizes")
    if (base > 240).any():
        raise ValueError("CSF row bases must be in 0..240")
    out = np.empty(rows * cols, dtype=np.uint8)
    out[cells] = np.stack((packed & 0xF, packed >> 4), -1).ravel() + base[cells // cols]

    first = np.flatnonzero(np.diff(page, prepend=-1))
    ends = origins[page].astype(np.int64) // 2 + index[np.arange(r0.size) + page + 1]
    starts = np.concatenate(([0], ends[:-1]))
    if (
        (index[first + np.arange(num_pages)] != 0).any()
        or (origins != 2 * starts[first]).any()
        or (ends < starts).any()
        or (ends - starts > 32 * width).any()
        or ends[-1] != words.size
    ):
        raise ValueError("Invalid csf_32x2_v1 exception index")
    block = np.repeat(np.arange(r0.size), ends - starts)
    pos = (words & 0xFF).astype(np.int64)
    if (
        (pos >= 64).any()
        or (pos % 2 >= width[block]).any()
        or ((pos[1:] <= pos[:-1]) & (block[1:] == block[:-1])).any()
    ):
        raise ValueError("CSF exception positions must be increasing and in range")
    out[(r0[block] + pos // 2) * cols + c0[block] + pos % 2] = words >> 8
    return out.reshape(rows, cols)


# weight_scale_encoding -> (stream suffixes in decoder argument order, decoder)
CSF_SCALE_ENCODINGS: dict[str, tuple[tuple[str, ...], Callable[..., np.ndarray]]] = {
    "csf": (("nvfp4_csf_fixed", "nvfp4_csf_exceptions"), _decode_csf),
    "csf_32x2_v1": (
        tuple(
            f"csf32_{name}"
            for name in ("bases", "codes", "page_bases", "offsets", "exceptions")
        ),
        _decode_csf_32x2,
    ),
}


def decode_csf_scale(encoding: str, streams: dict[str, torch.Tensor]) -> torch.Tensor:
    """Rebuild one ``[rows, cols]`` float8_e4m3fn block-scale matrix."""
    names, decode = CSF_SCALE_ENCODINGS[encoding]
    out = decode(*(streams[name].cpu() for name in names))
    return torch.from_numpy(out).view(torch.float8_e4m3fn)


def csf_weights_transform(
    encodings: Iterable[str],
) -> Callable[[Iterable[tuple[str, torch.Tensor]]], Iterable[tuple[str, torch.Tensor]]]:
    """Pair CSF scale streams back into canonical E4M3 scale tensors.

    The returned transform is lazy over a ``(name, tensor)`` weight stream:
    tensors named ``<name>.<stream>`` for a declared encoding are buffered per
    scale and, once the encoding's stream set completes, replaced by the
    decoded ``<name>`` tensor; everything else passes through unchanged.
    """
    suffixes = {
        f".{stream}": (encoding, stream)
        for encoding in encodings
        for stream in CSF_SCALE_ENCODINGS[encoding][0]
    }

    def transform(
        weights: Iterable[tuple[str, torch.Tensor]],
    ) -> Iterable[tuple[str, torch.Tensor]]:
        pending: dict[str, tuple[str, dict[str, torch.Tensor]]] = {}
        decoded = 0
        for name, tensor in weights:
            for suffix, (encoding, stream) in suffixes.items():
                if name.endswith(suffix):
                    base = name[: -len(suffix)]
                    break
            else:
                yield name, tensor
                continue
            if base in pending and pending[base][0] != encoding:
                raise ValueError(f"Mixed CSF scale encodings for {base}")
            parts = pending.setdefault(base, (encoding, {}))[1]
            parts[stream] = tensor
            if len(parts) == len(CSF_SCALE_ENCODINGS[encoding][0]):
                del pending[base]
                yield base, decode_csf_scale(encoding, parts)
                decoded += 1
        if pending:
            raise ValueError(
                f"CSF scale streams without a partner: {sorted(pending)[:4]}"
            )
        if decoded:
            logger.info("Decoded %d CSF scale tensors at load", decoded)

    return transform
