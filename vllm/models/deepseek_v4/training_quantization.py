"""Thin BF16-master adapters over vLLM's block-FP8 deployment path."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn

try:
    import triton
    import triton.language as tl
except ImportError:  # CPU-only import/test environments
    triton = tl = None

BLOCK_SHAPE = (128, 128)


@dataclass(frozen=True)
class CanonicalBlockFP8Weight:
    qweight: torch.Tensor
    scales: torch.Tensor
    block_shape: tuple[int, int] = BLOCK_SHAPE


@dataclass(frozen=True)
class PackedBlockFP8Weight:
    qweight: torch.Tensor
    scales: torch.Tensor
    cache_key: object | None = None


@dataclass(frozen=True)
class PackedBlockFP8Activation:
    qactivation: torch.Tensor
    scales: torch.Tensor


PackedGroupedBlockFP8Weight = PackedBlockFP8Weight


def _entry(module: str, name: str):
    try:
        return getattr(importlib.import_module(module), name)
    except (ImportError, AttributeError) as error:
        raise RuntimeError(f"vLLM deployment entry point {module}.{name} is unavailable") from error


def _key(weight: nn.Parameter):
    return (id(weight), weight._version, weight.device, weight.dtype, tuple(weight.shape))


def _validate_weight(weight: torch.Tensor) -> None:
    if weight.dtype != torch.bfloat16 or weight.ndim != 2:
        raise TypeError("block-FP8 master weight must be a 2-D BF16 tensor")
    if any(size % block for size, block in zip(weight.shape, BLOCK_SHAPE, strict=True)):
        raise ValueError(f"weight shape {tuple(weight.shape)} must be divisible by {BLOCK_SHAPE}")


if triton is not None:

    @triton.jit
    def _requantize(master, scales, output, columns, scale_columns, elements, BLOCK: tl.constexpr):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < elements
        rows = offsets // columns
        columns_in_row = offsets - rows * columns
        scale_offsets = (rows // 128) * scale_columns + columns_in_row // 128
        values = tl.load(master + offsets, mask=mask).to(tl.float32)
        tl.store(output + offsets, values / tl.load(scales + scale_offsets, mask=mask), mask=mask)


def requantize_block_fp8_weight(weight: torch.Tensor, scales: torch.Tensor):
    """Recover release FP8 bytes using the release scales, without rescaling."""
    _validate_weight(weight)
    expected = (weight.shape[0] // 128, weight.shape[1] // 128)
    if scales.dtype != torch.float32 or tuple(scales.shape) != expected or scales.device != weight.device:
        raise ValueError(f"fixed scales must be float32 {expected} on the weight device")
    if not bool(torch.all(torch.isfinite(scales) & (scales > 0))):
        raise ValueError("fixed scales must be finite and positive")
    with torch.inference_mode():
        if weight.is_cuda and triton is not None:
            qweight = torch.empty_like(weight, dtype=torch.float8_e4m3fn)
            _requantize[(triton.cdiv(weight.numel(), 256),)](
                weight, scales.contiguous(), qweight, weight.shape[1], scales.shape[1], weight.numel(), BLOCK=256
            )
        else:
            expanded = scales.repeat_interleave(128, 0).repeat_interleave(128, 1)
            qweight = (weight.float() / expanded).to(torch.float8_e4m3fn)
    return CanonicalBlockFP8Weight(qweight, scales)


def quantize_block_fp8_weight(weight: torch.Tensor):
    _validate_weight(weight)
    scales = getattr(weight, "_fp8_source_scales", None)
    if scales is not None and getattr(weight, "_fp8_source_scale_version", None) == weight._version:
        return requantize_block_fp8_weight(weight, scales)
    with torch.inference_mode():
        qweight, scales = _entry("vllm.utils.deep_gemm", "per_block_cast_to_fp8")(
            weight.detach(), block_size=list(BLOCK_SHAPE), use_ue8m0=False
        )
    return CanonicalBlockFP8Weight(qweight, scales)


def _post_process(qweight, scales):
    with torch.inference_mode():
        return _entry(
            "vllm.model_executor.layers.quantization.utils.fp8_utils",
            "deepgemm_post_process_fp8_weight_block",
        )(wq=qweight, ws=scales, quant_block_shape=BLOCK_SHAPE, use_e8m0=True)


def pack_block_fp8_weight(weight: nn.Parameter):
    canonical = quantize_block_fp8_weight(weight)
    qweight, scales = _post_process(canonical.qweight, canonical.scales)
    return PackedBlockFP8Weight(qweight, scales, _key(weight))


def pack_grouped_block_fp8_weight(weights: Iterable[nn.Parameter]):
    weights = tuple(weights)
    if not weights:
        raise ValueError("grouped block-FP8 packing requires at least one expert")
    canonical = tuple(quantize_block_fp8_weight(weight) for weight in weights)
    qweight, scales = _post_process(
        torch.stack([item.qweight for item in canonical]),
        torch.stack([item.scales for item in canonical]),
    )
    return PackedBlockFP8Weight(qweight, scales, tuple(_key(weight) for weight in weights))


def pack_block_fp8_activation(x: torch.Tensor):
    if x.dtype != torch.bfloat16 or x.ndim != 2 or x.shape[1] % 128 or x.stride(-1) != 1:
        raise ValueError("block-FP8 activation must be contiguous 2-D BF16 with K divisible by 128")
    oracle = _entry("vllm.utils.deep_gemm", "DeepGemmQuantScaleFMT").from_oracle()
    packed = getattr(oracle, "name", "") == "UE8M0"
    quantize = _entry(
        "vllm.model_executor.layers.quantization.utils.fp8_utils",
        "per_token_group_quant_fp8_packed_for_deepgemm" if packed else "per_token_group_quant_fp8",
    )
    with torch.inference_mode():
        if packed:
            qactivation, scales = quantize(x, 128, use_ue8m0=True)
        else:
            tma = bool(_entry("vllm.envs", "VLLM_USE_DEEP_GEMM_TMA_ALIGNED_SCALES"))
            qactivation, scales = quantize(
                x, 128, use_ue8m0=True, column_major_scales=True, tma_aligned_scales=tma
            )
    return PackedBlockFP8Activation(qactivation, scales)


def fp8_gemm_nt(x: torch.Tensor, weight: PackedBlockFP8Weight):
    if x.ndim != 2 or x.shape[1] != weight.qweight.shape[-1]:
        raise ValueError("activation K does not match packed weight K")
    activation = pack_block_fp8_activation(x)
    output = torch.empty((x.shape[0], weight.qweight.shape[-2]), dtype=torch.bfloat16, device=x.device)
    with torch.inference_mode():
        _entry("vllm.utils.deep_gemm", "fp8_gemm_nt")(
            (activation.qactivation, activation.scales),
            (weight.qweight, weight.scales),
            output,
            is_deep_gemm_e8m0_used=True,
        )
    return output


class DeploymentBlockFP8Adapter:
    def __init__(self, *, cache_weight: bool = False):
        self.cache_weight = cache_weight
        self._cached_weight = None

    def clear_cache(self):
        self._cached_weight = None

    def pack_weight(self, weight: nn.Parameter):
        key = _key(weight)
        if self.cache_weight and self._cached_weight is not None and self._cached_weight.cache_key == key:
            return self._cached_weight
        packed = pack_block_fp8_weight(weight)
        if self.cache_weight:
            self._cached_weight = packed
        return packed

    def __call__(self, x, weight):
        return fp8_gemm_nt(x, self.pack_weight(weight))


class DeploymentFusedBlockFP8Adapter(DeploymentBlockFP8Adapter):
    def pack_weight(self, weights):
        weights = tuple(weights)
        key = tuple(_key(weight) for weight in weights)
        if self.cache_weight and self._cached_weight is not None and self._cached_weight.cache_key == key:
            return self._cached_weight
        canonical = tuple(quantize_block_fp8_weight(weight) for weight in weights)
        qweight, scales = _post_process(
            torch.cat([item.qweight for item in canonical]),
            torch.cat([item.scales for item in canonical]),
        )
        packed = PackedBlockFP8Weight(qweight, scales, key)
        if self.cache_weight:
            self._cached_weight = packed
        return packed

    def __call__(self, x, *weights):
        return fp8_gemm_nt(x, self.pack_weight(weights))


__all__ = [
    "BLOCK_SHAPE", "CanonicalBlockFP8Weight", "DeploymentBlockFP8Adapter",
    "DeploymentFusedBlockFP8Adapter", "PackedBlockFP8Activation",
    "PackedBlockFP8Weight", "PackedGroupedBlockFP8Weight", "fp8_gemm_nt",
    "pack_block_fp8_activation", "pack_block_fp8_weight",
    "pack_grouped_block_fp8_weight", "quantize_block_fp8_weight",
    "requantize_block_fp8_weight",
]
