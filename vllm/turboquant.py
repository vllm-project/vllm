"""
TurboQuant: Efficient KV Cache Quantization for Large Language Models

This module provides PyTorch-native implementations of TurboQuant quantization
techniques for efficient KV cache compression. Supports integer and fractional
bit-widths for flexible accuracy/compression trade-offs.
"""
from __future__ import annotations

import math
from functools import lru_cache
from typing import Dict, NamedTuple, Optional, Tuple

import numpy as np
import torch

__version__ = "1.0.0"
__all__ = [
    "TurboQuantKVCache",
    "TurboQuantMSEState",
    "TurboQuantProdState",
    "TurboQuantPolarState",
    "TurboQuantPolarProdState",
    "TurboQuantSplitState",
    "turboquant_enabled",
    "_TurboQuantMSECodec",
    "_TurboQuantProdCodec",
    "_SplitCodec",
    "_build_codec",
    "DEFAULT_TURBOQUANT_SEED",
]

DEFAULT_TURBOQUANT_SEED = 0
_EPS = 1e-6
_POLAR_MAX_LEVELS = 4

# Global cache for codec instances to avoid recomputation
_CODEC_CACHE: Dict[Tuple[int, int, str, int], object] = {}


@lru_cache(maxsize=None)
def _rotation_matrix(dim: int, seed: int) -> torch.Tensor:
    """Generate a random orthonormal rotation matrix."""
    if dim <= 0:
        return torch.zeros((0, 0), dtype=torch.float32)
    if dim == 1:
        return torch.ones((1, 1), dtype=torch.float32)

    rng = np.random.default_rng(seed + dim * 7919)
    matrix = rng.standard_normal((dim, dim), dtype=np.float32)
    q, r = np.linalg.qr(matrix)
    q *= np.sign(np.diag(r))
    return torch.tensor(q.astype(np.float32))


@lru_cache(maxsize=None)
def _projection_matrix(dim: int, seed: int) -> torch.Tensor:
    """Generate a random projection matrix."""
    if dim <= 0:
        return torch.zeros((0, 0), dtype=torch.float32)
    rng = np.random.default_rng(seed + dim * 2971 + 17)
    matrix = rng.standard_normal((dim, dim), dtype=np.float32)
    return torch.tensor(matrix.astype(np.float32))


def _beta_pdf(grid: np.ndarray, dim: int) -> np.ndarray:
    """Beta probability density function for dimension dim."""
    if dim <= 1:
        pdf = np.ones_like(grid)
    else:
        coeff = math.gamma(dim / 2) / (
            math.sqrt(math.pi) * math.gamma((dim - 1) / 2)
        )
        pdf = coeff * np.power(np.clip(1.0 - grid**2, 0.0, None), (dim - 3) / 2)
    pdf_sum = pdf.sum()
    if pdf_sum == 0:
        return np.full_like(grid, 1.0 / len(grid))
    return pdf / pdf_sum


@lru_cache(maxsize=None)
def _codebook(dim: int, bits: int) -> torch.Tensor:
    """Generate codebook centroids using k-means with beta distribution weighting."""
    if bits <= 0:
        return torch.zeros((0,), dtype=torch.float32)
    levels = 1 << bits
    if dim <= 1:
        centroids = np.linspace(-1.0, 1.0, levels, dtype=np.float32)
        return torch.tensor(centroids)

    grid = np.linspace(-1.0 + 1e-6, 1.0 - 1e-6, 32768, dtype=np.float32)
    weights = _beta_pdf(grid, dim)
    cdf = np.cumsum(weights)
    quantiles = (np.arange(levels, dtype=np.float32) + 0.5) / levels
    centroids = np.interp(quantiles, cdf, grid).astype(np.float32)

    for _ in range(100):
        boundaries = np.empty(levels + 1, dtype=np.float32)
        boundaries[0] = -1.0
        boundaries[-1] = 1.0
        boundaries[1:-1] = 0.5 * (centroids[:-1] + centroids[1:])
        new_centroids = centroids.copy()
        for i in range(levels):
            if i == levels - 1:
                mask = (grid >= boundaries[i]) & (grid <= boundaries[i + 1])
            else:
                mask = (grid >= boundaries[i]) & (grid < boundaries[i + 1])
            bucket_weights = weights[mask]
            if bucket_weights.size == 0:
                continue
            total_weight = bucket_weights.sum()
            if total_weight > 0:
                new_centroids[i] = np.sum(bucket_weights * grid[mask]) / total_weight
        if np.max(np.abs(new_centroids - centroids)) < 1e-6:
            centroids = new_centroids
            break
        centroids = new_centroids

    return torch.tensor(centroids.astype(np.float32))


def _polar_angle_pdf(grid: np.ndarray, level: int) -> np.ndarray:
    """Probability density function for polar angles."""
    if level <= 1:
        pdf = np.ones_like(grid)
    else:
        exponent = (1 << (level - 1)) - 1
        pdf = np.power(np.clip(np.sin(2.0 * grid), 0.0, None), exponent)
    pdf_sum = pdf.sum()
    if pdf_sum == 0:
        return np.full_like(grid, 1.0 / len(grid))
    return pdf / pdf_sum


@lru_cache(maxsize=None)
def _polar_angle_codebook(level: int, bits: int) -> torch.Tensor:
    """Generate codebook for polar angles."""
    if bits <= 0:
        return torch.zeros((0,), dtype=torch.float32)

    level_count = 1 << bits
    if level <= 1:
        step = (2.0 * math.pi) / level_count
        centroids = np.arange(level_count, dtype=np.float32) * step + step / 2.0
        return torch.tensor(centroids.astype(np.float32))

    grid = np.linspace(1e-6, math.pi / 2 - 1e-6, 32768, dtype=np.float32)
    weights = _polar_angle_pdf(grid, level)
    cdf = np.cumsum(weights)
    quantiles = (np.arange(level_count, dtype=np.float32) + 0.5) / level_count
    centroids = np.interp(quantiles, cdf, grid).astype(np.float32)

    for _ in range(100):
        boundaries = np.empty(level_count + 1, dtype=np.float32)
        boundaries[0] = 0.0
        boundaries[-1] = math.pi / 2
        boundaries[1:-1] = 0.5 * (centroids[:-1] + centroids[1:])
        new_centroids = centroids.copy()
        for i in range(level_count):
            if i == level_count - 1:
                mask = (grid >= boundaries[i]) & (grid <= boundaries[i + 1])
            else:
                mask = (grid >= boundaries[i]) & (grid < boundaries[i + 1])
            bucket_weights = weights[mask]
            if bucket_weights.size == 0:
                continue
            total_weight = bucket_weights.sum()
            if total_weight > 0:
                new_centroids[i] = np.sum(bucket_weights * grid[mask]) / total_weight
        if np.max(np.abs(new_centroids - centroids)) < 1e-6:
            centroids = new_centroids
            break
        centroids = new_centroids

    return torch.tensor(centroids.astype(np.float32))


def _packed_width(length: int, bits: int) -> int:
    """Calculate the number of 32-bit words needed to store packed bits."""
    if length == 0 or bits == 0:
        return 0
    return (length * bits + 31) // 32


def _pack_lowbit(values: torch.Tensor, bits: int) -> torch.Tensor:
    """Pack values into low-bit integers."""
    if bits == 0:
        return torch.zeros((*values.shape[:-1], 0), dtype=torch.uint32)

    values = values.to(torch.uint32)
    length = values.shape[-1]
    packed_width = _packed_width(length, bits)
    
    if values.ndim == 1:
        values = values.unsqueeze(0)
    
    original_shape = values.shape
    flat = values.reshape((-1, length))
    packed = torch.zeros((flat.shape[0], packed_width), dtype=torch.uint32, device=values.device)

    for idx in range(length):
        bit_offset = idx * bits
        word_idx = bit_offset // 32
        offset = bit_offset % 32
        packed[:, word_idx] |= (flat[:, idx] & ((1 << bits) - 1)) << offset
        spill = offset + bits - 32
        if spill > 0:
            packed[:, word_idx + 1] |= (flat[:, idx] >> (bits - spill)) & ((1 << spill) - 1)

    return packed.reshape((*original_shape[:-1], packed_width))


def _unpack_lowbit(packed: torch.Tensor, bits: int, length: int) -> torch.Tensor:
    """Unpack low-bit integers from packed representation."""
    if bits == 0:
        return torch.zeros((*packed.shape[:-1], 0), dtype=torch.uint32)

    packed = packed.to(torch.uint32)
    flat = packed.reshape((-1, packed.shape[-1]))
    unpacked = torch.zeros((flat.shape[0], length), dtype=torch.uint32, device=packed.device)
    mask = (1 << bits) - 1

    for idx in range(length):
        bit_offset = idx * bits
        word_idx = bit_offset // 32
        offset = bit_offset % 32
        value = (flat[:, word_idx] >> offset) & mask
        spill = offset + bits - 32
        if spill > 0:
            value |= ((flat[:, word_idx + 1] & ((1 << spill) - 1)) << (bits - spill))
        unpacked[:, idx] = value

    return unpacked.reshape((*packed.shape[:-1], length))


def _validate_bits(bits: float) -> float:
    """Validate and normalize bit-width specification."""
    bits = float(bits)
    if bits < 1:
        raise ValueError("TurboQuant requires kv_bits >= 1.")
    rounded = round(bits * 2) / 2
    if not math.isclose(bits, rounded, abs_tol=1e-6):
        raise ValueError(
            f"TurboQuant currently supports integer and .5 bit-widths, got {bits}."
        )
    return rounded


def turboquant_enabled(bits: Optional[float], scheme: Optional[str] = None) -> bool:
    """Check if TurboQuant should be enabled."""
    if bits is None:
        return False
    if scheme == "turboquant":
        return True
    bits = float(bits)
    return not math.isclose(bits, round(bits), abs_tol=1e-6)


def _is_power_of_two(value: int) -> bool:
    """Check if value is a power of two."""
    return value > 0 and (value & (value - 1)) == 0


def _polar_levels(dim: int) -> int:
    """Calculate number of polar decomposition levels."""
    if dim <= 1:
        return 0
    return min(_POLAR_MAX_LEVELS, int(math.log2(dim)))


def _polar_level_bits(dim: int, bits: int) -> tuple[int, ...]:
    """Get bit allocation for each polar level."""
    if bits != 4:
        raise ValueError(f"PolarQuant key codec currently expects 4 bits, got {bits}.")
    levels = _polar_levels(dim)
    if levels == 0:
        return ()
    return (4,) + (2,) * (levels - 1)


class TurboQuantMSEState(NamedTuple):
    """State for MSE-based quantization."""
    norms: torch.Tensor
    indices: torch.Tensor


class TurboQuantProdState(NamedTuple):
    """State for product quantization."""
    norms: torch.Tensor
    mse_indices: torch.Tensor
    residual_norms: torch.Tensor
    qjl_signs: torch.Tensor


class TurboQuantPolarState(NamedTuple):
    """State for polar quantization."""
    radii: torch.Tensor
    level_indices: tuple[torch.Tensor, ...]


class TurboQuantPolarProdState(NamedTuple):
    """State for polar product quantization."""
    norms: torch.Tensor
    polar_state: TurboQuantPolarState
    residual_norms: torch.Tensor
    qjl_signs: torch.Tensor


class TurboQuantSplitState(NamedTuple):
    """State for split (mixed-precision) quantization."""
    low: object
    high: object


class _TurboQuantMSECodec:
    """Mean Squared Error based quantization codec."""
    
    def __init__(self, dim: int, bits: int, seed: int):
        self.dim = dim
        self.bits = bits
        self.rotation = _rotation_matrix(dim, seed)
        self.rotation_t = self.rotation.T if dim > 0 else self.rotation
        self.codebook = _codebook(dim, bits)

    def _quantize_unit_with_estimate(
        self, unit_vectors: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Quantize unit vectors and return estimate."""
        if self.bits == 0:
            return (
                torch.zeros((*unit_vectors.shape[:-1], 0), dtype=torch.uint32),
                torch.zeros(unit_vectors.shape, dtype=torch.float32),
            )

        rotated = torch.matmul(unit_vectors, self.rotation_t)
        distances = torch.abs(rotated.unsqueeze(-1) - self.codebook)
        indices = torch.argmin(distances, dim=-1).to(torch.uint32)
        packed = _pack_lowbit(indices, self.bits)
        estimated_rotated = self.codebook[indices]
        return packed, torch.matmul(estimated_rotated, self.rotation)

    def _quantize_unit(self, unit_vectors: torch.Tensor) -> torch.Tensor:
        """Quantize unit vectors."""
        packed, _ = self._quantize_unit_with_estimate(unit_vectors)
        return packed

    def _dequantize_unit(self, packed_indices: torch.Tensor) -> torch.Tensor:
        """Dequantize unit vectors."""
        if self.bits == 0:
            return torch.zeros((*packed_indices.shape[:-1], self.dim), dtype=torch.float32)

        indices = _unpack_lowbit(packed_indices, self.bits, self.dim).to(torch.int32)
        rotated = self.codebook[indices]
        return torch.matmul(rotated, self.rotation)

    def quantize(self, vectors: torch.Tensor) -> TurboQuantMSEState:
        """Quantize vectors."""
        vectors_f32 = vectors.to(torch.float32)
        norms = torch.linalg.norm(vectors_f32, dim=-1)
        safe_norms = torch.maximum(norms.unsqueeze(-1), torch.tensor(_EPS, device=vectors.device))
        unit_vectors = torch.where(
            norms.unsqueeze(-1) > 0,
            vectors_f32 / safe_norms,
            torch.zeros(vectors.shape, dtype=torch.float32, device=vectors.device),
        )
        return TurboQuantMSEState(
            norms.to(vectors.dtype),
            self._quantize_unit(unit_vectors),
        )

    def dequantize(self, state: TurboQuantMSEState) -> torch.Tensor:
        """Dequantize vectors."""
        unit_vectors = self._dequantize_unit(state.indices)
        return state.norms.unsqueeze(-1).to(unit_vectors.dtype) * unit_vectors


class _TurboQuantProdCodec:
    """Product quantization codec combining MSE and projection-based components."""
    
    def __init__(self, dim: int, bits: int, seed: int):
        self.dim = dim
        self.bits = bits
        self.mse_codec = _TurboQuantMSECodec(dim, max(bits - 1, 0), seed)
        self.projection = _projection_matrix(dim, seed + 1)
        self.projection_t = (
            self.projection.T if dim > 0 else self.projection
        )
        self.query_transform_t = (
            torch.cat([self.mse_codec.rotation_t, self.projection_t], dim=-1)
            if dim > 0
            else torch.zeros((0, 0), dtype=torch.float32)
        )
        self.scale = math.sqrt(math.pi / 2) / dim if dim > 0 else 0.0
        self.scale_array = torch.tensor([self.scale], dtype=torch.float32)

    def quantize(self, vectors: torch.Tensor) -> TurboQuantProdState:
        """Quantize vectors using product quantization."""
        vectors_f32 = vectors.to(torch.float32)
        norms = torch.linalg.norm(vectors_f32, dim=-1)
        safe_norms = torch.maximum(norms.unsqueeze(-1), torch.tensor(_EPS, device=vectors.device))
        unit_vectors = torch.where(
            norms.unsqueeze(-1) > 0,
            vectors_f32 / safe_norms,
            torch.zeros(vectors.shape, dtype=torch.float32, device=vectors.device),
        )

        mse_indices, mse_unit = self.mse_codec._quantize_unit_with_estimate(
            unit_vectors
        )
        residual = unit_vectors - mse_unit
        residual_norms = torch.linalg.norm(residual, dim=-1)
        projected = torch.matmul(residual, self.projection_t)
        signs = torch.where(projected >= 0, 1, 0).to(torch.uint32)

        return TurboQuantProdState(
            norms.to(vectors.dtype),
            mse_indices,
            residual_norms.to(vectors.dtype),
            _pack_lowbit(signs, 1),
        )

    def dequantize(self, state: TurboQuantProdState) -> torch.Tensor:
        """Dequantize vectors using product quantization."""
        mse_unit = self.mse_codec._dequantize_unit(state.mse_indices)
        sign_bits = _unpack_lowbit(state.qjl_signs, 1, self.dim).to(torch.float32)
        signs = sign_bits * 2.0 - 1.0
        qjl_unit = self.scale * state.residual_norms.unsqueeze(-1).to(
            torch.float32
        ) * torch.matmul(signs, self.projection)
        return state.norms.unsqueeze(-1).to(torch.float32) * (mse_unit + qjl_unit)


def _select_outlier_indices(
    tensor: torch.Tensor, avg_bits: float
) -> tuple[np.ndarray, np.ndarray]:
    """Select outlier(high-magnitude) dimensions for mixed-precision quantization."""
    lower_bits = math.floor(avg_bits)
    upper_bits = math.ceil(avg_bits)
    if lower_bits == upper_bits:
        raise ValueError("Mixed-precision selection requires a fractional bit-width.")

    dim = tensor.shape[-1]
    high_count = int(round((avg_bits - lower_bits) * dim / (upper_bits - lower_bits)))
    high_count = max(1, min(dim - 1, high_count))

    scores = torch.mean(torch.abs(tensor.to(torch.float32)), dim=(0, 1, 2) if tensor.ndim == 4 else tuple(range(tensor.ndim - 1)))
    order = np.argsort(scores.cpu().numpy())
    high_idx = np.sort(order[-high_count:].astype(np.int32))
    low_mask = np.ones(dim, dtype=bool)
    low_mask[high_idx] = False
    low_idx = np.nonzero(low_mask)[0].astype(np.int32)
    return low_idx, high_idx


class _SplitCodec:
    """Mixed-precision codec splitting dimensions between low and high bit-widths."""
    
    def __init__(self, tensor: torch.Tensor, bits: float, mode: str, seed: int):
        self.bits = bits
        self.mode = mode
        self.dim = tensor.shape[-1]
        self.lower_bits = math.floor(bits)
        self.upper_bits = math.ceil(bits)
        low_idx, high_idx = _select_outlier_indices(tensor, bits)
        self.low_idx = torch.tensor(low_idx, dtype=torch.int32)
        self.high_idx = torch.tensor(high_idx, dtype=torch.int32)

        concat_order = np.concatenate([low_idx, high_idx])
        self.restore_order = torch.tensor(np.argsort(concat_order), dtype=torch.int32)

        codec_cls = _TurboQuantProdCodec if mode == "prod" else _TurboQuantMSECodec
        self.low_codec = codec_cls(len(low_idx), self.lower_bits, seed)
        self.high_codec = codec_cls(len(high_idx), self.upper_bits, seed + 97)

    def quantize(self, tensor: torch.Tensor) -> TurboQuantSplitState:
        """Quantize using split codec."""
        low_tensor = torch.index_select(tensor, -1, self.low_idx)
        high_tensor = torch.index_select(tensor, -1, self.high_idx)
        return TurboQuantSplitState(
            self.low_codec.quantize(low_tensor),
            self.high_codec.quantize(high_tensor),
        )

    def dequantize(self, state: TurboQuantSplitState) -> torch.Tensor:
        """Dequantize using split codec."""
        low_tensor = self.low_codec.dequantize(state.low)
        high_tensor = self.high_codec.dequantize(state.high)
        merged = torch.cat([low_tensor, high_tensor], dim=-1)
        return torch.index_select(merged, -1, self.restore_order)


def _build_codec(tensor: torch.Tensor, bits: float, mode: str, seed: int):
    """Build appropriate codec for given bit-width with caching.
    
    Args:
        tensor: Input tensor to infer dimension from
        bits: Quantization bit-width (integer or fractional)
        mode: Quantization mode ("prod" or "mse")
        seed: Random seed for reproducibility
        
    Returns:
        Codec instance (_TurboQuantMSECodec, _TurboQuantProdCodec, or _SplitCodec)
    """
    bits = _validate_bits(bits)
    dim = tensor.shape[-1]
    
    # Create cache key (without tensor reference to allow caching)
    cache_key = (dim, int(bits * 10), mode, seed)
    
    # Check cache for existing codec
    if cache_key in _CODEC_CACHE:
        return _CODEC_CACHE[cache_key]
    
    # Build new codec
    if math.isclose(bits, round(bits), abs_tol=1e-6):
        codec_cls = _TurboQuantProdCodec if mode == "prod" else _TurboQuantMSECodec
        codec = codec_cls(dim, int(round(bits)), seed)
    else:
        codec = _SplitCodec(tensor, bits, mode, seed)
    
    # Cache the codec
    _CODEC_CACHE[cache_key] = codec
    return codec


class TurboQuantKVCache:
    """KV cache with TurboQuant compression."""
    
    def __init__(self, bits: float, seed: int = DEFAULT_TURBOQUANT_SEED):
        self.bits = _validate_bits(bits)
        self.seed = seed
        self.offset = 0
        self.keys = None
        self.values = None
        self.key_codec = None
        self.value_codec = None

    def _ensure_codecs(self, keys: torch.Tensor, values: torch.Tensor):
        """Lazily initialize codecs."""
        if self.key_codec is None:
            self.key_codec = _build_codec(keys, self.bits, mode="prod", seed=self.seed)
        if self.value_codec is None:
            self.value_codec = _build_codec(
                values, self.bits, mode="mse", seed=self.seed + 1
            )

    def update_and_fetch(self, keys: torch.Tensor, values: torch.Tensor):
        """Update cache with new key-value pairs."""
        self._ensure_codecs(keys, values)
        new_keys = self.key_codec.quantize(keys)
        new_values = self.value_codec.quantize(values)

        # Simplified state management - in production, would need proper buffering
        self.keys = new_keys
        self.values = new_values
        self.offset += keys.shape[2] if keys.ndim == 4 else 1

        return self.keys, self.values

    def dequantize(
        self, keys_state: Optional[object] = None, values_state: Optional[object] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Dequantize keys and values."""
        if keys_state is None or values_state is None:
            keys_state, values_state = self.keys, self.values
        
        keys = self.key_codec.dequantize(keys_state).to(torch.float32)
        values = self.value_codec.dequantize(values_state).to(torch.float32)
        return keys, values

    @property
    def state(self) -> tuple[object, object]:
        """Get current state."""
        return self.keys, self.values

    @state.setter
    def state(self, value: tuple[object, object]):
        """Set current state."""
        self.keys, self.values = value

    @property
    def nbytes(self) -> int:
        """Estimate memory usage of compressed cache."""
        nbytes = 0
        if self.keys is not None:
            if isinstance(self.keys, torch.Tensor):
                nbytes += self.keys.nbytes
            if isinstance(self.values, torch.Tensor):
                nbytes += self.values.nbytes
        return nbytes


# ============================================================================
# Convenience Functions for Easy Usage
# ============================================================================

def clear_cache() -> int:
    """Clear the codec cache to free memory.
    
    Returns:
        Number of cached codecs cleared
    """
    global _CODEC_CACHE
    num_cleared = len(_CODEC_CACHE)
    _CODEC_CACHE.clear()
    return num_cleared


def cache_size() -> int:
    """Get the number of cached codec instances.
    
    Returns:
        Number of cached codecs
    """
    return len(_CODEC_CACHE)


def quantize_kv_cache(
    keys: torch.Tensor, 
    values: torch.Tensor, 
    bits: float = 3.5,
    seed: int = DEFAULT_TURBOQUANT_SEED
) -> Tuple[TurboQuantKVCache, Tuple[object, object]]:
    """Convenience function to quantize KV cache in one call.
    
    Args:
        keys: Key tensor (B, H, T, D)
        values: Value tensor (B, H, T, D)
        bits: Quantization bit-width
        seed: Random seed
        
    Returns:
        Tuple of (cache, (keys_state, values_state))
    """
    cache = TurboQuantKVCache(bits=bits, seed=seed)
    keys_state, values_state = cache.update_and_fetch(keys, values)
    return cache, (keys_state, values_state)


def dequantize_kv_cache(
    cache: TurboQuantKVCache
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convenience function to dequantize KV cache.
    
    Args:
        cache: TurboQuantKVCache instance
        
    Returns:
        Tuple of (keys, values)
    """
    return cache.dequantize()


def estimate_compression(
    original_size_mb: float,
    bits: float
) -> float:
    """Estimate compression ratio for given bit-width.
    
    Args:
        original_size_mb: Original size in MB (float32)
        bits: Quantization bit-width
        
    Returns:
        Estimated final size in MB
    """
    # float32 = 32 bits per value
    byte_ratio = bits / 32.0
    return original_size_mb * byte_ratio


# ============================================================================
# Module Initialization
# ============================================================================

def _init_module() -> None:
    """Initialize TurboQuant module with optimizations."""
    # Pre-warm up LRU caches with common values
    for dim in [64, 128, 256, 512, 1024, 2048, 4096]:
        for bits in [1, 2, 3, 4]:
            try:
                _rotation_matrix(dim, 0)
                _projection_matrix(dim, 0)
                _codebook(dim, bits)
            except Exception:
                pass  # Silently skip errors


# Initialize on module load
try:
    _init_module()
except Exception:
    pass  # Module still functional even if init fails

