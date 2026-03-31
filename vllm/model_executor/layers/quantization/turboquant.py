# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
TurboQuant KV cache quantization.

Implements PolarQuant polar coordinate decomposition and QJL (Quantized
Johnson-Lindenstrauss) sign-bit residual correction for overhead-free 2-bit
and 3-bit KV cache quantization.

Reference: Zandieh et al., "TurboQuant: Redefining AI Efficiency with
Extreme Compression", ICLR 2026 (arXiv:2504.19874)

Supported KV cache dtypes:
  - pq4: PolarQuant 4-bit angles, no QJL (fastest encode, ~4x compression)
  - tq3: PolarQuant 3-bit + 1-bit QJL residual (best quality/compression, ~4x)
  - tq2: PolarQuant 2-bit + 1-bit QJL residual (maximum compression, ~5.3x)

Usage:
  vllm serve <model> --kv-cache-dtype tq3
"""

from dataclasses import dataclass

from vllm.logger import init_logger

logger = init_logger(__name__)

# TurboQuant dtypes recognized by the system
TURBOQUANT_DTYPES = ("tq2", "tq3", "pq4")


def is_turboquant_kv_cache(kv_cache_dtype: str) -> bool:
    """Check if the KV cache dtype is a TurboQuant type."""
    return kv_cache_dtype in TURBOQUANT_DTYPES


@dataclass
class TurboQuantConfig:
    """Configuration for TurboQuant KV cache quantization.

    This is not a weight quantization config (not a QuantizationConfig subclass)
    — TurboQuant operates purely on the KV cache at inference time with no
    model weight changes and no calibration data.
    """

    kv_cache_dtype: str
    """One of 'tq2', 'tq3', 'pq4'."""

    qjl_residual: bool
    """Whether QJL residual correction is enabled (True for tq2/tq3)."""

    angle_bits: int
    """Number of bits for quantized angles (2, 3, or 4)."""

    qjl_projection_dim: int | None
    """Number of QJL sign-bit projections. Defaults to head_dim."""

    rotation_seed: int
    """Base seed for per-layer Hadamard rotation. Derived per-head."""

    @classmethod
    def from_kv_cache_dtype(
        cls,
        kv_cache_dtype: str,
        rotation_seed: int = 42,
        qjl_projection_dim: int | None = None,
    ) -> "TurboQuantConfig":
        """Create config from a KV cache dtype string."""
        if kv_cache_dtype == "pq4":
            return cls(
                kv_cache_dtype=kv_cache_dtype,
                qjl_residual=False,
                angle_bits=4,
                qjl_projection_dim=None,
                rotation_seed=rotation_seed,
            )
        elif kv_cache_dtype == "tq3":
            return cls(
                kv_cache_dtype=kv_cache_dtype,
                qjl_residual=True,
                angle_bits=3,
                qjl_projection_dim=qjl_projection_dim,
                rotation_seed=rotation_seed,
            )
        elif kv_cache_dtype == "tq2":
            return cls(
                kv_cache_dtype=kv_cache_dtype,
                qjl_residual=True,
                angle_bits=2,
                qjl_projection_dim=qjl_projection_dim,
                rotation_seed=rotation_seed,
            )
        else:
            raise ValueError(
                f"Unknown TurboQuant dtype: {kv_cache_dtype}. "
                f"Supported: {TURBOQUANT_DTYPES}"
            )

    def effective_bits_per_element(self, head_size: int) -> float:
        """Calculate effective bits per KV element including all overhead.

        For a head of dimension d:
          - angles: (d-1) * angle_bits
          - radius: 16 bits (fp16)
          - QJL: d bits (1 bit per projection, proj_dim defaults to d)
          - Total per head per token: (d-1)*angle_bits + 16 + d*qjl
          - Per element: total / d
        """
        d = head_size
        qjl_dim = self.qjl_projection_dim or d
        angle_total = (d - 1) * self.angle_bits
        radius_total = 16  # fp16
        qjl_total = qjl_dim if self.qjl_residual else 0
        total_bits = angle_total + radius_total + qjl_total
        return total_bits / d

    def _padded_angle_bytes(self, head_size: int) -> int:
        """Angle bytes padded to next even number for fp16 alignment."""
        raw = ((head_size - 1) * self.angle_bits + 7) // 8
        return (raw + 1) & ~1  # Round up to even

    def bytes_per_token_per_head(self, head_size: int) -> int:
        """Calculate storage bytes per token per KV head."""
        d = head_size
        qjl_dim = self.qjl_projection_dim or d
        angle_bytes = self._padded_angle_bytes(d)
        radius_bytes = 2  # fp16
        qjl_bytes = (qjl_dim + 7) // 8 if self.qjl_residual else 0
        residual_norm_bytes = 2 if self.qjl_residual else 0  # fp16
        return angle_bytes + radius_bytes + qjl_bytes + residual_norm_bytes

    def block_bytes(
        self, num_kv_heads: int, head_size: int, block_size: int
    ) -> int:
        """Calculate total bytes per KV cache block."""
        return (
            num_kv_heads
            * block_size
            * self.bytes_per_token_per_head(head_size)
        )

    def derive_layer_seed(self, layer_idx: int) -> int:
        """Derive a per-layer rotation seed from the base seed."""
        # Use golden ratio hash for good distribution
        return self.rotation_seed ^ (layer_idx * 2654435761 & 0xFFFFFFFF)
