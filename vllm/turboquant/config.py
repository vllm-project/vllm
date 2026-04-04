# SPDX-License-Identifier: Apache-2.0
"""TurboQuant configuration."""

import math
from dataclasses import dataclass


# Named TQ presets: each maps to frozen config parameters.
# These are the 4 validated configs with quality benchmarks.
TQ_PRESETS: dict[str, dict] = {
    "tq-k8v4": {
        "key_quant_bits": 8,  # FP8 keys
        "total_bits": 4,
        "value_quant_bits": 4,
        "norm_correction": False,
    },
    "tq-t4nc": {
        "key_quant_bits": 0,  # 4-bit MSE keys (16 centroids)
        "total_bits": 4,
        "value_quant_bits": 4,
        "norm_correction": True,
    },
    "tq-k3v4nc": {
        "key_quant_bits": 0,  # 3-bit MSE keys (8 centroids)
        "total_bits": 3,
        "value_quant_bits": 4,
        "norm_correction": True,
    },
    "tq-t3nc": {
        "key_quant_bits": 0,  # 3-bit MSE keys (8 centroids)
        "total_bits": 3,
        "value_quant_bits": 3,
        "norm_correction": True,
    },
}


@dataclass
class TurboQuantConfig:
    """Configuration for TurboQuant KV-cache quantization.

    Uses PolarQuant (WHT rotation + Lloyd-Max scalar quantization) for keys
    and uniform quantization for values. QJL is intentionally omitted —
    community consensus (5+ independent groups) found it hurts attention
    quality by amplifying variance through softmax.

    Named presets (use via --kv-cache-dtype):
        tq-k8v4:   FP8 keys + 4-bit values, 2.6x compression, +1.17% PPL
        tq-t4nc:   4-bit MSE keys + 4-bit values + NC, 3.8x, +2.71% PPL
        tq-k3v4nc: 3-bit MSE keys + 4-bit values + NC, ~3.5x, +10.63% PPL
        tq-t3nc:   3-bit MSE keys + 3-bit values + NC, 4.9x, +20.59% PPL

    Args:
        head_dim: Attention head dimension (e.g. 64, 96, 128).
        total_bits: Bits per coordinate for key MSE quantization (3 or 4).
        key_quant_bits: Override bits for key quantization.
            0 = use total_bits (default). 8 = FP8 keys (hybrid mode).
        value_quant_bits: Bits per value dimension for uniform quantization.
            3 = 8 levels, 4 = 16 levels (default).
        seed: Base seed for deterministic random matrix generation.
            Actual seed per layer = seed + layer_idx * 1337.
        norm_correction: Re-normalize centroid vectors to unit norm before
            inverse rotation during dequant. Fixes quantization-induced norm
            distortion, improving PPL by ~0.8% at 4-bit.
    """
    head_dim: int = 128
    total_bits: int = 3
    key_quant_bits: int = 0  # 0 = use total_bits (default), 8 = FP8 keys
    value_quant_bits: int = 4  # 4 = 4-bit uniform, 8 = FP8 (E4M3)
    seed: int = 42
    norm_correction: bool = False

    @property
    def mse_bits(self) -> int:
        """MSE bits per coordinate — always equals total_bits."""
        return self.total_bits

    @property
    def effective_key_quant_bits(self) -> int:
        """Actual bits used for key storage."""
        if self.key_quant_bits == 8:
            return 8
        if self.key_quant_bits > 0:
            return self.key_quant_bits
        return self.mse_bits

    @property
    def key_fp8(self) -> bool:
        """Whether keys are stored as FP8 — no rotation/quantization needed."""
        return self.effective_key_quant_bits == 8

    @property
    def key_mse_bits(self) -> int:
        """MSE bits actually used for key quantization."""
        if self.key_fp8:
            return 0
        return self.effective_key_quant_bits

    @property
    def n_centroids(self) -> int:
        return 2 ** self.mse_bits

    @property
    def key_packed_size(self) -> int:
        """Packed bytes for a single KEY vector.

        FP8 mode (key_quant_bits=8):
          head_dim bytes (1 byte per element, no overhead).

        TQ mode:
          - MSE indices: ceil(head_dim * key_mse_bits / 8) bytes
          - vec_norm:     2 bytes (float16)
          - res_norm:     2 bytes (float16)
        """
        if self.key_fp8:
            return self.head_dim  # 1 byte per element
        mse_bytes = math.ceil(self.head_dim * self.key_mse_bits / 8)
        norm_bytes = 4  # 2x float16
        return mse_bytes + norm_bytes

    @property
    def effective_value_quant_bits(self) -> int:
        """Actual bits used for value storage."""
        return self.value_quant_bits

    @property
    def value_packed_size(self) -> int:
        """Packed bytes for a single VALUE vector.

        Uniform quantization: ceil(head_dim * bits / 8) + 4 bytes (scale + zero fp16).
        """
        data_bytes = math.ceil(self.head_dim * self.value_quant_bits / 8)
        return data_bytes + 4  # +2 scale(fp16) +2 zero(fp16)

    @property
    def slot_size(self) -> int:
        """Total packed bytes per head per position (key + value combined).

        Layout: [key_packed | value_packed]
        """
        return self.key_packed_size + self.value_packed_size

    @property
    def padded_slot_size(self) -> int:
        """Slot size rounded up to next power of 2.

        Power-of-2 is required for hybrid attention+mamba models (e.g.
        Qwen3.5) where page sizes must align across attention and mamba
        layers.  Also satisfies the even-number requirement for
        effective_head_size = padded_slot_size // 2.
        """
        raw = self.slot_size
        return 1 << (raw - 1).bit_length()  # next power of 2

    @staticmethod
    def get_boundary_skip_layers(num_layers: int, n: int) -> list[str]:
        """Get layer indices to skip TQ compression (boundary protection).

        Returns first N and last N layer indices as strings, suitable for
        kv_cache_dtype_skip_layers.
        """
        if n <= 0 or num_layers <= 0:
            return []
        n = min(n, num_layers // 2)  # don't skip more than half
        first = list(range(n))
        last = list(range(num_layers - n, num_layers))
        # Deduplicate (if num_layers <= 2*n)
        indices = sorted(set(first + last))
        return [str(i) for i in indices]

    @staticmethod
    def from_cache_dtype(cache_dtype: str,
                         head_dim: int) -> "TurboQuantConfig":
        """Create config from a named preset.

        Valid presets: tq-k8v4, tq-t4nc, tq-k3v4nc, tq-t3nc.
        """
        if cache_dtype not in TQ_PRESETS:
            valid = ", ".join(TQ_PRESETS.keys())
            raise ValueError(
                f"Unknown TurboQuant cache dtype: {cache_dtype!r}. "
                f"Valid presets: {valid}")
        preset = TQ_PRESETS[cache_dtype]
        return TurboQuantConfig(
            head_dim=head_dim,
            total_bits=preset["total_bits"],
            key_quant_bits=preset["key_quant_bits"],
            value_quant_bits=preset["value_quant_bits"],
            norm_correction=preset["norm_correction"],
        )
