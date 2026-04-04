# SPDX-License-Identifier: Apache-2.0
"""TurboQuant configuration."""

import math
import os
from dataclasses import dataclass


@dataclass
class TurboQuantConfig:
    """Configuration for TurboQuant KV-cache quantization.

    Uses PolarQuant (WHT rotation + Lloyd-Max scalar quantization) for keys
    and uniform quantization for values. QJL is intentionally omitted —
    community consensus (5+ independent groups) found it hurts attention
    quality by amplifying variance through softmax.

    Args:
        head_dim: Attention head dimension (e.g. 64, 96, 128).
        total_bits: Bits per coordinate for key MSE quantization (3 or 4).
            tq3 = 3 bits (8 centroids, ~4.3x compression).
            tq4 = 4 bits (16 centroids, ~3.8x compression).
        key_quant_bits: Override bits for key quantization.
            0 = use total_bits (default). 8 = FP8 keys (hybrid mode).
        value_quant_bits: Bits per value dimension for uniform quantization.
            4 = 16 levels (default, good quality).
            8 = FP8 (E4M3), no packing needed.
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
    def value_fp8(self) -> bool:
        """Whether values are stored as FP8 (E4M3) — no packing needed."""
        return self.effective_value_quant_bits == 8

    @property
    def value_packed_size(self) -> int:
        """Packed bytes for a single VALUE vector.

        FP8 mode: head_dim bytes (1 byte per element, no scale/zero).
        Uniform mode: ceil(head_dim * bits / 8) + 4 bytes (scale + zero fp16).
        """
        if self.value_fp8:
            return self.head_dim  # 1 byte per element, no overhead
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

    @property
    def packed_size(self) -> int:
        """Alias for slot_size (backward compat)."""
        return self.slot_size

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
    def from_cache_dtype(cache_dtype: str, head_dim: int,
                         value_quant_bits: int = 4) -> "TurboQuantConfig":
        # Allow env var overrides for power users
        vqb_env = os.environ.get("TQ_VALUE_BITS")
        if vqb_env is not None:
            value_quant_bits = int(vqb_env)

        kqb_env = os.environ.get("TQ_KEY_BITS")
        key_quant_bits = int(kqb_env) if kqb_env is not None else 0

        norm_correction = os.environ.get(
            "TQ_NORM_CORRECTION", "1") == "1"

        if cache_dtype == "tq3":
            return TurboQuantConfig(head_dim=head_dim, total_bits=3,
                                    key_quant_bits=key_quant_bits,
                                    value_quant_bits=value_quant_bits,
                                    norm_correction=norm_correction)
        elif cache_dtype == "tq4":
            return TurboQuantConfig(head_dim=head_dim, total_bits=4,
                                    key_quant_bits=key_quant_bits,
                                    value_quant_bits=value_quant_bits,
                                    norm_correction=norm_correction)
        else:
            raise ValueError(f"Unknown TurboQuant cache dtype: {cache_dtype}")
