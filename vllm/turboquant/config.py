# SPDX-License-Identifier: Apache-2.0
"""TurboQuant configuration."""

import math
import os
from dataclasses import dataclass


@dataclass
class TurboQuantConfig:
    """Configuration for TurboQuant KV-cache quantization.

    Args:
        head_dim: Attention head dimension (e.g. 64, 96, 128).
        total_bits: Total bits per coordinate (3 or 4).
            MSE stage uses (total_bits - 1) bits, QJL uses 1 bit.
        value_quant_bits: Bits per value dimension for uniform quantization.
            2 = 4 levels (aggressive, ~6x compression).
            4 = 16 levels (higher quality, ~3x compression).
        hybrid: If True, store values as FP8 regardless of value_quant_bits.
            Keys use full TQ compression, values use FP8 E4M3.
            Gives ~2x compression with faster decode (no nibble dequant).
        seed: Base seed for deterministic random matrix generation.
            Actual seed per layer = seed + layer_idx * 1337.
    """
    head_dim: int = 128
    total_bits: int = 3
    value_quant_bits: int = 4  # 8 = FP8 (E4M3), 2 = 2-bit uniform, 4 = 4-bit uniform
    hybrid: bool = False  # K=TQ3 + V=FP8 mode
    seed: int = 42

    @property
    def mse_bits(self) -> int:
        return max(self.total_bits - 1, 1)

    @property
    def n_centroids(self) -> int:
        return 2 ** self.mse_bits

    @property
    def key_packed_size(self) -> int:
        """Packed bytes for a single compressed KEY vector.

        Layout:
          - MSE indices: ceil(head_dim * mse_bits / 8) bytes
          - QJL signs:   ceil(head_dim / 8) bytes
          - vec_norm:     2 bytes (float16)
          - res_norm:     2 bytes (float16)
        """
        mse_bytes = math.ceil(self.head_dim * self.mse_bits / 8)
        qjl_bytes = math.ceil(self.head_dim / 8)
        norm_bytes = 4  # 2x float16
        return mse_bytes + qjl_bytes + norm_bytes

    @property
    def effective_value_quant_bits(self) -> int:
        """Actual bits used for value storage (8 in hybrid mode)."""
        if self.hybrid:
            return 8
        return self.value_quant_bits

    @property
    def value_fp8(self) -> bool:
        """Whether values are stored as FP8 (E4M3) — no packing needed."""
        return self.effective_value_quant_bits == 8

    @property
    def value_packed_size(self) -> int:
        """Packed bytes for a single VALUE vector.

        FP8/hybrid mode: head_dim bytes (1 byte per element, no scale/zero).
        Uniform mode: ceil(head_dim * bits / 8) + 4 bytes (scale + zero fp16).
        """
        if self.value_fp8:
            return self.head_dim  # 1 byte per element, no overhead
        data_bytes = math.ceil(self.head_dim * self.value_quant_bits / 8)
        return data_bytes + 4  # +2 scale(fp16) +2 zero(fp16)

    @property
    def slot_size(self) -> int:
        """Total packed bytes per head per position (key + value combined).

        Layout: [key_packed | value_fp16]
        For tq3 head_dim=256: 100 + 512 = 612 bytes.
        """
        return self.key_packed_size + self.value_packed_size

    @property
    def padded_slot_size(self) -> int:
        """Slot size rounded up to next power of 2 for page alignment.

        Ensures that num_kv_heads * padded_slot_size produces page sizes
        with clean integer ratios for unify_kv_cache_spec_page_size.
        """
        raw = self.slot_size
        s = 1
        while s < raw:
            s <<= 1
        return s

    @property
    def packed_size(self) -> int:
        """Alias for slot_size (backward compat)."""
        return self.slot_size

    @staticmethod
    def from_cache_dtype(cache_dtype: str, head_dim: int,
                         value_quant_bits: int = 4) -> "TurboQuantConfig":
        # Allow env var override for value quantization bits
        vqb_env = os.environ.get("TQ_VALUE_BITS")
        if vqb_env is not None:
            value_quant_bits = int(vqb_env)
        # Hybrid mode: K=TQ3 + V=FP8
        hybrid = os.environ.get("TQ_HYBRID", "0") == "1"
        if cache_dtype == "tq3":
            return TurboQuantConfig(head_dim=head_dim, total_bits=3,
                                    value_quant_bits=value_quant_bits,
                                    hybrid=hybrid)
        elif cache_dtype == "tq4":
            return TurboQuantConfig(head_dim=head_dim, total_bits=4,
                                    value_quant_bits=value_quant_bits,
                                    hybrid=hybrid)
        else:
            raise ValueError(f"Unknown TurboQuant cache dtype: {cache_dtype}")
