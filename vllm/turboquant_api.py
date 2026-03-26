"""
TurboQuant Convenience API

This module provides simplified access to TurboQuant functionality
for common use cases. It acts as a wrapper for easier importing and usage.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch

from vllm.turboquant import (
    DEFAULT_TURBOQUANT_SEED,
    TurboQuantKVCache,
    TurboQuantMSEState,
    TurboQuantProdState,
    TurboQuantSplitState,
    _TurboQuantMSECodec,
    _TurboQuantProdCodec,
    _build_codec,
    _SplitCodec,
    cache_size,
    clear_cache,
    dequantize_kv_cache,
    estimate_compression,
    quantize_kv_cache,
    turboquant_enabled,
)

__all__ = [
    "TurboQuantKVCache",
    "quantize_kv_cache",
    "dequantize_kv_cache",
    "estimate_compression",
    "clear_cache",
    "cache_size",
    "turboquant_enabled",
    "get_codec",
]


def get_codec(
    keys: torch.Tensor,
    bits: float,
    mode: str = "prod",
    seed: int = DEFAULT_TURBOQUANT_SEED,
):
    """Get or create a quantization codec.
    
    Args:
        keys: Reference tensor for dimension inference
        bits: Quantization bit-width
        mode: "prod" for keys, "mse" for values
        seed: Random seed
        
    Returns:
        Quantization codec instance
    """
    return _build_codec(keys, bits, mode, seed)


if __name__ == "__main__":
    print("TurboQuant API ready for use")
    print(f"Available functions: {', '.join(__all__)}")
