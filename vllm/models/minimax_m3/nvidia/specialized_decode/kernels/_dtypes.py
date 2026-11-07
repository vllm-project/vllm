# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cache-format checks shared by every stage that reads or writes the KV
cache or the indexer K cache.

The kernels are parametrized by a `CacheFormat` (a compile-time
parameter of the CuTe DSL kernels). Only `FP8_E4M3` has device code today; `BF16`
is a declared format with no kernel instantiation yet, so selecting it is
refused with a clear message instead of silently reinterpreting bytes.

NVFP4 caches are rejected explicitly: an nvfp4 cache is stored as uint8 with
element size 1, the same as fp8, so a size-only check would accept it and the
kernels would read 4-bit codes as e4m3.
"""

from __future__ import annotations

import enum


class CacheFormat(enum.Enum):
    FP8_E4M3 = "fp8_e4m3"
    BF16 = "bf16"  # declared; no kernel instantiation yet


# Formats with device code in this package.
IMPLEMENTED_FORMATS = frozenset({CacheFormat.FP8_E4M3})

_KV_NAMES = {
    "fp8": CacheFormat.FP8_E4M3,
    "fp8_e4m3": CacheFormat.FP8_E4M3,
    "auto": CacheFormat.BF16,  # upstream MiniMax M3 default
    "bfloat16": CacheFormat.BF16,
    "bf16": CacheFormat.BF16,
}
_INDEXER_NAMES = {
    "fp8": CacheFormat.FP8_E4M3,
    "auto": CacheFormat.BF16,  # upstream MiniMax M3 default
    "bf16": CacheFormat.BF16,
}


class CacheDtypeError(ValueError):
    pass


def _resolve(kind: str, name: str, table: dict) -> CacheFormat:
    n = str(name).lower()
    if "nvfp4" in n or "fp4" in n:
        raise CacheDtypeError(
            f"{kind} cache dtype {name!r}: NVFP4/FP4 caches are not supported by "
            "the MiniMax-M3 specialized decode kernels (they read fp8 e4m3)"
        )
    fmt = table.get(n)
    if fmt is None:
        raise CacheDtypeError(f"{kind} cache dtype {name!r} is not supported")
    if fmt not in IMPLEMENTED_FORMATS:
        raise CacheDtypeError(
            f"{kind} cache dtype {name!r} ({fmt.value}) has no kernel "
            "instantiation yet; run with fp8 or use the stock decode"
        )
    return fmt


def kv_cache_format(cache_dtype: str) -> CacheFormat:
    """From `CacheConfig.cache_dtype`."""
    return _resolve("KV", cache_dtype, _KV_NAMES)


def indexer_cache_format(indexer_kv_dtype: str) -> CacheFormat:
    """From `AttentionConfig.indexer_kv_dtype` (before 'auto' resolution
    is fine: 'auto' maps to the model default bf16)."""
    return _resolve("indexer K", indexer_kv_dtype, _INDEXER_NAMES)


def check_cache_tensor(kind: str, t, fmt: CacheFormat) -> None:
    """Runtime tensor check: the view the kernels get must match `fmt`."""
    import torch

    ok = {
        CacheFormat.FP8_E4M3: (torch.float8_e4m3fn, torch.uint8),
        CacheFormat.BF16: (torch.bfloat16,),
    }[fmt]
    if t.dtype not in ok:
        raise CacheDtypeError(
            f"{kind} cache tensor dtype {t.dtype} does not match {fmt.value}"
        )


__all__ = [
    "IMPLEMENTED_FORMATS",
    "CacheDtypeError",
    "CacheFormat",
    "check_cache_tensor",
    "indexer_cache_format",
    "kv_cache_format",
]
