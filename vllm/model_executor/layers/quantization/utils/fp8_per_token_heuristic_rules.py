# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Launch rules measured on discovery shapes, independent of tuning files."""

from functools import lru_cache

_KEYS = (
    "engine",
    "block_size_m",
    "block_size_n",
    "block_size_k",
    "num_warps",
    "num_stages",
    "group_m",
    "n_stripe",
    "unmasked_loads",
    "variant",
    "split_k",
    "programs",
)
_PALETTE = (
    ("contiguous", 128, 128, 128, 8, 2, 4, 0, False, "persistent", 1, 64),
    ("shared", 128, 128, 128, 4, 2, 1, 0, False, "gemm", 1, 32),
    ("shared", 128, 128, 128, 4, 2, 1, 0, True, "gemm", 1, 32),
    ("shared", 128, 128, 128, 8, 1, 1, 8, False, "gemm", 1, 32),
    ("shared", 128, 128, 128, 4, 2, 1, 8, False, "gemm", 1, 32),
    ("shared", 128, 128, 128, 8, 3, 1, 8, False, "gemm", 1, 32),
    ("shared", 128, 128, 128, 8, 1, 2, 0, False, "gemm", 1, 32),
    ("shared", 128, 32, 128, 4, 1, 1, 8, False, "gemm", 1, 32),
    ("shared", 128, 64, 128, 4, 3, 1, 0, False, "gemm", 1, 32),
    ("contiguous", 16, 32, 128, 4, 2, 1, 0, False, "gemm", 4, 32),
    ("contiguous", 16, 32, 128, 4, 2, 1, 0, False, "gemm", 8, 32),
    ("contiguous", 16, 64, 128, 4, 2, 1, 0, False, "gemm", 1, 32),
    ("contiguous", 32, 128, 128, 4, 2, 1, 0, False, "gemm", 1, 32),
    ("contiguous", 32, 32, 128, 4, 2, 1, 0, False, "gemm", 1, 32),
    ("shared", 32, 32, 128, 4, 2, 1, 0, False, "gemm", 1, 32),
    ("shared", 32, 64, 128, 4, 2, 1, 0, False, "gemm", 1, 32),
    ("contiguous", 64, 128, 128, 4, 2, 1, 0, False, "gemm", 1, 32),
    ("contiguous", 64, 128, 128, 4, 2, 4, 0, False, "persistent", 1, 64),
    ("shared", 64, 128, 128, 8, 2, 1, 0, False, "gemm", 1, 32),
    ("contiguous", 64, 32, 128, 4, 2, 1, 0, False, "gemm", 1, 32),
    ("shared", 64, 32, 128, 8, 3, 1, 0, False, "gemm", 1, 32),
    ("baseline", 64, 64, 128, 4, 2, 1, 0, False, "gemm", 1, 32),
    ("contiguous", 64, 64, 128, 4, 2, 1, 0, False, "gemm", 1, 32),
    ("shared", 16, 128, 256, 8, 1, 1, 0, False, "gemm", 1, 32),
    ("contiguous", 16, 64, 256, 4, 2, 1, 0, False, "gemm", 1, 32),
    ("shared", 16, 64, 256, 4, 1, 1, 0, False, "gemm", 1, 32),
    ("shared", 16, 64, 256, 4, 2, 1, 0, False, "gemm", 1, 32),
    ("contiguous", 32, 64, 256, 4, 2, 1, 0, False, "gemm", 1, 32),
    ("shared", 32, 64, 256, 4, 3, 1, 0, False, "gemm", 1, 32),
    ("shared", 64, 64, 256, 4, 2, 1, 0, False, "gemm", 1, 32),
    ("shared", 64, 64, 256, 8, 3, 1, 2, False, "gemm", 1, 32),
    ("shared", 64, 64, 256, 8, 3, 2, 0, False, "gemm", 1, 32),
    ("contiguous", 128, 64, 64, 8, 2, 1, 0, False, "gemm", 1, 32),
    ("contiguous", 256, 64, 64, 4, 2, 1, 0, False, "gemm", 1, 32),
    ("shared", 128, 128, 128, 8, 1, 1, 0, False, "gemm", 1, 32),
    ("shared", 256, 64, 64, 4, 1, 4, 0, False, "gemm", 1, 32),
    ("shared", 128, 128, 128, 8, 1, 4, 0, False, "gemm", 1, 32),
    ("contiguous", 32, 64, 256, 8, 2, 1, 0, False, "gemm", 1, 32),
    ("contiguous", 64, 64, 256, 4, 2, 1, 0, False, "gemm", 1, 32),
    ("shared", 128, 128, 64, 8, 1, 4, 0, False, "gemm", 1, 32),
    ("shared", 128, 64, 128, 4, 2, 4, 0, False, "gemm", 1, 32),
)
_CHOICES = tuple(dict(zip(_KEYS, row)) for row in _PALETTE)


def _legacy_config(m: int, n: int) -> dict:
    tile = (
        (64, 64 if n < 8192 else 128, 256)
        if m <= 32
        else (64, 64, 256)
        if m <= 64
        else (64, 128, 128)
        if m <= 128
        else (128, 128, 128)
    )
    return dict(
        engine="shared",
        block_size_m=tile[0],
        block_size_n=tile[1],
        block_size_k=tile[2],
        num_warps=4,
        num_stages=2,
    )


def _shape_config(
    m: int, n: int, k: int, num_processors: int, cache_size_bytes: int
) -> dict:
    output_elements = m * n
    tile_waves = ((m + 63) // 64) * ((n + 63) // 64) / num_processors
    if tile_waves <= 7.5:
        if output_elements <= 131072:
            if n / k <= 0.0625:
                if m <= 16:
                    tiles = ((m + 15) // 16) * ((n + 31) // 32)
                    if k >= 4096 and tiles < 4 * num_processors:
                        return _CHOICES[10 if n <= 256 else 9]
                    return _CHOICES[13]
                return _CHOICES[24 if m <= 64 else 13]
            if n <= 1024:
                if output_elements <= 8192:
                    return _CHOICES[27]
                return _CHOICES[24 if output_elements <= 32768 else 11]
            if output_elements <= 4096:
                return _CHOICES[26 if k <= 3840 else 25]
            return _CHOICES[14 if m <= 32 else 19]
        if output_elements <= 327680:
            if n / k > 1.3333333333333333:
                return _CHOICES[13]
            return _CHOICES[15 if n / k <= 0.0625 else 12]
        if n <= 4096:
            if output_elements <= 786432:
                return _CHOICES[16 if m <= 128 else 17]
            return _CHOICES[6 if k <= 4096 else 1]
        return _CHOICES[22 if output_elements <= 655360 else 18]

    if m <= 64:
        if n <= 30720:
            if output_elements <= 262144:
                return _CHOICES[28 if n <= 16384 else 23]
            return _CHOICES[29] if m <= 48 else _legacy_config(m, n)
        if m <= 16:
            return _CHOICES[20]
        return _CHOICES[31 if m <= 32 else 30]
    if n <= 24576:
        if cache_size_bytes > 0 and n * k * 64 <= cache_size_bytes * 15:
            if n / k <= 32.0:
                return _CHOICES[16] if n <= 512 else _legacy_config(m, n)
            return _CHOICES[32 if m <= 128 else 33]
        if output_elements <= 8388608:
            return _CHOICES[1]
        if output_elements <= 33554432:
            return _CHOICES[6]
        return _CHOICES[3 if m <= 4096 else 2]
    if m <= 128:
        return _CHOICES[8]
    if n <= 30720:
        return _CHOICES[5 if m <= 1024 else 0]
    return _CHOICES[7 if m <= 1024 else 4]


@lru_cache(maxsize=2048)
def get_heuristic_config(
    m: int,
    n: int,
    k: int,
    num_processors: int = 32,
    cache_size_bytes: int = 64 * 1024**2,
) -> dict:
    """Select by shape, processor count and last-level data-cache capacity."""
    if num_processors <= 0:
        raise ValueError("num_processors must be positive")
    if cache_size_bytes < 0:
        raise ValueError("cache_size_bytes must be nonnegative")
    choice = _shape_config(m, n, k, num_processors, cache_size_bytes)
    if m >= 128 and n >= 24576 and k <= 1024:
        if k <= 512:
            return _CHOICES[33] if 192 < m <= 1024 else _CHOICES[22]
        return _CHOICES[34]
    if cache_size_bytes > 0 and n >= 16384 and n * k >= cache_size_bytes:
        if n <= 24576 and 64 < m <= 128:
            return _CHOICES[39 if n * k * 8 <= cache_size_bytes * 9 else 40]
        if 128 < m <= 512:
            if n < 20480 or (n <= 24576 and 256 < m <= 384):
                return _CHOICES[36]
            if n <= 24576 or m <= 256:
                return _CHOICES[35]
        if (
            m > 4096
            and k <= 8192
            and not choice.get("n_stripe", 0)
            and choice.get("variant") != "persistent"
        ):
            return _CHOICES[35]
    if (
        cache_size_bytes > 0
        and 8 < m <= 64
        and k >= 8192
        and n <= k // 2
        and n * k * 16 >= cache_size_bytes * 15
    ):
        return _CHOICES[24 if m <= 16 else 37 if m <= 32 else 38]
    if choice.get("variant") == "persistent":
        return dict(choice, programs=2 * num_processors)
    return choice
