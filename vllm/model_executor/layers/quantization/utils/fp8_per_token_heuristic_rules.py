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


def _shape_config(m: int, n: int, k: int) -> dict:
    if (((m + 63) // 64) * ((n + 63) // 64)) / 32 <= 7.5:
        if m * n <= 131072:
            if n / k <= 0.0625:
                if m <= 16:
                    if n <= 256:
                        if (
                            m <= 32
                            and k >= 4096
                            and ((m + 15) // 16) * ((n + 31) // 32) < 128
                        ):
                            return _CHOICES[10]
                        return _CHOICES[13]
                    else:
                        if (
                            m <= 32
                            and k >= 4096
                            and ((m + 15) // 16) * ((n + 31) // 32) < 128
                        ):
                            return _CHOICES[9]
                        return _CHOICES[13]
                else:
                    if m <= 64:
                        return _CHOICES[24]
                    else:
                        return _CHOICES[13]
            else:
                if n <= 1024:
                    if m * n <= 8192:
                        return _CHOICES[27]
                    else:
                        if m * n <= 32768:
                            return _CHOICES[24]
                        else:
                            return _CHOICES[11]
                else:
                    if m * n <= 4096:
                        if k <= 3840:
                            return _CHOICES[26]
                        else:
                            return _CHOICES[25]
                    else:
                        if m <= 32:
                            return _CHOICES[14]
                        else:
                            return _CHOICES[19]
        else:
            if m * n <= 327680:
                if n / k <= 1.3333333333333333:
                    if n / k <= 0.0625:
                        return _CHOICES[15]
                    else:
                        if n <= 3840:
                            return _CHOICES[12]
                        else:
                            return _CHOICES[12]
                else:
                    return _CHOICES[13]
            else:
                if n <= 4096:
                    if m * n <= 786432:
                        if m <= 128:
                            return _CHOICES[16]
                        else:
                            return _CHOICES[17]
                    else:
                        if k <= 4096:
                            return _CHOICES[6]
                        else:
                            return _CHOICES[1]
                else:
                    if m * n <= 655360:
                        return _CHOICES[22]
                    else:
                        return _CHOICES[18]
    else:
        if m <= 64:
            if n <= 30720:
                if m * n <= 262144:
                    if n <= 16384:
                        return _CHOICES[28]
                    else:
                        return _CHOICES[23]
                else:
                    if m <= 48:
                        return _CHOICES[29]
                    else:
                        return _legacy_config(m, n)
            else:
                if m <= 16:
                    return _CHOICES[20]
                else:
                    if m <= 32:
                        return _CHOICES[31]
                    else:
                        return _CHOICES[30]
        else:
            if n <= 24576:
                if n * k <= 15728640:
                    if n / k <= 32.0:
                        if n <= 512:
                            return _CHOICES[16]
                        else:
                            return _legacy_config(m, n)
                    else:
                        if m <= 128:
                            return _CHOICES[32]
                        else:
                            return _CHOICES[33]
                else:
                    if m * n <= 33554432:
                        if m * n <= 8388608:
                            return _CHOICES[1]
                        else:
                            return _CHOICES[6]
                    else:
                        if m <= 4096:
                            return _CHOICES[3]
                        else:
                            return _CHOICES[2]
            else:
                if m <= 128:
                    return _CHOICES[8]
                else:
                    if n <= 30720:
                        if m <= 1024:
                            return _CHOICES[5]
                        else:
                            return _CHOICES[0]
                    else:
                        if m <= 1024:
                            return _CHOICES[7]
                        else:
                            return _CHOICES[4]


@lru_cache(maxsize=2048)
def get_heuristic_config(m: int, n: int, k: int) -> dict:
    choice = _shape_config(m, n, k)
    if m >= 128 and n >= 24576 and k <= 1024:
        if k <= 512:
            return _CHOICES[33] if 192 < m <= 1024 else _CHOICES[22]
        return _CHOICES[34]
    if n >= 16384 and n * k >= 64 * 1024**2:
        if n <= 24576 and 64 < m <= 128:
            return _CHOICES[39 if n * k <= 72 * 1024**2 else 40]
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
    if 8 < m <= 64 and k >= 8192 and n <= k // 2 and n * k >= 60 * 1024**2:
        return _CHOICES[24 if m <= 16 else 37 if m <= 32 else 38]
    return choice
