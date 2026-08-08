# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for CacheDType string ↔ int codes used by reshape_and_cache_flash.

The C++ op takes an int across the stable ABI to avoid host leaks from
eager str→std::string unboxing (vllm#50150). Python still exposes the
string API and maps via kv_cache_dtype_to_code.
"""

from __future__ import annotations

from typing import get_args

import pytest

from vllm._custom_ops import KV_CACHE_DTYPE_TO_CODE, kv_cache_dtype_to_code
from vllm.config.cache import CacheDType


def test_codes_cover_all_cache_dtypes() -> None:
    declared = set(get_args(CacheDType))
    coded = set(KV_CACHE_DTYPE_TO_CODE)
    assert declared == coded, (
        f"KV_CACHE_DTYPE_TO_CODE out of sync with CacheDType: "
        f"missing={declared - coded} extra={coded - declared}"
    )


def test_codes_are_unique_and_dense() -> None:
    codes = list(KV_CACHE_DTYPE_TO_CODE.values())
    assert len(codes) == len(set(codes))
    assert sorted(codes) == list(range(len(codes)))


@pytest.mark.parametrize(
    "name,code",
    [
        ("auto", 0),
        ("fp8", 3),
        ("fp8_e5m2", 5),
        ("nvfp4", 15),
        ("nvfp4_4over6", 16),
    ],
)
def test_known_mappings(name: str, code: int) -> None:
    assert kv_cache_dtype_to_code(name) == code


def test_unknown_dtype_raises() -> None:
    with pytest.raises(ValueError, match="Unsupported kv_cache_dtype"):
        kv_cache_dtype_to_code("not_a_real_dtype")
