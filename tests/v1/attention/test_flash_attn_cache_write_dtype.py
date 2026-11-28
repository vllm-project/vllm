# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The dtype string reshape_and_cache_flash hands to the cache write op."""

import pytest
import torch

from vllm import _custom_ops as ops


@pytest.mark.parametrize(
    ("configured", "kv_dtype", "passed"),
    [
        ("auto", torch.bfloat16, "auto"),
        ("bfloat16", torch.bfloat16, "auto"),
        ("float16", torch.float16, "auto"),
        # A plain name that is not the K/V's dtype is not the same write: the
        # op gets it as it is.
        ("float16", torch.bfloat16, "float16"),
        ("fp8_e4m3", torch.bfloat16, "fp8_e4m3"),
        ("fp8", torch.bfloat16, "fp8"),
    ],
)
def test_cache_write_gets_a_dtype_the_op_knows(
    monkeypatch: pytest.MonkeyPatch,
    configured: str,
    kv_dtype: torch.dtype,
    passed: str,
) -> None:
    seen: list[str] = []
    monkeypatch.setattr(
        torch.ops._C_cache_ops,
        "reshape_and_cache_flash",
        lambda *args: seen.append(args[5]),
        raising=False,
    )
    key = torch.zeros(4, 1, 8, dtype=kv_dtype)
    cache = torch.zeros(1, 4, 1, 8, dtype=kv_dtype)
    ops.reshape_and_cache_flash(
        key, key, cache, cache, torch.arange(4), configured, None, None
    )
    assert seen == [passed]
