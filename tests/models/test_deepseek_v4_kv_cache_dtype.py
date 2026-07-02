# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Resolution of ``--kv-cache-dtype`` for the DeepSeek V4 fp8_ds_mla layout.

Regression for the SM120 crash where the default ``auto`` was asserted instead
of resolving to ``fp8_ds_mla`` (the layout has no non-fp8 variant), so
``vllm serve <dsv4-model>`` with no flag failed on Blackwell.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm.models.deepseek_v4.attention import _resolve_dsv4_kv_cache_dtype


@pytest.mark.parametrize("kv_cache_dtype", ["auto", None, "fp8", "fp8_e4m3"])
def test_fp8_ds_mla_layout_resolves_to_ds_mla(kv_cache_dtype):
    cache_config = SimpleNamespace(cache_dtype=kv_cache_dtype)
    resolved, torch_dtype = _resolve_dsv4_kv_cache_dtype(
        True, kv_cache_dtype, cache_config
    )
    assert resolved == "fp8_ds_mla"
    assert torch_dtype is torch.uint8
    # canonical string is written back so the page-size spec picks the 576B slot
    assert cache_config.cache_dtype == "fp8_ds_mla"


def test_fp8_ds_mla_layout_rejects_non_fp8():
    with pytest.raises(AssertionError):
        _resolve_dsv4_kv_cache_dtype(True, "bfloat16", SimpleNamespace(cache_dtype=""))


@pytest.mark.parametrize(
    "kv_cache_dtype, expected_dtype",
    [
        ("auto", torch.bfloat16),
        ("bfloat16", torch.bfloat16),
        ("fp8", torch.float8_e4m3fn),
    ],
)
def test_plain_row_layout_unchanged(kv_cache_dtype, expected_dtype):
    resolved, torch_dtype = _resolve_dsv4_kv_cache_dtype(False, kv_cache_dtype, None)
    assert resolved == kv_cache_dtype
    assert torch_dtype is expected_dtype
