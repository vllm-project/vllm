# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.models.deepseek_v4.attention import _resolve_dsv4_kv_cache_dtype


@pytest.mark.parametrize("kv_cache_dtype", ["auto", "fp8", "fp8_e4m3", "fp8_ds_mla"])
def test_fp8_ds_mla_layout_resolves_to_fp8_ds_mla(kv_cache_dtype: str):
    # The fp8_ds_mla layout architecturally requires fp8 storage; "auto" and
    # fp8 aliases resolve to the canonical string and are written back to the
    # cache config so page-size specs pick the 576B per-token slot.
    cache_config = SimpleNamespace(cache_dtype=kv_cache_dtype)
    resolved, torch_dtype = _resolve_dsv4_kv_cache_dtype(
        True, kv_cache_dtype, cache_config
    )
    assert resolved == "fp8_ds_mla"
    assert torch_dtype is torch.uint8
    assert cache_config.cache_dtype == "fp8_ds_mla"


def test_fp8_ds_mla_layout_rejects_explicit_non_fp8():
    with pytest.raises(AssertionError, match="only supports fp8"):
        _resolve_dsv4_kv_cache_dtype(True, "bfloat16", None)


def test_plain_layout_keeps_auto_as_bf16():
    assert _resolve_dsv4_kv_cache_dtype(False, "auto", None) == (
        "auto",
        torch.bfloat16,
    )
    assert _resolve_dsv4_kv_cache_dtype(False, "fp8", None) == (
        "fp8",
        torch.float8_e4m3fn,
    )
