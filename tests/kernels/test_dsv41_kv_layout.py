# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

from vllm.models.deepseek_v4_1.attention import (
    DSV4_KV_LAYOUTS,
    _resolve_dsv4_kv_cache_dtype,
)


def test_layout_table_page_geometry():
    fp8, fp4 = DSV4_KV_LAYOUTS["fp8_ds_mla"], DSV4_KV_LAYOUTS["nvfp4_ds_mla"]
    assert (fp8.swa_bytes, fp8.compressed_bytes) == (584, 584)
    assert (fp4.swa_bytes, fp4.compressed_bytes) == (528, 288)
    # SWA pages (32 tokens) and compressed pages (128 / 64 states) must be
    # multiples of the FlashMLA TMA stride of their format.
    assert (32 * fp4.swa_bytes) % fp4.swa_alignment == 0
    assert (64 * fp4.compressed_bytes) % fp4.compressed_alignment == 0
    assert (128 * fp4.compressed_bytes) % fp4.compressed_alignment == 0


@pytest.mark.parametrize("dtype", ["auto", "fp8", "fp8_ds_mla"])
def test_resolve_fp8_ds_mla(dtype):
    layout = _resolve_dsv4_kv_cache_dtype(True, dtype, None)
    assert layout is DSV4_KV_LAYOUTS["fp8_ds_mla"]
    assert layout.torch_dtype == torch.uint8


def test_resolve_nvfp4_ds_mla_requires_ds_mla_layout():
    assert (
        _resolve_dsv4_kv_cache_dtype(True, "nvfp4_ds_mla", None)
        is DSV4_KV_LAYOUTS["nvfp4_ds_mla"]
    )
    with pytest.raises(ValueError, match="nvfp4_ds_mla"):
        _resolve_dsv4_kv_cache_dtype(False, "nvfp4_ds_mla", None)


def test_resolve_plain_rows():
    bf16 = _resolve_dsv4_kv_cache_dtype(False, "auto", None)
    assert bf16.torch_dtype == torch.bfloat16 and bf16.swa_bytes is None
    fp8 = _resolve_dsv4_kv_cache_dtype(False, "fp8", None)
    assert fp8.torch_dtype == torch.float8_e4m3fn and fp8.swa_alignment == 512
