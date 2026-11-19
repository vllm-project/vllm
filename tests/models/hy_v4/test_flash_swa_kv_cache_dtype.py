# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.model_executor.layers.attention import MLAAttention
from vllm.models.hy_v4_flash.nvidia.attention import HYV4FlashMLAAttention


@pytest.mark.parametrize("layer_cls", [MLAAttention, HYV4FlashMLAAttention])
@pytest.mark.parametrize("sliding_window", [None, 512])
@pytest.mark.parametrize("kv_cache_dtype", ["auto", "fp8_ds_mla", "nvfp4_ds_mla"])
def test_sliding_window_kv_cache_dtype(layer_cls, sliding_window, kv_cache_dtype):
    rejected = sliding_window is not None and (
        kv_cache_dtype == "nvfp4_ds_mla"
        or (kv_cache_dtype == "fp8_ds_mla" and layer_cls is MLAAttention)
    )
    if rejected:
        with pytest.raises(ValueError, match=f"kv_cache_dtype={kv_cache_dtype}"):
            layer_cls._validate_sliding_window_kv_cache_dtype(
                sliding_window, kv_cache_dtype
            )
    else:
        layer_cls._validate_sliding_window_kv_cache_dtype(sliding_window, kv_cache_dtype)
