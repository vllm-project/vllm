# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import Mock, patch

import pytest

from vllm.config import CacheConfig
from vllm.model_executor.layers.attention import MLAAttention
from vllm.model_executor.layers.linear import ColumnParallelLinear
from vllm.models.hy_v4_flash.nvidia.attention import HYV4FlashMLAAttention
from vllm.models.hy_v4_flash.nvidia.swa import FlashSparseBackend, FlashSWABackend
from vllm.v1.attention.backends.mla.flashmla_sparse import FlashMLASparseBackend


@pytest.mark.parametrize(
    "layer_cls,kv_cache_dtype,reject_swa",
    [
        (MLAAttention, "auto", False),
        (MLAAttention, "fp8_ds_mla", True),
        (MLAAttention, "nvfp4_ds_mla", True),
        (HYV4FlashMLAAttention, "auto", False),
        (HYV4FlashMLAAttention, "fp8_ds_mla", False),
        (HYV4FlashMLAAttention, "nvfp4_ds_mla", True),
    ],
)
@pytest.mark.parametrize("sliding_window", [None, 512])
def test_sliding_window_kv_cache_dtype(
    layer_cls, kv_cache_dtype, reject_swa, sliding_window
):
    if layer_cls is MLAAttention:
        backend = FlashMLASparseBackend
    else:
        backend = FlashSWABackend if sliding_window else FlashSparseBackend

    # Exercise the real constructor guard without initializing GPU kernels.
    with patch(
        "vllm.model_executor.layers.attention.mla_attention._init_kv_cache_quant",
        side_effect=RuntimeError("KV cache initialization reached"),
    ) as init_quant:
        rejected = sliding_window is not None and reject_swa
        error = ValueError if rejected else RuntimeError
        match = (
            f"kv_cache_dtype={kv_cache_dtype}"
            if rejected
            else "KV cache initialization reached"
        )
        with pytest.raises(error, match=match):
            layer_cls(
                num_heads=8,
                scale=1.0,
                qk_nope_head_dim=128,
                qk_rope_head_dim=64,
                v_head_dim=128,
                q_lora_rank=512,
                kv_lora_rank=512,
                kv_b_proj=Mock(spec=ColumnParallelLinear),
                cache_config=CacheConfig(cache_dtype=kv_cache_dtype),
                attn_backend=backend,
                use_sparse=True,
                sliding_window=sliding_window,
            )
        assert init_quant.call_count == (0 if rejected else 1)
