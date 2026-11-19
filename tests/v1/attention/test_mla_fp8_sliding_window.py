# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.model_executor.layers.attention.mla_attention import (
    _reject_unsupported_quantized_sliding_mla,
)
from vllm.models.hy_v4_flash.nvidia.swa import FlashSparseBackend, FlashSWABackend
from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.attention.backends.mla.flashmla_sparse import FlashMLASparseBackend


def test_default_backend_rejects_packed_quantized_sliding_mla():
    with pytest.raises(ValueError, match="fp8_ds_mla"):
        _reject_unsupported_quantized_sliding_mla(
            FlashMLASparseBackend, "fp8_ds_mla", sliding_window=512
        )
    with pytest.raises(ValueError, match="nvfp4_ds_mla"):
        _reject_unsupported_quantized_sliding_mla(
            FlashSWABackend, "nvfp4_ds_mla", sliding_window=512
        )


def test_unquantized_and_non_sliding_layouts_stay_allowed():
    _reject_unsupported_quantized_sliding_mla(
        FlashMLASparseBackend, "bfloat16", sliding_window=512
    )
    _reject_unsupported_quantized_sliding_mla(
        FlashMLASparseBackend, "fp8_ds_mla", sliding_window=None
    )
    _reject_unsupported_quantized_sliding_mla(
        FlashSWABackend, "fp8_ds_mla", sliding_window=512
    )


def test_only_hy4_flash_swa_opts_into_fp8_ds_mla():
    assert FlashSWABackend.supports_fp8_ds_mla_sliding_window()
    assert not FlashSparseBackend.supports_fp8_ds_mla_sliding_window()
    assert not FlashMLASparseBackend.supports_fp8_ds_mla_sliding_window()
    assert not AttentionBackend.supports_fp8_ds_mla_sliding_window()
