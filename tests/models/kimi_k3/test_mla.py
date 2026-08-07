# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import patch

import pytest

from vllm.models.kimi_k3.nvidia.mla import _validate_plain_fp8_prefill_support


@pytest.mark.parametrize("cache_dtype", ["fp8", "fp8_e4m3"])
def test_plain_fp8_requires_prefill_query_quantization(cache_dtype: str):
    with (
        patch(
            "vllm.models.kimi_k3.nvidia.mla."
            "backend_supports_prefill_query_quantization",
            return_value=False,
        ),
        pytest.raises(ValueError, match="--kv-cache-dtype fp8_ds_mla"),
    ):
        _validate_plain_fp8_prefill_support(cache_dtype)


@pytest.mark.parametrize("cache_dtype", ["fp8", "fp8_e4m3"])
def test_plain_fp8_allowed_when_prefill_quantization_is_supported(cache_dtype: str):
    with patch(
        "vllm.models.kimi_k3.nvidia.mla.backend_supports_prefill_query_quantization",
        return_value=True,
    ):
        _validate_plain_fp8_prefill_support(cache_dtype)


@pytest.mark.parametrize("cache_dtype", ["auto", "bfloat16", "fp8_ds_mla"])
def test_other_cache_dtypes_do_not_require_fp8_prefill(cache_dtype: str):
    with patch(
        "vllm.models.kimi_k3.nvidia.mla.backend_supports_prefill_query_quantization",
        return_value=False,
    ):
        _validate_plain_fp8_prefill_support(cache_dtype)
