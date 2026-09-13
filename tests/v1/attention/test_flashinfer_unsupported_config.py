# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.utils.torch_utils import is_quantized_kv_cache


def test_is_quantized_kv_cache_requires_string_cache_dtype():
    """FlashInferMetadataBuilder must pass cache_dtype, not kv_cache_dtype op dtype."""
    assert is_quantized_kv_cache("fp8_e4m3") is True
    assert is_quantized_kv_cache("nvfp4") is True
    with pytest.raises(AttributeError):
        is_quantized_kv_cache(torch.float8_e4m3fn)


@pytest.mark.parametrize(
    (
        "can_use_trtllm",
        "disable_q_quant",
        "kv_cache_dtype",
        "reorder_batch_threshold",
        "has_swa",
        "expected",
    ),
    [
        # Issue #47847 repro: all conditions met → unsupported.
        (True, True, "fp8_e4m3", 8, True, True),
        (True, True, "nvfp4", 8, True, True),
        # FP8 Q (flag off) is supported.
        (True, False, "fp8_e4m3", 8, True, False),
        # Single-token decode (no spec-as-decode).
        (True, True, "fp8_e4m3", 1, True, False),
        # Full-attention-only models.
        (True, True, "fp8_e4m3", 8, False, False),
        # Non-quantized KV.
        (True, True, "auto", 8, True, False),
        # Pre-SM100 / TRTLLM unavailable.
        (False, True, "fp8_e4m3", 8, True, False),
    ],
)
def test_is_unsupported_bf16_q_spec_swa_config(
    can_use_trtllm: bool,
    disable_q_quant: bool,
    kv_cache_dtype: str,
    reorder_batch_threshold: int,
    has_swa: bool,
    expected: bool,
):
    pytest.importorskip("flashinfer")
    from vllm.v1.attention.backends.flashinfer import (
        _is_unsupported_bf16_q_spec_swa_config,
    )

    assert (
        _is_unsupported_bf16_q_spec_swa_config(
            can_use_trtllm=can_use_trtllm,
            disable_flashinfer_q_quantization=disable_q_quant,
            kv_cache_dtype=kv_cache_dtype,
            reorder_batch_threshold=reorder_batch_threshold,
            has_sliding_window_layer=has_swa,
        )
        is expected
    )
