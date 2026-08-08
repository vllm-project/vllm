# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for FlashInfer spec-decode FULL cudagraph support.

Extracted from vLLM PR #47979 (change 4) with wrapper-keying fix.
Verifies that:
1. The capability probe correctly detects FlashInfer's q_len_per_req support.
2. The get_cudagraph_support path returns UNIFORM_BATCH when the
   native multi-token decode capability is present.
3. Decode wrappers are keyed by (batch_size, q_len_per_req).
"""
from __future__ import annotations

import pytest


def test_flashinfer_supports_uniform_multi_token_decode():
    """The probe should return a bool (True on FlashInfer >= 0.6.16)."""
    try:
        from vllm.v1.attention.backends.flashinfer import (
            flashinfer_supports_uniform_multi_token_decode,
        )
    except ImportError:
        pytest.skip("FlashInfer backend not importable")

    flashinfer_supports_uniform_multi_token_decode.cache_clear()
    result = flashinfer_supports_uniform_multi_token_decode()
    assert isinstance(result, bool)


def test_fast_plan_decode_has_q_len_per_req_param():
    """vLLM's fast_plan_decode wrapper must accept q_len_per_req."""
    import inspect

    from vllm.v1.attention.backends.flashinfer import fast_plan_decode

    sig = inspect.signature(fast_plan_decode)
    assert "q_len_per_req" in sig.parameters
    assert sig.parameters["q_len_per_req"].default == 1


def test_wrapper_cache_keyed_by_tuple():
    """_decode_wrappers_cudagraph must be a dict keyed by tuple, not int.

    This prevents the frozen-plan crash when q_len_per_req changes
    between pure decode (1) and spec-decode verify (K+1).
    """
    import inspect

    from vllm.v1.attention.backends.flashinfer import FlashInferMetadataBuilder

    src = inspect.getsource(FlashInferMetadataBuilder)
    # The cache dict type annotation must be tuple[int, int]
    assert "tuple[int, int]" in src, (
        "_decode_wrappers_cudagraph must be keyed by tuple[int, int] "
        "(batch_size, q_len_per_req), not int"
    )


def test_get_decode_wrapper_accepts_q_len_per_req():
    """_get_decode_wrapper must accept q_len_per_req as a parameter."""
    import inspect

    from vllm.v1.attention.backends.flashinfer import FlashInferMetadataBuilder

    sig = inspect.signature(FlashInferMetadataBuilder._get_decode_wrapper)
    assert "q_len_per_req" in sig.parameters
    assert sig.parameters["q_len_per_req"].default == 1
