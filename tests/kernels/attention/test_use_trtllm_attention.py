# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import patch

import pytest
import torch

from vllm.utils.flashinfer import (
    can_use_trtllm_attention,
    supports_trtllm_attention,
    use_trtllm_attention,
)

DEFAULT_KWARGS = dict(
    num_qo_heads=64,
    num_kv_heads=8,
    num_tokens=128,
    max_seq_len=4096,
    dcp_world_size=1,
    kv_cache_dtype="auto",
    q_dtype=torch.bfloat16,
    is_prefill=False,
    force_use_trtllm=None,
    has_sinks=False,
    has_spec=False,
)


def _call(**overrides) -> bool:
    kwargs = {**DEFAULT_KWARGS, **overrides}
    return use_trtllm_attention(**kwargs)


@pytest.fixture(autouse=True)
def _clear_supports_cache():
    """Clear functools.cache to ensure each test runs independently."""
    supports_trtllm_attention.cache_clear()


# supports_trtllm_attention


@patch("vllm.utils.flashinfer.vllm_is_batch_invariant", return_value=True)
def test_supports_batch_invariant_disables(_mock):
    assert supports_trtllm_attention() is False


@patch("vllm.utils.flashinfer.vllm_is_batch_invariant", return_value=False)
@patch(
    "vllm.utils.flashinfer.current_platform.is_device_capability_family",
    return_value=True,
)
@patch("vllm.utils.flashinfer.has_nvidia_artifactory", return_value=True)
def test_supports_sm100_with_artifactory(_art, _cap, _bi):
    assert supports_trtllm_attention() is True


@patch("vllm.utils.flashinfer.vllm_is_batch_invariant", return_value=False)
@patch(
    "vllm.utils.flashinfer.current_platform.is_device_capability_family",
    return_value=False,
)
def test_supports_non_sm100_platform(_cap, _bi):
    assert supports_trtllm_attention() is False


@patch("vllm.utils.flashinfer.vllm_is_batch_invariant", return_value=False)
@patch(
    "vllm.utils.flashinfer.current_platform.is_device_capability_family",
    return_value=True,
)
@patch("vllm.utils.flashinfer.has_nvidia_artifactory", return_value=False)
def test_supports_sm100_without_artifactory(_art, _cap, _bi):
    assert supports_trtllm_attention() is False


# can_use_trtllm_attention


@patch("vllm.utils.flashinfer.force_use_trtllm_attention", return_value=False)
def test_can_use_force_disabled(_mock):
    assert can_use_trtllm_attention(64, 8) is False


@patch("vllm.utils.flashinfer.force_use_trtllm_attention", return_value=None)
@patch("vllm.utils.flashinfer.supports_trtllm_attention", return_value=True)
def test_can_use_compatible_heads(_sup, _force):
    assert can_use_trtllm_attention(64, 8) is True


@patch("vllm.utils.flashinfer.force_use_trtllm_attention", return_value=None)
@patch("vllm.utils.flashinfer.supports_trtllm_attention", return_value=True)
def test_can_use_incompatible_heads(_sup, _force):
    assert can_use_trtllm_attention(40, 6) is False


@patch("vllm.utils.flashinfer.force_use_trtllm_attention", return_value=None)
@patch("vllm.utils.flashinfer.supports_trtllm_attention", return_value=False)
def test_can_use_platform_unsupported(_sup, _force):
    assert can_use_trtllm_attention(64, 8) is False


# use_trtllm_attention


@patch("vllm.utils.flashinfer.supports_trtllm_attention", return_value=True)
def test_use_force_off(_mock):
    assert _call(force_use_trtllm=False) is False


@patch("vllm.utils.flashinfer.supports_trtllm_attention", return_value=True)
def test_use_dcp_fallback(_mock):
    assert _call(dcp_world_size=2) is False


@patch("vllm.utils.flashinfer.supports_trtllm_attention", return_value=False)
def test_use_platform_unsupported(_mock):
    assert _call() is False


@patch("vllm.utils.flashinfer.supports_trtllm_attention", return_value=False)
def test_use_platform_unsupported_force_on_still_false(_mock):
    assert _call(force_use_trtllm=True) is False


@patch("vllm.utils.flashinfer.supports_trtllm_attention", return_value=True)
def test_use_incompatible_heads(_mock):
    assert _call(num_qo_heads=40, num_kv_heads=6) is False


@patch("vllm.utils.flashinfer.supports_trtllm_attention", return_value=True)
def test_use_incompatible_heads_force_on_still_false(_mock):
    assert _call(num_qo_heads=40, num_kv_heads=6, force_use_trtllm=True) is False


@patch("vllm.utils.flashinfer.supports_trtllm_attention", return_value=True)
def test_use_spec_decode_enables(_mock):
    assert _call(has_spec=True, is_prefill=False) is True


@patch("vllm.utils.flashinfer.supports_trtllm_attention", return_value=True)
@patch(
    "vllm.utils.flashinfer.current_platform.fp8_dtype",
    return_value=torch.float8_e4m3fn,
)
def test_use_fp8_query_forces_trtllm(_fp8, _sup):
    assert _call(q_dtype=torch.float8_e4m3fn) is True


@patch("vllm.utils.flashinfer.supports_trtllm_attention", return_value=True)
def test_use_sinks_force_trtllm(_mock):
    assert _call(has_sinks=True) is True


@patch("vllm.utils.flashinfer.supports_trtllm_attention", return_value=True)
def test_use_auto_prefill_kv_auto(_mock):
    assert _call(is_prefill=True, kv_cache_dtype="auto") is True


@patch("vllm.utils.flashinfer.supports_trtllm_attention", return_value=True)
def test_use_auto_prefill_kv_fp8(_mock):
    assert _call(is_prefill=True, kv_cache_dtype="fp8") is False


@patch("vllm.utils.flashinfer.supports_trtllm_attention", return_value=True)
def test_use_auto_decode_small_batch(_mock):
    assert _call(is_prefill=False, num_tokens=128, kv_cache_dtype="auto") is True


@patch("vllm.utils.flashinfer.supports_trtllm_attention", return_value=True)
def test_use_auto_decode_large_batch(_mock):
    assert _call(is_prefill=False, num_tokens=512, kv_cache_dtype="auto") is False


@patch("vllm.utils.flashinfer.supports_trtllm_attention", return_value=True)
def test_use_force_on(_mock):
    assert _call(force_use_trtllm=True) is True
