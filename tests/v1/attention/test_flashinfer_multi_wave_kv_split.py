# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only tests for VLLM_FLASHINFER_MULTI_WAVE_KV_SPLIT."""

import os

import pytest

from vllm.platforms import current_platform

if not current_platform.is_cuda():
    pytest.skip("FlashInfer backend requires a CUDA platform.", allow_module_level=True)

from vllm.v1.attention.backends.flashinfer import (
    FLASHINFER_MULTI_WAVE_KV_SPLIT_ENV,
    FlashInferDecodeKernel,
    _maybe_enable_multi_wave_kv_split,
)

# None is the native FlashInfer fallback (no XQA/trtllm-gen API decode).
_OTHER_DECODE_KERNELS = [FlashInferDecodeKernel.XQA, None]


@pytest.fixture(autouse=True)
def _clean_flashinfer_env(monkeypatch):
    monkeypatch.delenv(FLASHINFER_MULTI_WAVE_KV_SPLIT_ENV, raising=False)


@pytest.mark.parametrize(
    "decode_kernel", [FlashInferDecodeKernel.TRTLLM_GEN, *_OTHER_DECODE_KERNELS]
)
def test_flag_off_leaves_env_untouched(monkeypatch, decode_kernel):
    monkeypatch.delenv("VLLM_FLASHINFER_MULTI_WAVE_KV_SPLIT", raising=False)
    _maybe_enable_multi_wave_kv_split(decode_kernel)
    assert FLASHINFER_MULTI_WAVE_KV_SPLIT_ENV not in os.environ


def test_flag_on_with_trtllm_gen_sets_env(monkeypatch):
    monkeypatch.setenv("VLLM_FLASHINFER_MULTI_WAVE_KV_SPLIT", "1")
    _maybe_enable_multi_wave_kv_split(FlashInferDecodeKernel.TRTLLM_GEN)
    assert os.environ[FLASHINFER_MULTI_WAVE_KV_SPLIT_ENV] == "1"


@pytest.mark.parametrize("decode_kernel", _OTHER_DECODE_KERNELS)
def test_flag_on_with_other_decode_leaves_env_untouched(monkeypatch, decode_kernel):
    monkeypatch.setenv("VLLM_FLASHINFER_MULTI_WAVE_KV_SPLIT", "1")
    _maybe_enable_multi_wave_kv_split(decode_kernel)
    assert FLASHINFER_MULTI_WAVE_KV_SPLIT_ENV not in os.environ


@pytest.mark.parametrize("user_value", ["0", "", "1"])
def test_user_set_flashinfer_env_wins(monkeypatch, user_value):
    monkeypatch.setenv("VLLM_FLASHINFER_MULTI_WAVE_KV_SPLIT", "1")
    monkeypatch.setenv(FLASHINFER_MULTI_WAVE_KV_SPLIT_ENV, user_value)
    _maybe_enable_multi_wave_kv_split(FlashInferDecodeKernel.TRTLLM_GEN)
    assert os.environ[FLASHINFER_MULTI_WAVE_KV_SPLIT_ENV] == user_value
