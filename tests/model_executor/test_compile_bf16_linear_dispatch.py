# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the opt-in BF16 unquantized linear torch.compile path."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch

from vllm.model_executor.layers import utils
from vllm.platforms import current_platform


@pytest.fixture(autouse=True)
def _reset_compile_state(monkeypatch):
    utils._unquant_bf16_linear_cache.clear()
    utils._unquant_bf16_linear_capture_safe_keys.clear()
    monkeypatch.setattr(utils, "_unquant_bf16_linear_torch_compile_configured", False)
    monkeypatch.setattr(utils, "_unquant_bf16_linear_torch_compile_disabled", False)
    monkeypatch.setattr(utils, "_inductor_max_autotune_gemm_forced", True)
    monkeypatch.setattr(utils, "_dynamo_compile_caches_forced", True)
    yield
    utils._unquant_bf16_linear_cache.clear()
    utils._unquant_bf16_linear_capture_safe_keys.clear()


class _Tensorish:
    def __init__(
        self,
        *,
        dtype: torch.dtype = torch.bfloat16,
        is_cuda: bool = True,
        contiguous: bool = True,
    ):
        self.dtype = dtype
        self.is_cuda = is_cuda
        self._contiguous = contiguous

    def is_contiguous(self):
        return self._contiguous


def test_dispatch_env_on_cuda_returns_compile(monkeypatch):
    monkeypatch.setenv("VLLM_ENABLE_UNQUANT_BF16_LINEAR_TORCH_COMPILE", "1")
    monkeypatch.setattr(current_platform, "is_rocm", lambda: False)
    monkeypatch.setattr(current_platform, "is_cpu", lambda: False)
    monkeypatch.setattr(current_platform, "is_cuda", lambda: True)

    assert (
        utils.dispatch_unquantized_gemm() is utils._torch_compile_bf16_unquantized_gemm
    )


def test_dispatch_env_off_returns_default(monkeypatch):
    monkeypatch.setenv("VLLM_ENABLE_UNQUANT_BF16_LINEAR_TORCH_COMPILE", "0")
    monkeypatch.setattr(current_platform, "is_rocm", lambda: False)
    monkeypatch.setattr(current_platform, "is_cpu", lambda: False)
    monkeypatch.setattr(current_platform, "is_cuda", lambda: True)

    assert utils.dispatch_unquantized_gemm() is utils.default_unquantized_gemm


def test_dispatch_non_cuda_does_not_select_compile(monkeypatch):
    monkeypatch.setenv("VLLM_ENABLE_UNQUANT_BF16_LINEAR_TORCH_COMPILE", "1")
    monkeypatch.setattr(current_platform, "is_rocm", lambda: False)
    monkeypatch.setattr(current_platform, "is_cpu", lambda: True)
    monkeypatch.setattr(current_platform, "is_cuda", lambda: False)

    assert utils.dispatch_unquantized_gemm() is not (
        utils._torch_compile_bf16_unquantized_gemm
    )


def test_predicate_rejects_cpu_tensor():
    x = _Tensorish(is_cuda=False)
    w = _Tensorish()

    assert not utils._should_use_unquant_bf16_linear_torch_compile(x, w, None)


def test_predicate_rejects_non_bf16_dtype():
    x = _Tensorish(dtype=torch.float16)
    w = _Tensorish()

    assert not utils._should_use_unquant_bf16_linear_torch_compile(x, w, None)


def test_predicate_rejects_bias_dtype_mismatch():
    x = _Tensorish()
    w = _Tensorish()
    bias = _Tensorish(dtype=torch.float32)

    assert not utils._should_use_unquant_bf16_linear_torch_compile(x, w, bias)


def test_predicate_rejects_non_contiguous_input():
    x = _Tensorish(contiguous=False)
    w = _Tensorish()

    assert not utils._should_use_unquant_bf16_linear_torch_compile(x, w, None)


def test_predicate_rejects_while_dynamo_is_compiling(monkeypatch):
    x = _Tensorish()
    w = _Tensorish()
    monkeypatch.setattr(torch._dynamo, "is_compiling", lambda: True)

    assert not utils._should_use_unquant_bf16_linear_torch_compile(x, w, None)


def test_cache_key_is_static_shape_specific():
    x_a = torch.zeros(4, 8)
    x_b = torch.zeros(64, 8)
    w = torch.zeros(16, 8)

    assert utils._unquant_bf16_linear_cache_key(
        x_a, w, None
    ) != utils._unquant_bf16_linear_cache_key(x_b, w, None)


def test_compile_gemm_falls_back_to_f_linear_when_ineligible():
    x = torch.randn(4, 8, dtype=torch.bfloat16)
    w = torch.randn(16, 8, dtype=torch.bfloat16)

    out = utils._torch_compile_bf16_unquantized_gemm(None, x, w, None)

    torch.testing.assert_close(
        out, torch.nn.functional.linear(x, w, None), atol=0.0, rtol=0.0
    )


def test_capture_skips_uncompiled_shape(monkeypatch):
    x = torch.randn(4, 8, dtype=torch.bfloat16)
    w = torch.randn(16, 8, dtype=torch.bfloat16)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)

    assert utils._apply_unquant_bf16_linear_torch_compile(x, w, None) is None


def test_capture_reuses_warmed_shape(monkeypatch):
    x = torch.randn(4, 8, dtype=torch.bfloat16)
    w = torch.randn(16, 8, dtype=torch.bfloat16)
    key = utils._unquant_bf16_linear_cache_key(x, w, None)
    cached = MagicMock(return_value=torch.nn.functional.linear(x, w, None))
    utils._unquant_bf16_linear_cache[key] = cached
    utils._unquant_bf16_linear_capture_safe_keys.add(key)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)

    utils._apply_unquant_bf16_linear_torch_compile(x, w, None)

    cached.assert_called_once_with(x, w)


def test_compile_failure_disables_fast_path(monkeypatch):
    x = torch.randn(4, 8, dtype=torch.bfloat16)
    w = torch.randn(16, 8, dtype=torch.bfloat16)

    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(
        utils,
        "_get_or_create_unquant_bf16_linear_kernel",
        lambda x, weight, bias, allow_new_compile: MagicMock(
            side_effect=RuntimeError("inductor boom")
        ),
    )

    assert utils._apply_unquant_bf16_linear_torch_compile(x, w, None) is None
    assert utils._unquant_bf16_linear_torch_compile_disabled is True


@pytest.fixture
def _restore_inductor_state():
    pytest.importorskip("torch._inductor.utils")
    pytest.importorskip("torch._inductor.config")
    pytest.importorskip("torch._inductor.codegen.cuda")
    import torch._dynamo.config as dynamo_config
    import torch._inductor.config as inductor_config
    import torch._inductor.utils as inductor_utils
    from torch._inductor.codegen.cuda import cuda_env

    originals = {
        "is_big_gpu": inductor_utils.is_big_gpu,
        "is_datacenter_blackwell_arch": cuda_env.is_datacenter_blackwell_arch,
        "max_autotune": inductor_config.max_autotune,
        "max_autotune_gemm": inductor_config.max_autotune_gemm,
        "max_autotune_gemm_backends": inductor_config.max_autotune_gemm_backends,
        "coordinate_descent_tuning": inductor_config.coordinate_descent_tuning,
        "cache_size_limit": getattr(dynamo_config, "cache_size_limit", None),
        "accumulated_cache_size_limit": getattr(
            dynamo_config, "accumulated_cache_size_limit", None
        ),
    }
    yield inductor_utils, cuda_env, inductor_config, dynamo_config
    inductor_utils.is_big_gpu = originals["is_big_gpu"]
    cuda_env.is_datacenter_blackwell_arch = originals["is_datacenter_blackwell_arch"]
    inductor_config.max_autotune = originals["max_autotune"]
    inductor_config.max_autotune_gemm = originals["max_autotune_gemm"]
    inductor_config.max_autotune_gemm_backends = originals["max_autotune_gemm_backends"]
    inductor_config.coordinate_descent_tuning = originals["coordinate_descent_tuning"]
    if originals["cache_size_limit"] is not None:
        dynamo_config.cache_size_limit = originals["cache_size_limit"]
    if originals["accumulated_cache_size_limit"] is not None:
        dynamo_config.accumulated_cache_size_limit = originals[
            "accumulated_cache_size_limit"
        ]


def test_force_autotune_patches_inductor(monkeypatch, _restore_inductor_state):
    inductor_utils, _, inductor_config, _ = _restore_inductor_state
    monkeypatch.setattr(utils, "_inductor_max_autotune_gemm_forced", False)

    utils.force_inductor_max_autotune_gemm_on_small_gpus()

    assert inductor_utils.is_big_gpu() is True
    assert getattr(inductor_utils.is_big_gpu, "__vllm_big_gpu_override__", False)
    assert inductor_config.max_autotune is True
    assert inductor_config.max_autotune_gemm is True
    assert inductor_config.max_autotune_gemm_backends == "ATEN,TRITON"
    assert inductor_config.coordinate_descent_tuning is True


def test_force_large_dynamo_caches_raises_limits(monkeypatch, _restore_inductor_state):
    _, _, _, dynamo_config = _restore_inductor_state
    monkeypatch.setattr(utils, "_dynamo_compile_caches_forced", False)
    if hasattr(dynamo_config, "cache_size_limit"):
        dynamo_config.cache_size_limit = 1
    dynamo_config.accumulated_cache_size_limit = 1

    utils.force_large_dynamo_compile_caches()

    if hasattr(dynamo_config, "cache_size_limit"):
        assert dynamo_config.cache_size_limit >= utils._DYNAMO_CACHE_SIZE_LIMIT
    assert (
        dynamo_config.accumulated_cache_size_limit
        >= utils._DYNAMO_ACCUMULATED_CACHE_SIZE_LIMIT
    )


_cuda_bf16_ok = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8


@pytest.mark.skipif(not _cuda_bf16_ok, reason="CUDA BF16 required")
@pytest.mark.parametrize("has_bias", [False, True])
def test_compile_path_matches_f_linear_cuda(has_bias, monkeypatch):
    torch.manual_seed(0)
    x = torch.randn(4, 64, dtype=torch.bfloat16, device="cuda")
    w = torch.randn(128, 64, dtype=torch.bfloat16, device="cuda")
    bias = torch.randn(128, dtype=torch.bfloat16, device="cuda") if has_bias else None
    monkeypatch.setattr(
        utils,
        "_should_use_unquant_bf16_linear_torch_compile",
        lambda x, w, b: True,
    )

    out = utils._torch_compile_bf16_unquantized_gemm(None, x, w, bias)

    torch.testing.assert_close(
        out,
        torch.nn.functional.linear(x, w, bias),
        atol=2e-2,
        rtol=2e-2,
    )
