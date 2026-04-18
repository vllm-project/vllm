# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the torch.compile-wrapped BF16 linear fast path.

The fast path wraps ``F.linear`` in ``torch.compile(mode=
"max-autotune-no-cudagraphs", dynamic=True)`` so inductor benchmarks
triton/aten/cutlass candidates per shape without compiling the whole
model. It is opt-in via ``VLLM_ENABLE_UNQUANT_BF16_LINEAR_TORCH_COMPILE``.

These tests exercise the dispatch decision, eligibility predicate,
per-shape cache keys, CUDA-graph capture safety, and the optional
``is_big_gpu`` override. The only CUDA-required tests are the end-to-end
numerical correctness checks; the rest run on CPU-only hosts.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm.model_executor.layers import utils
from vllm.platforms import current_platform


@pytest.fixture(autouse=True)
def _reset_compile_state(monkeypatch):
    """Isolate each test from module-level compile caches."""
    utils._unquant_bf16_linear_cache.clear()
    utils._unquant_bf16_linear_capture_safe_keys.clear()
    monkeypatch.setattr(
        utils, "_unquant_bf16_linear_torch_compile_disabled", False
    )
    monkeypatch.setattr(
        utils, "_unquant_bf16_linear_torch_compile_configured", True
    )
    yield
    utils._unquant_bf16_linear_cache.clear()
    utils._unquant_bf16_linear_capture_safe_keys.clear()


# --------------------------------------------------------------------- #
# dispatch_unquantized_gemm routing
# --------------------------------------------------------------------- #


def test_dispatch_prefers_compile_path_when_env_set(monkeypatch):
    monkeypatch.setenv("VLLM_ENABLE_UNQUANT_BF16_LINEAR_TORCH_COMPILE", "1")
    monkeypatch.setattr(current_platform, "is_rocm", lambda: False)
    monkeypatch.setattr(current_platform, "is_cpu", lambda: False)
    monkeypatch.setattr(current_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(utils, "_TINYGEMM_AVAILABLE", True)

    assert (
        utils.dispatch_unquantized_gemm()
        is utils._torch_compile_bf16_unquantized_gemm
    )


def test_dispatch_env_off_falls_back_to_tinygemm(monkeypatch):
    monkeypatch.setenv("VLLM_ENABLE_UNQUANT_BF16_LINEAR_TORCH_COMPILE", "0")
    monkeypatch.setattr(current_platform, "is_rocm", lambda: False)
    monkeypatch.setattr(current_platform, "is_cpu", lambda: False)
    monkeypatch.setattr(current_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(utils, "_TINYGEMM_AVAILABLE", True)

    assert (
        utils.dispatch_unquantized_gemm()
        is utils._tinygemm_unquantized_gemm
    )


def test_dispatch_env_on_but_non_cuda_does_not_select_compile(monkeypatch):
    monkeypatch.setenv("VLLM_ENABLE_UNQUANT_BF16_LINEAR_TORCH_COMPILE", "1")
    monkeypatch.setattr(current_platform, "is_rocm", lambda: True)
    monkeypatch.setattr(current_platform, "is_cpu", lambda: False)
    monkeypatch.setattr(current_platform, "is_cuda", lambda: False)

    assert (
        utils.dispatch_unquantized_gemm()
        is not utils._torch_compile_bf16_unquantized_gemm
    )


# --------------------------------------------------------------------- #
# _should_use_unquant_bf16_linear_torch_compile predicate
# --------------------------------------------------------------------- #


def _bf16_cuda(shape, contiguous=True):
    # No real CUDA needed: fake an is_cuda=True tensor via monkeypatch on the
    # .is_cuda attribute of a CPU tensor. BUT: is_cuda is a property, so we
    # patch the predicate helper's view of it via a stub tensor class.
    class _Stub(torch.Tensor):
        pass

    t = torch.zeros(shape, dtype=torch.bfloat16).as_subclass(_Stub)
    object.__setattr__(t, "_fake_is_cuda", True)
    if not contiguous and t.ndim >= 2:
        t = t.transpose(-1, -2)
    return t


def test_predicate_rejects_non_cuda_tensor():
    x = torch.zeros(4, 8, dtype=torch.bfloat16)  # CPU tensor → is_cuda=False
    w = torch.zeros(16, 8, dtype=torch.bfloat16)
    assert not utils._should_use_unquant_bf16_linear_torch_compile(x, w, None)


def test_predicate_rejects_non_bf16_dtype():
    x = torch.zeros(4, 8, dtype=torch.float32)
    w = torch.zeros(16, 8, dtype=torch.float32)
    # Even if we faked is_cuda, wrong dtype should exit early.
    with patch.object(type(x), "is_cuda", property(lambda self: True)):
        assert not utils._should_use_unquant_bf16_linear_torch_compile(
            x, w, None
        )


def test_predicate_rejects_bias_dtype_mismatch():
    x = torch.zeros(4, 8, dtype=torch.bfloat16)
    w = torch.zeros(16, 8, dtype=torch.bfloat16)
    bias = torch.zeros(16, dtype=torch.float32)
    with patch.object(type(x), "is_cuda", property(lambda self: True)):
        assert not utils._should_use_unquant_bf16_linear_torch_compile(
            x, w, bias
        )


def test_predicate_rejects_non_contiguous_input():
    x = torch.zeros(4, 16, dtype=torch.bfloat16)[:, ::2]
    w = torch.zeros(16, 8, dtype=torch.bfloat16)
    with patch.object(type(x), "is_cuda", property(lambda self: True)):
        assert not utils._should_use_unquant_bf16_linear_torch_compile(
            x, w, None
        )


def test_predicate_rejects_when_dynamo_is_compiling(monkeypatch):
    x = torch.zeros(4, 8, dtype=torch.bfloat16)
    w = torch.zeros(16, 8, dtype=torch.bfloat16)
    monkeypatch.setattr(torch._dynamo, "is_compiling", lambda: True)
    with patch.object(type(x), "is_cuda", property(lambda self: True)):
        assert not utils._should_use_unquant_bf16_linear_torch_compile(
            x, w, None
        )


def test_predicate_rejects_when_previously_disabled(monkeypatch):
    x = torch.zeros(4, 8, dtype=torch.bfloat16)
    w = torch.zeros(16, 8, dtype=torch.bfloat16)
    monkeypatch.setattr(
        utils, "_unquant_bf16_linear_torch_compile_disabled", True
    )
    with patch.object(type(x), "is_cuda", property(lambda self: True)):
        assert not utils._should_use_unquant_bf16_linear_torch_compile(
            x, w, None
        )


# --------------------------------------------------------------------- #
# _unquant_bf16_linear_cache_key
# --------------------------------------------------------------------- #


def test_cache_key_includes_ndim_weight_shape_and_bias_presence():
    x_2d = torch.zeros(4, 8)
    x_3d = torch.zeros(2, 4, 8)
    w1 = torch.zeros(16, 8)
    w2 = torch.zeros(32, 8)
    bias = torch.zeros(16)

    k_2d = utils._unquant_bf16_linear_cache_key(x_2d, w1, None)
    k_3d = utils._unquant_bf16_linear_cache_key(x_3d, w1, None)
    k_w2 = utils._unquant_bf16_linear_cache_key(x_2d, w2, None)
    k_bias = utils._unquant_bf16_linear_cache_key(x_2d, w1, bias)

    # Different ndim, different weight shape, and bias-presence all split.
    assert len({k_2d, k_3d, k_w2, k_bias}) == 4


def test_cache_key_ignores_batch_dim():
    # Two 2D inputs that differ only in the leading (M) dim share a key:
    # dynamic=True compile lets a single kernel serve varying M.
    x_a = torch.zeros(4, 8)
    x_b = torch.zeros(64, 8)
    w = torch.zeros(16, 8)
    assert utils._unquant_bf16_linear_cache_key(
        x_a, w, None
    ) == utils._unquant_bf16_linear_cache_key(x_b, w, None)


# --------------------------------------------------------------------- #
# _torch_compile_bf16_unquantized_gemm fallback behavior
# --------------------------------------------------------------------- #


def test_compile_gemm_falls_back_to_f_linear_when_ineligible():
    # CPU bf16 tensors: predicate returns False → must pass through F.linear.
    x = torch.randn(4, 8, dtype=torch.bfloat16)
    w = torch.randn(16, 8, dtype=torch.bfloat16)
    out = utils._torch_compile_bf16_unquantized_gemm(None, x, w, None)
    torch.testing.assert_close(
        out, torch.nn.functional.linear(x, w, None), atol=0.0, rtol=0.0
    )


# --------------------------------------------------------------------- #
# CUDA-graph capture safety (simulated via monkeypatch)
# --------------------------------------------------------------------- #


def test_capture_skips_uncompiled_shape(monkeypatch):
    """During capture, a shape whose key is not in the warmed-set returns None
    and the outer function falls back to F.linear."""
    x = torch.randn(4, 8, dtype=torch.bfloat16)
    w = torch.randn(16, 8, dtype=torch.bfloat16)

    # Pretend we're in a CUDA stream capture.
    monkeypatch.setattr(
        torch.cuda, "is_current_stream_capturing", lambda: True
    )

    # Force the eligibility predicate to pass.
    monkeypatch.setattr(
        utils, "_should_use_unquant_bf16_linear_torch_compile",
        lambda x, w, b: True,
    )

    sentinel_compile = MagicMock()
    monkeypatch.setattr(
        utils,
        "_get_or_create_unquant_bf16_linear_kernel",
        lambda key, allow_new_compile: (
            None if not allow_new_compile else sentinel_compile
        ),
    )

    out = utils._torch_compile_bf16_unquantized_gemm(None, x, w, None)

    # Must have fallen back — the compiled kernel was NOT invoked.
    sentinel_compile.assert_not_called()
    torch.testing.assert_close(
        out, torch.nn.functional.linear(x, w, None), atol=0.0, rtol=0.0
    )


def test_capture_reuses_warmed_shape(monkeypatch):
    """During capture, a shape already in the warmed-set uses the cached
    compiled kernel."""
    x = torch.randn(4, 8, dtype=torch.bfloat16)
    w = torch.randn(16, 8, dtype=torch.bfloat16)
    key = utils._unquant_bf16_linear_cache_key(x, w, None)

    # Pretend the kernel was warmed in an earlier non-capture call.
    cached = MagicMock(return_value=torch.nn.functional.linear(x, w, None))
    utils._unquant_bf16_linear_cache[key] = cached
    utils._unquant_bf16_linear_capture_safe_keys.add(key)

    monkeypatch.setattr(
        torch.cuda, "is_current_stream_capturing", lambda: True
    )
    monkeypatch.setattr(
        utils, "_should_use_unquant_bf16_linear_torch_compile",
        lambda x, w, b: True,
    )

    utils._torch_compile_bf16_unquantized_gemm(None, x, w, None)
    cached.assert_called_once()


def test_compile_failure_disables_fast_path(monkeypatch):
    """A raising compile should flip the module-level disabled flag and
    fall through to F.linear instead of propagating."""
    x = torch.randn(4, 8, dtype=torch.bfloat16)
    w = torch.randn(16, 8, dtype=torch.bfloat16)

    monkeypatch.setattr(
        utils, "_should_use_unquant_bf16_linear_torch_compile",
        lambda x, w, b: True,
    )
    monkeypatch.setattr(
        torch.cuda, "is_current_stream_capturing", lambda: False
    )

    # The cached kernel raises when called — the apply() helper's try/except
    # must catch it, set the disabled flag, and return None.
    fake_kernel = MagicMock(
        side_effect=RuntimeError("inductor boom")
    )
    monkeypatch.setattr(
        utils,
        "_get_or_create_unquant_bf16_linear_kernel",
        lambda key, allow_new_compile: fake_kernel,
    )

    out = utils._torch_compile_bf16_unquantized_gemm(None, x, w, None)
    torch.testing.assert_close(
        out, torch.nn.functional.linear(x, w, None), atol=0.0, rtol=0.0
    )
    assert utils._unquant_bf16_linear_torch_compile_disabled is True


# --------------------------------------------------------------------- #
# is_big_gpu override
# --------------------------------------------------------------------- #


def test_big_gpu_override_noop_when_env_off(monkeypatch):
    monkeypatch.setenv("VLLM_INDUCTOR_OVERRIDE_BIG_GPU", "0")
    monkeypatch.setattr(utils, "_inductor_big_gpu_override_applied", False)

    pytest.importorskip("torch._inductor.utils")
    import torch._inductor.utils as inductor_utils

    before = inductor_utils.is_big_gpu
    utils._maybe_override_inductor_is_big_gpu()
    assert inductor_utils.is_big_gpu is before


def test_big_gpu_override_patches_when_env_on(monkeypatch):
    pytest.importorskip("torch._inductor.utils")
    import torch._inductor.utils as inductor_utils

    original = inductor_utils.is_big_gpu
    try:
        monkeypatch.setenv("VLLM_INDUCTOR_OVERRIDE_BIG_GPU", "1")
        monkeypatch.setattr(
            utils, "_inductor_big_gpu_override_applied", False
        )

        utils._maybe_override_inductor_is_big_gpu()

        # Patched function always returns True regardless of device.
        assert inductor_utils.is_big_gpu() is True
        assert inductor_utils.is_big_gpu(0) is True
        assert getattr(
            inductor_utils.is_big_gpu, "__vllm_big_gpu_override__", False
        )
    finally:
        inductor_utils.is_big_gpu = original


def test_big_gpu_override_idempotent(monkeypatch):
    pytest.importorskip("torch._inductor.utils")
    import torch._inductor.utils as inductor_utils

    original = inductor_utils.is_big_gpu
    try:
        monkeypatch.setenv("VLLM_INDUCTOR_OVERRIDE_BIG_GPU", "1")
        monkeypatch.setattr(
            utils, "_inductor_big_gpu_override_applied", False
        )

        utils._maybe_override_inductor_is_big_gpu()
        first = inductor_utils.is_big_gpu
        utils._maybe_override_inductor_is_big_gpu()
        assert inductor_utils.is_big_gpu is first
    finally:
        inductor_utils.is_big_gpu = original


# --------------------------------------------------------------------- #
# End-to-end numerical correctness (CUDA, BF16, SM>=9)
# --------------------------------------------------------------------- #

_cuda_ok = (
    torch.cuda.is_available()
    and torch.cuda.get_device_capability()[0] >= 8  # bf16 on CUDA
)


@pytest.mark.skipif(not _cuda_ok, reason="CUDA BF16 required")
@pytest.mark.parametrize(
    "m,n,k,has_bias",
    [
        (4, 128, 64, False),
        (4, 128, 64, True),
        (17, 256, 128, False),  # odd M to exercise dynamic shape
    ],
)
def test_compile_path_matches_f_linear(m, n, k, has_bias, monkeypatch):
    """One full end-to-end run: compile the kernel, invoke it, compare
    against plain F.linear. Small shapes keep autotune cost bounded."""
    torch.manual_seed(0)
    x = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
    w = torch.randn(n, k, dtype=torch.bfloat16, device="cuda")
    bias = (
        torch.randn(n, dtype=torch.bfloat16, device="cuda")
        if has_bias
        else None
    )

    # Force the predicate to accept (it would accept anyway on CUDA bf16
    # contiguous, but be explicit to avoid flakiness from other state).
    monkeypatch.setattr(
        utils,
        "_should_use_unquant_bf16_linear_torch_compile",
        lambda x, w, b: True,
    )

    out = utils._torch_compile_bf16_unquantized_gemm(None, x, w, bias)
    ref = torch.nn.functional.linear(x, w, bias)

    # Inductor selects different backends than eager cuBLAS, so allow
    # small numerical drift consistent with BF16 accumulation.
    torch.testing.assert_close(out, ref, atol=2e-2, rtol=2e-2)


@pytest.mark.skipif(not _cuda_ok, reason="CUDA BF16 required")
def test_compile_path_reuses_cache_for_same_shape(monkeypatch):
    torch.manual_seed(0)
    x = torch.randn(4, 64, dtype=torch.bfloat16, device="cuda")
    w = torch.randn(128, 64, dtype=torch.bfloat16, device="cuda")

    monkeypatch.setattr(
        utils,
        "_should_use_unquant_bf16_linear_torch_compile",
        lambda x, w, b: True,
    )

    utils._torch_compile_bf16_unquantized_gemm(None, x, w, None)
    size_after_first = len(utils._unquant_bf16_linear_cache)

    # Same shape, new tensors — must reuse the cached kernel.
    x2 = torch.randn_like(x)
    utils._torch_compile_bf16_unquantized_gemm(None, x2, w, None)
    assert len(utils._unquant_bf16_linear_cache) == size_after_first

    # Different weight shape → new cache entry.
    w3 = torch.randn(256, 64, dtype=torch.bfloat16, device="cuda")
    utils._torch_compile_bf16_unquantized_gemm(None, x, w3, None)
    assert len(utils._unquant_bf16_linear_cache) == size_after_first + 1
