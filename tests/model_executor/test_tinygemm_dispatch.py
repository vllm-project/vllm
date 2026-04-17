# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for FlashInfer tinygemm BF16 dispatch predicate and routing."""

from unittest.mock import MagicMock

import pytest
import torch

from vllm.model_executor.layers import utils
from vllm.platforms import current_platform


@pytest.fixture(scope="module")
def _tinygemm_op():
    """Ensure torch.ops.vllm.tinygemm_bf16 is registered.

    On hosts without FlashInfer/SM90+ the real op is not registered by
    ``_init_tinygemm``; register a CPU fallback so the dispatch predicate
    can be exercised in CI.
    """
    if hasattr(torch.ops.vllm, "tinygemm_bf16"):
        yield
        return

    lib = torch.library.Library("vllm", "FRAGMENT")
    lib.define(
        "tinygemm_bf16(Tensor input, Tensor weight, Tensor bias) -> Tensor"
    )
    lib.impl(
        "tinygemm_bf16",
        lambda x, w, b: torch.nn.functional.linear(x, w, b),
        "CompositeExplicitAutograd",
    )
    yield
    lib._destroy()


@pytest.fixture
def tinygemm_spy(_tinygemm_op, monkeypatch):
    spy = MagicMock(
        side_effect=lambda x, w, b: torch.nn.functional.linear(x, w, b)
    )
    monkeypatch.setattr(torch.ops.vllm, "tinygemm_bf16", spy)
    return spy


@pytest.mark.parametrize("num_tokens", [1, 8])
def test_small_m_bf16_uses_tinygemm(tinygemm_spy, num_tokens):
    x = torch.randn(num_tokens, 16, dtype=torch.bfloat16)
    w = torch.randn(32, 16, dtype=torch.bfloat16)

    out = utils._tinygemm_unquantized_gemm(None, x, w)

    tinygemm_spy.assert_called_once()
    ref = torch.nn.functional.linear(x, w, None)
    torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)


def test_m_greater_than_8_falls_back(tinygemm_spy):
    x = torch.randn(9, 16, dtype=torch.bfloat16)
    w = torch.randn(32, 16, dtype=torch.bfloat16)

    out = utils._tinygemm_unquantized_gemm(None, x, w)

    tinygemm_spy.assert_not_called()
    torch.testing.assert_close(
        out, torch.nn.functional.linear(x, w, None), atol=1e-2, rtol=1e-2
    )


def test_non_bf16_falls_back(tinygemm_spy):
    x = torch.randn(1, 16, dtype=torch.float32)
    w = torch.randn(32, 16, dtype=torch.float32)

    utils._tinygemm_unquantized_gemm(None, x, w)

    tinygemm_spy.assert_not_called()


def test_n_not_multiple_of_16_falls_back(tinygemm_spy):
    x = torch.randn(1, 16, dtype=torch.bfloat16)
    w = torch.randn(15, 16, dtype=torch.bfloat16)

    utils._tinygemm_unquantized_gemm(None, x, w)

    tinygemm_spy.assert_not_called()


def test_non_contiguous_input_falls_back(tinygemm_spy):
    x = torch.randn(1, 32, dtype=torch.bfloat16)[:, ::2]
    w = torch.randn(32, 16, dtype=torch.bfloat16)

    utils._tinygemm_unquantized_gemm(None, x, w)

    tinygemm_spy.assert_not_called()


def test_3d_input_reshapes_correctly(tinygemm_spy):
    x = torch.randn(2, 3, 16, dtype=torch.bfloat16)  # 6 tokens total
    w = torch.randn(32, 16, dtype=torch.bfloat16)

    out = utils._tinygemm_unquantized_gemm(None, x, w)

    tinygemm_spy.assert_called_once()
    call_args = tinygemm_spy.call_args
    assert call_args.args[0].shape == (6, 16)
    assert out.shape == (2, 3, 32)
    torch.testing.assert_close(
        out, torch.nn.functional.linear(x, w, None), atol=1e-2, rtol=1e-2
    )


def test_bias_passed_through(tinygemm_spy):
    x = torch.randn(1, 16, dtype=torch.bfloat16)
    w = torch.randn(32, 16, dtype=torch.bfloat16)
    bias = torch.randn(32, dtype=torch.bfloat16)

    out = utils._tinygemm_unquantized_gemm(None, x, w, bias)

    tinygemm_spy.assert_called_once()
    passed_bias = tinygemm_spy.call_args.args[2]
    torch.testing.assert_close(passed_bias, bias)
    torch.testing.assert_close(
        out, torch.nn.functional.linear(x, w, bias), atol=1e-2, rtol=1e-2
    )


def test_non_bf16_bias_falls_back(tinygemm_spy):
    x = torch.randn(1, 16, dtype=torch.bfloat16)
    w = torch.randn(32, 16, dtype=torch.bfloat16)
    bias = torch.randn(32, dtype=torch.float32)

    utils._tinygemm_unquantized_gemm(None, x, w, bias)

    tinygemm_spy.assert_not_called()


def test_dispatch_unavailable_returns_default(monkeypatch):
    monkeypatch.setattr(utils, "_TINYGEMM_AVAILABLE", False)
    monkeypatch.setattr(current_platform, "is_rocm", lambda: False)
    monkeypatch.setattr(current_platform, "is_cpu", lambda: False)

    assert utils.dispatch_unquantized_gemm() is utils.default_unquantized_gemm


def test_dispatch_available_returns_tinygemm(monkeypatch):
    monkeypatch.setattr(utils, "_TINYGEMM_AVAILABLE", True)
    monkeypatch.setattr(current_platform, "is_rocm", lambda: False)
    monkeypatch.setattr(current_platform, "is_cpu", lambda: False)

    assert utils.dispatch_unquantized_gemm() is utils._tinygemm_unquantized_gemm
