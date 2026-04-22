# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import MagicMock

import pytest
import torch

from vllm.model_executor.layers import utils
from vllm.platforms import current_platform


def _mock_flashinfer_mm(a, b, bias=None, backend="auto"):
    return torch.nn.functional.linear(a, b.t(), bias)


def test_maybe_flashinfer_bf16_unquantized_gemm_uses_flashinfer_on_sm100(
    monkeypatch,
):
    x = torch.randn(2, 3, 16, dtype=torch.bfloat16)
    weight = torch.randn(8, 16, dtype=torch.bfloat16)
    bias = torch.randn(8, dtype=torch.bfloat16)
    expected = torch.nn.functional.linear(x, weight, bias)

    monkeypatch.setattr(current_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(
        current_platform, "is_device_capability_family", lambda cc: cc == 100
    )

    flashinfer_mm_mock = MagicMock(side_effect=_mock_flashinfer_mm)
    monkeypatch.setattr(
        "vllm.utils.flashinfer.has_flashinfer_bf16_gemm", lambda: True
    )
    monkeypatch.setattr("vllm.utils.flashinfer.flashinfer_bf16_mm", flashinfer_mm_mock)

    out = utils.maybe_flashinfer_bf16_unquantized_gemm(x, weight, bias, None)

    flashinfer_mm_mock.assert_called_once()
    assert flashinfer_mm_mock.call_args.kwargs["backend"] == "auto"
    torch.testing.assert_close(out, expected)


def test_maybe_flashinfer_bf16_unquantized_gemm_allows_forced_cudnn_off_sm100(
    monkeypatch,
):
    x = torch.randn(2, 16, dtype=torch.bfloat16)
    weight = torch.randn(8, 16, dtype=torch.bfloat16)
    bias = torch.randn(8, dtype=torch.bfloat16)
    expected = torch.nn.functional.linear(x, weight, bias)

    monkeypatch.setattr(current_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(current_platform, "is_device_capability_family", lambda _: False)

    flashinfer_mm_mock = MagicMock(side_effect=_mock_flashinfer_mm)
    monkeypatch.setattr(
        "vllm.utils.flashinfer.has_flashinfer_bf16_gemm", lambda: True
    )
    monkeypatch.setattr("vllm.utils.flashinfer.flashinfer_bf16_mm", flashinfer_mm_mock)

    out = utils.maybe_flashinfer_bf16_unquantized_gemm(x, weight, bias, "cudnn")

    flashinfer_mm_mock.assert_called_once()
    assert flashinfer_mm_mock.call_args.kwargs["backend"] == "cudnn"
    torch.testing.assert_close(out, expected)


def test_default_unquantized_gemm_cpu_falls_back_to_torch():
    x = torch.randn(2, 16, dtype=torch.bfloat16)
    weight = torch.randn(8, 16, dtype=torch.bfloat16)
    bias = torch.randn(8, dtype=torch.bfloat16)
    expected = torch.nn.functional.linear(x, weight, bias)

    out = utils.default_unquantized_gemm(None, x, weight, bias)

    torch.testing.assert_close(out, expected)


def test_maybe_flashinfer_bf16_unquantized_gemm_propagates_forced_backend(
    monkeypatch,
):
    x = torch.randn(2, 16, dtype=torch.bfloat16)
    weight = torch.randn(8, 16, dtype=torch.bfloat16)
    flashinfer_mm_mock = MagicMock(side_effect=RuntimeError("backend unsupported"))

    monkeypatch.setattr(current_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(current_platform, "is_device_capability_family", lambda _: False)
    monkeypatch.setattr(
        "vllm.utils.flashinfer.has_flashinfer_bf16_gemm", lambda: True
    )
    monkeypatch.setattr("vllm.utils.flashinfer.flashinfer_bf16_mm", flashinfer_mm_mock)

    with pytest.raises(RuntimeError, match="backend unsupported"):
        utils.maybe_flashinfer_bf16_unquantized_gemm(x, weight, None, "tgv")
