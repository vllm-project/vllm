# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for tinygemm_bf16 optimization in default_unquantized_gemm."""

from unittest.mock import MagicMock, patch

import torch

from vllm.model_executor.layers import utils


class TestTinygemmBf16Availability:
    """Tests for _tinygemm_bf16_available function."""

    @patch("vllm.model_executor.layers.utils.has_flashinfer")
    def test_returns_false_when_flashinfer_unavailable(self, mock_has_flashinfer):
        mock_has_flashinfer.return_value = False
        result = utils._tinygemm_bf16_available()
        assert result is False

    @patch("vllm.model_executor.layers.utils.has_flashinfer")
    @patch("vllm.model_executor.layers.utils.current_platform")
    def test_returns_false_when_device_capability_below_9(
        self, mock_platform, mock_has_flashinfer
    ):
        mock_has_flashinfer.return_value = True
        mock_capability = MagicMock()
        mock_capability.__getitem__ = lambda self, key: 8 if key == 0 else 0
        mock_platform.get_device_capability.return_value = mock_capability

        result = utils._tinygemm_bf16_available()
        assert result is False

    @patch("vllm.model_executor.layers.utils.has_flashinfer")
    @patch("vllm.model_executor.layers.utils.current_platform")
    def test_returns_true_when_flashinfer_and_sm90_available(
        self, mock_platform, mock_has_flashinfer
    ):
        mock_has_flashinfer.return_value = True
        mock_capability = MagicMock()
        mock_capability.__getitem__ = lambda self, key: 9 if key == 0 else 0
        mock_platform.get_device_capability.return_value = mock_capability

        result = utils._tinygemm_bf16_available()
        assert result is True

    @patch("vllm.model_executor.layers.utils.has_flashinfer")
    @patch("vllm.model_executor.layers.utils.current_platform")
    def test_returns_false_when_exception_raised(
        self, mock_platform, mock_has_flashinfer
    ):
        mock_has_flashinfer.return_value = True
        mock_platform.get_device_capability.side_effect = RuntimeError("CUDA error")

        result = utils._tinygemm_bf16_available()
        assert result is False


class TestDefaultUnquantizedGemm:
    """Tests for default_unquantized_gemm function with tinygemm_bf16."""

    def test_falls_back_to_linear_without_flashinfer(self):
        layer = torch.nn.Linear(16, 8, bias=False)
        x = torch.randn(2, 16, dtype=torch.bfloat16)
        weight = torch.randn(8, 16, dtype=torch.bfloat16)

        with patch.object(utils, "_tinygemm_bf16_available", return_value=False):
            out = utils.default_unquantized_gemm(layer, x, weight, None)
            expected = torch.nn.functional.linear(x, weight, None)

        torch.testing.assert_close(out, expected)

    @patch("vllm.model_executor.layers.utils.flashinfer_tinygemm_bf16")
    @patch("vllm.model_executor.layers.utils._tinygemm_bf16_available")
    def test_uses_tinygemm_when_all_conditions_met(self, mock_available, mock_tinygemm):
        mock_available.return_value = True
        mock_tinygemm.return_value = None

        layer = torch.nn.Linear(16, 16, bias=False)
        x = torch.randn(2, 16, dtype=torch.bfloat16)
        weight = torch.randn(16, 16, dtype=torch.bfloat16)

        utils.default_unquantized_gemm(layer, x, weight, None)

        mock_tinygemm.assert_called_once()

    @patch("vllm.model_executor.layers.utils._tinygemm_bf16_available")
    def test_skips_tinygemm_when_batch_gt_8(self, mock_available):
        mock_available.return_value = True

        layer = torch.nn.Linear(16, 16, bias=False)
        x = torch.randn(16, 16, dtype=torch.bfloat16)
        weight = torch.randn(16, 16, dtype=torch.bfloat16)

        with patch("vllm.model_executor.layers.utils.flashinfer_tinygemm_bf16") as mock:
            out = utils.default_unquantized_gemm(layer, x, weight, None)
            mock.assert_not_called()

        expected = torch.nn.functional.linear(x, weight, None)
        torch.testing.assert_close(out, expected)

    @patch("vllm.model_executor.layers.utils._tinygemm_bf16_available")
    def test_skips_tinygemm_when_dtype_not_bf16(self, mock_available):
        mock_available.return_value = True

        layer = torch.nn.Linear(16, 16, bias=False)
        x = torch.randn(2, 16, dtype=torch.float16)
        weight = torch.randn(16, 16, dtype=torch.float16)

        with patch("vllm.model_executor.layers.utils.flashinfer_tinygemm_bf16") as mock:
            out = utils.default_unquantized_gemm(layer, x, weight, None)
            mock.assert_not_called()

        expected = torch.nn.functional.linear(x, weight, None)
        torch.testing.assert_close(out, expected)

    @patch("vllm.model_executor.layers.utils._tinygemm_bf16_available")
    def test_skips_tinygemm_when_weight_not_divisible_by_16(self, mock_available):
        mock_available.return_value = True

        layer = torch.nn.Linear(16, 8, bias=False)
        x = torch.randn(2, 16, dtype=torch.bfloat16)
        weight = torch.randn(8, 16, dtype=torch.bfloat16)

        with patch("vllm.model_executor.layers.utils.flashinfer_tinygemm_bf16") as mock:
            out = utils.default_unquantized_gemm(layer, x, weight, None)
            mock.assert_not_called()

        expected = torch.nn.functional.linear(x, weight, None)
        torch.testing.assert_close(out, expected)

    @patch("vllm.model_executor.layers.utils._tinygemm_bf16_available")
    def test_skips_tinygemm_when_input_not_contiguous(self, mock_available):
        mock_available.return_value = True

        layer = torch.nn.Linear(16, 16, bias=False)
        x = torch.randn(2, 16, dtype=torch.bfloat16).transpose(0, 1)
        weight = torch.randn(16, 16, dtype=torch.bfloat16)

        assert not x.is_contiguous()

        with patch("vllm.model_executor.layers.utils.flashinfer_tinygemm_bf16") as mock:
            out = utils.default_unquantized_gemm(layer, x, weight, None)
            mock.assert_not_called()

        expected = torch.nn.functional.linear(x, weight, None)
        torch.testing.assert_close(out, expected)

    @patch("vllm.model_executor.layers.utils.flashinfer_tinygemm_bf16")
    @patch("vllm.model_executor.layers.utils._tinygemm_bf16_available")
    def test_handles_bias_correctly(self, mock_available, mock_tinygemm):
        mock_available.return_value = True
        mock_tinygemm.return_value = None

        layer = torch.nn.Linear(16, 16, bias=True)
        x = torch.randn(2, 16, dtype=torch.bfloat16)
        weight = torch.randn(16, 16, dtype=torch.bfloat16)
        bias = torch.randn(16, dtype=torch.bfloat16)

        utils.default_unquantized_gemm(layer, x, weight, bias)

        call_kwargs = mock_tinygemm.call_args.kwargs
        assert call_kwargs.get("bias") is not None

    @patch("vllm.model_executor.layers.utils.flashinfer_tinygemm_bf16")
    @patch("vllm.model_executor.layers.utils._tinygemm_bf16_available")
    def test_handles_3d_input_tensor(self, mock_available, mock_tinygemm):
        mock_available.return_value = True
        mock_tinygemm.return_value = None

        layer = torch.nn.Linear(16, 16, bias=False)
        x = torch.randn(2, 4, 16, dtype=torch.bfloat16)
        weight = torch.randn(16, 16, dtype=torch.bfloat16)

        out = utils.default_unquantized_gemm(layer, x, weight, None)

        assert out.shape == (2, 4, 16)

    @patch("vllm.model_executor.layers.utils.flashinfer_tinygemm_bf16")
    @patch("vllm.model_executor.layers.utils._tinygemm_bf16_available")
    def test_skips_tinygemm_when_bias_wrong_dtype(self, mock_available, mock_tinygemm):
        mock_available.return_value = True
        mock_tinygemm.return_value = None

        layer = torch.nn.Linear(16, 16, bias=True)
        x = torch.randn(2, 16, dtype=torch.bfloat16)
        weight = torch.randn(16, 16, dtype=torch.bfloat16)
        bias = torch.randn(16, dtype=torch.float16)

        out = utils.default_unquantized_gemm(layer, x, weight, bias)

        expected = torch.nn.functional.linear(x, weight, bias)
        torch.testing.assert_close(out, expected)
        mock_tinygemm.assert_not_called()
