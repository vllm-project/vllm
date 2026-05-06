# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tests for deep_gemm_warmup guard when DeepGEMM is not available.

_fp8_linear_may_use_deep_gemm was calling get_mk_alignment_for_contiguous_layout()
unconditionally before the FP8 type check, causing RuntimeError for any model
when deep_gemm is not installed — including standard bf16 models with no FP8 layers.

Regression test for https://github.com/vllm-project/vllm/issues/41849.
"""

from unittest.mock import MagicMock, patch

import pytest
import torch


def _make_plain_linear():
    """A plain torch.nn.Linear — no FP8 quantization."""
    return torch.nn.Linear(64, 64)


def _make_mock_fp8_linear():
    """A minimal mock that looks like an Fp8-quantized LinearBase."""
    from vllm.model_executor.layers.linear import LinearBase
    from vllm.model_executor.layers.quantization.fp8 import Fp8LinearMethod

    m = MagicMock(spec=LinearBase)
    qm = MagicMock(spec=Fp8LinearMethod)
    qm.block_quant = True
    qm.use_marlin = False
    m.quant_method = qm
    return m


def test_fp8_linear_returns_false_when_deep_gemm_unavailable_plain_module():
    """
    _fp8_linear_may_use_deep_gemm must return False (not raise) for a plain
    nn.Linear when is_deep_gemm_supported() is False.

    Before the fix: RuntimeError: DeepGEMM backend is not available
    After the fix:  returns False immediately via early-return guard
    """
    with patch(
        "vllm.model_executor.warmup.deep_gemm_warmup.is_deep_gemm_supported",
        return_value=False,
    ):
        from vllm.model_executor.warmup.deep_gemm_warmup import (
            _fp8_linear_may_use_deep_gemm,
        )
        module = _make_plain_linear()
        result = _fp8_linear_may_use_deep_gemm(module)
        assert result is False, (
            "Expected False when deep_gemm is unavailable, "
            f"got {result!r}"
        )


def test_fp8_linear_returns_false_when_deep_gemm_unavailable_mock_fp8():
    """
    Same as above but with a mock that passes isinstance checks — the guard
    must fire before get_mk_alignment_for_contiguous_layout() is called.
    """
    with patch(
        "vllm.model_executor.warmup.deep_gemm_warmup.is_deep_gemm_supported",
        return_value=False,
    ), patch(
        "vllm.model_executor.warmup.deep_gemm_warmup"
        ".get_mk_alignment_for_contiguous_layout"
    ) as mock_align:
        from vllm.model_executor.warmup.deep_gemm_warmup import (
            _fp8_linear_may_use_deep_gemm,
        )
        module = _make_mock_fp8_linear()
        result = _fp8_linear_may_use_deep_gemm(module)
        assert result is False
        mock_align.assert_not_called(), (
            "get_mk_alignment_for_contiguous_layout should not be called "
            "when is_deep_gemm_supported() is False"
        )


def test_deep_gemm_warmup_noop_when_unavailable():
    """
    deep_gemm_warmup must return without calling _count_warmup_iterations
    when is_deep_gemm_supported() returns False for every module.
    """
    model = torch.nn.Sequential(
        torch.nn.Linear(64, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 32),
    )

    with patch(
        "vllm.model_executor.warmup.deep_gemm_warmup.is_deep_gemm_supported",
        return_value=False,
    ), patch(
        "vllm.model_executor.warmup.deep_gemm_warmup"
        ".get_mk_alignment_for_contiguous_layout"
    ) as mock_align:
        from vllm.model_executor.warmup.deep_gemm_warmup import deep_gemm_warmup

        deep_gemm_warmup(model, max_tokens=512)
        mock_align.assert_not_called(), (
            "No deep_gemm calls expected for a bf16 model "
            "when deep_gemm is unavailable"
        )
