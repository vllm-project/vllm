# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FlashInfer warmup autotuning skips "fp4_gemm" only when the CuTe-DSL NVFP4
linear kernel is selected, with VLLM_FLASHINFER_AUTOTUNE_SKIP_OPS as an
override."""

from unittest.mock import MagicMock

import pytest
import torch.nn as nn

from vllm.model_executor.kernels.linear import FlashInferCuteDslNvFp4LinearKernel
from vllm.model_executor.warmup.kernel_warmup import _flashinfer_autotune_skip_ops


def _make_runner(kernel=None, holder_name="quant_method"):
    linear = nn.Linear(8, 8)
    if kernel is not None:
        holder = MagicMock()
        holder.kernel = kernel
        setattr(linear, holder_name, holder)
    runner = MagicMock()
    runner.get_model.return_value = nn.Sequential(linear, nn.ReLU())
    return runner


def _cutedsl_kernel():
    return FlashInferCuteDslNvFp4LinearKernel.__new__(
        FlashInferCuteDslNvFp4LinearKernel
    )


@pytest.mark.parametrize("holder_name", ["quant_method", "scheme"])
def test_skips_fp4_gemm_when_cutedsl_kernel_selected(holder_name):
    runner = _make_runner(_cutedsl_kernel(), holder_name)
    assert _flashinfer_autotune_skip_ops(runner) == {"fp4_gemm"}


def test_no_skip_without_cutedsl_kernel():
    assert _flashinfer_autotune_skip_ops(_make_runner()) is None


def test_env_var_overrides_auto_detection(monkeypatch):
    monkeypatch.setattr("vllm.envs.VLLM_FLASHINFER_AUTOTUNE_SKIP_OPS", ["custom_op"])
    runner = _make_runner(_cutedsl_kernel())
    assert _flashinfer_autotune_skip_ops(runner) == {"custom_op"}


def test_empty_env_var_disables_skipping(monkeypatch):
    monkeypatch.setattr("vllm.envs.VLLM_FLASHINFER_AUTOTUNE_SKIP_OPS", [])
    runner = _make_runner(_cutedsl_kernel())
    assert _flashinfer_autotune_skip_ops(runner) is None
