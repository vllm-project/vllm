# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.model_executor.layers.quantization.utils import fp8_utils


def test_required_batch_invariant_kernel_has_no_fallback(monkeypatch):
    monkeypatch.delenv("VLLM_BATCH_INVARIANT_KERNEL_LIB", raising=False)

    with pytest.raises(RuntimeError, match="refusing a non-BI MoE fallback"):
        fp8_utils.require_batch_invariant_quant_kernel()


def test_required_batch_invariant_kernel_loads_configured_library(monkeypatch):
    calls = []
    monkeypatch.setenv("VLLM_BATCH_INVARIANT_KERNEL_LIB", "/test/bi-kernel.so")
    monkeypatch.setattr(
        fp8_utils,
        "_load_batch_invariant_kernel_library",
        lambda path: calls.append(path),
    )

    fp8_utils.require_batch_invariant_quant_kernel()

    assert calls == ["/test/bi-kernel.so"]


def test_required_batch_invariant_kernel_rejects_missing_library(
    monkeypatch, tmp_path
):
    missing = tmp_path / "_vllm_batch_invariant_C.so"
    monkeypatch.setenv("VLLM_BATCH_INVARIANT_KERNEL_LIB", str(missing))

    with pytest.raises(RuntimeError, match="library does not exist"):
        fp8_utils.require_batch_invariant_quant_kernel()
