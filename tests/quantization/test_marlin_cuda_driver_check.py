# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Every Marlin weight-preparation op must run the CUDA driver/toolkit check.

All Marlin users (AWQ, GPTQ, compressed-tensors, FP8, FP4, MoE, ...) go
through these ops before the first Marlin GEMM, so the PTX-JIT diagnostic is
wired there once instead of in each quantization method. The Marlin kernels
themselves are replaced by fakes, so no GPU is needed.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

import vllm._custom_ops as ops


def _fake_repack(b_q_weight, size_k, size_n, num_bits, is_a_8bit=False):
    return torch.zeros((size_k // 16, size_n * (num_bits // 2)), dtype=torch.int32)


@pytest.fixture
def fake_marlin(monkeypatch):
    fake_c = SimpleNamespace(
        gptq_marlin_repack=_fake_repack,
        awq_marlin_repack=_fake_repack,
        marlin_int4_fp8_preprocess=lambda qweight, qzeros, inplace: qweight,
    )
    monkeypatch.setattr(
        ops, "torch", SimpleNamespace(ops=SimpleNamespace(_C=fake_c), empty=torch.empty)
    )
    check = MagicMock(return_value=False)
    monkeypatch.setattr(ops, "warn_if_cuda_driver_cannot_jit_ptx", check)
    return check


def _call_all_marlin_prep_ops():
    w = torch.zeros((16, 16), dtype=torch.int32)
    w_moe = torch.zeros((2, 16, 16), dtype=torch.int32)
    yield "gptq_marlin_repack", lambda: ops.gptq_marlin_repack(w, 16, 64, 4)
    yield "awq_marlin_repack", lambda: ops.awq_marlin_repack(w, 16, 64, 4)
    yield "gptq_marlin_moe_repack", lambda: ops.gptq_marlin_moe_repack(w_moe, 16, 64, 4)
    yield "awq_marlin_moe_repack", lambda: ops.awq_marlin_moe_repack(w_moe, 16, 64, 4)
    yield "marlin_int4_fp8_preprocess", lambda: ops.marlin_int4_fp8_preprocess(w)


@pytest.mark.parametrize("name", [name for name, _ in _call_all_marlin_prep_ops()])
def test_marlin_prep_ops_run_driver_check(monkeypatch, fake_marlin, name):
    monkeypatch.setattr(ops.current_platform, "is_cuda", lambda: True)
    fn = dict(_call_all_marlin_prep_ops())[name]
    fn()
    fake_marlin.assert_called_with("Marlin")


def test_marlin_prep_ops_skip_check_off_cuda(monkeypatch, fake_marlin):
    monkeypatch.setattr(ops.current_platform, "is_cuda", lambda: False)
    for _, fn in _call_all_marlin_prep_ops():
        fn()
    fake_marlin.assert_not_called()
