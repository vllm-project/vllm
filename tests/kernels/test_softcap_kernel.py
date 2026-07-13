# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the fused softcap Triton kernel."""

import pytest
import torch

from vllm.model_executor.layers.softcap_kernel import softcap_logits


def _ref_softcap(logits, cap):
    return cap * torch.tanh(logits / cap)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("shape", [(1, 256000), (32, 256000), (128, 32000)])
@pytest.mark.parametrize("cap", [30.0, 50.0])
def test_softcap_correctness(dtype, shape, cap):
    logits = torch.randn(shape, device="cuda", dtype=dtype)
    ref = _ref_softcap(logits, cap)
    out = softcap_logits(logits.clone(), cap, inplace=False)
    atol = 1e-5 if dtype == torch.float32 else 0.04
    torch.testing.assert_close(out, ref, atol=atol, rtol=0.01)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_softcap_inplace(dtype):
    logits = torch.randn(32, 256000, device="cuda", dtype=dtype)
    ref = _ref_softcap(logits, 30.0)
    ptr = logits.data_ptr()
    out = softcap_logits(logits, 30.0, inplace=True)
    assert out.data_ptr() == ptr
    atol = 1e-5 if dtype == torch.float32 else 0.04
    torch.testing.assert_close(out, ref, atol=atol, rtol=0.01)


@pytest.mark.parametrize("shape", [(1, 1), (7, 12345)])
def test_softcap_odd_shapes(shape):
    logits = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    ref = _ref_softcap(logits, 30.0)
    out = softcap_logits(logits.clone(), 30.0, inplace=False)
    torch.testing.assert_close(out, ref, atol=0.04, rtol=0.01)
