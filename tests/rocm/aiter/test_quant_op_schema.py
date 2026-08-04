# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Schema/aliasing tests for the AITER FP8 quantization custom ops.
#
# These use the shared opcheck helper, whose test_schema check catches custom
# ops whose implementation aliases an input that the registered schema declares
# as non-aliasing -- the failure mode behind the rocm_aiter_per_tensor_quant
# regression (a returned scale that aliased the input scale).
#
# Skipped if AITER is not installed or the platform is not ROCm.

import importlib.util

import pytest
import torch

from tests.kernels.utils import opcheck

# this import statement is needed to ensure the ops are registered
from vllm._aiter_ops import rocm_aiter_ops
from vllm.platforms import current_platform

aiter_available = importlib.util.find_spec("aiter") is not None

pytestmark = pytest.mark.skipif(
    not (current_platform.is_rocm() and aiter_available),
    reason="AITER ops are only available on ROCm with aiter package installed",
)

FP8_DTYPE = current_platform.fp8_dtype()


def _x(M=128, N=4096):
    return torch.randn((M, N), dtype=torch.float16, device="cuda")


def test_per_tensor_quant_static_schema():
    """Static per-tensor: caller provides scale (the aliasing regression)."""
    x = _x()
    out = torch.empty_like(x, dtype=FP8_DTYPE)
    scale = torch.ones(1, dtype=torch.float32, device="cuda")
    opcheck(
        torch.ops.vllm.rocm_aiter_per_tensor_quant,
        (out, x, scale, False),
    )


def test_per_tensor_quant_dynamic_schema():
    """Dynamic per-tensor: op computes scale into the caller's buffer."""
    x = _x()
    out = torch.empty_like(x, dtype=FP8_DTYPE)
    scale = torch.empty(1, dtype=torch.float32, device="cuda")
    opcheck(
        torch.ops.vllm.rocm_aiter_per_tensor_quant,
        (out, x, scale, True),
    )


def test_per_token_quant_dynamic_schema():
    """Dynamic per-token: op computes scale into a freshly allocated buffer."""
    x = _x()
    opcheck(
        torch.ops.vllm.rocm_aiter_per_token_quant,
        (x, FP8_DTYPE, None),
    )


def test_group_fp8_quant_schema():
    """Dynamic per-token-group quant."""
    x = _x()
    opcheck(
        torch.ops.vllm.rocm_aiter_group_fp8_quant,
        (x, 128),
    )


@pytest.mark.parametrize("dynamic", [True, False])
def test_per_tensor_quant_matches_native(dynamic):
    """Wrapper output matches the native scaled_fp8_quant reference."""
    from vllm import _custom_ops as ops

    torch.manual_seed(0)
    x = _x()
    if dynamic:
        scale_in = None
    else:
        scale_in = torch.tensor([0.5], dtype=torch.float32, device="cuda")

    out, scale = rocm_aiter_ops.per_tensor_quant(x, FP8_DTYPE, scale_in)
    ref_out, ref_scale = ops.scaled_fp8_quant(x, scale_in)

    assert out.shape == x.shape
    assert out.dtype == FP8_DTYPE
    assert scale.shape == ref_scale.shape
    deq = out.to(torch.float32) * scale
    if dynamic:
        # Dynamic mode: AITER and native each compute their own scale, so their
        # outputs differ and can't be compared. Just check that AITER's output
        # dequantizes back to the input, within fp8 rounding error.
        torch.testing.assert_close(deq, x.to(torch.float32), rtol=0.07, atol=5e-2)
    else:
        # Static mode: both use the caller's scale, so the outputs must match.
        assert torch.equal(scale, scale_in)
        ref_deq = ref_out.to(torch.float32) * ref_scale
        torch.testing.assert_close(deq, ref_deq, rtol=2e-2, atol=2e-2)


# A test_per_tensor_quant_torch_compile test previously lived here to validate
# the per-tensor aliasing contract. It existed because opcheck's test_schema
# could not check this op directly: test_schema compares the op's outputs with
# torch.allclose, but on fp8 outputs that comparison runs arithmetic fp8 does not
# support and raises "mul_cuda" is unimplemented for fp8. The fp8-safe opcheck
# helper fixes that by casting to double before the comparison, so the per-tensor
# schema tests above can now run test_schema directly. That makes this test
# redundant, so it has been removed.
