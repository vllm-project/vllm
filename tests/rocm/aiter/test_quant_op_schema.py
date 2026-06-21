# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Schema/aliasing tests for the AITER FP8 quantization custom ops.
#
# These use torch.library.opcheck, whose test_schema check catches custom ops
# whose implementation aliases an input that the registered schema declares as
# non-aliasing -- the failure mode behind the rocm_aiter_per_tensor_quant
# regression (a returned scale that aliased the input scale).
#
# Skipped if AITER is not installed or the platform is not ROCm.

import importlib.util

import pytest
import torch

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


# The in-place per-tensor op takes the fp8 output buffer as an input, which
# opcheck's test_schema cannot exercise ("mul_cuda" is unimplemented for fp8),
# so restrict to the utils that run on fp8 inputs. The aliasing contract for
# this op is instead covered by test_per_tensor_quant_torch_compile below.
_INPLACE_OPCHECK_UTILS = (
    "test_faketensor",
    "test_aot_dispatch_dynamic",
    "test_autograd_registration",
)


def test_per_tensor_quant_static_schema():
    """Static per-tensor: caller provides scale (the aliasing regression)."""
    x = _x()
    out = torch.empty_like(x, dtype=FP8_DTYPE)
    scale = torch.ones(1, dtype=torch.float32, device="cuda")
    torch.library.opcheck(
        torch.ops.vllm.rocm_aiter_per_tensor_quant,
        (out, x, scale, False),
        test_utils=_INPLACE_OPCHECK_UTILS,
    )


def test_per_tensor_quant_dynamic_schema():
    """Dynamic per-tensor: op computes scale into the caller's buffer."""
    x = _x()
    out = torch.empty_like(x, dtype=FP8_DTYPE)
    scale = torch.empty(1, dtype=torch.float32, device="cuda")
    torch.library.opcheck(
        torch.ops.vllm.rocm_aiter_per_tensor_quant,
        (out, x, scale, True),
        test_utils=_INPLACE_OPCHECK_UTILS,
    )


def test_per_token_quant_dynamic_schema():
    """Dynamic per-token: op computes scale into a freshly allocated buffer."""
    x = _x()
    torch.library.opcheck(
        torch.ops.vllm.rocm_aiter_per_token_quant,
        (x, FP8_DTYPE, None),
    )


def test_group_fp8_quant_schema():
    """Dynamic per-token-group quant."""
    x = _x()
    torch.library.opcheck(
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
    if not dynamic:
        # static scale is passed through unchanged
        assert torch.equal(scale, scale_in)
    # Compare dequantized values to be robust to 1-ULP fp8 boundary flips.
    deq = out.to(torch.float32) * scale
    ref_deq = ref_out.to(torch.float32) * ref_scale
    torch.testing.assert_close(deq, ref_deq, rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("dynamic", [True, False])
def test_per_tensor_quant_torch_compile(monkeypatch, dynamic):
    """per_tensor_quant compiles under inductor without an aliasing error.

    Forces the custom-op aliasing check to error (it is otherwise only a
    warning outside CI), so a regression that returns an input-aliasing
    scale fails here regardless of the CI env var.
    """
    aliasing_cfg = pytest.importorskip("torch._functorch.config")
    monkeypatch.setattr(
        aliasing_cfg, "error_on_custom_op_aliasing", True, raising=False
    )

    x = _x()
    scale = None if dynamic else torch.tensor([0.5], dtype=torch.float32, device="cuda")

    def fn(x, s):
        return rocm_aiter_ops.per_tensor_quant(x, FP8_DTYPE, s)

    compiled = torch.compile(fn, fullgraph=True, backend="inductor", dynamic=False)

    out_eager, scale_eager = fn(x, scale)
    out_compiled, scale_compiled = compiled(x, scale)

    assert out_compiled.shape == out_eager.shape
    torch.testing.assert_close(
        out_compiled.to(torch.float32) * scale_compiled,
        out_eager.to(torch.float32) * scale_eager,
        rtol=2e-2,
        atol=2e-2,
    )
