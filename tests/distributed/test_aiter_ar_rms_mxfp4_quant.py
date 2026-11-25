# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
from torch import nn

import vllm._aiter_ops as aiter_ops
from vllm.distributed.device_communicators.aiter_custom_all_reduce import (
    AiterCustomAllreduce,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import kMxfp4Dynamic
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm(), reason="AITER custom all-reduce is ROCm-only"
)

HIDDEN = 5120


def _fake_allreduce(monkeypatch: pytest.MonkeyPatch, *, stage, declined=False):
    """An AiterCustomAllreduce whose AITER kernel returns marked tensors."""
    calls = []

    class FakeAiterCA:
        disabled = False

        def custom_fused_ar_rms_mxfp4_quant(self, inp, residual, weight, eps, **kw):
            calls.append(kw)
            if declined:
                return None
            m, k = inp.shape
            x_q = torch.empty(m, k // 2, dtype=torch.uint8)
            x_s = torch.empty(m, k // 32, dtype=torch.uint8)
            return x_q, residual + 1, x_s, inp + 1

    aiter_ar = AiterCustomAllreduce.__new__(AiterCustomAllreduce)
    aiter_ar._impl = FakeAiterCA()
    monkeypatch.setattr(aiter_ar, "mxfp4_fused_ar_rms_stage", lambda inp: stage)
    monkeypatch.setattr(
        aiter_ops.rocm_aiter_ops, "get_aiter_allreduce", lambda: aiter_ar
    )
    return calls


@pytest.mark.parametrize(
    ("m", "stage", "batch_invariant", "declined", "fused"),
    [
        (4, True, False, False, True),
        (16, False, False, False, True),
        # From M=32 the preshuffled GEMM reads shuffled scales, which the
        # fused kernel does not write.
        (32, True, False, False, False),
        # Batch-invariant mode skips the preshuffled GEMM.
        (4, True, True, False, False),
        # Neither fused kernel takes the shape.
        (4, None, False, False, False),
        # Custom all-reduce declines the input.
        (4, True, False, True, False),
    ],
)
def test_fused_allreduce_rmsnorm_mxfp4_quant_gate(
    monkeypatch: pytest.MonkeyPatch, m, stage, batch_invariant, declined, fused
):
    calls = _fake_allreduce(monkeypatch, stage=stage, declined=declined)
    monkeypatch.setattr(aiter_ops.envs, "VLLM_BATCH_INVARIANT", batch_invariant)
    inp = torch.randn(m, HIDDEN, dtype=torch.bfloat16)
    residual = torch.randn_like(inp)

    out = aiter_ops.rocm_aiter_ops.fused_allreduce_rmsnorm_mxfp4_quant(
        inp, residual, torch.randn(HIDDEN), 1e-6, gemma_norm=True
    )

    if not fused:
        assert out is None
        return
    x_q, x_s, normed, residual_out = out
    assert x_q.shape == (m, HIDDEN // 2) and x_s.shape == (m, HIDDEN // 32)
    torch.testing.assert_close(normed, inp + 1)
    torch.testing.assert_close(residual_out, residual + 1)
    assert calls[0]["use_1stage"] is stage
    assert calls[0]["emit_bf16"] is True and calls[0]["gemma_norm"] is True


def _mxfp4_linear(*, use_asm_gemm=True, hadamard=False) -> nn.Module:
    from vllm.model_executor.kernels.linear.mxfp4.aiter import (
        AiterMxfp4LinearKernel,
    )

    kernel = AiterMxfp4LinearKernel.__new__(AiterMxfp4LinearKernel)
    kernel.use_asm_gemm = use_asm_gemm
    kernel.is_hadamard_fused = hadamard
    linear = nn.Module()
    linear.scheme = SimpleNamespace(ocp_mx_linear=kernel)
    linear._input_quant_key = kMxfp4Dynamic
    return linear


@pytest.mark.parametrize(
    ("linear_kwargs", "declined", "quantized"),
    [
        ({}, False, True),
        # The fused kernel does not rotate the activation.
        ({"hadamard": True}, False, False),
        # The Triton GEMM reads dynamic_mxfp4_quant's layout.
        ({"use_asm_gemm": False}, False, False),
        # AITER declines: the plain fused all-reduce + norm.
        ({}, True, False),
    ],
)
def test_qwen3_5_norm_quantizes_for_mxfp4_consumer(
    monkeypatch: pytest.MonkeyPatch, linear_kwargs, declined, quantized
):
    import vllm.models.qwen3_5.amd.model as model

    _fake_allreduce(monkeypatch, stage=True, declined=declined)
    monkeypatch.setattr(model, "dbo_enabled", lambda name: False)
    monkeypatch.setattr(
        model,
        "fused_allreduce_gemma_rms_norm",
        lambda x, residual, norm: (x + 2, residual + 2),
    )
    norm = model.Qwen3_5RMSNorm(HIDDEN, reduce_inputs=True)
    x = torch.randn(4, HIDDEN, dtype=torch.bfloat16)
    residual = torch.randn_like(x)

    qa, y, residual_out = norm.forward_fused(
        x, residual, _mxfp4_linear(**linear_kwargs)
    )

    if quantized:
        assert qa.quant_key == kMxfp4Dynamic and qa.orig_shape == x.shape
        torch.testing.assert_close(y, x + 1)
        torch.testing.assert_close(residual_out, residual + 1)
    else:
        assert qa is None
        torch.testing.assert_close(y, x + 2)
        torch.testing.assert_close(residual_out, residual + 2)


@pytest.mark.parametrize(
    ("m", "k", "dtype", "expected"),
    [
        (4, 8192, torch.bfloat16, True),  # one-stage
        (16, 8192, torch.bfloat16, False),  # two-stage
        (64, 8192, torch.bfloat16, None),  # neither launcher accepts it
        (4, 8192, torch.float32, None),
    ],
)
def test_mxfp4_fused_ar_rms_stage_at_tp8(m, k, dtype, expected):
    ar = AiterCustomAllreduce.__new__(AiterCustomAllreduce)
    ar._impl = SimpleNamespace(world_size=8)

    assert ar.mxfp4_fused_ar_rms_stage(torch.empty(m, k, dtype=dtype)) is expected
