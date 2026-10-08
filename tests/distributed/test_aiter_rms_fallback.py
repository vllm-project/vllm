# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import sys
import types
from types import SimpleNamespace

import pytest
import torch

import vllm._aiter_ops as aiter_ops
import vllm.distributed as vllm_distributed
import vllm.kernels  # noqa: F401
from vllm.platforms import current_platform


def test_rocm_aiter_gemma_rmsnorm_falls_back_after_allreduce(
    monkeypatch: pytest.MonkeyPatch,
):
    class FakeAiterCA:
        world_size = 8
        fully_connected = True

        def custom_fused_ar_rms(self, *args, **kwargs):
            return None

    class FakeAiterAllReduce:
        aiter_ca = FakeAiterCA()

        def use_1stage_fused_ar_rms(self, inp):
            return False

    class FakeQuickReduce:
        disabled = False

        def should_fused_allreduce_rmsnorm(self, inp, residual, weight):
            return False

        def uses_flydsl_int4(self, inp):
            return False

        def should_quick_allreduce(self, inp):
            raise AssertionError("Gemma norm must not use fused QR+RMSNorm")

    device_comm = SimpleNamespace(qr_comm=FakeQuickReduce())
    monkeypatch.setattr(
        vllm_distributed,
        "get_tp_group",
        lambda: SimpleNamespace(device_communicator=device_comm),
    )
    monkeypatch.setattr(
        aiter_ops.rocm_aiter_ops,
        "get_aiter_allreduce",
        lambda: FakeAiterAllReduce(),
    )
    monkeypatch.setattr(
        vllm_distributed,
        "tensor_model_parallel_all_reduce",
        lambda inp: inp + 2,
    )

    calls = []
    fake_aiter = types.SimpleNamespace()

    def fake_rmsnorm2d_fwd_with_add(
        out,
        inp,
        residual,
        residual_out,
        weight,
        epsilon,
        *,
        gemma_norm,
    ):
        calls.append((inp, weight, epsilon, gemma_norm))
        residual_out.copy_(inp + residual)
        out.copy_(residual_out)

    fake_aiter.rmsnorm2d_fwd_with_add = fake_rmsnorm2d_fwd_with_add
    monkeypatch.setitem(sys.modules, "aiter", fake_aiter)

    # On the platform's device, not the CPU. The fallback leaves Python through
    # `torch.ops.vllm_aiter.fused_add_rms_norm`, which is registered for the
    # platform's dispatch key alone, so CPU tensors have nothing to dispatch to
    # on a GPU build. The kernel behind it is still the fake above.
    device = current_platform.device_type
    inp = torch.randn(8192, 16, dtype=torch.bfloat16, device=device)
    residual = torch.randn_like(inp)
    weight = torch.randn(16, dtype=torch.bfloat16, device=device)

    out, residual_out = aiter_ops._rocm_aiter_fused_allreduce_rmsnorm_impl(
        inp, residual, weight, 1e-6, gemma_norm=True
    )

    assert len(calls) == 1
    assert calls[0][0] is not inp
    assert calls[0][1] is weight
    assert calls[0][2] == 1e-6
    assert calls[0][3] is True
    torch.testing.assert_close(residual_out, inp + 2 + residual)
    torch.testing.assert_close(out, residual_out)


def test_rocm_aiter_fused_rmsnorm_skips_fused_kernels_for_flydsl_payloads(
    monkeypatch: pytest.MonkeyPatch,
):
    """A payload FlyDSL QuickReduce owns but its fused RMSNorm kernel declines
    takes the unfused all-reduce + RMSNorm, not the fused HIP QR or custom-AR
    kernels."""

    class FakeAiterCA:
        def custom_fused_ar_rms(self, *args, **kwargs):
            raise AssertionError("FlyDSL payloads must not use fused custom AR")

    class FakeAiterAllReduce:
        aiter_ca = FakeAiterCA()

        def use_1stage_fused_ar_rms(self, inp):
            return False

    class FakeQuickReduce:
        disabled = False

        def should_fused_allreduce_rmsnorm(self, inp, residual, weight):
            return False

        def uses_flydsl_int4(self, inp):
            return True

        def should_quick_allreduce(self, inp):
            raise AssertionError("FlyDSL payloads must not use fused HIP QR+RMSNorm")

    monkeypatch.setattr(
        vllm_distributed,
        "get_tp_group",
        lambda: SimpleNamespace(
            device_communicator=SimpleNamespace(qr_comm=FakeQuickReduce())
        ),
    )
    monkeypatch.setattr(
        aiter_ops.rocm_aiter_ops, "get_aiter_allreduce", lambda: FakeAiterAllReduce()
    )
    monkeypatch.setattr(
        vllm_distributed, "tensor_model_parallel_all_reduce", lambda inp: inp + 2
    )
    unfused = []

    def fake_fused_add_rms_norm(*args):
        unfused.append(args)
        return "out", "residual_out"

    monkeypatch.setattr(
        torch.ops.vllm_aiter,
        "fused_add_rms_norm",
        fake_fused_add_rms_norm,
        raising=False,
    )

    inp = torch.zeros(4, 16, dtype=torch.bfloat16)
    result = aiter_ops._rocm_aiter_fused_allreduce_rmsnorm_impl(
        inp, torch.zeros_like(inp), torch.ones(16, dtype=torch.bfloat16), 1e-6
    )

    assert result == ("out", "residual_out")
    assert len(unfused) == 1
    torch.testing.assert_close(unfused[0][0], inp + 2)
