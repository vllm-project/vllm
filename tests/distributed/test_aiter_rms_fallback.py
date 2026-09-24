# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import sys
import types
from types import SimpleNamespace

import pytest
import torch

import vllm._aiter_ops as aiter_ops
import vllm.distributed as vllm_distributed


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

    inp = torch.randn(8192, 16, dtype=torch.bfloat16)
    residual = torch.randn_like(inp)
    weight = torch.randn(16, dtype=torch.bfloat16)

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
