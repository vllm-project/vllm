# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

import vllm._aiter_ops as aiter_ops
import vllm.distributed as vllm_distributed
from vllm.distributed.device_communicators.aiter_custom_all_reduce import (
    AiterCustomAllreduce,
)

HIDDEN = 8192
OUT = 4608


def _inputs(m: int):
    inp = torch.randn(m, HIDDEN, dtype=torch.bfloat16)
    residual = torch.randn_like(inp)
    norm_weight = torch.randn(HIDDEN, dtype=torch.bfloat16)
    weight = torch.empty(OUT, HIDDEN // 2, dtype=torch.uint8)
    weight_scale = torch.empty(OUT, HIDDEN // 32, dtype=torch.uint8)
    return inp, residual, norm_weight, weight, weight_scale


def _patch(
    monkeypatch: pytest.MonkeyPatch,
    *,
    stage,
    batch_invariant: bool = False,
    qr_comm=None,
):
    calls: dict[str, list] = {"stage": [], "fused_quant": [], "gemm": [], "ar_rms": []}

    class FakeAiterCA:
        def fused_ar_rms_mxfp4_quant(self, inp, residual, **kwargs):
            calls["fused_quant"].append(kwargs)
            m, k = inp.shape
            x_q = torch.empty(m, k // 2, dtype=torch.uint8)
            x_s = torch.empty(m, k // 32, dtype=torch.uint8)
            return x_q, residual + 1, x_s, inp + 1

    class FakeAiterAllReduce:
        aiter_ca = FakeAiterCA()

        def use_1stage_fused_ar_rms(self, inp):
            return inp.shape[0] <= 80

        def mxfp4_fused_ar_rms_stage(self, inp):
            calls["stage"].append(inp)
            return stage

    def fake_ar_rms(input_, residual, weight, epsilon, gemma_norm):
        calls["ar_rms"].append(gemma_norm)
        return input_ + 2, residual + 2

    def fake_gemm(x, weight, weight_scale, *args, **kwargs):
        calls["gemm"].append((x, args, kwargs))
        return torch.empty(x.shape[0], weight.shape[0], dtype=torch.bfloat16)

    monkeypatch.setattr(
        aiter_ops.rocm_aiter_ops, "get_aiter_allreduce", lambda: FakeAiterAllReduce()
    )
    monkeypatch.setattr(aiter_ops.envs, "VLLM_BATCH_INVARIANT", batch_invariant)
    monkeypatch.setattr(
        aiter_ops, "_rocm_aiter_fused_allreduce_rmsnorm_impl", fake_ar_rms
    )
    monkeypatch.setattr(
        torch.ops.vllm, "gemm_with_dynamic_quant", fake_gemm, raising=False
    )
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(
        vllm_distributed,
        "get_tp_group",
        lambda: SimpleNamespace(device_communicator=SimpleNamespace(qr_comm=qr_comm)),
    )
    return calls


class _FakeFlyDSLQuickReduce:
    def __init__(self, accepts: bool):
        self.accepts = accepts
        self.calls: list[tuple] = []

    def should_fused_allreduce_rmsnorm_mxfp4(self, inp, residual, weight):
        return self.accepts

    def fused_allreduce_rmsnorm_mxfp4(self, inp, residual, weight, eps, gemma_norm):
        self.calls.append((inp, residual, weight, eps, gemma_norm))
        m, k = inp.shape
        x_q = torch.empty(m, k // 2, dtype=torch.uint8)
        x_s = torch.empty((m + 255) // 256 * 256, k // 32, dtype=torch.uint8)
        return inp + 3, residual + 3, x_q, x_s


@pytest.mark.parametrize("gemma_norm", [True, False])
def test_decode_quantizes_in_the_allreduce_kernel(
    monkeypatch: pytest.MonkeyPatch, gemma_norm: bool
):
    calls = _patch(monkeypatch, stage=True)
    inp, residual, norm_weight, weight, weight_scale = _inputs(4)

    out, norm_out, residual_out = (
        aiter_ops._rocm_aiter_fused_allreduce_rmsnorm_mxfp4_gemm_impl(
            inp,
            residual,
            norm_weight,
            1e-6,
            gemma_norm,
            weight,
            weight_scale,
            torch.bfloat16,
        )
    )

    assert calls["ar_rms"] == []
    assert len(calls["fused_quant"]) == 1
    kwargs = calls["fused_quant"][0]
    assert kwargs["use_1stage"] is True
    assert kwargs["emit_bf16"] is True
    assert kwargs["gemma_norm"] is gemma_norm
    assert kwargs["w"] is norm_weight
    (x, _, gemm_kwargs) = calls["gemm"][0]
    assert x.dtype == torch.uint8
    assert gemm_kwargs["x_scales"] is not None
    assert out.shape == (4, OUT)
    torch.testing.assert_close(norm_out, inp + 1)
    torch.testing.assert_close(residual_out, residual + 1)


@pytest.mark.parametrize(
    ("m", "stage", "batch_invariant"),
    [
        # From M=32 the preshuffled GEMM takes shuffled scales, which the
        # fused epilogue does not write.
        (32, True, False),
        # Batch-invariant mode skips the preshuffled GEMM.
        (4, True, True),
        # Neither fused AR+RMSNorm+MXFP4 launcher accepts the shape.
        (16, None, False),
    ],
)
def test_falls_back_to_the_unfused_sequence(
    monkeypatch: pytest.MonkeyPatch, m: int, stage, batch_invariant: bool
):
    calls = _patch(monkeypatch, stage=stage, batch_invariant=batch_invariant)
    inp, residual, norm_weight, weight, weight_scale = _inputs(m)

    out, norm_out, residual_out = (
        aiter_ops._rocm_aiter_fused_allreduce_rmsnorm_mxfp4_gemm_impl(
            inp,
            residual,
            norm_weight,
            1e-6,
            True,
            weight,
            weight_scale,
            torch.bfloat16,
        )
    )

    assert calls["fused_quant"] == []
    assert calls["ar_rms"] == [True]
    (x, _, gemm_kwargs) = calls["gemm"][0]
    assert x.dtype == torch.bfloat16
    assert "x_scales" not in gemm_kwargs
    assert out.shape == (m, OUT)
    torch.testing.assert_close(norm_out, inp + 2)
    torch.testing.assert_close(residual_out, residual + 2)


@pytest.mark.parametrize("gemma_norm", [True, False])
def test_prefill_quantizes_in_the_flydsl_quickreduce_kernel(
    monkeypatch: pytest.MonkeyPatch, gemma_norm: bool
):
    qr_comm = _FakeFlyDSLQuickReduce(accepts=True)
    calls = _patch(monkeypatch, stage=None, qr_comm=qr_comm)
    inp, residual, norm_weight, weight, weight_scale = _inputs(256)

    out, norm_out, residual_out = (
        aiter_ops._rocm_aiter_fused_allreduce_rmsnorm_mxfp4_gemm_impl(
            inp,
            residual,
            norm_weight,
            1e-6,
            gemma_norm,
            weight,
            weight_scale,
            torch.bfloat16,
        )
    )

    assert calls["ar_rms"] == [] and calls["fused_quant"] == []
    ((got_inp, got_res, got_w, eps, got_gemma),) = qr_comm.calls
    assert got_inp is inp and got_res is residual and got_w is norm_weight
    assert (eps, got_gemma) == (1e-6, gemma_norm)
    (x, _, gemm_kwargs) = calls["gemm"][0]
    assert x.dtype == torch.uint8
    assert gemm_kwargs["x_scales"].shape == (256, HIDDEN // 32)
    assert gemm_kwargs["rocm_use_aiter_fp4_asm_gemm"] is True
    assert out.shape == (256, OUT)
    torch.testing.assert_close(norm_out, inp + 3)
    torch.testing.assert_close(residual_out, residual + 3)


@pytest.mark.parametrize(
    ("m", "accepts"),
    [
        # FlyDSL QuickReduce declines (below its window, unsupported hidden size).
        (256, False),
        # The one-stage custom all-reduce owns small batches.
        (64, True),
    ],
)
def test_prefill_falls_back_when_flydsl_quickreduce_does_not_run(
    monkeypatch: pytest.MonkeyPatch, m: int, accepts: bool
):
    qr_comm = _FakeFlyDSLQuickReduce(accepts=accepts)
    calls = _patch(monkeypatch, stage=None, qr_comm=qr_comm)
    inp, residual, norm_weight, weight, weight_scale = _inputs(m)

    aiter_ops._rocm_aiter_fused_allreduce_rmsnorm_mxfp4_gemm_impl(
        inp, residual, norm_weight, 1e-6, True, weight, weight_scale, torch.bfloat16
    )

    assert qr_comm.calls == []
    assert calls["ar_rms"] == [True]
    (x, _, gemm_kwargs) = calls["gemm"][0]
    assert x.dtype == torch.bfloat16
    assert "x_scales" not in gemm_kwargs


@pytest.mark.parametrize(
    ("m", "k", "dtype", "expected"),
    [
        (4, 8192, torch.bfloat16, True),  # one-stage
        (16, 8192, torch.bfloat16, False),  # two-stage
        (64, 8192, torch.bfloat16, None),  # neither launcher accepts it
        (4, 8192, torch.float32, None),
    ],
)
def test_mxfp4_fused_ar_rms_stage_at_tp8(
    monkeypatch: pytest.MonkeyPatch, m, k, dtype, expected
):
    monkeypatch.setattr(
        AiterCustomAllreduce,
        "build_supports_gemma_mxfp4_quant",
        staticmethod(lambda: True),
    )
    ar = AiterCustomAllreduce.__new__(AiterCustomAllreduce)
    ar._impl = SimpleNamespace(world_size=8)

    assert ar.mxfp4_fused_ar_rms_stage(torch.empty(m, k, dtype=dtype)) is expected


def test_mxfp4_fused_ar_rms_stage_needs_the_gemma_kernel(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        AiterCustomAllreduce,
        "build_supports_gemma_mxfp4_quant",
        staticmethod(lambda: False),
    )
    ar = AiterCustomAllreduce.__new__(AiterCustomAllreduce)
    ar._impl = SimpleNamespace(world_size=8)

    assert ar.mxfp4_fused_ar_rms_stage(torch.empty(4, 8192)) is None
