# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dispatch tests for ``fused_ar_rms_norm_quant``.

The helper picks the highest-fusion kernel tier that applies and must fall back
cleanly when one is unavailable. These tests pin the tier selection (and that
exactly one all-reduce happens) with fakes, so they need no GPU or process
group.
"""

import pytest
import torch
import torch.nn as nn

import vllm.model_executor.layers.fusion.ar_rms_quant as arq
from vllm.model_executor.layers.fusion.ar_rms_quant import (
    _FUSED_RMS_QUANT_IMPLS,
    fused_ar_rms_norm_quant,
)
from vllm.model_executor.layers.fusion.quant_activation import QuantizedActivation
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kFp8StaticTensorSym,
)

HIDDEN = 64


class _Patterns:
    kARResidualRMSNorm = 1
    kARResidualRMSNormFP8Quant = 2
    kARResidualRMSNormFP4Quant = 3


def _rms(x: torch.Tensor, gamma: torch.Tensor, eps: float) -> torch.Tensor:
    dtype = x.dtype
    x = x.float()
    out = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * gamma.float()
    return out.to(dtype)


class _FakeFusedKernel:
    """Stand-in for ``flashinfer_trtllm_fused_allreduce_norm``.

    Mirrors the buffer contract: norm goes to ``norm_out`` when the caller
    allocated one (no prior residual), otherwise in place into
    ``allreduce_in``; ``residual`` receives the running sum when it is not a
    throw-away zero buffer.
    """

    def __init__(self, world_size: int = 2) -> None:
        self.world_size = world_size
        self.calls: list[dict] = []

    def __call__(
        self,
        *,
        allreduce_in,
        residual,
        norm_out,
        quant_out,
        scale_out,
        rms_gamma,
        rms_eps,
        pattern_code,
        scale_factor,
        world_size,
        launch_with_pdl,
        fp32_acc,
        max_token_num,
    ) -> None:
        self.calls.append(
            {
                "pattern_code": pattern_code,
                "max_token_num": max_token_num,
                "scale_factor": scale_factor,
                "world_size": world_size,
            }
        )
        assert world_size == self.world_size
        summed = allreduce_in * world_size + residual
        if norm_out is not None:
            norm_out.copy_(_rms(summed, rms_gamma, rms_eps))
        else:
            # In-place contract: the running sum lands in ``residual`` and the
            # normalized activation overwrites ``allreduce_in``.
            residual.copy_(summed)
            allreduce_in.copy_(_rms(summed, rms_gamma, rms_eps))
        if quant_out is not None:
            quant_out.fill_(1.0)


class _ReduceCounter:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        return x * 2


class _FakeConsumer(nn.Module):
    def __init__(self, quant_key=None) -> None:
        super().__init__()
        if quant_key is not None:
            self.input_quant_key = quant_key
            self.input_scale = torch.tensor(0.5)


@pytest.fixture
def dist(monkeypatch):
    """TP=2 with counting all-reduce and no fused kernel by default."""
    counter = _ReduceCounter()
    monkeypatch.setattr(arq, "get_tensor_model_parallel_world_size", lambda: 2)
    monkeypatch.setattr(arq, "tensor_model_parallel_all_reduce", counter)
    monkeypatch.setattr(arq, "flashinfer_trtllm_fused_allreduce_norm", None)
    monkeypatch.setattr(arq, "ar_fusion_patterns", _Patterns)
    monkeypatch.setattr(arq, "_FLASHINFER_AR_MAX_SIZE_MB", {})
    return counter


@pytest.fixture
def fused_kernel(dist, monkeypatch):
    """TP=2 with the FlashInfer fused kernel standing in for the collective.

    Depends on ``dist`` so the fused kernel wins regardless of fixture order.
    """
    kernel = _FakeFusedKernel()
    monkeypatch.setattr(arq, "flashinfer_trtllm_fused_allreduce_norm", kernel)
    monkeypatch.setattr(arq, "ar_fusion_patterns", _Patterns)
    # 2 MiB workspace / (64 hidden * 4 bytes) = 8192 tokens.
    monkeypatch.setattr(arq, "_FLASHINFER_AR_MAX_SIZE_MB", {2: 2.0})
    return kernel


def test_rejects_non_rmsnorm():
    with pytest.raises(AssertionError, match="plain RMSNorm"):
        fused_ar_rms_norm_quant(
            torch.randn(4, HIDDEN),
            None,
            nn.Identity(),
            None,
            do_allreduce=False,
        )


def test_no_allreduce_returns_plain_norm(default_vllm_config, dist):
    norm = RMSNorm(HIDDEN)
    x = torch.randn(4, HIDDEN)

    out, residual = fused_ar_rms_norm_quant(x, None, norm, None, do_allreduce=False)

    assert isinstance(out, torch.Tensor)
    assert residual is x
    assert torch.equal(out, norm(x))
    assert dist.calls == 0


def test_allreduce_without_flashinfer_uses_explicit_allreduce(
    default_vllm_config, dist
):
    norm = RMSNorm(HIDDEN)
    x = torch.randn(4, HIDDEN)
    prior = torch.randn(4, HIDDEN)
    expected = prior + 2 * x

    out, residual = fused_ar_rms_norm_quant(x, prior, norm, None, do_allreduce=True)

    assert dist.calls == 1
    assert torch.equal(residual, expected)
    assert torch.equal(out, norm(expected))


def test_allreduce_prefers_fused_kernel(default_vllm_config, fused_kernel, dist):
    norm = RMSNorm(HIDDEN)
    x = torch.randn(4, HIDDEN)
    prior = torch.randn(4, HIDDEN)
    expected = prior + 2 * x

    out, residual = fused_ar_rms_norm_quant(x, prior, norm, None, do_allreduce=True)

    assert dist.calls == 0  # no fallback all-reduce
    assert len(fused_kernel.calls) == 1
    assert fused_kernel.calls[0]["pattern_code"] == _Patterns.kARResidualRMSNorm
    assert torch.allclose(out, norm(expected))


def test_allreduce_falls_back_when_batch_exceeds_workspace(
    default_vllm_config, fused_kernel, dist, monkeypatch
):
    """Batches past the FlashInfer workspace cap must use the unfused path."""
    monkeypatch.setattr(arq, "_FLASHINFER_AR_MAX_SIZE_MB", {2: 1e-6})
    norm = RMSNorm(HIDDEN)
    x = torch.randn(4, HIDDEN)

    fused_ar_rms_norm_quant(x, None, norm, None, do_allreduce=True)

    assert fused_kernel.calls == []
    assert dist.calls == 1


def test_fp8_quant_key_fuses_quant_into_fused_allreduce(
    default_vllm_config, fused_kernel, dist
):
    norm = RMSNorm(HIDDEN)
    x = torch.randn(4, HIDDEN)
    consumer = _FakeConsumer(kFp8StaticTensorSym)

    out, _ = fused_ar_rms_norm_quant(x, None, norm, consumer, do_allreduce=True)

    assert isinstance(out, QuantizedActivation)
    assert out.quant_key is kFp8StaticTensorSym
    assert out.orig_shape == x.shape
    assert out.orig_dtype == x.dtype
    assert fused_kernel.calls[0]["pattern_code"] == (
        _Patterns.kARResidualRMSNormFP8Quant
    )
    assert dist.calls == 0


def test_fp8_quant_at_tp1_uses_rms_norm_quant(default_vllm_config, dist, monkeypatch):
    seen = []

    def fake_rms_quant(norm, x, residual, linear):
        seen.append(linear)
        qa = QuantizedActivation(
            data=x,
            scale=linear.input_scale,
            orig_dtype=x.dtype,
            orig_shape=x.shape,
            quant_key=kFp8StaticTensorSym,
        )
        return qa, x if residual is None else residual

    monkeypatch.setitem(_FUSED_RMS_QUANT_IMPLS, kFp8StaticTensorSym, fake_rms_quant)
    norm = RMSNorm(HIDDEN)
    consumer = _FakeConsumer(kFp8StaticTensorSym)

    out, _ = fused_ar_rms_norm_quant(
        torch.randn(4, HIDDEN), None, norm, consumer, do_allreduce=False
    )

    assert isinstance(out, QuantizedActivation)
    assert seen == [consumer]
    assert dist.calls == 0


def test_unsupported_quant_key_degrades_to_plain_norm(default_vllm_config, dist):
    norm = RMSNorm(HIDDEN)
    consumer = _FakeConsumer(quant_key=object())

    out, _ = fused_ar_rms_norm_quant(
        torch.randn(4, HIDDEN), None, norm, consumer, do_allreduce=False
    )

    assert isinstance(out, torch.Tensor)
