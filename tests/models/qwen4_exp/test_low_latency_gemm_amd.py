# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the AMD Qwen4Exp tuned-shape `tgemm` selection."""

import sys
import types
from unittest.mock import MagicMock

import pytest
import torch

from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm(), reason="AMD Qwen4Exp GEMM selection needs ROCm"
)

TUNED_SHAPE = (512, 2560)
UNTUNED_SHAPE = (64, 128)


@pytest.fixture
def llg():
    from vllm.models.qwen4_exp.amd import low_latency_gemm

    low_latency_gemm._has_tuned_row.cache_clear()
    yield low_latency_gemm
    low_latency_gemm._has_tuned_row.cache_clear()


def _fake_aiter(monkeypatch, row_for):
    lookup = MagicMock(side_effect=lambda M, N, K, *args: row_for(M, N, K))
    mm = MagicMock(name="tgemm.mm")
    aiter_tuned_gemm = types.SimpleNamespace(
        get_GEMM_A16W16_tuned_config=lookup, tgemm=types.SimpleNamespace(mm=mm)
    )
    monkeypatch.setitem(sys.modules, "aiter.tuned_gemm", aiter_tuned_gemm)
    return lookup, mm


@pytest.mark.parametrize(
    ("num_tokens", "row", "guard_permits", "use_tgemm"),
    [
        (8, {"libtype": "opus"}, True, True),
        (8, {"libtype": "torch"}, True, False),
        (8, None, True, False),
        (4, {"libtype": "opus"}, True, False),
        # LM-head-sized weight: the standard path skips skinny at 2-5 tokens.
        (4, {"libtype": "opus"}, False, True),
    ],
)
def test_op_uses_tgemm_only_for_tuned_rows(
    monkeypatch, llg, num_tokens, row, guard_permits, use_tgemm
):
    lookup, mm = _fake_aiter(monkeypatch, lambda M, N, K: row)
    standard = MagicMock(name="rocm_unquantized_gemm_impl")
    monkeypatch.setattr(llg, "rocm_unquantized_gemm_impl", standard)
    monkeypatch.setattr(
        llg, "lmhead_skinny_guard_permits", lambda weight, n: guard_permits
    )
    x = torch.empty(num_tokens, TUNED_SHAPE[1], dtype=torch.bfloat16, device="meta")
    weight = torch.empty(TUNED_SHAPE, dtype=torch.bfloat16, device="meta")

    llg.qwen4_exp_rocm_tgemm(x, weight)

    if num_tokens > llg._SKINNY_MAX_TOKENS or not guard_permits:
        lookup.assert_called_once_with(
            num_tokens, *TUNED_SHAPE, False, "torch.bfloat16", "torch.bfloat16"
        )
    else:
        lookup.assert_not_called()
    if use_tgemm:
        mm.assert_called_once_with(x, weight)
        standard.assert_not_called()
    else:
        mm.assert_not_called()
        standard.assert_called_once_with(x, weight)


@pytest.mark.parametrize(("down_tgemm", "up_tgemm"), [(False, True), (True, False)])
def test_hc_fallback_uses_tgemm_for_switched_weights(monkeypatch, down_tgemm, up_tgemm):
    from vllm.models.qwen4_exp.amd import hyperconnection as hc

    hc_count, lowrank, hidden, down_rows, num_tokens = 4, 8, 16, 16, 8
    calls = []

    def gemm(name):
        def fn(x, w):
            calls.append((name, tuple(w.shape)))
            return x.new_zeros((x.shape[0], w.shape[0]))

        return fn

    monkeypatch.setattr(hc, "qwen4_exp_rocm_tgemm", gemm("tgemm"))
    monkeypatch.setattr(hc, "rocm_unquantized_gemm_impl", gemm("standard"))
    monkeypatch.setattr(hc, "_hc_combine_norm", lambda r, b, i, w, eps, n: (r, r))
    monkeypatch.setattr(hc, "_hc_silu", lambda x, n: x)
    monkeypatch.setattr(hc, "_hc_gate_mix", lambda x, g, n: x[:, :hidden])
    residual = torch.zeros(num_tokens, hc_count * hidden)
    w_down = torch.zeros(down_rows, hc_count * hidden)
    w_up = torch.zeros(hc_count * hidden, lowrank)

    hc._hc_two_stage_combine_and_mix(
        residual,
        residual,
        None,
        None,
        w_down,
        None,
        w_up,
        1e-6,
        hc_count,
        lowrank,
        num_tokens - 1,
        down_tgemm,
        up_tgemm,
    )

    assert calls == [
        ("tgemm" if down_tgemm else "standard", tuple(w_down.shape)),
        ("tgemm" if up_tgemm else "standard", tuple(w_up.shape)),
    ]


def _linears():
    from vllm.model_executor.layers.linear import ReplicatedLinear

    module = torch.nn.Module()
    module.tuned = ReplicatedLinear(
        TUNED_SHAPE[1],
        TUNED_SHAPE[0],
        bias=False,
        params_dtype=torch.bfloat16,
        disable_tp=True,
    )
    module.untuned = ReplicatedLinear(
        UNTUNED_SHAPE[1],
        UNTUNED_SHAPE[0],
        bias=False,
        params_dtype=torch.bfloat16,
        disable_tp=True,
    )
    return module


@pytest.mark.parametrize(
    ("cdna_version", "cu_count", "expect_swap"),
    [(3, 304, True), (3, 80, False), (4, 256, False)],
)
def test_enable_swaps_only_tuned_linears_on_cdna3(
    default_vllm_config,
    dist_init,
    monkeypatch,
    llg,
    cdna_version,
    cu_count,
    expect_swap,
):
    import vllm.platforms.rocm as rocm_platform
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod

    _fake_aiter(
        monkeypatch,
        lambda M, N, K: {"libtype": "opus"} if (N, K) == TUNED_SHAPE else None,
    )
    monkeypatch.setattr(rocm_platform, "get_cdna_version", lambda: cdna_version)
    monkeypatch.setattr(llg, "num_compute_units", lambda: cu_count)
    monkeypatch.setattr(llg.rocm_aiter_ops, "is_linear_enabled", lambda: True)
    module = _linears()

    llg.enable_qwen4_exp_low_latency_gemm(module, torch.bfloat16)

    expected = llg.Qwen4ExpTgemmLinearMethod if expect_swap else UnquantizedLinearMethod
    assert type(module.tuned.quant_method) is expected
    assert type(module.untuned.quant_method) is UnquantizedLinearMethod
