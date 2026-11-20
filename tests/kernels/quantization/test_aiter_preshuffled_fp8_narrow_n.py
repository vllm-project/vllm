# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kernel selection of AiterPreshuffledPerTokenFp8ScaledMMLinearKernel for N < 128.

The (N, K) tuned check passes when any M is tuned. For N < 128, AITER's default
bpreshuffle config has no kernel at large M, so the kernel must also have a
tuned row at a large M. These checks run before any GPU work, and
can_implement has no platform check (that is in is_supported), so the test
needs no GPU.
"""

import pytest
import torch

from vllm._aiter_ops import rocm_aiter_ops
from vllm.model_executor.kernels.linear.scaled_mm import aiter as aiter_mm
from vllm.model_executor.kernels.linear.scaled_mm.ScaledMMLinearKernel import (
    FP8ScaledMMLinearLayerConfig,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kFp8DynamicTokenSym,
    kFp8StaticChannelSym,
)

KERNEL = aiter_mm.AiterPreshuffledPerTokenFp8ScaledMMLinearKernel


def _config(n: int, k: int) -> FP8ScaledMMLinearLayerConfig:
    return FP8ScaledMMLinearLayerConfig(
        weight_quant_key=kFp8StaticChannelSym,
        activation_quant_key=kFp8DynamicTokenSym,
        weight_shape=(n, k),
        input_dtype=torch.bfloat16,
        out_dtype=torch.bfloat16,
    )


def _tables(monkeypatch: pytest.MonkeyPatch, tuned_m: dict[tuple[int, int], set]):
    """Fake bpreshuffle table: (N, K) -> set of tuned M."""
    monkeypatch.setattr(
        aiter_mm.current_platform, "fp8_dtype", lambda: torch.float8_e4m3fn
    )
    monkeypatch.setattr(
        rocm_aiter_ops,
        "is_shuffled_per_token_w8a8_gemm_tuned",
        staticmethod(lambda n, k, dtype: bool(tuned_m.get((n, k)))),
    )
    monkeypatch.setattr(
        rocm_aiter_ops,
        "is_shuffled_per_token_w8a8_gemm_tuned_at_m",
        staticmethod(lambda m, n, k, dtype: m in tuned_m.get((n, k), set())),
    )


def test_refuses_narrow_n_tuned_only_at_small_m(monkeypatch):
    # The Qwen3.8-Flash-Next merged GDN a/b projection: one tuned row at M=64.
    _tables(monkeypatch, {(96, 2560): {64}})
    ok, _ = KERNEL.can_implement(_config(96, 2560))
    assert not ok


def test_accepts_narrow_n_tuned_at_large_m(monkeypatch):
    # Narrow shapes tuned across M (N=64, K=4096 in the Qwen3.5-397B table) keep
    # the kernel.
    _tables(monkeypatch, {(64, 4096): {1, 64, 16384, 32768}})
    ok, reason = KERNEL.can_implement(_config(64, 4096))
    assert ok, reason


@pytest.mark.parametrize("n", [128, 2048])
def test_wide_n_needs_only_one_tuned_m(monkeypatch, n: int):
    _tables(monkeypatch, {(n, 2560): {64}})
    ok, reason = KERNEL.can_implement(_config(n, 2560))
    assert ok, reason


def test_untuned_shape_is_refused(monkeypatch):
    _tables(monkeypatch, {})
    ok, _ = KERNEL.can_implement(_config(2048, 2560))
    assert not ok
