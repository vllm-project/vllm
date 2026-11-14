# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.kernels.mhc import warmup as mhc_warmup
from vllm.model_executor.warmup.jit_warmup import JitWarmupRegistry


@pytest.fixture
def raw_splits(monkeypatch):
    """132 SMs, capped at 64 for k=16384: grid 1..8 -> 64, 64, 44, ..., 16."""
    monkeypatch.setattr(
        mhc_warmup,
        "compute_num_split",
        lambda block_k, k, grid_size: max(1, min(132 // grid_size, k // 64 // 4)),
    )


def _keys(num_splits, max_tokens=512, k=16384):
    return mhc_warmup.HCPrenormGemmKernel().get_warmup_keys(
        max_tokens=max_tokens, n=24, k=k, num_splits=num_splits
    )


def test_v4_warms_every_raw_split(raw_splits) -> None:
    keys = _keys(mhc_warmup.compute_mhc_num_splits)

    assert [key.n_splits for key in keys] == [16, 18, 22, 26, 33, 44, 64]
    assert {(key.n, key.k) for key in keys} == {(24, 16384)}


def test_v41_warms_only_its_rounded_splits(raw_splits) -> None:
    keys = _keys(mhc_warmup.compute_mhc_pre_num_splits)

    assert [key.n_splits for key in keys] == [16]
    small_k = _keys(mhc_warmup.compute_mhc_pre_num_splits, k=256)
    assert [key.n_splits for key in small_k] == [1]


def test_warmup_splits_match_every_runtime_token_count(raw_splits) -> None:
    for num_splits in (
        mhc_warmup.compute_mhc_num_splits,
        mhc_warmup.compute_mhc_pre_num_splits,
    ):
        warmed = {key.n_splits for key in _keys(num_splits, max_tokens=1000)}
        runtime = {num_splits(16384, t) for t in range(1, 1001)}
        assert warmed == runtime


def test_compile_matches_runtime_layout_once(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(
        "vllm.utils.deep_gemm.tf32_hc_prenorm_gemm",
        lambda x, fn, out, sqr_sum, splits: calls.append((x, fn, out, sqr_sum, splits)),
    )
    monkeypatch.setattr(
        mhc_warmup, "current_platform", SimpleNamespace(device_type="cpu")
    )
    kernel = mhc_warmup.HCPrenormGemmKernel()
    key = kernel.CompileKey(n=24, k=16384, n_splits=3)

    kernel.compile_many([key, key])

    ((x, fn, out, sqr_sum, splits),) = calls
    assert splits == 3
    assert x.shape == (64, 16384) and x.dtype == torch.bfloat16
    assert fn.shape == (24, 16384) and fn.dtype == torch.float32
    assert out.shape == (3, 64, 24) and out.dtype == torch.float32
    assert sqr_sum.shape == (3, 64) and sqr_sum.dtype == torch.float32


def test_identical_layer_registrations_collapse() -> None:
    registry = JitWarmupRegistry(vllm_config=None)
    with registry.activate():
        for _ in range(43):  # one per decoder layer
            mhc_warmup.HC_PRENORM_GEMM_KERNEL.register_warmup(
                max_tokens=8192,
                n=24,
                k=16384,
                num_splits=mhc_warmup.compute_mhc_pre_num_splits,
            )
    assert len(registry) == 1
