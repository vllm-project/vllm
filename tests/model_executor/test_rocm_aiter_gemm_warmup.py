# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace
from typing import Any

import pytest

from vllm.model_executor.warmup.kernel_warmup import (
    _aiter_gemm_m_buckets,
    rocm_aiter_gemm_warmup,
)

pytestmark = pytest.mark.cpu_test


def _aiter_padded_m(m: int) -> int:
    # AITER getPaddedM(M, N, K, gl=0) for M >= 16.
    if m <= 256:
        return (m + 15) // 16 * 16
    if m <= 1024:
        return (m + 31) // 32 * 32
    if m <= 4096:
        return (m + 63) // 64 * 64
    return (m + 127) // 128 * 128


@pytest.mark.parametrize("max_m", [1, 10, 16, 255, 1000, 4096, 5000, 8192])
def test_buckets_reach_every_padded_m(max_m: int) -> None:
    buckets = _aiter_gemm_m_buckets(max_m)
    assert buckets == sorted(set(buckets))
    assert buckets[-1] == max_m
    assert all(_aiter_padded_m(b) == b for b in buckets[:-1] if b >= 16)
    reachable = {_aiter_padded_m(m) for m in range(16, max_m + 1)}
    assert reachable <= {_aiter_padded_m(b) for b in buckets if b >= 16}


WarmupCall = tuple[int, dict[str, Any]]


def _fake_runner(calls: list[WarmupCall], enabled: bool = True) -> SimpleNamespace:
    return SimpleNamespace(
        vllm_config=SimpleNamespace(
            kernel_config=SimpleNamespace(enable_rocm_aiter_gemm_warmup=enabled),
            scheduler_config=SimpleNamespace(max_num_batched_tokens=512),
        ),
        _dummy_run=lambda num_tokens, **kwargs: calls.append((num_tokens, kwargs)),
    )


@pytest.mark.parametrize("enabled", [False, True])
def test_warmup_runs_each_size_once_when_enabled(enabled: bool) -> None:
    calls: list[WarmupCall] = []
    rocm_aiter_gemm_warmup(_fake_runner(calls, enabled))
    if not enabled:
        assert calls == []
        return
    sizes = [num_tokens for num_tokens, _ in calls]
    assert sizes == sorted(set(sizes))
    assert {1, 2, 4, 8, 504, 511, 512} <= set(sizes)
    assert all(
        kwargs == {"skip_eplb": True, "skip_attn": False, "is_profile": False}
        for _, kwargs in calls
    )


def test_warmup_skips_attention_for_a_kv_cacheless_caller() -> None:
    """The weight cache daemon warms AITER before any KV cache exists, so its
    dummy runs must skip attention; `_dummy_run` only allows that under
    `is_profile`."""
    calls: list[WarmupCall] = []
    rocm_aiter_gemm_warmup(_fake_runner(calls), skip_attn=True)
    assert calls
    assert all(
        kwargs == {"skip_eplb": True, "skip_attn": True, "is_profile": True}
        for _, kwargs in calls
    )
