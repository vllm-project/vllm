# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest

from vllm.model_executor.warmup.kernel_warmup import (
    _aiter_gemm_m_buckets,
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
