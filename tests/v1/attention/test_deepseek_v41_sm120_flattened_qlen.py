# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM120 admission to the flattened device-query-lens path.

The DSA indexer has two ways to take per-request query lengths off device
tensors: the SM100 varlen paged MQA logits kernel, or Hopper's per-token
flattening. SM120 supports neither gate today, so `enable_adaptive_verification`
cannot start on RTX PRO 6000 (capability 12.0) even though DeepGEMM already
schedules arbitrary `next_n` there and the MQA logits kernel takes the same
fp32-scale inputs as Hopper.

These tests pin the allowlist so the gate cannot silently regress, and pin the
SM100 gate so SM120 does not acquire the MX-scale path by accident.
"""

from types import SimpleNamespace

import pytest

from vllm.v1.attention.backends.mla import indexer
from vllm.v1.attention.backends.mla.indexer import (
    DeepseekV4IndexerBackend,
    DeepseekV41IndexerBackend,
)


def _mock_cuda_platform(monkeypatch: pytest.MonkeyPatch, family: int) -> None:
    """CUDA host whose only capability family is *family* (90, 100 or 120)."""
    monkeypatch.setattr(indexer.current_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(indexer.current_platform, "is_rocm", lambda: False)
    monkeypatch.setattr(
        indexer.current_platform,
        "is_device_capability_family",
        lambda cap, device_id=0: cap // 10 == family // 10,
    )


def _mock_deep_gemm(monkeypatch: pytest.MonkeyPatch, present: bool) -> None:
    monkeypatch.setattr(indexer, "has_deep_gemm", lambda: present)


@pytest.mark.cpu_test
@pytest.mark.parametrize("family", [90, 120])
def test_flattened_device_query_lens_supported_on_hopper_and_sm120(
    monkeypatch: pytest.MonkeyPatch, family: int
) -> None:
    _mock_cuda_platform(monkeypatch, family)
    _mock_deep_gemm(monkeypatch, True)

    assert indexer._supports_flattened_device_query_lens()


@pytest.mark.cpu_test
def test_flattened_device_query_lens_not_supported_on_sm100(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SM100 takes the varlen kernel instead; it must not enter this path."""
    _mock_cuda_platform(monkeypatch, 100)
    _mock_deep_gemm(monkeypatch, True)

    assert not indexer._supports_flattened_device_query_lens()


@pytest.mark.cpu_test
@pytest.mark.parametrize("family", [90, 120])
def test_flattened_device_query_lens_requires_deep_gemm(
    monkeypatch: pytest.MonkeyPatch, family: int
) -> None:
    _mock_cuda_platform(monkeypatch, family)
    _mock_deep_gemm(monkeypatch, False)

    assert not indexer._supports_flattened_device_query_lens()


@pytest.mark.cpu_test
def test_sm120_does_not_gain_the_sm100_varlen_kernel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The SM100 gate is deliberately untouched by this change."""
    _mock_cuda_platform(monkeypatch, 120)
    _mock_deep_gemm(monkeypatch, True)

    assert not indexer._supports_varlen_paged_mqa_logits()


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    "backend_cls", [DeepseekV4IndexerBackend, DeepseekV41IndexerBackend]
)
def test_sm120_indexer_backend_accepts_device_query_lens_mismatch(
    monkeypatch: pytest.MonkeyPatch, backend_cls: type[DeepseekV4IndexerBackend]
) -> None:
    """This is the predicate `enable_adaptive_verification` is gated on."""
    _mock_cuda_platform(monkeypatch, 120)
    _mock_deep_gemm(monkeypatch, True)

    assert backend_cls.supports_device_cpu_query_lens_mismatch()


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    "backend_cls", [DeepseekV4IndexerBackend, DeepseekV41IndexerBackend]
)
def test_sm120_indexer_backend_rejects_without_deep_gemm(
    monkeypatch: pytest.MonkeyPatch, backend_cls: type[DeepseekV4IndexerBackend]
) -> None:
    _mock_cuda_platform(monkeypatch, 120)
    _mock_deep_gemm(monkeypatch, False)

    assert not backend_cls.supports_device_cpu_query_lens_mismatch()


@pytest.mark.cpu_test
def test_sm120_adaptive_verification_selects_flattening(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With the allowlist widened, the adaptive branch of `_use_flattening`
    has to hold on SM120 — that is what the gate was blocking."""
    _mock_cuda_platform(monkeypatch, 120)
    _mock_deep_gemm(monkeypatch, True)

    adaptive = SimpleNamespace(
        speculative_config=SimpleNamespace(enable_adaptive_verification=True),
        num_speculative_tokens=1,
    )
    fixed = SimpleNamespace(
        speculative_config=SimpleNamespace(enable_adaptive_verification=False),
        num_speculative_tokens=1,
    )

    assert indexer._use_flattening(adaptive)
    # next_n == 2 is native off SM90/SM100, so the fixed arm does not flatten.
    assert not indexer._use_flattening(fixed)
