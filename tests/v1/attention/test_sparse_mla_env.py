# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for sparse MLA environment helpers."""

from vllm.v1.attention.backends.mla import sparse_mla_env


def test_prefill_topk_uses_sm12x_multi_request_guard(monkeypatch):
    monkeypatch.delenv("VLLM_TRITON_MLA_SPARSE_TOPK_CHUNK_SIZE", raising=False)
    monkeypatch.setattr(
        sparse_mla_env.current_platform,
        "is_device_capability_family",
        lambda family: family == 120,
    )

    assert (
        sparse_mla_env.triton_sparse_mla_prefill_topk_chunk_size(
            combined_topk_size=1152,
            compress_ratio=128,
            request_count=2,
        )
        == 256
    )


def test_prefill_topk_relaxes_sm12x_single_request_c128a(monkeypatch):
    monkeypatch.delenv("VLLM_TRITON_MLA_SPARSE_TOPK_CHUNK_SIZE", raising=False)
    monkeypatch.setattr(
        sparse_mla_env.current_platform,
        "is_device_capability_family",
        lambda family: family == 120,
    )

    assert (
        sparse_mla_env.triton_sparse_mla_prefill_topk_chunk_size(
            combined_topk_size=1152,
            compress_ratio=128,
            request_count=1,
        )
        == 1024
    )


def test_prefill_topk_keeps_default_for_other_lower_risk_shapes(monkeypatch):
    monkeypatch.delenv("VLLM_TRITON_MLA_SPARSE_TOPK_CHUNK_SIZE", raising=False)
    monkeypatch.setattr(
        sparse_mla_env.current_platform,
        "is_device_capability_family",
        lambda family: family == 120,
    )

    assert (
        sparse_mla_env.triton_sparse_mla_prefill_topk_chunk_size(
            combined_topk_size=640,
            compress_ratio=4,
            request_count=2,
        )
        == 512
    )
    assert (
        sparse_mla_env.triton_sparse_mla_prefill_topk_chunk_size(
            combined_topk_size=128,
            compress_ratio=1,
            request_count=2,
        )
        == 128
    )


def test_prefill_topk_honors_explicit_env_override(monkeypatch):
    monkeypatch.setenv("VLLM_TRITON_MLA_SPARSE_TOPK_CHUNK_SIZE", "512")
    monkeypatch.setattr(
        sparse_mla_env.current_platform,
        "is_device_capability_family",
        lambda family: family == 120,
    )

    assert (
        sparse_mla_env.triton_sparse_mla_prefill_topk_chunk_size(
            combined_topk_size=1152,
            compress_ratio=128,
            request_count=2,
        )
        == 512
    )
