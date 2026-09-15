# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only tests for the FlashInfer sparse MLA workspace sizing (#50781)."""

from unittest.mock import patch

from vllm.v1.attention.backends.mla.flashinfer_mla_sparse import (
    _DEFAULT_WORKSPACE_BUFFER_SIZE,
    _required_workspace_bytes,
    compute_trtllm_sparse_mla_workspace_bytes,
)

# Reporter config from vllm#50781: GLM-5.2 (64 q heads), TP=8, DCP=8,
# max_num_batched_tokens=16384, FlashInfer 0.6.14.
_REPORTER_HEADS_PER_RANK = 64 // 8
_REPORTER_DCP = 8
_REPORTER_MAX_TOKENS = 16384
# From the reported crash: trtllm-gen requested exactly this many bytes for
# trtllm_gen_softmax_workspace after an 8 MiB counter carve had left
# 404,750,336 of the 413,138,944-byte default buffer.
_REPORTER_OBSERVED_SOFTMAX_BYTES = 1_611_661_312
_REPORTER_OBSERVED_PRECARVE_BYTES = 8_388_608


def test_reporter_softmax_carve_is_reproduced_byte_exactly():
    # The crashing step scheduled 12288 tokens (a single 12K-token request);
    # the softmax slab formula must reproduce the kernel's request exactly:
    # 8 * (8 heads/rank * 8 dcp) * 12288 tokens * 256 + 1 MiB guard.
    softmax_bytes = compute_trtllm_sparse_mla_workspace_bytes(
        base_workspace_bytes=0,
        dcp_world_size=_REPORTER_DCP,
        num_heads_per_rank=_REPORTER_HEADS_PER_RANK,
        max_num_batched_tokens=12288,
    )
    assert softmax_bytes == _REPORTER_OBSERVED_SOFTMAX_BYTES


def test_reporter_config_covers_observed_overflow():
    computed = compute_trtllm_sparse_mla_workspace_bytes(
        base_workspace_bytes=_DEFAULT_WORKSPACE_BUFFER_SIZE,
        dcp_world_size=_REPORTER_DCP,
        num_heads_per_rank=_REPORTER_HEADS_PER_RANK,
        max_num_batched_tokens=_REPORTER_MAX_TOKENS,
    )
    assert computed >= (
        _REPORTER_OBSERVED_SOFTMAX_BYTES + _REPORTER_OBSERVED_PRECARVE_BYTES
    )


def test_non_dcp_size_is_unchanged():
    computed = compute_trtllm_sparse_mla_workspace_bytes(
        base_workspace_bytes=_DEFAULT_WORKSPACE_BUFFER_SIZE,
        dcp_world_size=1,
        num_heads_per_rank=128,
        max_num_batched_tokens=65536,
    )
    assert computed == _DEFAULT_WORKSPACE_BUFFER_SIZE


def test_default_constant_matches_envs_default(monkeypatch):
    from vllm import envs

    monkeypatch.delenv("VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE", raising=False)
    assert _DEFAULT_WORKSPACE_BUFFER_SIZE == envs.VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE


def test_env_unset_returns_computed(monkeypatch):
    monkeypatch.delenv("VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE", raising=False)
    required = _required_workspace_bytes(
        dcp_world_size=_REPORTER_DCP,
        num_heads_per_rank=_REPORTER_HEADS_PER_RANK,
        max_num_batched_tokens=_REPORTER_MAX_TOKENS,
    )
    assert required == compute_trtllm_sparse_mla_workspace_bytes(
        _DEFAULT_WORKSPACE_BUFFER_SIZE,
        _REPORTER_DCP,
        _REPORTER_HEADS_PER_RANK,
        _REPORTER_MAX_TOKENS,
    )


def test_env_override_below_computed_is_respected_with_warning(monkeypatch):
    override = 100 * 1024 * 1024
    monkeypatch.setenv("VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE", str(override))
    with patch(
        "vllm.v1.attention.backends.mla.flashinfer_mla_sparse.logger"
    ) as mock_logger:
        required = _required_workspace_bytes(
            dcp_world_size=_REPORTER_DCP,
            num_heads_per_rank=_REPORTER_HEADS_PER_RANK,
            max_num_batched_tokens=_REPORTER_MAX_TOKENS,
        )
    assert required == override
    mock_logger.warning_once.assert_called_once()


def test_env_override_above_computed_is_respected_without_warning(monkeypatch):
    override = 8 * 1024 * 1024 * 1024
    monkeypatch.setenv("VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE", str(override))
    with patch(
        "vllm.v1.attention.backends.mla.flashinfer_mla_sparse.logger"
    ) as mock_logger:
        required = _required_workspace_bytes(
            dcp_world_size=_REPORTER_DCP,
            num_heads_per_rank=_REPORTER_HEADS_PER_RANK,
            max_num_batched_tokens=_REPORTER_MAX_TOKENS,
        )
    assert required == override
    mock_logger.warning_once.assert_not_called()
