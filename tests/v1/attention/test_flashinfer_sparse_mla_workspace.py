# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only tests for FlashInfer sparse MLA backend constraints."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm.v1.attention.backends.mla.flashinfer_mla_sparse import (
    _DEFAULT_WORKSPACE_BUFFER_SIZE,
    FlashInferMLASparseTRTLLMBackend,
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


def test_combined_pcp_dcp_rejects_flashinfer_sparse():
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            prefill_context_parallel_size=4,
            decode_context_parallel_size=4,
        ),
        model_config=None,
    )
    with patch("vllm.config.get_current_vllm_config", return_value=config):
        reason = FlashInferMLASparseTRTLLMBackend.supports_combination(
            head_size=576,
            dtype=torch.bfloat16,
            kv_cache_dtype="fp8",
            block_size=64,
            use_mla=True,
            has_sink=False,
            use_sparse=True,
            use_mm_prefix=False,
            device_capability=SimpleNamespace(major=10),
        )

    assert reason is not None
    assert "use FLASHMLA_SPARSE" in reason


@pytest.mark.parametrize("pcp_size,dcp_size", [(4, 1), (1, 4)])
def test_single_context_parallel_mode_allows_flashinfer_sparse(
    pcp_size: int, dcp_size: int
):
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            prefill_context_parallel_size=pcp_size,
            decode_context_parallel_size=dcp_size,
        ),
        model_config=None,
    )
    with patch("vllm.config.get_current_vllm_config", return_value=config):
        reason = FlashInferMLASparseTRTLLMBackend.supports_combination(
            head_size=576,
            dtype=torch.bfloat16,
            kv_cache_dtype="fp8",
            block_size=64,
            use_mla=True,
            has_sink=False,
            use_sparse=True,
            use_mm_prefix=False,
            device_capability=SimpleNamespace(major=10),
        )

    assert reason is None


@pytest.mark.parametrize("dcp", [1, 2])
def test_trtllm_builder_presizes_workspace_only_under_dcp(monkeypatch, dcp):
    """Only DCP pre-sizes the workspace, and before the first graph capture."""
    from vllm.model_executor.layers.attention.sparse_mla_attention import (
        SparseMLACommonMetadataBuilder,
    )
    from vllm.v1.attention.backends.mla import flashinfer_mla_sparse as fi_sparse

    def stub_base_init(self, kv_cache_spec, layer_names, vllm_config, device):
        # Only the reorder-threshold and workspace inputs matter here.
        self.vllm_config = vllm_config
        self.dcp_world_size = dcp

    monkeypatch.setattr(SparseMLACommonMetadataBuilder, "__init__", stub_base_init)
    workspace = MagicMock()
    monkeypatch.setattr(fi_sparse, "_get_workspace_buffer", workspace)
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(get_num_attention_heads=lambda _: 16),
        parallel_config=SimpleNamespace(decode_context_parallel_size=dcp),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=8192),
        speculative_config=None,
        attention_config=SimpleNamespace(hisparse_config=None),
    )

    fi_sparse.FlashInferMLASparseTRTLLMMetadataBuilder(
        None, ["layer"], vllm_config, torch.device("cpu")
    )

    assert [call.args[1] for call in workspace.call_args_list] == (
        [] if dcp == 1 else [_required_workspace_bytes(dcp, 16, 8192)]
    )


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
