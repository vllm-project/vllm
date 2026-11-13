# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MiniMax-M3 builders advertise varlen decode through the bounded contract."""

from types import SimpleNamespace

import pytest

from vllm.models.minimax_m3.amd import indexer_msa, sparse_attention_nvfp4
from vllm.models.minimax_m3.common import indexer, sparse_attention
from vllm.platforms import current_platform
from vllm.v1.attention.backend import AttentionCGSupport
from vllm.v1.worker.gpu.attn_utils import get_varlen_cudagraph_unsupported_backend

NUM_SPEC_TOKENS = 7


def _vllm_config(adaptive: bool):
    return SimpleNamespace(
        speculative_config=SimpleNamespace(
            enable_adaptive_verification=adaptive,
            num_speculative_tokens=NUM_SPEC_TOKENS,
            # DSpark drafts in a separate pass and adds no target-batch rows.
            max_num_new_slots_for_drafting=0,
        ),
    )


@pytest.fixture(autouse=True)
def _rocm_without_aiter_sparse_pa(monkeypatch):
    monkeypatch.setattr(current_platform, "is_rocm", lambda: True)
    monkeypatch.setattr(
        sparse_attention, "_minimax_m3_aiter_sparse_pa_requested", lambda: False
    )


@pytest.mark.parametrize(
    "backend",
    [indexer.MiniMaxM3IndexerBackend, sparse_attention.MiniMaxM3SparseBackend],
)
def test_backends_allow_query_lens_mismatch_only_with_varlen_builders(
    backend, monkeypatch
):
    # The selector gates adaptive verification on this; only the ROCm Triton
    # builders read per-request query lengths from the device.
    assert backend.supports_device_cpu_query_lens_mismatch()
    monkeypatch.setattr(current_platform, "is_rocm", lambda: False)
    assert not backend.supports_device_cpu_query_lens_mismatch()


@pytest.mark.parametrize(
    "builder",
    [
        indexer.MiniMaxM3IndexerTritonMetadataBuilder,
        sparse_attention.MiniMaxM3SparseMetadataBuilder,
    ],
)
def test_varlen_builders_pass_adaptive_gate(builder):
    config = _vllm_config(adaptive=True)
    assert (
        builder.get_cudagraph_support(config, None) == AttentionCGSupport.UNIFORM_BATCH
    )
    assert builder.get_varlen_cudagraph_max_query_len(config, None) == (
        1 + NUM_SPEC_TOKENS
    )
    group = SimpleNamespace(
        layer_names=["layer"],
        kv_cache_spec=None,
        backend=SimpleNamespace(__name__=builder.__name__),
        get_metadata_builder=lambda _: builder,
    )
    assert (
        get_varlen_cudagraph_unsupported_backend(
            [[group]],  # type: ignore[list-item]
            config,
            max_query_len=1 + NUM_SPEC_TOKENS,
        )
        is None
    )
    # Without adaptive verification they keep plain uniform-batch graphs.
    assert builder.get_varlen_cudagraph_max_query_len(_vllm_config(False), None) is None


@pytest.mark.parametrize(
    "backend",
    [
        indexer_msa.MiniMaxM3IndexerMSABackend,
        sparse_attention_nvfp4.MiniMaxM3SparseRocmNvfp4Backend,
    ],
)
def test_uniform_only_rocm_builders_do_not_claim_varlen(backend):
    # Their decode kernels score uniform per-request query rows.
    builder = backend.get_builder_cls()
    assert not builder.supports_varlen_decode()
    assert builder.get_varlen_cudagraph_max_query_len(_vllm_config(True), None) is None
    assert not backend.supports_device_cpu_query_lens_mismatch()
