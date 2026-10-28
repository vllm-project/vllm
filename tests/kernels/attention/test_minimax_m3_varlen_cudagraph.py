# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MiniMax-M3 builders advertise varlen decode through the bounded contract."""

from types import SimpleNamespace

import pytest

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
            parallel_drafting=False,
        ),
        use_v2_model_runner=True,
    )


@pytest.fixture(autouse=True)
def _rocm_without_aiter_sparse_pa(monkeypatch):
    monkeypatch.setattr(current_platform, "is_rocm", lambda: True)
    monkeypatch.setattr(
        sparse_attention, "_minimax_m3_aiter_sparse_pa_requested", lambda: False
    )


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
            [[group]], config, max_query_len=1 + NUM_SPEC_TOKENS
        )
        is None
    )
    # Without adaptive verification they keep plain uniform-batch graphs.
    assert builder.get_varlen_cudagraph_max_query_len(_vllm_config(False), None) is None


def test_aiter_msa_indexer_builder_does_not_claim_varlen():
    indexer_msa = pytest.importorskip("vllm.models.minimax_m3.amd.indexer_msa")
    builder = indexer_msa.MiniMaxM3IndexerMSAMetadataBuilder
    assert not builder.supports_varlen_decode()
    assert builder.get_varlen_cudagraph_max_query_len(_vllm_config(True), None) is None
