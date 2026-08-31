# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton MLA non-causal multi-token DCP capability and its two read sites."""

from types import SimpleNamespace

from vllm.platforms import current_platform
from vllm.v1.attention.backend import AttentionCGSupport
from vllm.v1.attention.backends.mla import triton_mla
from vllm.v1.attention.backends.mla.triton_mla import (
    TritonMLABackend,
    TritonMLAMetadataBuilder,
)


def test_noncausal_multitoken_dcp_capability_tracks_platform():
    """The capability is ROCm-scoped, and both read sites must agree on it.

    ``supports_non_causal_dcp`` gates backend eligibility for non-causal MLA
    with DCP on every platform that can select Triton MLA (CUDA and XPU as well
    as ROCm), so this assertion is deliberately not platform gated.
    """
    expected = current_platform.is_rocm()

    assert TritonMLAMetadataBuilder.supports_non_causal_multi_token_decode
    assert TritonMLABackend.supports_non_causal()
    assert TritonMLAMetadataBuilder.supports_non_causal_multi_token_dcp is expected
    assert TritonMLABackend.supports_non_causal_dcp() is expected


def test_reorder_threshold_keeps_the_draft_block_under_dcp(monkeypatch):
    """A non-causal draft group must stay a decode group when DCP is on.

    ``_init_reorder_batch_threshold`` forces the threshold back to 1 whenever
    DCP is enabled and the backend does not claim varlen DCP support, which
    would push the draft's multi-token block onto the prefill path. The builder
    has to pass the same capability constant it advertises, or the two silently
    disagree.
    """
    builder = object.__new__(TritonMLAMetadataBuilder)
    builder.vllm_config = SimpleNamespace(
        speculative_config=SimpleNamespace(
            num_speculative_tokens=3, parallel_drafting=False
        ),
        parallel_config=SimpleNamespace(decode_context_parallel_size=4),
    )

    builder._init_reorder_batch_threshold(
        1,
        supports_spec_as_decode=True,
        supports_dcp_with_varlen=triton_mla._SUPPORTS_NONCAUSAL_MULTITOKEN_DCP,
    )

    expected = 4 if current_platform.is_rocm() else 1
    assert builder.reorder_batch_threshold == expected


def test_cudagraph_support_tracks_multitoken_decode_group():
    single_token_spec = SimpleNamespace(non_causal_multi_token_decode=False)
    multi_token_spec = SimpleNamespace(non_causal_multi_token_decode=True)

    assert (
        TritonMLAMetadataBuilder.get_cudagraph_support(None, single_token_spec)
        == AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE
    )
    assert (
        TritonMLAMetadataBuilder.get_cudagraph_support(None, multi_token_spec)
        == AttentionCGSupport.UNIFORM_BATCH
    )
