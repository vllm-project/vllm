# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for MLA full-CUDAGraph cudagraph-support gating.

These tests guard the fix for using an MLA backend that only supports
single-token decode in full CUDA graphs (``query_len_support`` is
``SINGLE_ONLY``) together with speculative decoding. Such a backend must
advertise ``UNIFORM_SINGLE_TOKEN_DECODE`` (not ``UNIFORM_BATCH``) so the
dispatcher downgrades ``FULL_AND_PIECEWISE -> PIECEWISE`` before
``build_for_cudagraph_capture`` is ever reached with a multi-token batch.
"""

from types import SimpleNamespace

import pytest

from vllm.config.compilation import (
    CompilationConfig,
    CompilationMode,
    CUDAGraphMode,
)
from vllm.model_executor.layers.attention.mla_attention import (
    MLACommonMetadataBuilder,
    QueryLenSupport,
)
from vllm.v1.attention.backend import AttentionCGSupport


def _call_build_for_cudagraph_capture(
    *,
    reorder_batch_threshold: int,
    query_len_support: QueryLenSupport,
    max_query_len: int,
    num_reqs: int = 1,
    num_actual_tokens: int | None = None,
):
    """Invoke ``build_for_cudagraph_capture`` on a minimal fake builder.

    We don't need (or want) to spin up a full ``VllmConfig``-backed
    builder just to exercise the guard. The method only touches a small
    set of attributes on ``self`` and a few fields on the metadata.
    """
    if num_actual_tokens is None:
        num_actual_tokens = num_reqs * max_query_len

    fake_builder = SimpleNamespace(
        reorder_batch_threshold=reorder_batch_threshold,
        query_len_support=query_len_support,
        build=lambda _common_prefix_len, m: m,
    )
    fake_metadata = SimpleNamespace(
        num_reqs=num_reqs,
        num_actual_tokens=num_actual_tokens,
        max_query_len=max_query_len,
    )

    return MLACommonMetadataBuilder.build_for_cudagraph_capture(
        fake_builder,
        fake_metadata,  # type: ignore[arg-type]
    )


def test_decode_only_capture_passes_through():
    """Pure-decode capture should call ``build`` unchanged."""
    result = _call_build_for_cudagraph_capture(
        reorder_batch_threshold=1,
        query_len_support=QueryLenSupport.SINGLE_ONLY,
        max_query_len=1,
        num_reqs=4,
    )
    assert result.max_query_len == 1


def test_uniform_spec_decode_capture_within_threshold_passes():
    """Backends that bumped ``reorder_batch_threshold`` past 1 (i.e. they
    advertise UNIFORM/VARLEN ``query_len_support``) should still capture
    spec-decode shapes when ``max_query_len`` fits within the bumped
    threshold."""
    result = _call_build_for_cudagraph_capture(
        reorder_batch_threshold=4,  # 1 + num_speculative_tokens=3
        query_len_support=QueryLenSupport.UNIFORM,
        max_query_len=4,
        num_reqs=2,
    )
    assert result.max_query_len == 4


def test_capture_size_must_not_exceed_max_num_seq():
    """The pre-existing num_reqs/num_actual_tokens check still fires."""
    with pytest.raises(AssertionError, match="decode-only full CUDAGraph capture"):
        _call_build_for_cudagraph_capture(
            reorder_batch_threshold=1,
            query_len_support=QueryLenSupport.SINGLE_ONLY,
            max_query_len=1,
            num_reqs=8,
            num_actual_tokens=4,  # fewer tokens than reqs at threshold=1
        )


def test_triton_mla_advertises_decode_only_cudagraph_support():
    """Triton MLA only supports single-token (query_len==1) decodes, so it
    must advertise ``UNIFORM_SINGLE_TOKEN_DECODE`` (decode-only full cudagraph)
    rather than ``UNIFORM_BATCH``. ``UNIFORM_BATCH`` would wrongly claim
    multi-token (spec-decode) capture support and crash in
    ``build_for_cudagraph_capture``; ``UNIFORM_SINGLE_TOKEN_DECODE`` lets the
    dispatcher downgrade FULL -> PIECEWISE under spec decode instead. (Cutlass
    MLA, another SINGLE_ONLY backend, follows the same convention.)"""
    from vllm.v1.attention.backends.mla.triton_mla import TritonMLAMetadataBuilder

    assert TritonMLAMetadataBuilder.query_len_support is QueryLenSupport.SINGLE_ONLY
    support = TritonMLAMetadataBuilder.get_cudagraph_support(
        vllm_config=None,  # type: ignore[arg-type]
        kv_cache_spec=None,  # type: ignore[arg-type]
    )
    assert support is AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE


def _make_piecewise_capable_compilation_config(
    cudagraph_mode: CUDAGraphMode,
) -> CompilationConfig:
    """Build a CompilationConfig whose splitting ops include every
    attention op known to the dispatcher (the test for ``setting
    cudagraph_mode=PIECEWISE`` short-circuits to ``NONE`` if
    ``splitting_ops_contain_attention()`` is ``False``, and that helper
    requires *all* ``_attention_ops`` to be present)."""
    return CompilationConfig(
        mode=CompilationMode.VLLM_COMPILE,
        cudagraph_mode=cudagraph_mode,
        splitting_ops=list(CompilationConfig._attention_ops),
    )


def test_dispatcher_downgrades_full_to_piecewise_for_single_only_spec_decode():
    """End-to-end of the dispatcher logic: with a SINGLE_ONLY MLA
    backend reporting ``UNIFORM_SINGLE_TOKEN_DECODE`` and a spec-decode
    workload (``uniform_decode_query_len > 1``), the dispatcher must
    drop FULL_AND_PIECEWISE down to PIECEWISE rather than keep FULL and
    crash during capture."""
    cc = _make_piecewise_capable_compilation_config(CUDAGraphMode.FULL_AND_PIECEWISE)
    resolved = cc.resolve_cudagraph_mode_and_sizes(
        min_cg_support=AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE,
        min_cg_attn_backend="TritonMLAMetadataBuilder",
        uniform_decode_query_len=4,  # 1 + num_speculative_tokens=3
        tensor_parallel_size=1,
    )
    assert resolved is CUDAGraphMode.PIECEWISE


def test_dispatcher_keeps_full_for_uniform_backend_with_spec_decode():
    """Sanity check the other direction: a backend that reports
    ``UNIFORM_BATCH`` keeps FULL_AND_PIECEWISE under spec decode."""
    cc = _make_piecewise_capable_compilation_config(CUDAGraphMode.FULL_AND_PIECEWISE)
    resolved = cc.resolve_cudagraph_mode_and_sizes(
        min_cg_support=AttentionCGSupport.UNIFORM_BATCH,
        min_cg_attn_backend="RocmAiterMLAMetadataBuilder",
        uniform_decode_query_len=4,
        tensor_parallel_size=1,
    )
    assert resolved is CUDAGraphMode.FULL_AND_PIECEWISE
