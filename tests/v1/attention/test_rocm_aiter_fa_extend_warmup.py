# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The AITER flash-attention backend warms the kernels only an extend reaches."""

from unittest.mock import MagicMock

import pytest
import torch

from vllm.platforms import current_platform
from vllm.v1.attention.backends import rocm_aiter_fa
from vllm.v1.attention.backends.rocm_aiter_fa import AiterFlashAttentionExtendKernel

pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm(), reason="ROCm-specific tests"
)

NO_WINDOW = (-1, -1)


def _keys(**overrides):
    kwargs = dict(
        dtype=torch.bfloat16,
        head_dim=64,
        num_qo_heads=8,
        num_kv_heads=2,
        sliding_window=NO_WINDOW,
        has_alibi=False,
        has_sinks=False,
    )
    kwargs.update(overrides)
    return AiterFlashAttentionExtendKernel().get_warmup_keys(**kwargs)


def test_warms_the_causal_and_context_chunk_calls():
    causal, context = _keys()

    assert causal.causal and not context.causal
    assert causal.qhead_per_kvhead == context.qhead_per_kvhead == 4
    assert causal.head_dim == context.head_dim == 64


def test_sinks_only_reach_the_causal_call():
    """`extend_forward` adds the sinks once, so context chunks pass none."""
    causal, context = _keys(has_sinks=True)

    assert causal.has_sinks and not context.has_sinks


def test_sliding_window_layers_have_nothing_to_warm():
    """A window extend runs the same LSE-free call a new prompt does."""
    assert _keys(sliding_window=(1023, 0)) == []


def test_compile_runs_the_calls_extend_forward_makes(monkeypatch):
    calls = []
    ops = MagicMock()
    ops.flash_attn_varlen_func.side_effect = lambda **kwargs: calls.append(kwargs)
    monkeypatch.setattr(rocm_aiter_fa, "rocm_aiter_ops", ops)
    monkeypatch.setattr(current_platform, "device_type", "cpu")
    monkeypatch.setattr(torch.accelerator, "synchronize", lambda: None)

    kernel = AiterFlashAttentionExtendKernel()
    keys = kernel.get_warmup_keys(
        dtype=torch.bfloat16,
        head_dim=64,
        num_qo_heads=8,
        num_kv_heads=2,
        sliding_window=NO_WINDOW,
        has_alibi=True,
        has_sinks=True,
    )
    kernel.compile_many(keys)

    causal, context = calls
    for call, is_causal in ((causal, True), (context, False)):
        assert call["causal"] is is_causal
        assert call["return_lse"]
        assert call["min_seqlen_q"] == rocm_aiter_fa._MIN_SEQLEN_Q
        assert call["window_size"] == NO_WINDOW
        assert call["alibi_slopes"] is not None
        # AITER treats a one-token query as decode and drops `causal`.
        assert call["max_seqlen_q"] > 1
        assert call["q"].shape[1:] == (4, 64)
        assert call["k"].shape[1:] == (1, 64)
    assert causal["sink_ptr"] is not None and context["sink_ptr"] is None
    assert causal["max_seqlen_k"] == causal["max_seqlen_q"]
    assert context["max_seqlen_k"] > context["max_seqlen_q"]
