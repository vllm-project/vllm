# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression test for FlashInferMetadataBuilder.get_cudagraph_support.

vllm-project/vllm#55581: the class method derived the query-head count from the
*target* model for every KV-cache group. For a draft/spec-decode group whose
head geometry differs from the target, this mis-declared CUDA-graph support
(every graph switched off at an odd TP degree) even though the kernel-selection
path used the group's own per-layer head count. The fix derives the head count
from the group's layers (get_num_attention_heads_from_layers) when layer_names
is provided, falling back to the model-wide count otherwise.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.v1.attention.backend import AttentionCGSupport
from vllm.v1.attention.backends import flashinfer as fi
from vllm.v1.kv_cache_interface import AttentionSpec

pytestmark = pytest.mark.cpu_test


class _FakeImpl:
    """Minimal stand-in for an AttentionImpl exposing only ``num_heads``."""

    def __init__(self, num_heads: int):
        self.num_heads = num_heads


class _FakeLayer(AttentionLayerBase):
    """Concrete AttentionLayerBase stub carrying only ``impl.num_heads``."""

    def __init__(self, num_heads: int):
        self.impl = _FakeImpl(num_heads)

    def get_attn_backend(self):  # pragma: no cover - not exercised here
        raise NotImplementedError

    def get_kv_cache_spec(self, vllm_config):  # pragma: no cover
        raise NotImplementedError


def _make_config(target_heads_per_rank: int) -> SimpleNamespace:
    """Build a VllmConfig-like object with a populated forward context.

    Mirrors the issue's TP=3 arrangement: the target model reports 66 total
    heads (22/rank at TP=3), while the draft group's layers report 12/rank.
    """
    model_config = SimpleNamespace(
        get_num_attention_heads=lambda parallel_config: target_heads_per_rank
    )
    parallel_config = SimpleNamespace(tensor_parallel_size=3)
    return SimpleNamespace(
        model_config=model_config,
        parallel_config=parallel_config,
        compilation_config=SimpleNamespace(static_forward_context={}),
    )


def _draft_spec(num_kv_heads: int = 3, head_size: int = 128) -> AttentionSpec:
    return AttentionSpec(
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        dtype=torch.bfloat16,
        block_size=16,
    )


def test_draft_group_uses_its_own_head_count(monkeypatch):
    """Draft group (12 q-heads, 3 kv-heads) must declare UNIFORM_BATCH.

    The target model reports 22 q-heads/rank; 22 % 3 != 0, so the buggy
    target-head-derived path downgrades to UNIFORM_SINGLE_TOKEN_DECODE. The
    draft's own 12 % 3 == 0, so the fixed path declares UNIFORM_BATCH.
    """
    # Force the divisibility check to be the deciding factor, bypassing the
    # SM100/artifactory gating so the test is deterministic on any host.
    monkeypatch.setattr(
        fi, "can_use_trtllm_attention", lambda num_qo_heads, num_kv_heads: True
    )

    vllm_config = _make_config(target_heads_per_rank=22)
    draft_layer_names = ["model.layers.0.attn", "model.layers.1.attn"]
    for name in draft_layer_names:
        vllm_config.compilation_config.static_forward_context[name] = _FakeLayer(
            num_heads=12
        )

    support = fi.FlashInferMetadataBuilder.get_cudagraph_support(
        vllm_config, _draft_spec(), draft_layer_names
    )
    assert support == AttentionCGSupport.UNIFORM_BATCH


def test_falls_back_to_model_wide_heads_without_layer_names(monkeypatch):
    """Without layer_names, the model-wide target head count is used.

    22 % 3 != 0, so the fallback path downgrades, confirming the optional
    parameter is backward-compatible (other backends ignore it).
    """
    monkeypatch.setattr(
        fi, "can_use_trtllm_attention", lambda num_qo_heads, num_kv_heads: True
    )

    vllm_config = _make_config(target_heads_per_rank=22)
    support = fi.FlashInferMetadataBuilder.get_cudagraph_support(
        vllm_config, _draft_spec()
    )
    assert support == AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE
