# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU regression test for FlashInferMetadataBuilder.get_cudagraph_support.

Verifies that a draft/spec-decode KV-cache group with different head geometry
from the target model correctly reports UNIFORM_BATCH (not the wrong
UNIFORM_SINGLE_TOKEN_DECODE produced when the target model's global head
count is used instead of the group's own layers).

Fixes: https://github.com/vllm-project/vllm/issues/55581
"""

from unittest.mock import MagicMock, patch

import pytest

pytestmark = pytest.mark.cpu_test


class _FakeImpl:
    def __init__(self, num_heads: int):
        self.num_heads = num_heads


class _FakeLayer:
    def __init__(self, num_heads: int):
        self.impl = _FakeImpl(num_heads)


class _FakeAttentionSpec:
    def __init__(self, num_kv_heads: int):
        self.num_kv_heads = num_kv_heads


def _make_vllm_config(global_num_heads: int):
    cfg = MagicMock()
    cfg.model_config.get_num_attention_heads.return_value = global_num_heads
    cfg.parallel_config.decode_context_parallel_size = 1
    cfg.attention_config.use_non_causal = False
    return cfg


def _make_group_spec(layer_names: list[str], num_kv_heads: int):
    from vllm.v1.kv_cache_interface import UniformTypeKVCacheSpecs

    return UniformTypeKVCacheSpecs(
        block_size=16,
        kv_cache_specs={
            name: _FakeAttentionSpec(num_kv_heads) for name in layer_names
        },
    )


def _patch_get_layers(layer_map: dict):
    return patch(
        "vllm.v1.attention.backends.utils.get_layers_from_vllm_config",
        return_value=layer_map,
    )


def _patch_trtllm():
    return patch(
        "vllm.v1.attention.backends.flashinfer.can_use_trtllm_attention",
        side_effect=lambda **kwargs: (
            kwargs["num_qo_heads"] % kwargs["num_kv_heads"] == 0
        ),
    )


def _patch_attention_spec_type():
    return patch(
        "vllm.v1.attention.backends.flashinfer.AttentionSpec",
        _FakeAttentionSpec,
    )


def test_draft_group_uses_own_layer_heads_uniform_batch():
    """Draft group: 12 qo-heads / 3 kv-heads → divisible → UNIFORM_BATCH.

    Without the fix, get_cudagraph_support() used the *target* model's 22
    query-heads (22 % 3 != 0) and incorrectly returned
    UNIFORM_SINGLE_TOKEN_DECODE.
    """
    from vllm.v1.attention.backend import AttentionCGSupport
    from vllm.v1.attention.backends.flashinfer import FlashInferMetadataBuilder

    draft_layer_names = ["model.layers.0.self_attn", "model.layers.1.self_attn"]
    draft_layers = {n: _FakeLayer(num_heads=12) for n in draft_layer_names}

    vllm_cfg = _make_vllm_config(global_num_heads=22)
    kv_spec = _make_group_spec(draft_layer_names, num_kv_heads=3)

    with (
        _patch_get_layers(draft_layers),
        _patch_trtllm(),
        _patch_attention_spec_type(),
    ):
        result = FlashInferMetadataBuilder.get_cudagraph_support(vllm_cfg, kv_spec)

    assert result == AttentionCGSupport.UNIFORM_BATCH, (
        f"Expected UNIFORM_BATCH for draft group (12 q-heads / 3 kv-heads), "
        f"got {result}"
    )


def test_fallback_to_global_heads_when_no_layer_names():
    """Without layer_names, the global head count is used (backward compat).

    22 global q-heads, 3 kv-heads → trtllm unsupported → UNIFORM_SINGLE_TOKEN_DECODE.
    """
    from vllm.v1.attention.backend import AttentionCGSupport
    from vllm.v1.attention.backends.flashinfer import FlashInferMetadataBuilder

    vllm_cfg = _make_vllm_config(global_num_heads=22)
    kv_spec = _FakeAttentionSpec(num_kv_heads=3)

    with _patch_trtllm(), _patch_attention_spec_type():
        result = FlashInferMetadataBuilder.get_cudagraph_support(vllm_cfg, kv_spec)

    assert result == AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE


def test_target_group_uniform_batch_unchanged():
    """Target group with 24 q-heads / 3 kv-heads → UNIFORM_BATCH."""
    from vllm.v1.attention.backend import AttentionCGSupport
    from vllm.v1.attention.backends.flashinfer import FlashInferMetadataBuilder

    target_layer_names = [f"model.layers.{i}.self_attn" for i in range(5)]
    target_layers = {n: _FakeLayer(num_heads=24) for n in target_layer_names}

    vllm_cfg = _make_vllm_config(global_num_heads=24)
    kv_spec = _make_group_spec(target_layer_names, num_kv_heads=3)

    with (
        _patch_get_layers(target_layers),
        _patch_trtllm(),
        _patch_attention_spec_type(),
    ):
        result = FlashInferMetadataBuilder.get_cudagraph_support(vllm_cfg, kv_spec)

    assert result == AttentionCGSupport.UNIFORM_BATCH
