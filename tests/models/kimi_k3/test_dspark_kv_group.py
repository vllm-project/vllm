# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The K3 DSpark draft must not merge into the target's MLA KV cache group."""

from types import SimpleNamespace

import pytest

from vllm.models.kimi_k3.nvidia.mla import MultiHeadLatentAttention
from vllm.v1.kv_cache_interface import MLAAttentionSpec


def _get_spec(non_causal_multi_token_decode: bool) -> MLAAttentionSpec:
    mla = MultiHeadLatentAttention.__new__(MultiHeadLatentAttention)
    mla.kv_cache_dtype = "auto"
    mla.head_size = 576
    mla.non_causal_multi_token_decode = non_causal_multi_token_decode
    vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=64),
        model_config=None,
    )
    return mla.get_kv_cache_spec(vllm_config)  # type: ignore[arg-type]


def test_dspark_draft_spec_carries_model_version():
    spec = _get_spec(non_causal_multi_token_decode=True)
    assert spec.model_version == "kimi_k3_dspark"


def test_target_spec_has_no_model_version():
    spec = _get_spec(non_causal_multi_token_decode=False)
    assert spec.model_version is None


def test_dspark_draft_spec_cannot_merge_into_target_group():
    # The split matters because MLAAttentionSpec.merge ORs
    # non_causal_multi_token_decode, which would flag the causal target.
    with pytest.raises(AssertionError, match="model version"):
        MLAAttentionSpec.merge(
            [
                _get_spec(non_causal_multi_token_decode=False),
                _get_spec(non_causal_multi_token_decode=True),
            ]
        )
