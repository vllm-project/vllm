# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Contract tests for the shared spec-type classification."""

from __future__ import annotations

import pytest

from vllm.distributed.kv_transfer.kv_connector.v1.transfer_planning import (
    get_representative_spec_type,
    is_attention_spec,
    is_ssm_spec,
)
from vllm.v1.kv_cache_interface import (
    ChunkedLocalAttentionSpec,
    CircularBufferSpec,
    CrossAttentionSpec,
    FullAttentionSpec,
    MambaSpec,
    MLAAttentionSpec,
    SlidingWindowSpec,
    UniformTypeKVCacheSpecs,
)


class TestSpecClassification:
    """Pin the spec-type classification that transfer planning dispatches on."""

    @pytest.mark.parametrize(
        ("spec_type", "classification"),
        [
            (FullAttentionSpec, "attention"),
            (MLAAttentionSpec, "attention"),
            (SlidingWindowSpec, "attention"),
            (ChunkedLocalAttentionSpec, "attention"),
            (CircularBufferSpec, "attention"),
            (CrossAttentionSpec, "attention"),
            (MambaSpec, "ssm"),
        ],
    )
    def test_spec_type_classification(self, spec_type, classification):
        assert is_attention_spec(spec_type) == (classification == "attention")
        assert is_ssm_spec(spec_type) == (classification == "ssm")

    def test_representative_spec_type_unwraps_uniform_groups(self):
        inner = FullAttentionSpec.__new__(FullAttentionSpec)
        wrapper = UniformTypeKVCacheSpecs.__new__(UniformTypeKVCacheSpecs)
        # UniformTypeKVCacheSpecs is frozen; set its field map directly.
        object.__setattr__(wrapper, "kv_cache_specs", {"a": inner, "b": inner})
        assert get_representative_spec_type(wrapper) is FullAttentionSpec
        assert get_representative_spec_type(inner) is FullAttentionSpec
