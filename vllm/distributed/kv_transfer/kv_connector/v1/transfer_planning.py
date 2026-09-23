# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KV cache spec-type classification shared by KV transfer connectors."""

from __future__ import annotations

from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    KVCacheSpec,
    MambaSpec,
    UniformTypeKVCacheSpecs,
)


def is_attention_spec(spec_type: type[KVCacheSpec]) -> bool:
    return issubclass(spec_type, AttentionSpec)


def is_ssm_spec(spec_type: type[KVCacheSpec]) -> bool:
    return issubclass(spec_type, MambaSpec)


def get_representative_spec_type(spec: KVCacheSpec) -> type[KVCacheSpec]:
    """Return the concrete spec type behind ``spec``, unwrapping uniform-type groups."""
    if isinstance(spec, UniformTypeKVCacheSpecs):
        # All inner specs are the same type; pick any.
        inner = next(iter(spec.kv_cache_specs.values()))
        return type(inner)
    return type(spec)
