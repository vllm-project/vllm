# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# KV cache group spec variants — discriminated union on `spec_type`.
# All variants share the fields defined in _BaseGroupSpec.
# ---------------------------------------------------------------------------


class _KVCacheGroupBase(BaseModel):
    group_id: int
    layer_names: list[str]
    block_size: int
    page_size_bytes: int


class _FullAttentionBase(_KVCacheGroupBase):
    num_kv_heads: int
    head_size: int
    head_size_v: int
    dtype: str
    sliding_window: int | None = None
    attention_chunk_size: int | None = None


class FullAttentionGroupSpec(_FullAttentionBase):
    spec_type: Literal["FullAttentionSpec"] = "FullAttentionSpec"


class MLAAttentionGroupSpec(_FullAttentionBase):
    spec_type: Literal["MLAAttentionSpec"] = "MLAAttentionSpec"
    cache_dtype_str: str | None = None


class SlidingWindowGroupSpec(_KVCacheGroupBase):
    spec_type: Literal["SlidingWindowSpec"] = "SlidingWindowSpec"
    num_kv_heads: int
    head_size: int
    dtype: str
    sliding_window: int


class ChunkedLocalAttentionGroupSpec(_KVCacheGroupBase):
    spec_type: Literal["ChunkedLocalAttentionSpec"] = "ChunkedLocalAttentionSpec"
    num_kv_heads: int
    head_size: int
    dtype: str
    attention_chunk_size: int


class MambaGroupSpec(_KVCacheGroupBase):
    spec_type: Literal["MambaSpec"] = "MambaSpec"
    shapes: list[list[int]]
    dtypes: list[str]
    mamba_type: str
    mamba_cache_mode: str


class CrossAttentionGroupSpec(_KVCacheGroupBase):
    spec_type: Literal["CrossAttentionSpec"] = "CrossAttentionSpec"
    num_kv_heads: int
    head_size: int
    dtype: str


class SinkFullAttentionGroupSpec(_FullAttentionBase):
    spec_type: Literal["SinkFullAttentionSpec"] = "SinkFullAttentionSpec"
    sink_len: int | None = None


class UniformTypeGroupSpec(_KVCacheGroupBase):
    spec_type: Literal["UniformTypeKVCacheSpecs"] = "UniformTypeKVCacheSpecs"
    layer_specs: list[dict]


# Discriminated union — ``spec_type`` is the discriminator field.
KVCacheGroupInfo = Annotated[
    FullAttentionGroupSpec
    | MLAAttentionGroupSpec
    | SlidingWindowGroupSpec
    | ChunkedLocalAttentionGroupSpec
    | MambaGroupSpec
    | CrossAttentionGroupSpec
    | SinkFullAttentionGroupSpec
    | UniformTypeGroupSpec,
    Field(discriminator="spec_type"),
]


# ---------------------------------------------------------------------------
# Top-level response model for GET /v1/server/kv-cache
# ---------------------------------------------------------------------------


class KVCacheRuntimeInfo(BaseModel):
    kv_cache_size_tokens: int | None = None
    max_concurrency: float | None = None
    num_gpu_blocks: int | None = None
    num_cpu_blocks: int | None = None
    groups: list[KVCacheGroupInfo] = Field(default_factory=list)
