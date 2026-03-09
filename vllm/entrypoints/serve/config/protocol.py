# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# KV cache group spec variants — discriminated union on `spec_type`.
# All variants share the fields defined in _BaseGroupSpec.
# ---------------------------------------------------------------------------


class _BaseGroupSpec(BaseModel):
    """Fields common to every KV cache group spec."""

    layer_names: list[str]
    block_size: int
    page_size_bytes: int


class FullAttentionGroupSpec(_BaseGroupSpec):
    spec_type: Literal["FullAttentionSpec"] = "FullAttentionSpec"
    num_kv_heads: int
    head_size: int
    head_size_v: int
    dtype: str
    sliding_window: int | None = None
    attention_chunk_size: int | None = None


class MLAAttentionGroupSpec(_BaseGroupSpec):
    spec_type: Literal["MLAAttentionSpec"] = "MLAAttentionSpec"
    num_kv_heads: int
    head_size: int
    head_size_v: int
    dtype: str
    sliding_window: int | None = None
    attention_chunk_size: int | None = None
    cache_dtype_str: str | None = None


class SlidingWindowGroupSpec(_BaseGroupSpec):
    spec_type: Literal["SlidingWindowSpec"] = "SlidingWindowSpec"
    num_kv_heads: int
    head_size: int
    dtype: str
    sliding_window: int


class ChunkedLocalAttentionGroupSpec(_BaseGroupSpec):
    spec_type: Literal["ChunkedLocalAttentionSpec"] = "ChunkedLocalAttentionSpec"
    num_kv_heads: int
    head_size: int
    dtype: str
    attention_chunk_size: int


class MambaGroupSpec(_BaseGroupSpec):
    spec_type: Literal["MambaSpec"] = "MambaSpec"
    shapes: list[list[int]]
    dtypes: list[str]
    mamba_type: str
    mamba_cache_mode: str


class CrossAttentionGroupSpec(_BaseGroupSpec):
    spec_type: Literal["CrossAttentionSpec"] = "CrossAttentionSpec"
    num_kv_heads: int
    head_size: int
    dtype: str


class SinkFullAttentionGroupSpec(_BaseGroupSpec):
    spec_type: Literal["SinkFullAttentionSpec"] = "SinkFullAttentionSpec"
    num_kv_heads: int
    head_size: int
    head_size_v: int
    dtype: str
    sliding_window: int | None = None
    attention_chunk_size: int | None = None
    sink_len: int | None = None


class UniformTypeGroupSpec(_BaseGroupSpec):
    """Group whose layers share one block table but have different hidden sizes.

    ``layer_specs`` contains one serialised group-spec dict per layer,
    using the same schema as the top-level ``groups`` entries.
    """

    spec_type: Literal["UniformTypeKVCacheSpecs"] = "UniformTypeKVCacheSpecs"
    layer_specs: list[dict[str, Any]]


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
# Top-level response models
# ---------------------------------------------------------------------------


class KVCacheInfo(BaseModel):
    num_gpu_blocks: int | None
    num_cpu_blocks: int | None
    gpu_memory_utilization: float
    enable_prefix_caching: bool
    kv_offloading_enabled: bool
    kv_offloading_backend: str | None
    kv_offloading_size_gib: float | None
    groups: list[KVCacheGroupInfo]


class ParallelismInfo(BaseModel):
    tensor_parallel_size: int
    pipeline_parallel_size: int
    data_parallel_size: int
    data_parallel_rank: int
    data_parallel_master_ip: str
    data_parallel_master_port: int
    data_parallel_rpc_port: int


class HMAInfo(BaseModel):
    enabled: bool


class InferenceConfigResponse(BaseModel):
    kv_cache: KVCacheInfo
    parallelism: ParallelismInfo
    hma: HMAInfo
