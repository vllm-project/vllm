# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from pydantic import BaseModel


class ModelInfo(BaseModel):
    name: str
    served_names: list[str]
    dtype: str
    quantization: str | None
    max_model_len: int


class KVCacheInfo(BaseModel):
    gpu_memory_utilization: float
    dtype: str
    enable_prefix_caching: bool


class SchedulerInfo(BaseModel):
    max_num_seqs: int
    max_num_batched_tokens: int
    enable_chunked_prefill: bool
    policy: str


class ParallelismInfo(BaseModel):
    tensor_parallel_size: int
    pipeline_parallel_size: int
    data_parallel_size: int
    data_parallel_rank: int


class FeaturesInfo(BaseModel):
    speculative_decoding: bool
    lora: bool
    hma: bool


class ServerConfigResponse(BaseModel):
    model: ModelInfo
    scheduler: SchedulerInfo
    kv_cache: KVCacheInfo
    parallelism: ParallelismInfo
    features: FeaturesInfo
