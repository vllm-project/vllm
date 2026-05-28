# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Top-level response models
# ---------------------------------------------------------------------------


class ModelInfo(BaseModel):
    served_model_names: list[str]
    dtype: str
    quantization: str | None
    max_model_len: int
    max_logprobs: int


class KVCacheInfo(BaseModel):
    num_gpu_blocks: int | None
    num_cpu_blocks: int | None
    gpu_memory_utilization: float
    cache_dtype: str
    enable_prefix_caching: bool
    prefix_caching_hash_algo: str
    kv_offloading_enabled: bool
    kv_offloading_backend: str | None
    kv_offloading_size_gib: float | None


class SchedulerInfo(BaseModel):
    max_num_batched_tokens: int
    max_num_seqs: int
    max_num_partial_prefills: int
    enable_chunked_prefill: bool
    policy: str


class ParallelismInfo(BaseModel):
    tensor_parallel_size: int
    pipeline_parallel_size: int
    data_parallel_size: int
    data_parallel_size_local: int
    data_parallel_rank: int
    data_parallel_rank_local: int | None
    data_parallel_master_ip: str
    data_parallel_master_port: int
    data_parallel_rpc_port: int
    expert_parallel_enabled: bool
    prefill_context_parallel_size: int


class SpeculativeDecodingInfo(BaseModel):
    enabled: bool
    method: str | None
    num_speculative_tokens: int | None
    draft_model: str | None


class LoRAInfo(BaseModel):
    enabled: bool
    max_lora_rank: int | None
    max_loras: int | None
    max_cpu_loras: int | None
    lora_dtype: str | None


class HMAInfo(BaseModel):
    enabled: bool


class ConfigDataResponse(BaseModel):
    model: ModelInfo
    kv_cache: KVCacheInfo
    scheduler: SchedulerInfo
    parallelism: ParallelismInfo
    speculative_decoding: SpeculativeDecodingInfo
    lora: LoRAInfo
    hma: HMAInfo
