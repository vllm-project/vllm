# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from fastapi import APIRouter, FastAPI, Request

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.utils.torch_utils import kv_cache_dtype_str_to_dtype

from .protocol import (
    ChunkedLocalAttentionGroupSpec,
    ComputeCapability,
    CrossAttentionGroupSpec,
    DeviceInfo,
    FullAttentionGroupSpec,
    HMAInfo,
    InferenceConfigResponse,
    KVCacheGroupInfo,
    KVCacheInfo,
    LoRAInfo,
    MambaGroupSpec,
    MLAAttentionGroupSpec,
    ModelInfo,
    ParallelismInfo,
    SchedulerInfo,
    SinkFullAttentionGroupSpec,
    SlidingWindowGroupSpec,
    SpeculativeDecodingInfo,
    UniformTypeGroupSpec,
)

logger = init_logger(__name__)

router = APIRouter()

_SPEC_TYPE_TO_MODEL: dict[str, type] = {
    "FullAttentionSpec": FullAttentionGroupSpec,
    "MLAAttentionSpec": MLAAttentionGroupSpec,
    "SlidingWindowSpec": SlidingWindowGroupSpec,
    "ChunkedLocalAttentionSpec": ChunkedLocalAttentionGroupSpec,
    "MambaSpec": MambaGroupSpec,
    "CrossAttentionSpec": CrossAttentionGroupSpec,
    "SinkFullAttentionSpec": SinkFullAttentionGroupSpec,
    "UniformTypeKVCacheSpecs": UniformTypeGroupSpec,
}


def _build_group_spec(group: dict) -> KVCacheGroupInfo:
    """Map one serialized group dict to its API protocol counterpart."""
    spec_type = group["spec_type"]
    model_cls = _SPEC_TYPE_TO_MODEL.get(spec_type)
    if model_cls is None:
        raise ValueError(f"Unhandled KVCacheSpec type: {spec_type!r}")
    return model_cls(**group)


def _dtype_str(dtype) -> str:
    return str(dtype).replace("torch.", "")


def _build_response(
    vllm_config: VllmConfig,
    kv_cache_groups: list[dict] | None,
    devices: list[dict] | None,
) -> InferenceConfigResponse:
    model_cfg = vllm_config.model_config
    cache_cfg = vllm_config.cache_config
    parallel_cfg = vllm_config.parallel_config
    scheduler_cfg = vllm_config.scheduler_config
    spec_cfg = vllm_config.speculative_config
    lora_cfg = vllm_config.lora_config

    # Served model names: coerce to list, fall back to [model]
    raw_name = model_cfg.served_model_name
    if isinstance(raw_name, list):
        served_model_names = raw_name
    elif isinstance(raw_name, str):
        served_model_names = [raw_name]
    else:
        served_model_names = [model_cfg.model]

    # Resolved cache dtype (never "auto")
    resolved_cache_dtype = kv_cache_dtype_str_to_dtype(cache_cfg.cache_dtype, model_cfg)
    cache_dtype_str = _dtype_str(resolved_cache_dtype)

    groups: list[KVCacheGroupInfo] = (
        [_build_group_spec(g) for g in kv_cache_groups]
        if kv_cache_groups is not None
        else []
    )

    # Device info
    device_list: list[DeviceInfo] = []
    if devices is not None:
        for d in devices:
            cap = d.get("compute_capability")
            device_list.append(
                DeviceInfo(
                    rank=d["rank"],
                    name=d["name"],
                    total_memory_bytes=d["total_memory_bytes"],
                    compute_capability=(
                        ComputeCapability(major=cap["major"], minor=cap["minor"])
                        if cap is not None
                        else None
                    ),
                    num_compute_units=d["num_compute_units"],
                )
            )

    # Speculative decoding
    if spec_cfg is not None:
        spec_info = SpeculativeDecodingInfo(
            enabled=True,
            method=spec_cfg.method,
            num_speculative_tokens=spec_cfg.num_speculative_tokens,
            draft_model=spec_cfg.model,
        )
    else:
        spec_info = SpeculativeDecodingInfo(
            enabled=False,
            method=None,
            num_speculative_tokens=None,
            draft_model=None,
        )

    # LoRA
    if lora_cfg is not None:
        lora_dtype = lora_cfg.lora_dtype
        if isinstance(lora_dtype, torch.dtype):
            lora_dtype_str: str | None = _dtype_str(lora_dtype)
        elif isinstance(lora_dtype, str) and lora_dtype not in ("auto", None):
            lora_dtype_str = lora_dtype
        else:
            lora_dtype_str = None
        lora_info = LoRAInfo(
            enabled=True,
            max_lora_rank=lora_cfg.max_lora_rank,
            max_loras=lora_cfg.max_loras,
            max_cpu_loras=lora_cfg.max_cpu_loras,
            lora_dtype=lora_dtype_str,
        )
    else:
        lora_info = LoRAInfo(
            enabled=False,
            max_lora_rank=None,
            max_loras=None,
            max_cpu_loras=None,
            lora_dtype=None,
        )

    return InferenceConfigResponse(
        model=ModelInfo(
            served_model_names=served_model_names,
            dtype=_dtype_str(model_cfg.dtype),
            quantization=(
                str(model_cfg.quantization)
                if model_cfg.quantization is not None
                else None
            ),
            max_model_len=model_cfg.max_model_len,
            max_logprobs=model_cfg.max_logprobs,
        ),
        kv_cache=KVCacheInfo(
            num_gpu_blocks=cache_cfg.num_gpu_blocks,
            num_cpu_blocks=cache_cfg.num_cpu_blocks,
            gpu_memory_utilization=cache_cfg.gpu_memory_utilization,
            cache_dtype=cache_dtype_str,
            enable_prefix_caching=cache_cfg.enable_prefix_caching,
            prefix_caching_hash_algo=str(cache_cfg.prefix_caching_hash_algo),
            kv_offloading_enabled=cache_cfg.kv_offloading_size is not None,
            kv_offloading_backend=(
                cache_cfg.kv_offloading_backend
                if cache_cfg.kv_offloading_size is not None
                else None
            ),
            kv_offloading_size_gib=cache_cfg.kv_offloading_size,
            groups=groups,
        ),
        scheduler=SchedulerInfo(
            max_num_batched_tokens=scheduler_cfg.max_num_batched_tokens,
            max_num_seqs=scheduler_cfg.max_num_seqs,
            max_num_partial_prefills=scheduler_cfg.max_num_partial_prefills,
            enable_chunked_prefill=scheduler_cfg.enable_chunked_prefill,
            policy=str(scheduler_cfg.policy),
        ),
        parallelism=ParallelismInfo(
            tensor_parallel_size=parallel_cfg.tensor_parallel_size,
            pipeline_parallel_size=parallel_cfg.pipeline_parallel_size,
            data_parallel_size=parallel_cfg.data_parallel_size,
            data_parallel_size_local=parallel_cfg.data_parallel_size_local,
            data_parallel_rank=parallel_cfg.data_parallel_rank,
            data_parallel_rank_local=parallel_cfg.data_parallel_rank_local,
            data_parallel_master_ip=parallel_cfg.data_parallel_master_ip,
            data_parallel_master_port=parallel_cfg.data_parallel_master_port,
            data_parallel_rpc_port=parallel_cfg.data_parallel_rpc_port,
            expert_parallel_enabled=parallel_cfg.enable_expert_parallel,
            prefill_context_parallel_size=parallel_cfg.prefill_context_parallel_size,
        ),
        devices=device_list,
        speculative_decoding=spec_info,
        lora=lora_info,
        hma=HMAInfo(
            enabled=not bool(scheduler_cfg.disable_hybrid_kv_cache_manager),
        ),
    )


@router.get("/inference/v1/config")
async def get_inference_config(raw_request: Request) -> InferenceConfigResponse:
    vllm_config: VllmConfig = raw_request.app.state.vllm_config
    kv_cache_groups: list[dict] | None = raw_request.app.state.kv_cache_config
    devices: list[dict] | None = raw_request.app.state.devices
    return _build_response(vllm_config, kv_cache_groups, devices)


def attach_router(app: FastAPI) -> None:
    app.include_router(router)
