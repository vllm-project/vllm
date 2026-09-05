# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from fastapi import APIRouter, FastAPI, Request

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.utils.torch_utils import kv_cache_dtype_str_to_dtype

from .protocol import (
    FeaturesInfo,
    KVCacheInfo,
    ModelInfo,
    ParallelismInfo,
    SchedulerInfo,
    ServerConfigResponse,
)

logger = init_logger(__name__)

router = APIRouter()


def _dtype_str(dtype) -> str:
    return str(dtype).replace("torch.", "")


def _build_response(
    vllm_config: VllmConfig,
    served_names: list[str],
) -> ServerConfigResponse:
    model_cfg = vllm_config.model_config
    cache_cfg = vllm_config.cache_config
    parallel_cfg = vllm_config.parallel_config
    scheduler_cfg = vllm_config.scheduler_config
    spec_cfg = vllm_config.speculative_config
    lora_cfg = vllm_config.lora_config

    resolved_cache_dtype = kv_cache_dtype_str_to_dtype(cache_cfg.cache_dtype, model_cfg)

    return ServerConfigResponse(
        model=ModelInfo(
            name=served_names[0],
            served_names=served_names,
            dtype=_dtype_str(model_cfg.dtype),
            quantization=(
                str(model_cfg.quantization)
                if model_cfg.quantization is not None
                else None
            ),
            max_model_len=model_cfg.max_model_len,
        ),
        kv_cache=KVCacheInfo(
            gpu_memory_utilization=cache_cfg.gpu_memory_utilization,
            dtype=_dtype_str(resolved_cache_dtype),
            enable_prefix_caching=cache_cfg.enable_prefix_caching,
        ),
        scheduler=SchedulerInfo(
            max_num_seqs=scheduler_cfg.max_num_seqs,
            max_num_batched_tokens=scheduler_cfg.max_num_batched_tokens,
            enable_chunked_prefill=scheduler_cfg.enable_chunked_prefill,
            policy=str(scheduler_cfg.policy),
        ),
        parallelism=ParallelismInfo(
            tensor_parallel_size=parallel_cfg.tensor_parallel_size,
            pipeline_parallel_size=parallel_cfg.pipeline_parallel_size,
            data_parallel_size=parallel_cfg.data_parallel_size,
            data_parallel_rank=parallel_cfg.data_parallel_rank,
        ),
        features=FeaturesInfo(
            speculative_decoding=spec_cfg is not None,
            lora=lora_cfg is not None,
            hma=not bool(scheduler_cfg.disable_hybrid_kv_cache_manager),
        ),
    )


@router.get("/v1/server/config")
async def get_server_config(raw_request: Request) -> ServerConfigResponse:
    vllm_config: VllmConfig = raw_request.app.state.vllm_config
    args = raw_request.app.state.args
    if args.served_model_name:
        served_names = list(args.served_model_name)
    else:
        served_names = [vllm_config.model_config.served_model_name]
    return _build_response(vllm_config, served_names)


def attach_router(app: FastAPI) -> None:
    app.include_router(router)
