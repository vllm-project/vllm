# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from fastapi import APIRouter, FastAPI, Request

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.v1.kv_cache_interface import (
    ChunkedLocalAttentionSpec,
    CrossAttentionSpec,
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
    MLAAttentionSpec,
    SinkFullAttentionSpec,
    SlidingWindowSpec,
    UniformTypeKVCacheSpecs,
)

from .protocol import (
    ChunkedLocalAttentionGroupSpec,
    CrossAttentionGroupSpec,
    FullAttentionGroupSpec,
    HMAInfo,
    InferenceConfigResponse,
    KVCacheGroupInfo,
    KVCacheInfo,
    MambaGroupSpec,
    MLAAttentionGroupSpec,
    ParallelismInfo,
    SinkFullAttentionGroupSpec,
    SlidingWindowGroupSpec,
    UniformTypeGroupSpec,
)

logger = init_logger(__name__)

router = APIRouter()


def _dtype_str(dtype) -> str:
    """Convert a torch.dtype to a plain string, e.g. 'bfloat16'."""
    return str(dtype).replace("torch.", "")


def _build_group_spec(group: KVCacheGroupSpec) -> KVCacheGroupInfo:
    """Map one internal KVCacheGroupSpec to its API protocol counterpart."""
    spec = group.kv_cache_spec
    base = dict(
        layer_names=group.layer_names,
        block_size=spec.block_size,
        page_size_bytes=spec.page_size_bytes,
    )

    # Most-derived types must be checked before their base types.
    if isinstance(spec, MLAAttentionSpec):
        return MLAAttentionGroupSpec(
            **base,
            num_kv_heads=spec.num_kv_heads,
            head_size=spec.head_size,
            head_size_v=spec.head_size_v,
            dtype=_dtype_str(spec.dtype),
            sliding_window=spec.sliding_window,
            attention_chunk_size=spec.attention_chunk_size,
            cache_dtype_str=spec.cache_dtype_str,
        )
    if isinstance(spec, SinkFullAttentionSpec):
        return SinkFullAttentionGroupSpec(
            **base,
            num_kv_heads=spec.num_kv_heads,
            head_size=spec.head_size,
            head_size_v=spec.head_size_v,
            dtype=_dtype_str(spec.dtype),
            sliding_window=spec.sliding_window,
            attention_chunk_size=spec.attention_chunk_size,
            sink_len=spec.sink_len,
        )
    if isinstance(spec, FullAttentionSpec):
        return FullAttentionGroupSpec(
            **base,
            num_kv_heads=spec.num_kv_heads,
            head_size=spec.head_size,
            head_size_v=spec.head_size_v,
            dtype=_dtype_str(spec.dtype),
            sliding_window=spec.sliding_window,
            attention_chunk_size=spec.attention_chunk_size,
        )
    if isinstance(spec, SlidingWindowSpec):
        return SlidingWindowGroupSpec(
            **base,
            num_kv_heads=spec.num_kv_heads,
            head_size=spec.head_size,
            dtype=_dtype_str(spec.dtype),
            sliding_window=spec.sliding_window,
        )
    if isinstance(spec, ChunkedLocalAttentionSpec):
        return ChunkedLocalAttentionGroupSpec(
            **base,
            num_kv_heads=spec.num_kv_heads,
            head_size=spec.head_size,
            dtype=_dtype_str(spec.dtype),
            attention_chunk_size=spec.attention_chunk_size,
        )
    if isinstance(spec, CrossAttentionSpec):
        return CrossAttentionGroupSpec(
            **base,
            num_kv_heads=spec.num_kv_heads,
            head_size=spec.head_size,
            dtype=_dtype_str(spec.dtype),
        )
    if isinstance(spec, MambaSpec):
        return MambaGroupSpec(
            **base,
            shapes=[list(s) for s in spec.shapes],
            dtypes=[_dtype_str(d) for d in spec.dtypes],
            mamba_type=spec.mamba_type,
            mamba_cache_mode=spec.mamba_cache_mode,
        )
    if isinstance(spec, UniformTypeKVCacheSpecs):
        layer_specs = [
            _build_group_spec(KVCacheGroupSpec([name], sub_spec)).model_dump()
            for name, sub_spec in spec.kv_cache_specs.items()
        ]
        return UniformTypeGroupSpec(**base, layer_specs=layer_specs)

    raise ValueError(f"Unhandled KVCacheSpec type: {type(spec)!r}")


def _build_response(
    vllm_config: VllmConfig,
    kv_cache_config: KVCacheConfig | None,
) -> InferenceConfigResponse:
    cache_cfg = vllm_config.cache_config
    parallel_cfg = vllm_config.parallel_config
    scheduler_cfg = vllm_config.scheduler_config

    groups: list[KVCacheGroupInfo] = (
        [_build_group_spec(g) for g in kv_cache_config.kv_cache_groups]
        if kv_cache_config is not None
        else []
    )

    return InferenceConfigResponse(
        kv_cache=KVCacheInfo(
            num_gpu_blocks=cache_cfg.num_gpu_blocks,
            num_cpu_blocks=cache_cfg.num_cpu_blocks,
            gpu_memory_utilization=cache_cfg.gpu_memory_utilization,
            enable_prefix_caching=cache_cfg.enable_prefix_caching,
            kv_offloading_enabled=cache_cfg.kv_offloading_size is not None,
            kv_offloading_backend=(
                cache_cfg.kv_offloading_backend
                if cache_cfg.kv_offloading_size is not None
                else None
            ),
            kv_offloading_size_gib=cache_cfg.kv_offloading_size,
            groups=groups,
        ),
        parallelism=ParallelismInfo(
            tensor_parallel_size=parallel_cfg.tensor_parallel_size,
            pipeline_parallel_size=parallel_cfg.pipeline_parallel_size,
            data_parallel_size=parallel_cfg.data_parallel_size,
            data_parallel_rank=parallel_cfg.data_parallel_rank,
            data_parallel_master_ip=parallel_cfg.data_parallel_master_ip,
            data_parallel_master_port=parallel_cfg.data_parallel_master_port,
            data_parallel_rpc_port=parallel_cfg.data_parallel_rpc_port,
        ),
        hma=HMAInfo(
            enabled=not bool(scheduler_cfg.disable_hybrid_kv_cache_manager),
        ),
    )


@router.get("/inference/v1/config")
async def get_inference_config(raw_request: Request) -> InferenceConfigResponse:
    vllm_config: VllmConfig = raw_request.app.state.vllm_config
    kv_cache_config: KVCacheConfig | None = raw_request.app.state.kv_cache_config
    return _build_response(vllm_config, kv_cache_config)


def attach_router(app: FastAPI) -> None:
    app.include_router(router)
