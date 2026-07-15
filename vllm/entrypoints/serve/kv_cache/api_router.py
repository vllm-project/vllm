# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from fastapi import APIRouter, FastAPI, Request

from vllm.entrypoints.serve.kv_cache.protocol import (
    ChunkedLocalAttentionGroupSpec,
    CrossAttentionGroupSpec,
    FullAttentionGroupSpec,
    KVCacheGroupInfo,
    KVCacheRuntimeInfo,
    MambaGroupSpec,
    MLAAttentionGroupSpec,
    SinkFullAttentionGroupSpec,
    SlidingWindowGroupSpec,
    UniformTypeGroupSpec,
)

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


def _build_kv_cache_response(kv_cache_data: dict | None) -> KVCacheRuntimeInfo:
    if kv_cache_data is None:
        return KVCacheRuntimeInfo()

    groups: list[KVCacheGroupInfo] = [
        _build_group_spec(g) for g in kv_cache_data.get("groups", [])
    ]

    return KVCacheRuntimeInfo(
        kv_cache_size_tokens=kv_cache_data.get("kv_cache_size_tokens"),
        max_concurrency=kv_cache_data.get("max_concurrency"),
        num_gpu_blocks=kv_cache_data.get("num_gpu_blocks"),
        num_cpu_blocks=kv_cache_data.get("num_cpu_blocks"),
        groups=groups,
    )


@router.get("/v1/server/kv-cache")
async def get_kv_cache(raw_request: Request) -> KVCacheRuntimeInfo:
    kv_cache_data: dict | None = raw_request.app.state.kv_cache_config
    return _build_kv_cache_response(kv_cache_data)


def attach_router(app: FastAPI) -> None:
    app.include_router(router)
