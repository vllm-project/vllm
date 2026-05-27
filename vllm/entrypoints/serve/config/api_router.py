# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from fastapi import APIRouter, FastAPI, Request

from vllm.config import VllmConfig
from vllm.entrypoints.serve.config.protocol import (
    ChunkedLocalAttentionGroupSpec,
    ComputeCapability,
    CrossAttentionGroupSpec,
    DeviceInfo,
    FullAttentionGroupSpec,
    KVCacheGroupInfo,
    KVCacheRuntimeInfo,
    MambaGroupSpec,
    MLAAttentionGroupSpec,
    RuntimeDataResponse,
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


def _build_response(
    vllm_config: VllmConfig,
    kv_cache_groups: list[dict] | None,
    devices: list[dict] | None,
) -> RuntimeDataResponse:
    cache_config = vllm_config.cache_config
    
    num_gpu_blocks = cache_config.num_cpu_blocks
    num_cpu_blocks = cache_config.num_cpu_blocks

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

    return RuntimeDataResponse(
        kv_cache=KVCacheRuntimeInfo(
            num_gpu_blocks=num_gpu_blocks,
            num_cpu_blocks=num_cpu_blocks,
            groups=groups,
        ),
        devices=device_list,
    )


@router.get("/inference/v1/capabilities")
async def get_capabilities(raw_request: Request) -> RuntimeDataResponse:
    vllm_config: VllmConfig = raw_request.app.state.vllm_config
    kv_cache_groups: list[dict] | None = raw_request.app.state.kv_cache_config
    devices: list[dict] | None = raw_request.app.state.devices
    return _build_response(vllm_config, kv_cache_groups, devices)


def attach_router(app: FastAPI) -> None:
    app.include_router(router)
