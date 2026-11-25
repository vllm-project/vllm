# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Translate vLLM caches into the optional KVServe runtime API."""

from importlib import import_module
from typing import Any

import torch

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.utils import (
    get_current_attn_backends,
    get_current_attn_backends_and_specs,
)
from vllm.distributed.kv_transfer.kv_connector.v1.compression import ChunkIO
from vllm.distributed.parallel_state import get_tensor_model_parallel_rank
from vllm.model_executor.models.utils import extract_layer_index
from vllm.v1.core.single_type_kv_cache_manager import FullAttentionManager
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    KVCacheConfig,
    KVCacheGroup,
    KVCacheLayerConfig,
    MLAAttentionSpec,
)
from vllm.v1.worker.utils import select_common_block_size


def load_backend() -> Any:
    try:
        backend = import_module("kvserve_v1.compression.runtime")
    except ImportError as exc:
        raise ImportError(
            "Mooncake compression requires KVServe's runtime extra "
            "and the nvCOMP extra matching your CUDA version."
        ) from exc
    if getattr(backend, "API_VERSION", None) != 2:
        raise RuntimeError("Mooncake requires KVServe compression API version 2")
    return backend


def get_compression_config(extra: dict[str, Any]) -> Any:
    aggregate = extra.get("raw_aggregate", extra.get("raw_agg"))
    compression = extra.get("compression") or {}
    if (aggregate is None or aggregate is False) and not compression.get(
        "enable-compression-pipeline", False
    ):
        return None
    if extra.get("kv_lease_duration") is not None:
        raise ValueError(
            "Mooncake compression does not support kv_lease_duration; leave it unset."
        )
    if aggregate is not None and aggregate is not False:
        raise ValueError("Mooncake raw aggregation is not available in this version")
    return load_backend().CompressionConfig.from_extra_config(extra)


def build_memory_plan(
    vllm_config: VllmConfig, specs: dict[str, KVCacheLayerConfig]
) -> Any:
    transfer_config = vllm_config.kv_transfer_config
    if transfer_config is None:
        return None
    config = get_compression_config(transfer_config.kv_connector_extra_config)
    if config is None:
        return None
    backend = load_backend()
    cache_specs = {}
    for name, layer in specs.items():
        spec = layer.spec
        if layer.manager_cls is not FullAttentionManager:
            raise ValueError("Mooncake compression requires full-attention layers")
        if not isinstance(spec, AttentionSpec):
            raise ValueError(f"Invalid full-attention cache spec for {name}")
        if isinstance(spec, MLAAttentionSpec):
            raise ValueError("Mooncake compression does not support MLA KV caches")
        if (
            spec.tokens_per_state != 1
            or spec.num_head_slots is not None
            or spec.state_content_bytes is not None
            or spec.num_heads != spec.num_kv_heads
            or spec.state_content_size_bytes
            != (spec.head_size + spec.head_size_v) * spec.dtype.itemsize
        ):
            raise ValueError(
                "Mooncake compression requires dense K/V heads and one state "
                f"per token; layer {name} has an unsupported cache geometry."
            )
        cache_specs[name] = backend.CacheSpec(
            layer_index=extract_layer_index(name),
            block_size=spec.block_size,
            num_kv_heads=spec.num_kv_heads,
            head_size=spec.head_size,
            head_size_v=spec.head_size_v,
            dtype=spec.dtype,
        )
    if not cache_specs:
        raise ValueError("Mooncake compression requires FullAttention KV caches")
    block_sizes = {spec.block_size for spec in cache_specs.values()}
    if len(block_sizes) != 1:
        raise ValueError("Mooncake compression requires a common logical block size")
    backends, attn_specs, layer_names = get_current_attn_backends_and_specs(
        vllm_config,
        KVCacheConfig(
            num_blocks=0,
            kv_cache_tensors=[],
            kv_cache_groups=[
                KVCacheGroup.from_layers({name: layer}) for name, layer in specs.items()
            ],
        ),
        get_current_attn_backends(vllm_config),
    )
    kernel_block_size = select_common_block_size(
        vllm_config.cache_config.block_size, backends, attn_specs, layer_names
    )
    logical_block_size = block_sizes.pop()
    if logical_block_size % kernel_block_size:
        raise ValueError("Logical attention blocks must contain whole kernel blocks.")
    resource_requirements = getattr(backend, "resource_requirements", None)
    if resource_requirements is None:
        raise RuntimeError("KVServe runtime does not expose resource requirements")
    return resource_requirements(
        config,
        cache_specs,
        blocks_per_logical=logical_block_size // kernel_block_size,
        tp_size=vllm_config.parallel_config.tensor_parallel_size,
        tp_rank=get_tensor_model_parallel_rank(),
        num_layers=getattr(vllm_config.model_config.hf_config, "num_hidden_layers", 0),
        device=torch.device("cuda", torch.accelerator.current_device_index()),
    )


def create_worker_runtime(plan: Any, cache_layout: str, block_factor: int, device):
    """Bind both Mooncake integrations to the planned KVServe runtime."""
    backend = load_backend()
    factory = getattr(backend, "create_runtime", None)
    if factory is not None:
        return factory(plan, cache_layout, block_factor, device)
    return backend.CompressionRuntime(
        plan=plan,
        kv_cache_layout=cache_layout,
        blocks_per_logical=block_factor,
        device=device,
    )


def create_chunk_io(runtime: Any) -> ChunkIO:
    backend = load_backend()
    factory = getattr(backend, "ChunkIO", None)
    if factory is None:
        raise RuntimeError("KVServe runtime does not expose the shared chunk API")
    return factory(runtime)
