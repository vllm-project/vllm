# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Translate vLLM caches into the optional KVServe runtime API."""

from importlib import import_module
from typing import Any

import torch

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.utils import (
    get_attn_backends_and_specs_for_layers,
    get_current_attn_backends,
)
from vllm.distributed.parallel_state import get_tensor_model_parallel_rank
from vllm.model_executor.models.utils import extract_layer_index
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheSpec, MLAAttentionSpec
from vllm.v1.worker.utils import select_common_block_size


def load_backend() -> Any:
    try:
        backend = import_module("kvserve_v1.compression.runtime")
    except ImportError as exc:
        raise ImportError(
            "Mooncake compression requires KVServe's runtime extra "
            "and the nvCOMP extra matching your CUDA version."
        ) from exc
    if getattr(backend, "API_VERSION", None) != 1:
        raise RuntimeError("Unsupported KVServe compression API version")
    return backend


def get_compression_config(extra: dict[str, Any]) -> Any:
    aggregate = extra.get("raw_aggregate", extra.get("raw_agg"))
    compression = extra.get("compression") or {}
    if (aggregate is None or aggregate is False) and not compression.get(
        "enable-compression-pipeline", False
    ):
        return None
    return load_backend().CompressionConfig.from_extra_config(extra)


def build_memory_plan(vllm_config: VllmConfig, specs: dict[str, KVCacheSpec]) -> Any:
    transfer_config = vllm_config.kv_transfer_config
    if transfer_config is None:
        return None
    config = get_compression_config(transfer_config.kv_connector_extra_config)
    if config is None:
        return None
    backend = load_backend()
    cache_specs = {}
    for name, spec in specs.items():
        if not isinstance(spec, FullAttentionSpec):
            continue
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
    backends, attn_specs = get_attn_backends_and_specs_for_layers(
        vllm_config, specs, get_current_attn_backends(vllm_config)
    )
    kernel_block_size = select_common_block_size(
        vllm_config.cache_config.block_size, backends, attn_specs
    )
    logical_block_size = block_sizes.pop()
    if logical_block_size % kernel_block_size:
        raise ValueError("Logical attention blocks must contain whole kernel blocks.")
    return backend.build_memory_plan(
        config,
        cache_specs,
        blocks_per_logical=logical_block_size // kernel_block_size,
        tp_size=vllm_config.parallel_config.tensor_parallel_size,
        tp_rank=get_tensor_model_parallel_rank(),
        num_layers=getattr(vllm_config.model_config.hf_config, "num_hidden_layers", 0),
        device=torch.device("cuda", torch.accelerator.current_device_index()),
    )
