# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Translate vLLM KV cache metadata for native offloading backends."""

from typing import TYPE_CHECKING

from vllm.v1.core.kv_cache_utils import resolve_kv_cache_block_sizes
from vllm.v1.kv_cache_interface import FullAttentionSpec, MLAAttentionSpec
from vllm.v1.kv_offload.config import OffloadingConfig, OffloadingGroupConfig

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheTensor


def is_kv_cache_tensor_packed(kv_cache_tensor: "KVCacheTensor") -> bool:
    """Return whether a KV cache tensor uses a packed block stride."""
    return bool(kv_cache_tensor.block_stride)


def build_offloading_config(
    vllm_config: "VllmConfig",
    kv_cache_config: "KVCacheConfig",
) -> OffloadingConfig:
    """Translate vLLM configuration into the native offloading boundary."""
    kv_transfer_config = vllm_config.kv_transfer_config
    assert kv_transfer_config is not None
    extra_config = kv_transfer_config.kv_connector_extra_config

    parallel_config = vllm_config.parallel_config
    context_parallel_factor = (
        parallel_config.decode_context_parallel_size
        * parallel_config.prefill_context_parallel_size
    )
    groups = tuple(
        OffloadingGroupConfig(
            block_size=group.kv_cache_spec.block_size,
            gpu_block_size=(group.kv_cache_spec.block_size * context_parallel_factor),
            layer_names=tuple(group.layer_names),
            is_non_mla_full_attention=(
                isinstance(group.kv_cache_spec, FullAttentionSpec)
                and not isinstance(group.kv_cache_spec, MLAAttentionSpec)
            ),
        )
        for group in kv_cache_config.kv_cache_groups
    )

    _, hash_block_size = resolve_kv_cache_block_sizes(kv_cache_config, vllm_config)
    for group in groups:
        assert group.gpu_block_size % hash_block_size == 0, (
            f"gpu_block_size={group.gpu_block_size} not divisible by "
            f"hash_block_size={hash_block_size}. "
            f"Hybrid models (e.g. Mamba+Attention) need "
            f"--enable-prefix-caching to align block sizes."
        )

    block_size_factor = 1
    offloaded_block_size = extra_config.get("block_size")
    if offloaded_block_size is not None:
        offloaded_block_size_int = int(offloaded_block_size)
        unique_gpu_block_sizes = {group.gpu_block_size for group in groups}
        assert len(unique_gpu_block_sizes) == 1, (
            "If 'block_size' is specified in kv_connector_extra_config, "
            "there must be at least one KV cache group, "
            "and all groups must have the same block size."
        )
        gpu_block_size = unique_gpu_block_sizes.pop()
        assert offloaded_block_size_int % gpu_block_size == 0
        block_size_factor = offloaded_block_size_int // gpu_block_size

    worker_kv_bytes_per_gpu_block = 0
    if kv_cache_config.num_blocks > 0:
        packed_tensors = tuple(
            is_kv_cache_tensor_packed(tensor)
            for tensor in kv_cache_config.kv_cache_tensors
        )
        is_packed = any(packed_tensors)
        assert not is_packed or all(packed_tensors)
        total_gpu_kv_bytes = (
            kv_cache_config.kv_cache_tensors[0].size
            if is_packed
            else sum(tensor.size for tensor in kv_cache_config.kv_cache_tensors)
        )
        worker_kv_bytes_per_gpu_block = total_gpu_kv_bytes // kv_cache_config.num_blocks

    kv_events_config = vllm_config.kv_events_config
    return OffloadingConfig(
        groups=groups,
        hash_block_size=hash_block_size,
        block_size_factor=block_size_factor,
        num_gpu_blocks=kv_cache_config.num_blocks,
        worker_kv_bytes_per_gpu_block=worker_kv_bytes_per_gpu_block,
        world_size=parallel_config.world_size,
        enable_kv_cache_events=(
            kv_events_config is not None and kv_events_config.enable_kv_cache_events
        ),
        extra_config=extra_config,
        model_name=vllm_config.model_config.model,
        kv_cache_dtype=str(vllm_config.cache_config.cache_dtype).replace("torch.", ""),
        namespace_block_size=vllm_config.cache_config.block_size,
        tp_size=parallel_config.tensor_parallel_size,
        pp_size=parallel_config.pipeline_parallel_size,
        pcp_size=parallel_config.prefill_context_parallel_size,
        dcp_size=parallel_config.decode_context_parallel_size,
        rank=parallel_config.rank,
        use_v2_model_runner=vllm_config.use_v2_model_runner,
        engine_id=kv_transfer_config.engine_id,
    )
