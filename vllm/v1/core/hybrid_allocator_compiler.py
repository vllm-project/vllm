from collections import defaultdict
import math
from typing import Dict, List, Tuple
from vllm.config import VllmConfig
from vllm.v1.core.kv_cache_interface import KVCacheConfig, KVCacheTensor, KVCacheTensorSeperate, KVCacheTensorShareBuffer, KVCacheSpec, KVCacheSpec, FullAttentionSpec, SlidingWindowSpec
from vllm.logger import init_logger

logger = init_logger(__name__)


def get_kv_cache_config(vllm_config: VllmConfig, kv_cache_spec: KVCacheSpec,
                        available_memory: int) -> Tuple[KVCacheConfig, int]:
    check_enough_memory(vllm_config, kv_cache_spec, available_memory)
    if is_same_type(kv_cache_spec):
        # TODO(Chen): improve the readability of default path (self attn only models)
        return _get_kv_cache_config_same_type(vllm_config, kv_cache_spec,
                                              available_memory)
    elif is_same_hidden_size(kv_cache_spec):
        return _get_kv_cache_config_same_size(vllm_config, kv_cache_spec,
                                              available_memory)
    else:
        raise NotImplementedError


def _get_kv_cache_config_same_type(
        vllm_config: VllmConfig, kv_cache_spec: KVCacheSpec,
        available_memory: int) -> Tuple[KVCacheConfig, int]:
    page_sizes = {layer.page_size_bytes for layer in kv_cache_spec.values()}
    assert len(page_sizes) == 1
    page_size = page_sizes.pop()

    num_gpu_blocks = int(available_memory // page_size // len(kv_cache_spec))
    num_gpu_blocks = max(num_gpu_blocks, 0)

    logger.info("num_gpu_blocks=%d", num_gpu_blocks)

    if vllm_config.cache_config.num_gpu_blocks_override is not None:
        num_gpu_blocks_override = \
            vllm_config.cache_config.num_gpu_blocks_override
        logger.info(
            "Overriding num_gpu_blocks=%d with "
            "num_gpu_blocks_override=%d", num_gpu_blocks,
            num_gpu_blocks_override)
        num_gpu_blocks = num_gpu_blocks_override

    per_layer_size = page_size * num_gpu_blocks

    kv_cache_config = KVCacheConfig(
        buffer_size=-1,
        tensors={
            layer_name: KVCacheTensorSeperate(size=per_layer_size)
            for layer_name in kv_cache_spec.keys()
        },
        groups=[[layer_name for layer_name in kv_cache_spec.keys()]],
        kv_cache_spec=kv_cache_spec)
    # TODO(Chen): KVCacheTensorSeperate can be removed
    print("kv_cache_config", kv_cache_config)
    return kv_cache_config, num_gpu_blocks


def _get_kv_cache_config_same_size(
        vllm_config: VllmConfig, kv_cache_spec: KVCacheSpec,
        available_memory: int) -> Tuple[KVCacheConfig, int]:
    # Grouped allocation
    # TODO(Chen): explain it, need test

    page_sizes = {layer.page_size_bytes for layer in kv_cache_spec.values()}
    assert len(page_sizes) == 1
    page_size = page_sizes.pop()

    grouped_layers: Dict[str, List[str]] = defaultdict(
        list)  # key -> List[layer_name]
    for layer_name, layer_spec in kv_cache_spec.items():
        grouped_layers[layer_spec.key].append(layer_name)

    group_size_gcd = math.gcd(
        *[len(layers) for layers in grouped_layers.values()])

    allocator_page_size = page_size * group_size_gcd
    num_pages = available_memory // allocator_page_size
    buffer_size = num_pages * allocator_page_size

    if vllm_config.cache_config.num_gpu_blocks_override is not None:
        num_gpu_blocks_override = \
            vllm_config.cache_config.num_gpu_blocks_override
        logger.info(
            "Overriding num_gpu_blocks=%d with "
            "num_gpu_blocks_override=%d", num_pages, num_gpu_blocks_override)
        # TODO(Chen): num_page and num_block has different meaning
        num_pages = num_gpu_blocks_override

    logger.info("num_gpu_blocks=%d", num_pages)

    groups = []
    tensors: Dict[int, KVCacheTensor] = {}

    for layers in grouped_layers.values():
        for i in range(0, len(layers), group_size_gcd):
            new_group = []
            for idx_in_group, layer_name in enumerate(layers[i:i +
                                                             group_size_gcd]):
                new_group.append(layer_name)
                tensors[layer_name] = KVCacheTensorShareBuffer(
                    size=buffer_size - idx_in_group * page_size,
                    start_bias=idx_in_group * page_size,
                )
            groups.append(new_group)

    kv_cache_config = KVCacheConfig(
        buffer_size=num_pages * allocator_page_size,
        tensors=tensors,
        groups=groups,
        kv_cache_spec=kv_cache_spec,
    )

    print("kv_cache_config", kv_cache_config)
    return kv_cache_config, num_pages


def is_same_type(kv_cache_spec: KVCacheSpec) -> bool:
    layer_keys = set(layer.key for layer in kv_cache_spec.values())
    return len(layer_keys) == 1


def is_same_hidden_size(kv_cache_spec: KVCacheSpec):
    # TODO(Chen): needs more accurate check
    return all(
        isinstance(layer, (FullAttentionSpec, SlidingWindowSpec))
        for layer in kv_cache_spec.values())


def check_enough_memory(vllm_config: VllmConfig, kv_cache_spec: KVCacheSpec,
                        available_memory: int):
    if available_memory <= 0:
        raise ValueError("No available memory for the cache blocks. "
                         "Try increasing `gpu_memory_utilization` when "
                         "initializing the engine.")

    max_model_len = vllm_config.model_config.max_model_len
    needed_memory = 0
    for layer_spec in kv_cache_spec.values():
        needed_memory += layer_spec.bytes_for_tokens(max_model_len)

    if needed_memory > available_memory:
        # TODO(Chen): need unit test
        raise ValueError(
            f"To serve at least one request with the models's max seq len "
            f"({max_model_len}), ({needed_memory/1024/1024/1024} GB KV cache is"
            f"needed, which is larger than the available KV Cache memory "
            f"({available_memory/1024/1024/1024} GB). Try increasing "
            f"`gpu_memory_utilization` or decreasing `max_model_len` when "
            f"initializing the engine.")
