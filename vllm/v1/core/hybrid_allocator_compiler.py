from typing import Tuple
from vllm.config import VllmConfig
from vllm.v1.core.kv_cache_interface import KVCacheConfig, KVCacheTensorSeperate, LayerConfig, SelfAttentionCache, SlidingWindowCache
from vllm.logger import init_logger

logger = init_logger(__name__)


def get_kv_cache_config(vllm_config: VllmConfig, layer_config: LayerConfig,
                        available_memory: int) -> Tuple[KVCacheConfig, int]:
    check_enough_memory(vllm_config, layer_config, available_memory)
    if is_same_type(layer_config):
        return _get_kv_cache_config_same_type(vllm_config, layer_config,
                                              available_memory)
    elif is_same_hidden_size(layer_config):
        raise NotImplementedError
    else:
        raise NotImplementedError


def _get_kv_cache_config_same_type(
        vllm_config: VllmConfig, layer_config: LayerConfig,
        available_memory: int) -> Tuple[KVCacheConfig, int]:
    page_sizes = {lc.page_size for lc in layer_config.layers.values()}
    assert len(page_sizes) == 1
    page_size = page_sizes.pop()

    num_gpu_blocks = int(available_memory // page_size //
                         len(layer_config.layers))
    num_gpu_blocks = max(num_gpu_blocks, 0)

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
            layer_cnt: KVCacheTensorSeperate(size=per_layer_size)
            for layer_cnt in layer_config.layer_id_mapping.keys()
        },
        block_table_sharing={
            "default":
            [layer_id for layer_id in layer_config.layer_id_mapping.values()]
        },
        layer_config=layer_config)
    # TODO (Chen): KVCacheTensorSeperate can be removed
    print("kv_cache_config", kv_cache_config)
    return kv_cache_config, num_gpu_blocks


def is_same_type(layer_config: LayerConfig) -> bool:
    layer_keys = set(lc.key for lc in layer_config.layers.values())
    return len(layer_keys) == 1


def is_same_hidden_size(layer_config: LayerConfig):
    return all(
        isinstance(lc, (SelfAttentionCache, SlidingWindowCache))
        for lc in layer_config.layers.values())


def check_enough_memory(vllm_config: VllmConfig, layer_config: LayerConfig,
                        available_memory: int):
    if available_memory <= 0:
        raise ValueError("No available memory for the cache blocks. "
                         "Try increasing `gpu_memory_utilization` when "
                         "initializing the engine.")

    max_model_len = vllm_config.model_config.max_model_len
    needed_memory = 0
    for lc in layer_config.layers.values():
        needed_memory += lc.memory_size(max_model_len)

    if needed_memory > available_memory:
        # TODO (Chen): need unit test
        raise ValueError(
            f"The model's KV cache needed ({needed_memory/1024/1024/1024} "
            f"GB) to store one request with the models's max seq len "
            f"({max_model_len}) is larger than the available KV Cache memory "
            f"({available_memory/1024/1024/1024} GB). Try increasing "
            f"`gpu_memory_utilization` or decreasing `max_model_len` when "
            f"initializing the engine.")
