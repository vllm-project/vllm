'''
 Copyright (c) ByteDance Inc.
 Authors: 
  - Tongping Liu (tongping.liu@bytedance.com)
  - https://github.com/vllm-project/vllm/pull/6102/commits
'''
"""CacheEngine class for managing the KV cache."""
from typing import List, Dict, Tuple

import torch

from vllm.attention import get_attn_backend
from vllm.config import CacheConfig, ModelConfig, ParallelConfig, SchedulerConfig, DeviceConfig
from vllm.logger import init_logger
from vllm.utils import STR_DTYPE_TO_TORCH_DTYPE, is_pin_memory_available, get_dtype_size

from vllm import _dattn_ops as dattn

logger = init_logger(__name__)


class CacheEngineDAttn:
    """Manages the KV cache.

    This class is responsible for initializing and managing the GPU and CPU KV
    caches. It also provides methods for performing KV cache operations, such
    as swapping and copying.
    """

    def __init__(
        self,
        cache_config: CacheConfig,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        scheduler_config: SchedulerConfig,
        device_config: DeviceConfig,
    ) -> None:
        self.cache_config = cache_config
        self.model_config = model_config
        self.parallel_config = parallel_config
        self.scheduler_config = scheduler_config
        self.device_config = device_config
        
        if self.device_config.device_type != "cuda":
            raise RuntimeError("DATTN only support cuda device.")

        self.head_size = model_config.get_head_size()
        self.num_layers = model_config.get_num_layers(parallel_config)
        self.num_kv_heads = model_config.get_num_kv_heads(parallel_config)

        self.block_size = cache_config.block_size
        self.block_bytes_size = self.cache_config.block_bytes_size
        self.num_gpu_blocks = cache_config.num_gpu_blocks
        #self.num_cpu_blocks = cache_config.num_cpu_blocks

        print(f"self.block_bytes_size-{self.block_bytes_size}")
        if cache_config.cache_dtype == "auto":
            self.dtype = model_config.dtype
        else:
            self.dtype = STR_DTYPE_TO_TORCH_DTYPE[cache_config.cache_dtype]

        self.dtype_size = get_dtype_size(self.dtype)
        self.max_batch_size = self.scheduler_config.max_num_seqs
        self.max_seq_len = self.scheduler_config.max_model_len
        
        # If max_seq_len is not divisible by block_size,
        # round up to the nearest value that is.
        if self.max_seq_len % self.block_size != 0:
            logger.warning("Note: self.max_seq_len mod self.block_size != 0")
            exit(0)

        self.token_size = self.num_kv_heads * self.head_size
        self.sequence_buffer_size = self.max_seq_len * self.token_size
        self.sequence_buffer_bytes_size = self.sequence_buffer_size * self.dtype_size
        self.cache_space_size = self.max_batch_size * self.sequence_buffer_size
        self.cache_space_bytes_size = self.cache_space_size * self.dtype_size
    
        assert (self.cache_space_bytes_size) % self.block_bytes_size == 0, "cache_space_bytes_size must be divisible by block_bytes_size"
        
        self.cache_space_page_num = self.cache_space_bytes_size // self.block_bytes_size

        logger.info("CacheEngineDAttn basic info: { block_size: %d, dtype_size: %d, head_size: %d, "
                    "num_kv_heads: %d, max_seq_len: %d, max_batch_size: %d, num_layers: %d,"
                    "token_size: %d, sequence_buffer_size: %d, cache_space_size: %d, "
                    "cache_space_bytes_size: %d, cache_space_page_num: %d }",
                    self.block_size, self.dtype_size, self.head_size,
                    self.num_kv_heads, self.max_seq_len, self.max_batch_size, self.num_layers, 
                    self.token_size, self.sequence_buffer_size, self.cache_space_size,
                    self.cache_space_bytes_size, self.cache_space_page_num)

        self.device_cache_allocator = dattn.kvCacheAllocator(self.max_seq_len, self.num_layers, self.num_kv_heads,
                                                             self.head_size, self.block_size, self.dtype_size)

        # record the number of allocated blocks in a cache space for each request 
        self.allocated_block_counts = [0 for _ in range(self.max_batch_size)] 
        
        # Get attention backend.
        self.attn_backend = get_attn_backend(
            model_config.get_num_attention_heads(parallel_config),
            self.head_size,
            self.num_kv_heads,
            model_config.get_sliding_window(),
            model_config.dtype,
            cache_config.cache_dtype,
            self.block_size,
        )

        self.kv_cache_ptrs = self._reserve_gpu_kv_cache()
        self.gpu_cache = self._create_fake_kv_cache()

    """
    In dAttention's design, we are required to pass the layer index so
    that CUDA kernel could use it to get the kv_cache. For other mechanisms, like
    PagedAttention or vAttention, they are passing different kv_vache for different layers.
    """
    def _create_fake_kv_cache(self) -> List[torch.Tensor]: 
        fake_kv_caches = []

        for i in range(self.num_layers):
            fake_kv_caches.append(torch.tensor(i))

        return fake_kv_caches
    
    def _reserve_gpu_kv_cache(self) -> List[int]:
        kv_cache_ptrs = []

        for i in range(self.max_batch_size):
            kv_cache_ptrs.append(self.device_cache_allocator.reserve_cache_region(i))
            #print(f"i:{i}, virtual address:{hex(kv_cache[i])}")

        return kv_cache_ptrs

    def swap_in(self, src_to_dst: torch.Tensor) -> None:
        raise NotImplementedError("swap_in is not implemented for DATTN now.")

    def swap_out(self, src_to_dst: torch.Tensor) -> None:
        raise NotImplementedError("swap_out is not implemented for DATTN now.")

    # TODO: we need to implement the copy_blocks 
    def copy(self, src_to_dsts: torch.Tensor) -> None:
        self.attn_backend.copy_blocks(self.gpu_cache, src_to_dsts)

    @staticmethod
    def get_cache_block_size(
        cache_config: CacheConfig,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
    ) -> int:
        head_size = model_config.get_head_size()
        num_heads = model_config.get_num_kv_heads(parallel_config)
        num_attention_layers = model_config.get_num_attention_layers(
            parallel_config)

        key_cache_block = cache_config.block_size * num_heads * head_size
        value_cache_block = key_cache_block
        total = num_attention_layers * (key_cache_block + value_cache_block)
        #print(f"CacheEngineDAttn:head_size:{head_size}, num_heads:{num_heads}, num_attention_layers:{num_attention_layers}, block_size: {cache_config.block_size}, key_cache_block:{key_cache_block},total:{total/1024}KB")
        if cache_config.cache_dtype == "auto":
            dtype = model_config.dtype
        else:
            dtype = STR_DTYPE_TO_TORCH_DTYPE[cache_config.cache_dtype]
        dtype_size = get_dtype_size(dtype)
        #print(f"CacheEngineDAttn:cache_config.block_bytes_size:{dtype_size * total}")
        return dtype_size * total

    """
        This function tried to allocate the physical pages for all requests 
        Initially, vmm patch will allocate physical pages for each request. 
        That is, each request will invoke from python to C++ library function directly. 

        However, it is not a wise approach, as that can increase the overhead by 100X based on our experiments. 
        Instead, we should invoke c++ library function just once by passing an array with [req_id, new_blocks]

        Note that self.allocated_block_counts[buffer_id] will track the number of allocated blocks
        in this function. Let's utilize the same mechanism at the first step. 
        TODO: we may change this later. To my understanding, it is better to track the number of blocks at sequence 
    """
    def alloc_seqs(self, allocated_block_counts: Dict[int, int]):
        to_alloc_blocks = []
        """Allocate cache handles for the given number of blocks."""
        for buffer_id, num_blocks in allocated_block_counts.items():
            allocated_blocks = self.allocated_block_counts[buffer_id]
            num_blocks -= allocated_blocks
            #print(f"CacheEngineDAttn: buffer_id-{buffer_id}, num_blocks:{num_blocks}")
            if num_blocks > 0:
                to_alloc_blocks.append([buffer_id, num_blocks])
                self.allocated_block_counts[buffer_id] += num_blocks

        # Allocate physical blocks for all requests. 
        self.device_cache_allocator.alloc_cache_blocks(to_alloc_blocks) 


    def free_seqs(self, free_buffer_ids: List[int]):
        """Free cache handles for the given buffer ids."""
        for req in free_buffer_ids:
            print(f"BOOWWWW free_seqs with req:{req} with length:{len(free_buffer_ids)}")
        self.device_cache_allocator.release_cache_regions(free_buffer_ids)

def _get_dtype_size(dtype: torch.dtype) -> int:
    return torch.tensor([], dtype=dtype).element_size()
