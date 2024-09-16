import logging
import time
import numpy as np
from typing import Dict, List, NamedTuple, Optional, Set, Tuple

import torch
import torch.distributed

import vllm.envs as envs
from vllm.config import ModelConfig, ParallelConfig
from vllm.distributed.parallel_state import (get_tensor_model_parallel_group,
                                             get_tensor_model_parallel_rank,
                                             get_tensor_model_parallel_world_size)
from vllm.sequence import (SequenceData, SequenceGroupMetadata)
from vllm.utils import init_logger

try:
    from vineyard.llm import KVCache as VineyardKVCache
    from vineyard.llm import KVTensor as VineyardKVTensor
    from vineyard.llm import FileCacheConfig, VineyardCacheConfig
except ImportError:
    raise
    VineyardKVCache = None
    VineyardKVTensor = None

logger = init_logger(__name__)

class CacheServiceMetrics:
    hit_tokens: int = 0
    total_tokens: int = 0
    hit_blocks: int = 0
    total_blocks: int = 0
    counter: int = 0
     
    
class VineyardLLMCache:
    def __init__(
        self,
        head_size: int,
        num_kv_heads: int,
        cache_capacity: int = 1024,
        layer: int = 2,
        kv_cache_dtype: str = None,
        torch_dtype: torch.dtype = torch.bfloat16,
        metrics: CacheServiceMetrics = None
    ):
        self._init_vineyard_logger()

        self.head_size = head_size
        self.num_kv_heads = num_kv_heads
        self.cache_capacity = cache_capacity
        self.layer = layer
        self.kv_cache_dtype = kv_cache_dtype
        self.torch_dtype = torch_dtype
        self.tensor_nbytes = head_size * num_kv_heads * 2  # float16/bfloat16
        self.cache = VineyardKVCache(
            tensor_nbytes=self.tensor_nbytes,
            cache_capacity=self.cache_capacity,
            layer=self.layer,
            rank=get_tensor_model_parallel_rank(),
            world_size=get_tensor_model_parallel_world_size(),
        )
        self.chunk_size = self.cache.chunk_size
        self.token_capacity = 2**15
        self.buffer = torch.empty(
            (2, self.layer, self.token_capacity, self.num_kv_heads, self.head_size),
            dtype=torch_dtype, device='cpu',
        ).pin_memory()
        self.tensors = []
        for i in range(self.token_capacity):
            self.tensors.append([])
            for j in range(self.layer):
                k_tensor = self.buffer[0, j, i]
                v_tensor = self.buffer[1, j, i]
                self.tensors[-1].append((
                    VineyardKVTensor(k_tensor.data_ptr(), k_tensor.numel() * k_tensor.element_size()),
                    VineyardKVTensor(v_tensor.data_ptr(), v_tensor.numel() * v_tensor.element_size()),
                ))
        self.metrics = metrics
        self.time_query = []
        self.time_load = []
        self.time_unload = []
        self.time_update = []
        logger.info(f"VineyardLLMCache init {metrics}")

    def _init_vineyard_logger(self):
        import vineyard
        logging.basicConfig()

        vineyard.logger.setLevel(logger.getEffectiveLevel())
        vineyard.logger.handlers.clear()
        for handler in logger.handlers:
            vineyard.logger.addHandler(handler)

    @staticmethod
    def from_envs(
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        kv_cache_dtype: str,
        torch_dtype: torch.dtype = torch.bfloat16,
        metrics: CacheServiceMetrics = None,
    ) -> Optional["VineyardLLMCache"]:
        if VineyardKVCache is None:
            logger.warn("VineyardKVCache module is not available")
            return None

        if not envs.VLLM_USE_FLASH_ATTN_DECODING:
            logger.warn("VineyardLLMCache requires flash attention decoding")
            return None

        head_size = model_config.get_head_size()
        num_kv_heads = model_config.get_num_kv_heads(parallel_config)
        num_layers = model_config.get_num_layers(parallel_config)
        logger.info(f"VineyardLLMCache from_envs {metrics}")
        return VineyardLLMCache(
            head_size=head_size,
            num_kv_heads=num_kv_heads,
            cache_capacity=2**20,
            layer=num_layers,
            kv_cache_dtype=kv_cache_dtype,
            torch_dtype=torch_dtype,
            metrics = metrics
        )

    def prefetch_seq_kv_caches(
        self,
        seq_group_metadata: SequenceGroupMetadata,
        kv_caches: List[torch.Tensor],
        block_size: int,
    ) -> Tuple[str, int]:
        from vllm._custom_ops import reshape_and_cache_flash

        if seq_group_metadata is not None:
            seq_ids = list(seq_group_metadata.seq_data.keys())
            assert len(seq_ids) == 1
            seq_id = seq_ids[0]
            seq_data = seq_group_metadata.seq_data[seq_id]

            context_len = seq_data.get_num_computed_tokens()
            token_chunk_size = seq_group_metadata.token_chunk_size
            tokens = seq_data.get_prompt_token_ids()

            # leave at least one token unmatched
            token_chunk_size -= 1

            # alignment `context_len` to `self.chunk_size`
            query_context_len = context_len - context_len % self.chunk_size
            query_token_size = context_len + token_chunk_size - query_context_len

            query_prefix = tokens[:query_context_len]
            query_tokens = tokens[query_context_len:query_context_len + query_token_size]
            query_args = [
                seq_id,
                context_len,
                token_chunk_size,
                query_context_len,
                query_token_size,
                query_prefix,
                query_tokens,
            ]
            # torch.distributed.broadcast_object_list(query_args, src=0,
            #                                         group=get_tensor_model_parallel_group())
        else:
            query_args = [None, None, None, None, None, None, None]
            # torch.distributed.broadcast_object_list(query_args, src=0,
            #                                         group=get_tensor_model_parallel_group())
            (seq_id,
             context_len,
             token_chunk_size,
             query_context_len,
             query_token_size,
             query_prefix,
             query_tokens
            ) = query_args

        # start_time = time.time()
        matched = self.cache.query(
            prefix=query_prefix,
            tokens=query_tokens,
            kv_cache_list=self.tensors[:query_token_size],
        )
        # self.time_query.append(time.time() - start_time)
        # print(f"time query avg {np.mean(self.time_query)} std {np.std(self.time_query)}")
        # synchronized across tensor parallel ranks
        matched_tensor = torch.tensor([matched], dtype=torch.long, device='cuda')
        # torch.distributed.all_reduce(matched_tensor, op=torch.distributed.ReduceOp.MIN,
        #                              group=get_tensor_model_parallel_group())
        matched = matched_tensor[0].item()

        # shift
        offset = context_len % self.chunk_size
        matched -= offset

        matched = min(matched, token_chunk_size - 1)
        if matched <= 0:
            return seq_id, 0
        if seq_group_metadata is not None:
            block_table = seq_group_metadata.block_tables[seq_id]
            # print(f"prefetch_seq_kv_caches block_table {seq_group_metadata.block_tables[seq_id]} context_len {context_len} matched {matched} ")
            slot_mapping = []
            for i in range(context_len, context_len + matched):
                block_number = block_table[i // block_size]
                block_offset = i % block_size
                slot = block_number * block_size + block_offset
                slot_mapping.append(slot)
            slot_mapping = torch.tensor(slot_mapping, dtype=torch.long, device='cuda')
            # torch.distributed.broadcast(slot_mapping, src=0,
            #                             group=get_tensor_model_parallel_group())
        else:
            slot_mapping = torch.zeros((matched,), dtype=torch.long, device='cuda')
            # torch.distributed.broadcast(slot_mapping, src=0,
            #                             group=get_tensor_model_parallel_group())

        self.metrics.hit_tokens += matched
        self.metrics.total_tokens += len(tokens)
        self.metrics.hit_blocks += (matched // block_size)
        self.metrics.total_blocks += (- len(tokens) // (-block_size))
        self.metrics.counter += 1
        # logger.info(f"prefetch_seq_kv_caches metrics {self.metrics.hit_tokens} {self.metrics.total_tokens} {self.metrics.hit_blocks} {self.metrics.total_blocks} matched {matched} token_len {len(tokens)}")
        # save to GPU kv cache
        # start_time = time.time()
        buffer = self.buffer[:, :, offset:offset+matched].cuda()
        # self.time_load.append(time.time() - start_time)
        # print(f"time load avg {np.mean(self.time_load)} std {np.std(self.time_load)}")
        for j in range(self.layer):
            # use `reshape_and_cache_flash` rather than `copy_` as
            # the target kv cache slots is not contingous.
            reshape_and_cache_flash(
                buffer[0][j],
                buffer[1][j],
                kv_caches[j][0],
                kv_caches[j][1],
                slot_mapping,
                self.kv_cache_dtype,
                1.0,
                1.0
            )

        # update the seq_group_metadata's and seq's metadata
        if seq_group_metadata is not None:
            seq_data.update_num_computed_tokens(matched)
            seq_group_metadata.token_chunk_size -= matched

        return seq_id, matched

    def prefetch_kv_caches(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
        kv_caches: List[torch.Tensor],
        block_size: int,
    ) -> Dict[str, int]:
        ''' Returns a dict to indicate the matched kv cache lengths
            for each sequence group metadata.
        '''
        if block_size is None or kv_caches[0] is None:  # profile run
            return {}

        if seq_group_metadata_list is not None:
            prefill_requests = []
            for seq_group_meta in seq_group_metadata_list:
                if seq_group_meta.is_prompt:
                    prefill_requests.append(seq_group_meta)
            num_prefill_requests = [len(prefill_requests)]
            # torch.distributed.broadcast_object_list(num_prefill_requests, src=0,
            #                                         group=get_tensor_model_parallel_group())
        else:
            num_prefill_requests = [None]
            # torch.distributed.broadcast_object_list(num_prefill_requests, src=0,
            #                                         group=get_tensor_model_parallel_group())
            prefill_requests = [None] * num_prefill_requests[0]
        num_prefill_requests = num_prefill_requests[0]

        matched = {}
        for seq_group_meta in prefill_requests:
            seq_id, seq_matched = self.prefetch_seq_kv_caches(
                seq_group_meta, kv_caches, block_size,
            )
            matched[seq_id] = seq_matched
        if matched:
            logger.debug(f"prefetch_kv_caches: matched=%r", matched)
        return matched

    def update_seq_kv_caches(
        self,
        matched: Dict[str, int],
        seq_group_metadata: SequenceGroupMetadata,
        kv_caches: List[torch.Tensor],
        block_size: int,
    ) -> Tuple[str, int]:
        if seq_group_metadata is not None:
            seq_ids = list(seq_group_metadata.seq_data.keys())
            assert len(seq_ids) == 1
            seq_id = seq_ids[0]
            seq_data = seq_group_metadata.seq_data[seq_id]

            context_len = seq_data.get_num_computed_tokens()
            token_chunk_size = seq_group_metadata.token_chunk_size
            tokens = seq_data.get_prompt_token_ids()

            # alignment `context_len` to `self.chunk_size`
            update_context_len = context_len - context_len % self.chunk_size
            update_token_size = context_len + token_chunk_size - update_context_len
            update_token_size -= update_token_size % self.chunk_size
            update_prefix = tokens[:update_context_len]
            update_tokens = tokens[update_context_len:update_context_len+update_token_size]

            update_args = [
                seq_id,
                update_context_len,
                update_token_size,
                update_prefix,
                update_tokens,
            ]
            # torch.distributed.broadcast_object_list(update_args, src=0,
            #                                         group=get_tensor_model_parallel_group())
        else:
            update_args = [None, None, None, None, None]
            # torch.distributed.broadcast_object_list(update_args, src=0,
            #                                         group=get_tensor_model_parallel_group())
            (seq_id,
             update_context_len,
             update_token_size,
             update_prefix,
             update_tokens,
            ) = update_args

        if update_token_size <= 0:
            # restore the seq_group_metadata's and seq's metadata
            if seq_group_metadata is not None:
                seq_data.update_num_computed_tokens(-matched[seq_id])
                seq_group_metadata.token_chunk_size += matched[seq_id]
            return seq_id, 0

        if seq_group_metadata is not None:
            block_table = seq_group_metadata.block_tables[seq_id]
            slot_mapping = []
            for i in range(update_context_len, update_context_len + update_token_size):
                block_number = block_table[i // block_size]
                block_offset = i % block_size
                slot = block_number * block_size + block_offset
                slot_mapping.append(slot)
            slot_mapping = torch.tensor(slot_mapping, dtype=torch.long, device='cuda')
            # torch.distributed.broadcast(slot_mapping, src=0,
            #                             group=get_tensor_model_parallel_group())
        else:
            slot_mapping = torch.zeros((update_token_size,), dtype=torch.long, device='cuda')
            # torch.distributed.broadcast(slot_mapping, src=0,
            #                             group=get_tensor_model_parallel_group())

        # fetch from GPU kv cache
        # start_time = time.time()
        for j in range(self.layer):
            self.buffer[:, j, :update_token_size].copy_(
                kv_caches[j][:, slot_mapping // block_size, slot_mapping % block_size])
        # self.time_unload.append(time.time() - start_time)   
        # print(f"time unload avg {np.mean(self.time_unload)} std {np.std(self.time_unload)}")
        
        # start_time = time.time()
        # updates into vineyard
        updated = self.cache.update(
            prefix=update_prefix,
            tokens=update_tokens,
            kv_cache_list=self.tensors[:update_token_size],
        )
        # self.time_update.append(time.time() - start_time)   
        # print(f"time update avg {np.mean(self.time_update)} std {np.std(self.time_update)}")

        # restore the seq_group_metadata's and seq's metadata
        if seq_group_metadata is not None:
            seq_data.update_num_computed_tokens(-matched[seq_id])
            seq_group_metadata.token_chunk_size += matched[seq_id]

        return seq_id, updated

    def update_kv_caches(
        self,
        matched: Dict[int, int],
        seq_group_metadata_list: List[SequenceGroupMetadata],
        kv_caches: List[torch.Tensor],
        block_size: int,
    ) -> Dict[str, int]:
        if block_size is None or kv_caches[0] is None:  # profile run
            return {}

        if seq_group_metadata_list is not None:
            prefill_requests = []
            for seq_group_meta in seq_group_metadata_list:
                if seq_group_meta.is_prompt:
                    prefill_requests.append(seq_group_meta)
            num_prefill_requests = [len(prefill_requests)]
            # torch.distributed.broadcast_object_list(num_prefill_requests, src=0,
            #                                         group=get_tensor_model_parallel_group())
        else:
            num_prefill_requests = [None]
            # torch.distributed.broadcast_object_list(num_prefill_requests, src=0,
            #                                         group=get_tensor_model_parallel_group())
            prefill_requests = [None] * num_prefill_requests[0]
        num_prefill_requests = num_prefill_requests[0]

        updated = {}
        for seq_group_meta in prefill_requests:
            seq_id, seq_updated = self.update_seq_kv_caches(
                matched, seq_group_meta, kv_caches, block_size,
            )
            updated[seq_id] = seq_updated
        if updated:
            logger.debug(f"update_kv_caches: updated=%r", updated)
        return updated

    def __repr__(self):
        return (
            f'VineyardLLMCache('
            f'tensor_nbytes={self.tensor_nbytes}, '
            f'cache_capacity={self.cache_capacity}, '
            f'layer={self.layer}, '
            f'kv_cache_dtype={self.kv_cache_dtype}, '
            f'torch_dtype={self.torch_dtype}, '
            f'cache={self.cache})'
        )
