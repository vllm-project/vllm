import torch
import sys
from collections import deque
from vllm.config import CacheConfig, ModelConfig, ParallelConfig
from vllm.utils import (STR_DTYPE_TO_TORCH_DTYPE, get_dtype_size,
                        is_pin_memory_available)
from dataclasses import dataclass
from vllm import _custom_ops as ops
import time
from vllm.logger import init_logger
from typing import Optional, Union

logger = init_logger(__name__)

default_mem_size = 4 * 1024 * 1024 * 1024
batch_layers_transmission_to_GPU = False

@dataclass
class BlockMappingFromCPU:
    block_mapping: torch.Tensor  # 2-D tenso
    block_offset: torch.Tensor # 1-D tensor, like offset array in CSR format
                                 # the offset of each request in block_mapping
    request_ids: torch.Tensor    # request IDs
    def __init__(self,
                 block_mapping: list[list[int, int]],
                 block_offset: list[int],
                 request_ids: list[int]):
        self.block_mapping = torch.tensor(block_mapping,
                                          device="cpu",
                                          dtype=torch.int64).view(-1, 2)
        self.block_offset = torch.tensor(block_offset,
                                         device="cpu",
                                         dtype=torch.int64).view(-1)
        self.request_ids = torch.tensor(request_ids,
                                        device="cpu",
                                        dtype=torch.int64).view(-1)
    def __str__(self):
        return "block_mapping: " + str(self.block_mapping) + \
                " block_offset: " + str(self.block_offset) + \
                " request_ids: " + str(self.request_ids)

@dataclass
class KVStoreMeta:
    incomplete_put_block_ids: torch.Tensor # 4-D tensor:
                                    # vllm_block_id,
                                    # start_offset,end_offset,
                                    # store_block_id
    put_block_ids_mapping: torch.Tensor # 2-D tensor:
                                       # vllm_block_id, store_block_id
    request_ids: torch.Tensor # 1-D tensor

    @staticmethod
    def null():
        return KVStoreMeta(torch.Tensor(),
                           torch.Tensor(),
                           torch.Tensor())

    def __str__(self):
        return "incomplete_put_block_ids: " + str(self.incomplete_put_block_ids) + \
                " put_block_ids_mapping: " + str(self.put_block_ids_mapping) + \
                " request_ids: " + str(self.request_ids)

class BlockCount:
    def __init__(self, block_id, access_count, last_access, block_hash,
                 send_flag = False):
        # XXX: can remove it
        self.block_id = block_id
        self.access_count = access_count
        self.last_access = last_access
        self.block_hash = block_hash
        self.send_flag = send_flag
    def __str__(self):
        return "block_id: " + str(self.block_id) + \
                " access_count: " + str(self.access_count) + \
                " last_access: " + str(self.last_access) + \
                " block_hash: " + str(self.block_hash)

class KVBlockStoreManager:
    def __init__(self,
                 block_head_mem_size: int, # size of each block for key/value
                 num_layer: int,
                 num_block_slot: int, # number of slots for each block
                 mem_size : int = default_mem_size, # total memory size
                 ):

        t = 2 * num_layer * block_head_mem_size
        mem_size = (mem_size // t) * t
        self.num_block_slot = num_block_slot
        self.num_blocks = (mem_size // t)
        self.time_cnt = 0
        self.block_cnt = 0
        self.block_table = [BlockCount(0, 0, 0, 0)] * self.num_blocks
        self.hash_block_map: dict[int, int] = {} # hash -> store_block_id
        self.gpu_and_store_block_map: dict[int, int] = \
                {} # gpu_block_id -> store_block_id
        logger.info("KVBlockStore use %f GB memory per worker, "
                    "%d blocks, block size = %d",
                    mem_size / 1024 / 1024 / 1024,
                    self.num_blocks,
                    self.num_block_slot)
        self.is_prefill = True

    @classmethod
    def from_configs(cls,
                     cache_config: CacheConfig,
                     model_config: ModelConfig,
                     parallel_config: ParallelConfig):
        dtype = None
        if (cache_config.cache_dtype == "auto"):
            dtype = model_config.dtype
        else:
            dtype = STR_DTYPE_TO_TORCH_DTYPE[cache_config.cache_dtype]
        block_size = cache_config.block_size
        num_key_value_heads = model_config.get_num_kv_heads(parallel_config)
        head_dim = model_config.get_head_size()
        num_layers = model_config.get_num_layers(parallel_config)

        block_head_mem_size = (dtype.itemsize * block_size
                          * num_key_value_heads * head_dim)
        return cls(block_head_mem_size,
                   num_layers,
                   block_size,
                   cache_config.kv_store_space_bytes)


    # allocate a logical block in CPU, and map the GPU block to kv store block
    def allocate_block(self, block_hash, gpu_block_id) -> BlockCount:
        if (self.is_prefill == False):
            return None
        ret_block_id = self.block_cnt
        if (self.block_cnt == self.num_blocks):
            # XXX: use policy to evict
            #      least used and earliest block
            min_access_count = sys.maxsize
            for block_count in self.block_table:
                if block_count.access_count < min_access_count:
                    min_access_count = block_count.access_count
            min_access_blocks = []
            for block_count in self.block_table:
                if block_count.access_count == min_access_count:
                    min_access_blocks.append(block_count.block_id)
            min_last_access = sys.maxsize
            final_block_id = -1
            for store_block_id in min_access_blocks:
                block_count = self.block_table[store_block_id]
                if block_count.last_access < min_last_access:
                    min_last_access = block_count.last_access
                    final_block_id = store_block_id
            assert(final_block_id != -1)
            ret_block_id = final_block_id
            final_block_count = self.block_table[ret_block_id]
            # print("evict block_id: ", final_block_id)
            # print("block count data -> ", final_block_count)
            del self.hash_block_map[final_block_count.block_hash]
        else:
            self.block_cnt += 1
        self.hash_block_map[block_hash] = ret_block_id
        self.block_table[ret_block_id] = BlockCount(ret_block_id, 1,
                                                self.time_cnt, block_hash)
        self.gpu_and_store_block_map[gpu_block_id] = ret_block_id
        self.time_cnt += 1
        return self.block_table[ret_block_id]

    def has(self, block_hash: int) -> bool:
        if (self.is_prefill == False):
            return False
        return (block_hash != -1) and \
               (self.hash_block_map.get(block_hash) != None)

    def remap_block_id(self,
                       block_hash: int,
                       vllm_block_id: int):
        if (self.is_prefill == False):
            return
        assert(self.hash_block_map.get(block_hash) != None)
        store_block_id = self.hash_block_map[block_hash]
        self.gpu_and_store_block_map[vllm_block_id] = store_block_id

    def open_send_flag(self, block_id: int):
        if (self.is_prefill == False):
            return
        store_block_id = self.gpu_and_store_block_map[block_id]
        self.block_table[store_block_id].send_flag = True
        self.block_table[store_block_id].access_count += 1
        self.block_table[store_block_id].last_access = self.time_cnt
        self.time_cnt += 1

    def close_send_flags(self,
                         vllm_block_ids):
        if (len(vllm_block_ids) == 0):
            return
        # print("vllm_block_ids: ", vllm_block_ids)
        for block_id in vllm_block_ids:
            store_block_id = self.gpu_and_store_block_map[block_id]
            self.block_table[store_block_id].send_flag = False

    def get_put_blocks_mapping(self,
                               incomplete_ids: torch.Tensor,
                               block_ids: torch.Tensor) \
                                       -> (torch.Tensor, torch.Tensor):
        if (self.is_prefill == False) or \
                ((incomplete_ids.numel() == 0) and (block_ids.numel() == 0)):
            return torch.Tensor(), torch.Tensor()
        assert(incomplete_ids.is_cuda == False)
        assert(block_ids.is_cuda == False)
        # Note: the self.num_block_slot is equal to the vllm block size
        incomplete_ids_numpy = incomplete_ids.numpy()
        block_ids_numpy = block_ids.numpy()
        incomplete_store_ids = torch.empty([incomplete_ids_numpy.shape[0]],
                                           dtype=incomplete_ids.dtype)
        store_block_ids = torch.empty([block_ids_numpy.shape[0]],
                                      dtype=block_ids.dtype)
        incomplete_store_ids_cpu = incomplete_store_ids.numpy()
        store_block_ids_cpu = store_block_ids.numpy()
        for i, incomplete_id in enumerate(incomplete_ids_numpy):
            store_block_id = self.gpu_and_store_block_map[incomplete_id[0]]
            incomplete_store_ids_cpu[i] = store_block_id
        for i, block_id in enumerate(block_ids_numpy):
            store_block_id = self.gpu_and_store_block_map[block_id]
            store_block_ids_cpu[i] = store_block_id

        # XXX: need to pre-allocate the another dimension in attn_meta builder?
        return (torch.cat((incomplete_ids,
                           incomplete_store_ids.view(
                               incomplete_store_ids.shape[0], 1)),
                          dim=1),
                torch.stack((block_ids, store_block_ids), dim=1))

    def get_block_mapping_from_torch(self, vllm_block_ids: torch.Tensor) \
                                                            -> torch.Tensor:
        if (self.is_prefill == False) or \
                (vllm_block_ids.numel() == 0):
            return torch.Tensor()
        ret_block_ids = torch.empty(vllm_block_ids.view(-1).shape,
                              dtype=vllm_block_ids.dtype)
        ret_vllm_block_ids = torch.empty(vllm_block_ids.view(-1).shape,
                              dtype=vllm_block_ids.dtype)
        ret_block_ids_cpu = ret_block_ids.view(-1).numpy()
        ret_vllm_block_ids_cpu = ret_vllm_block_ids.view(-1).numpy()
        cnt = 0
        for i, vllm_block_id in \
                enumerate(vllm_block_ids.view(-1).cpu().numpy()):
            assert(self.gpu_and_store_block_map.get(vllm_block_id) != None)
            store_block_id = self.gpu_and_store_block_map[vllm_block_id]
            if (self.block_table[store_block_id].send_flag):
                ret_block_ids_cpu[cnt] = store_block_id
                ret_vllm_block_ids_cpu[cnt] = vllm_block_id
                cnt += 1
        ret_block_ids.resize_([cnt])
        ret_vllm_block_ids.resize_([cnt])
        ret_tensor = torch.stack((ret_block_ids, ret_vllm_block_ids), dim=1)
        return ret_tensor

    def get_block_mapping_from_python(self, vllm_block_ids: list[int]) \
            -> list[tuple[int, int]]:
        if (self.is_prefill == False) or \
                (len(vllm_block_ids) == 0):
            return []
        ret = []
        for vllm_block_id in vllm_block_ids:
            assert(self.gpu_and_store_block_map.get(vllm_block_id) != None)
            store_block_id = self.gpu_and_store_block_map[vllm_block_id]
            if (self.block_table[store_block_id].send_flag):
                ret.append([store_block_id, vllm_block_id])
        return ret

    def update_hash(self, old_hash: int, new_hash: int):
        if (self.is_prefill == False):
            return
        assert(self.hash_block_map.get(old_hash) != None)
        store_block_id = self.hash_block_map[old_hash]
        del self.hash_block_map[old_hash]
        self.hash_block_map[new_hash] = store_block_id
        self.block_table[store_block_id].block_hash = new_hash

    # used to add a block_hash mapping when turn mutable
    #                           to immutable in BlockManager v2
    def add_hash_map(self, block_hash: int, vllm_block_id: int):
        if (self.is_prefill == False):
            return
        assert(self.gpu_and_store_block_map.get(vllm_block_id) != None)
        store_block_id = self.gpu_and_store_block_map[vllm_block_id]
        self.hash_block_map[block_hash] = store_block_id
        self.block_table[store_block_id].block_hash = block_hash

class EventPool:
    def __init__(self,
                 reserve_num_requests: int,
                 num_layers: int,
                 device: torch.device):
        self.reserve_num_requests = reserve_num_requests
        self.num_layers = num_layers
        self.event_queue: deque[torch.cuda.Event] = deque()
        self.device = device
        with torch.cuda.device(device):
            for i in range(reserve_num_requests):
                event = torch.cuda.Event()
                # create the detail new event
                event.record()
                event.synchronize()
                self.event_queue.append(event)

    def get_event(self) -> torch.cuda.Event:
        if (len(self.event_queue) == 0):
            with torch.cuda.device(self.device):
                event = torch.cuda.Event()
                # create the detail new event
                event.record()
                event.synchronize()
                self.event_queue.append(event)
        return self.event_queue.popleft()

    def put_event(self, event: torch.cuda.Event):
        self.event_queue.append(event)

    def get_events(self, num_events: int) -> list[torch.cuda.Event]:
        ret = []
        for i in range(num_events):
            ret.append(self.get_event())
        return ret

    def put_events(self, events: list[torch.cuda.Event]):
        for event in events:
            self.event_queue.append(event)

class KVBlockStore:
    def __init__(self,
                 block_head_mem_size: int, # size of each block for key/value
                 num_layer: int,
                 num_block_slot: int, # number of slots for each block
                 data_type : torch.dtype,
                 device: torch.device,
                 mem_size : int = default_mem_size, # total memory size
                 ):

        t = 2 * num_layer * block_head_mem_size
        mem_size = (mem_size // t) * t
        assert(mem_size % (2 * num_layer * block_head_mem_size) == 0)
        assert(block_head_mem_size % data_type.itemsize == 0)
        num_item = (block_head_mem_size // data_type.itemsize // num_block_slot)
        self.block_head_mem_size = block_head_mem_size
        self.num_block_slot = num_block_slot
        self.num_blocks = (mem_size // t)
        self.num_item = num_item
        self.device = device
        self.num_layer = num_layer
        self.event_map: dict[int,
                             Optional[torch.cuda.Event,
                                      list[torch.cuda.Event]]] = {}
        self.batch_layers_to_GPU = batch_layers_transmission_to_GPU
        with torch.cuda.device(device):
            self.store = torch.empty([self.num_blocks,
                                      2,
                                      num_layer,
                                      num_block_slot,
                                      num_item],
                                     dtype=data_type,
                                     device="cpu").pin_memory()
            self.get_stream = torch.cuda.Stream()
            self.put_stream = torch.cuda.Stream()
            self.put_event = torch.cuda.Event()
            self.event_pool = EventPool(100, num_layer, device)

    @classmethod
    def from_configs(cls,
                     cache_config: CacheConfig,
                     model_config: ModelConfig,
                     parallel_config: ParallelConfig,
                     device: torch.device):
        dtype = None
        if (cache_config.cache_dtype == "auto"):
            dtype = model_config.dtype
        else:
            dtype = STR_DTYPE_TO_TORCH_DTYPE[cache_config.cache_dtype]
        block_size = cache_config.block_size
        num_key_value_heads = model_config.get_num_kv_heads(parallel_config)
        head_dim = model_config.get_head_size()
        num_layers = model_config.get_num_layers(parallel_config)

        block_head_mem_size = (dtype.itemsize * block_size
                          * num_key_value_heads * head_dim)
        return cls(block_head_mem_size,
                   num_layers,
                   block_size,
                   dtype,
                   device,
                   cache_config.kv_store_space_bytes)

    def put_block_layer(self,
                        incomplete_block_ids: torch.Tensor,
                        block_ids_mapping: torch.Tensor,
                        layer_id: int,
                        kv_cache: torch.Tensor,
                        forward_stream: torch.cuda.Stream):
        if (incomplete_block_ids.numel() == 0) and \
                (block_ids_mapping.numel() == 0):
            return
        assert(incomplete_block_ids.is_cuda == False)
        assert(block_ids_mapping.is_cuda == False)
        incomplete_block_ids_numpy = incomplete_block_ids.numpy()
        self.put_event.record(forward_stream)
        self.put_event.wait(self.put_stream)
        if (block_ids_mapping.numel() != 0):
            with torch.cuda.stream(self.put_stream):
                ops.kv_store_copy_blocks2CPU(
                        kv_cache, self.store, layer_id,
                        block_ids_mapping)
        if (incomplete_block_ids.numel() != 0):
            with torch.cuda.stream(self.put_stream):
                ops.kv_store_copy_incomplete_blocks(kv_cache, self.store,
                                           layer_id,
                                           incomplete_block_ids)

    def get_blocks(self,
                   block_mapping_from_cpu: BlockMappingFromCPU,
                   kv_caches: list[torch.Tensor]):
        block_mapping_tensor = block_mapping_from_cpu.block_mapping
        block_offset_tensor = block_mapping_from_cpu.block_offset
        request_ids_tensor = block_mapping_from_cpu.request_ids
        request_ids_numpy = block_mapping_from_cpu.request_ids.numpy()
        if (block_mapping_tensor.numel() == 0) or \
                (len(request_ids_numpy) == 0):
            return
        is_batch_layer = self.batch_layers_to_GPU
        event_list = []
        if (is_batch_layer):
            # if batch layer, we need to allocate one event for each request
            request_last_events = []
            for idx, req_id in enumerate(request_ids_numpy):
                event = self.event_pool.get_event()
                self.event_map[req_id] = event
                event_list.append(event)
        else:
            # if not batch layer, we need to allocate the events for each layer
            for req_id in request_ids_numpy:
                event_list_tmp = self.event_pool.get_events(self.num_layer)
                self.event_map[req_id] = event_list_tmp
                event_list.extend(event_list_tmp)
        with torch.cuda.stream(self.get_stream):
            ops.kv_store_copy_blocks2GPU(
                    self.store, kv_caches,
                    self.num_layer,
                    block_mapping_tensor,
                    block_offset_tensor,
                    request_ids_tensor,
                    [event.cuda_event for event in event_list],
                    is_batch_layer)

    # pair used with get_blocks_batch
    def get_stream_sync(self, request_ids: torch.Tensor):
        if (request_ids.numel() == 0):
            return
        for req_id in request_ids.numpy():
            if (self.event_map.get(req_id) == None):
                continue
            event = self.event_map[req_id]
            event.synchronize()
            # recycle the events
            self.event_pool.put_event(event)
            del self.event_map[req_id]

    # pair used with get_layer_blocks/get_blocks
    def get_stream_layer_sync(self,
                              layer_id: int,
                              request_ids: torch.Tensor):
        if (request_ids.numel() == 0):
            return
        for req_id in request_ids.numpy():
            if (self.event_map.get(req_id) == None):
                continue
            self.event_map[req_id][layer_id].synchronize()
        if (layer_id == self.num_layer - 1):
            # recycle the events
            for req_id in request_ids.numpy():
                if (self.event_map.get(req_id) == None):
                    continue
                event_list = self.event_map[req_id]
                self.event_pool.put_events(event_list)
                del self.event_map[req_id]

    def put_stream_sync(self):
        self.put_stream.synchronize()
