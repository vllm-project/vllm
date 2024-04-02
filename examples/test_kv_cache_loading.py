import torch
from vllm import LLM, SamplingParams
from vllm.worker.cache_engine import CacheEngine, CacheEngineManager
from vllm.core.block_manager import BlockAllocator
import pdb

# Create an LLM.
llm = LLM(model="meta-llama/Llama-2-7b-chat-hf")

# get block
cache_engine = CacheEngineManager.GetCacheEngine()
gpu_cache = cache_engine.gpu_cache
layer_id = 0
block = llm.llm_engine.scheduler.block_manager.gpu_allocator.allocate(0, 1)

# fill with K and V. K: all 1 tensor, V: all 2 tensor
num_heads, head_size, block_size = cache_engine.get_value_block_shape()
key_tensor = torch.full((block_size, num_heads, head_size), 1).half().cuda()
v_tensor = torch.full((block_size, num_heads, head_size), 2).half().cuda()

# put K and V into the block
BlockAllocator.fill_kv_block(
        (key_tensor, v_tensor),
        gpu_cache[layer_id],
        block)
# check if the block is filled with K and V
assert torch.isclose(gpu_cache[layer_id][0][block.block_number].mean(), torch.tensor(1.).half())
assert torch.isclose(gpu_cache[layer_id][1][block.block_number].mean(), torch.tensor(2.).half())


key_tensor_new = torch.ones_like(key_tensor)
v_tensor_new = torch.ones_like(v_tensor)
pdb.set_trace()
BlockAllocator.load_kv_block(
        (key_tensor_new, v_tensor_new),
        gpu_cache[layer_id],
        block)
pdb.set_trace()

assert torch.isclose(key_tensor_new.mean(), torch.tensor(1.).half())
assert torch.isclose(v_tensor_new.mean(), torch.tensor(2.).half())

#assert torch.isclose(gpu_cache[layer_id][0][block.block_number].mean(), torch.tensor(1.).half())
#assert torch.isclose(gpu_cache[layer_id][1][block.block_number].mean(), torch.tensor(2.).half())
print("finish")

