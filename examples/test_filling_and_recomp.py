import torch
from vllm import LLM, SamplingParams
from vllm.worker.cache_engine import CacheEngine, CacheEngineManager
from vllm.core.block_manager import BlockAllocator
import pdb

'''
# Sample prompts.
prompt1 = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
    "Hey there I want to talk with you",
    "Can you write some random python code for me?"
]
'''

# Create an LLM.
llm = LLM(model="meta-llama/Llama-2-7b-chat-hf")
pdb.set_trace()

#FIXME(Jiayi): need a tokenizer to manully change max tokens
tokenizer = llm.llm_engine.tokenizer

prompt1 = [
    "Hello, my name is Eric."
]

max_tokens_1 = len(tokenizer.encode(prompt1[0])) + 1
# Create a sampling params object.
sampling_params_1 = SamplingParams(temperature=0.0, max_tokens=max_tokens_1)
outputs = llm.generate(prompt1, sampling_params_1)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

pdb.set_trace()

# prompt token ids

#llm = LLM(model="lmsys/longchat-7b-16k")
#llm = LLM(model="gpt2")
# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.

#FIXME(Jiayi): we need a clear cache method, kv cache, block tables, slot mappings, etc.

#FIXME(Jiayi): need a hack to input tokens
#FIXME(Jiayi): use maxgen 1 first

print("end")


'''
# Create an LLM.
llm = LLM(model="meta-llama/Llama-2-7b-chat-hf")
sampling_params = SamplingParams(temperature=0.0)
pdb.set_trace()

# get block
cache_engine = llm.llm_engine.model_executor.driver_worker.cache_engine
gpu_cache = cache_engine.gpu_cache
layer_id = 0
block = llm.llm_engine.scheduler.block_manager.gpu_allocator.allocate(num_hashed_tokens=1)
pdb.set_trace()

# llm.llm_engine.model_executor.driver_worker.cache_engine.gpu_cache
# llm.llm_engine.model_executor.driver_worker.gpu_cache

# fill with K and V. K: all 1 tensor, V: all 2 tensor
num_heads, head_size, block_size = cache_engine.get_value_block_shape()
key_tensor = torch.full((block_size, num_heads, head_size), 1).half().cuda()
v_tensor = torch.full((block_size, num_heads, head_size), 2).half().cuda()
pdb.set_trace()

# put K and V into the block
BlockAllocator.fill_kv_block(
        (key_tensor, v_tensor),
        gpu_cache[layer_id],
        block)
pdb.set_trace()

# check if the block is filled with K and V
assert torch.isclose(gpu_cache[layer_id][0][block.block_number].mean(), torch.tensor(1.).half())
assert torch.isclose(gpu_cache[layer_id][1][block.block_number].mean(), torch.tensor(2.).half())
'''
