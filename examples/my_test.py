import torch
from vllm import LLM, SamplingParams
from vllm.worker.cache_engine import CacheEngine, CacheEngineManager
from vllm.core.block_manager import BlockAllocator

# Sample prompts.
prompts = [
    "Hello, my name is",
]
# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

# Create an LLM.
llm = LLM(model="facebook/opt-125m")
#llm = LLM(model="lmsys/longchat-7b-16k")
#llm = LLM(model="gpt2")
# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
outputs = llm.generate(prompts, sampling_params)
# Print the outputs.
#for output in outputs:
#    prompt = output.prompt
#    generated_text = output.outputs[0].text
#    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")



cache_engine = CacheEngineManager.GetCacheEngine()
gpu_cache = cache_engine.gpu_cache
layer_id = 0
block = llm.llm_engine.scheduler.block_manager.gpu_allocator.allocate(0, 1)

num_heads, head_size, block_size = cache_engine.get_value_block_shape()
key_tensor = torch.full((block_size, num_heads, head_size), 1).cuda()
v_tensor = torch.full((block_size, num_heads, head_size), 2).cuda()

BlockAllocator.fill_kv_block(
        (key_tensor, v_tensor),
        gpu_cache[layer_id],
        block)
