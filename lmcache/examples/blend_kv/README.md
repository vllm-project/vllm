# KV blending example
This is a minimal example demonstrating the KV blending functionality of LMCache.

The KV blending functionality is enabled by setting `enable_blending: True` in the configuration yaml.

In `blend_kv.py`, the following code will first calculate the KV cache of two text chunks.
```python
offline_precompute = OfflineKVPreCompute(llm)
for chunk in chunks:
    offline_precompute.precompute_kv(chunk)
```

Then, the text chunks are concatenated together, prepended with a system prompt, and appended with a user's quest.
```python
user_prompt= [sys_prompt, chunks[0], chunks[1], question]
user_prompt = combine_input_prompt_chunks(user_prompt)
```

Finally, the prompt will be sent to the serving engine and the KV blending module will blend the KV for the text chunks.


## How to run
### Offline
```
LMCACHE_CONFIG_FILE=example_blending.yaml LMCACHE_USE_EXPERIMENTAL=False python3 blend_kv.py
LMCACHE_CONFIG_FILE=example_blending.yaml LMCACHE_USE_EXPERIMENTAL=False python3 batched_kv.py
LMCACHE_CONFIG_FILE=example_blending.yaml LMCACHE_USE_EXPERIMENTAL=False VLLM_WORKER_MULTIPROC_METHOD=spawn python3 tp_kv.py
LMCACHE_CONFIG_FILE=example_blending.yaml VLLM_WORKER_MULTIPROC_METHOD=spawn LMCACHE_USE_EXPERIMENTAL=False python3 batched_tp_kv.py
```
### Online
```
LMCACHE_CONFIG_FILE=example_blending.yaml LMCACHE_USE_EXPERIMENTAL=False CUDA_VISIBLE_DEVICES=0 python3 -m lmcache_vllm.vllm.entrypoints.openai.api_server --model mistralai/Mistral-7B-Instruct-v0.2 --gpu-memory-utilization 0.8 --port 8000
python3 online_kv.py 8000
```
```
LMCACHE_CONFIG_FILE=example_blending.yaml LMCACHE_USE_EXPERIMENTAL=False CUDA_VISIBLE_DEVICES=0,1 VLLM_WORKER_MULTIPROC_METHOD=spawn python3 -m lmcache_vllm.vllm.entrypoints.openai.api_server --model mistralai/Mistral-7B-Instruct-v0.2 --gpu-memory-utilization 0.8 --port 8000 --tensor-parallel-size 2
python3 online_kv.py 8000
```
