#!/bin/bash

# More documentation: https://docs.vllm.ai/en/v0.8.3/serving/openai_compatible_server.html#vllm-serve
export VLLM_USE_V1="1"
# Specify base model (and optionally loras) to load in when starting the server.
vllm serve ibm-granite/granite-3.2-8b-instruct \
    --enable-lora \
    --lora-modules '{"name": "new_alora", "path": "/proj/dmfexp/statllm/users/kgreenewald/.cache/huggingface/models/hub/models--ibm-granite--granite-3.2-8b-alora-uncertainty/snapshots/6109ad88201426003e696d023ec67c19e7f3d444", "base_model_name": "ibm-granite/granite-3.2-8b-instruct"}' \
    --dtype bfloat16 \
    --max-lora-rank 64 \
    --enable-prefix-caching
#--no-enable-prefix-caching
# Check that the lora model is listed along with other models.
#curl localhost:8000/v1/models | jq .

###########################################

# A second option is to enable dynamic adapter loading instead of at start-up.
#export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True

#curl -X POST http://localhost:8000/v1/load_lora_adapter \
#-H "Content-Type: application/json" \
#-d '{
#    "lora_name": "new_alora",
#    "lora_path": "/path/to/new_alora"
#}'
# Should return "200 OK - Success: LoRA adapter 'new_alora' added successfully"

# Example of dynamically unloading an adapter.
# curl -X POST http://localhost:8000/v1/unload_lora_adapter \
# -H "Content-Type: application/json" \
# -d '{
#     "lora_name": "new_alora"
# }'

###########################################

# Send a request using the new aLoRA
#curl http://localhost:8000/v1/completions \
#    -H "Content-Type: application/json" \
#    -d '{
#        "model": "new_alora",
#        "prompt": ""What is MIT?"",
#        "max_tokens": 600,
#        "temperature": 0
#    }' | jq
