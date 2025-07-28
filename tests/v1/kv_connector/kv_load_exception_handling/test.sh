#!/bin/bash

SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
export PYTHONPATH=$PYTHONPATH:$SCRIPT_DIR

vllm serve DeepSeek-V2-Lite-Chat \
--trust-remote-code \
--served-model-name vllm_cpu_offload \
--max-model-len 32768 \
--no-enable-prefix-caching \
--max-seq-len-to-capture 10000 \
--max-num-seqs 64 \
--gpu-memory-utilization 0.9 \
--host 0.0.0.0 \
-tp 2 \
--kv-transfer-config '{"kv_connector":"RandomDropConnector","kv_role":"kv_both","kv_connector_module_path":"random_drop_connector"}' 
