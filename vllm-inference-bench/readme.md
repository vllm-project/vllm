# Inference Benchmarking using vLLM 


VLLM_USE_PRECOMPILED=1 pip install --editable .


Llama 4 scout: meta-llama/Llama-4-Scout-17B-16E-Instruct

DeepSeek-v2 : deepseek-ai/DeepSeek-V2-Chat-0628

Llama 3-8b: meta-llama/Llama-3.1-8B-Instruct

vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --trust-remote-code \
  --tensor-parallel-size 8 \


# Prompt-heavy benchmark (8k/1k)
vllm bench serve \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --dataset-name random \
  --random-input-len 8000 \
  --random-output-len 1000 \
  --request-rate 10000 \
  --num-prompts 16 \
  --ignore-eos

# Single node EP deployment with pplx backend
VLLM_ALL2ALL_BACKEND=pplx VLLM_USE_DEEP_GEMM=1 \
    vllm serve deepseek-ai/DeepSeek-V3-0324 \
    --tensor-parallel-size 1 \      
    --data-parallel-size 8 \         
    --enable-expert-parallel


VLLM_ALL2ALL_BACKEND=pplx VLLM_USE_DEEP_GEMM=1 vllm serve deepseek-ai/DeepSeek-V2-Chat-0628 --tensor-parallel-size=1 --data-parallel-size=8 --enable-expert-parallel

VLLM_ALL2ALL_BACKEND=deepep_low_latency VLLM_USE_DEEP_GEMM=1 vllm serve deepseek-ai/DeepSeek-V2-Chat-0628 --tensor-parallel-size=1 --data-parallel-size=8 --enable-expert-parallel --gpu_memory_utilization=0.6

VLLM_ALL2ALL_BACKEND=deepep_low_latency VLLM_USE_DEEP_GEMM=1 vllm serve meta-llama/Llama-4-Scout-17B-16E-Instruct --tensor-parallel-size=1 --data-parallel-size=8 --enable-expert-parallel --gpu_memory_utilization=0.6


meta-llama/Llama-4-Scout-17B-16E-Instruct