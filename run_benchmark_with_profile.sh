export VLLM_TORCH_PROFILER_DIR=$PWD/profile

VLLM_USE_V1=1 python3 benchmarks/benchmark_throughput.py --model facebook/opt-125m --dataset_name random  --enforce-eager --max-num-seqs 32 --gpu-memory-util 0.8 --num-prompts 16 --max-model-len 2000 --input-len 1024 --output-len 10 --max-num-batched-tokens 32768  --disable-sliding-window --dtype float16 --profile