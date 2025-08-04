# python benchmarks/benchmark_throughput.py \
#     --model meta-llama/Meta-Llama-3-8B-Instruct \
#     --dataset-name sonnet \
#     --dataset-path /data/lily/batch-sd/benchmarks/sonnet.txt \
#     --prefix-len 0 \
#     --output-len 512 \
#     --num-prompts 200 \
#     --speculative_config '{"method": "ngram", "num_speculative_tokens": 20, "prompt_lookup_min": 2, "prompt_lookup_max": 5}'


# python benchmarks/benchmark_throughput.py \
#     --model meta-llama/Meta-Llama-3-8B-Instruct \
#     --dataset-name sharegpt \
#     --dataset-path /data/lily/ShareGPT_V3_unfiltered_cleaned_split.json  \
#     --prefix-len 0 \
#     --output-len 512 \
#     --num-prompts 200 \
#     --speculative_config '{"method": "ngram", "num_speculative_tokens": 20, "prompt_lookup_min": 2, "prompt_lookup_max": 5}'

# python benchmarks/benchmark_throughput.py \
#     --model meta-llama/Meta-Llama-3-8B-Instruct \
#     --dataset-name hf \
#     --dataset-path likaixin/InstructCoder  \
#     --prefix-len 0 \
#     --output-len 512 \
#     --num-prompts 200 \
#     --speculative_config '{"method": "ngram", "num_speculative_tokens": 20, "prompt_lookup_min": 2, "prompt_lookup_max": 5}'


# python benchmarks/benchmark_throughput.py \
#     --model meta-llama/Meta-Llama-3-8B-Instruct \
#     --dataset-name sonnet \
#     --dataset-path /data/lily/batch-sd/benchmarks/sonnet.txt \
#     --prefix-len 0 \
#     --output-len 512 \
#     --num-prompts 200 \
#     --speculative_config '{"method": "eagle", "model": "yuhuili/EAGLE-LLaMA3-Instruct-8B", "num_speculative_tokens": 20}'

# python benchmarks/benchmark_throughput.py \
#     --model meta-llama/Meta-Llama-3-8B-Instruct \
#     --dataset-name sharegpt \
#     --dataset-path /data/lily/ShareGPT_V3_unfiltered_cleaned_split.json \
#     --prefix-len 0 \
#     --output-len 512 \
#     --num-prompts 200 \
#     --speculative_config '{"method": "eagle", "model": "yuhuili/EAGLE-LLaMA3-Instruct-8B", "num_speculative_tokens": 20}'


# python benchmarks/benchmark_throughput.py \
#     --model meta-llama/Meta-Llama-3-8B-Instruct \
#     --dataset-name hf \
#     --dataset-path likaixin/InstructCoder \
#     --prefix-len 0 \
#     --output-len 512 \
#     --num-prompts 200 \
#     --speculative_config '{"method": "eagle", "model": "yuhuili/EAGLE-LLaMA3-Instruct-8B", "num_speculative_tokens": 20}'


# python benchmarks/benchmark_throughput.py \
#     --model meta-llama/Meta-Llama-3.1-8B-Instruct \
#     --dataset-name hf \
#     --dataset-path likaixin/InstructCoder \
#     --prefix-len 0 \
#     --output-len 512 \
#     --num-prompts 200 \
#     --speculative_config '{"method": "eagle3", "model": "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B", "num_speculative_tokens": 20}'


python benchmarks/benchmark_throughput.py \
    --model meta-llama/Meta-Llama-3.1-8B-Instruct\
    --dataset-name hf \
    --dataset-path philschmid/mt-bench  \
    --prefix-len 0 \
    --output-len 512 \
    --num-prompts 200 \
    --speculative_config '{"method": "eagle3", "model": "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B", "num_speculative_tokens": 20}'


# python benchmarks/benchmark_throughput.py \
#     --model meta-llama/Meta-Llama-3.1-8B-Instruct \
#     --dataset-name sharegpt \
#     --dataset-path /data/lily/ShareGPT_V3_unfiltered_cleaned_split.json \
#     --prefix-len 0 \
#     --output-len 512 \
#     --num-prompts 200 \
#     --speculative_config '{"method": "eagle3", "model": "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B", "num_speculative_tokens": 20}'

# python benchmarks/benchmark_throughput.py \
#     --model meta-llama/Meta-Llama-3.1-8B-Instruct \
#     --dataset-name sonnet \
#     --dataset-path /data/lily/batch-sd/benchmarks/sonnet.txt \
#     --prefix-len 0 \
#     --output-len 512 \
#     --num-prompts 200 \
#     --speculative_config '{"method": "eagle3", "model": "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B", "num_speculative_tokens": 20}'


# python benchmarks/benchmark_throughput.py \
#     --model meta-llama/Meta-Llama-3.1-8B-Instruct \
#     --dataset-name hf \
#     --dataset-path likaixin/InstructCoder \
#     --prefix-len 0 \
#     --output-len 512 \
#     --num-prompts 200 \
#     --speculative_config '{"method": "eagle", "model": "yuhuili/EAGLE-LLaMA3.1-Instruct-8B", "num_speculative_tokens": 20}'

# python benchmarks/benchmark_throughput.py \
#     --model meta-llama/Meta-Llama-3.1-8B-Instruct \
#     --dataset-name sharegpt \
#     --dataset-path /data/lily/ShareGPT_V3_unfiltered_cleaned_split.json \
#     --prefix-len 0 \
#     --output-len 512 \
#     --num-prompts 200 \
#     --speculative_config '{"method": "eagle", "model": "yuhuili/EAGLE-LLaMA3.1-Instruct-8B", "num_speculative_tokens": 20}'

# python benchmarks/benchmark_throughput.py \
#     --model meta-llama/Meta-Llama-3.1-8B-Instruct \
#     --dataset-name hf \
#     --dataset-path likaixin/InstructCoder  \
#     --prefix-len 0 \
#     --output-len 512 \
#     --num-prompts 200 \
#     --speculative_config '{"method": "ngram", "num_speculative_tokens": 20, "prompt_lookup_min": 2, "prompt_lookup_max": 5}'


# python benchmarks/benchmark_throughput.py \
#     --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
#     --dataset-name hf \
#     --dataset-path AI-MO/aimo-validation-aime \
#     --prefix-len 0 \
#     --output-len 1024 \
#     --num-prompts 90 \
#     --speculative_config '{"method": "eagle3", "num_speculative_tokens": 20, "model": "yuhuili/EAGLE3-DeepSeek-R1-Distill-LLaMA-8B"}'



# python benchmarks/benchmark_throughput.py \
#     --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
#     --dataset-name hf \
#     --dataset-path AI-MO/aimo-validation-aime \
#     --prefix-len 0 \
#     --output-len 1024 \
#     --num-prompts 90 \
#     --speculative_config '{"method": "ngram", "num_speculative_tokens": 20, "prompt_lookup_min": 2, "prompt_lookup_max": 5}'



# python benchmarks/benchmark_throughput.py \
#     --model meta-llama/Meta-Llama-3.1-8B-Instruct \
#     --dataset-name sharegpt \
#     --dataset-path /data/lily/ShareGPT_V3_unfiltered_cleaned_split.json  \
#     --prefix-len 0 \
#     --output-len 512 \
#     --num-prompts 200 \
#     --speculative_config '{"method": "ngram", "num_speculative_tokens": 20, "prompt_lookup_min": 2, "prompt_lookup_max": 5}'


# python benchmarks/benchmark_throughput.py \
#     --model meta-llama/Meta-Llama-3.1-8B-Instruct \
#     --dataset-name hf \
#     --dataset-path philschmid/mt-bench \
#     --prefix-len 0 \
#     --output-len 512 \
#     --num-prompts 200 \
#     --speculative_config '{"method": "ngram", "num_speculative_tokens": 20, "prompt_lookup_min": 2, "prompt_lookup_max": 5}'

# python benchmarks/benchmark_throughput.py \
#     --model meta-llama/Meta-Llama-3.1-8B-Instruct \
#     --dataset-name hf \
#     --dataset-path philschmid/mt-bench \
#     --prefix-len 0 \
#     --output-len 512 \
#     --num-prompts 200 \
#     --speculative_config '{"method": "eagle", "model": "yuhuili/EAGLE-LLaMA3-Instruct-8B", "num_speculative_tokens": 20}'


# python benchmarks/benchmark_throughput.py \
#     --model meta-llama/Meta-Llama-3.1-8B-Instruct \
#     --dataset-name hf \
#     --dataset-path abisee/cnn_dailymail \
#     --prefix-len 0 \
#     --output-len 512 \
#     --num-prompts 200 \
#     --speculative_config '{"method": "eagle", "model": "yuhuili/EAGLE-LLaMA3-Instruct-8B", "num_speculative_tokens": 20}'

# python benchmarks/benchmark_throughput.py \
#     --model meta-llama/Meta-Llama-3.1-8B-Instruct \
#     --dataset-name hf \
#     --dataset-path abisee/cnn_dailymail \
#     --prefix-len 0 \
#     --output-len 512 \
#     --num-prompts 200 \
#     --speculative_config '{"method": "ngram", "num_speculative_tokens": 20, "prompt_lookup_min": 2, "prompt_lookup_max": 5}'


# python benchmarks/benchmark_throughput.py \
#     --model meta-llama/Meta-Llama-3.1-8B-Instruct \
#     --dataset-name hf \
#     --dataset-path philschmid/mt-bench \
#     --prefix-len 0 \
#     --output-len 512 \
#     --num-prompts 10 \
#     --speculative_config '{"method": "eagle3", "model": "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B", "num_speculative_tokens": 20}'


# python benchmarks/benchmark_throughput.py \
#     --model meta-llama/Meta-Llama-3.1-8B-Instruct \
#     --dataset-name hf \
#     --dataset-path abisee/cnn_dailymail \
#     --prefix-len 0 \
#     --output-len 512 \
#     --num-prompts 200 \
#     --speculative_config '{"method": "ngram", "num_speculative_tokens": 20, "prompt_lookup_min": 2, "prompt_lookup_max": 5}'
