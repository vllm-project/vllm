# Benchmarking vLLM

## Dataset Overview

| **Dataset** | **Online** | **Offline** | **Data Path**                                                                                                                                                  |
|-----------------|:----------:|:-----------:|----------------------------------------------------------------------------------------------------------------------------------------------------------------|
| ShareGPT        | ✅         | ✅          | download from [ShareGPT URL](https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json) |
| BurstGPT        | ✅         | ✅          | download from [BurstGPT CSV](https://github.com/HPMLL/BurstGPT/releases/download/v1.1/BurstGPT_without_fails_2.csv)                                            |
| Sonnet          | ✅         | ✅          | in the repo `benchmarks/sonnet.txt`                                                                                                                            |
| Random          | ✅         | ✅          | `synthetic`                                                                                                                                                    |
| HuggingFace     | ✅         | 🚧          | `specify your dataset path on huggingface`                                                                                                                     |
| VisionArena     | ✅         | 🚧          | `lmarena-ai/vision-arena-bench-v0.1`                                                                                                                           |

✅: supported  
🚧: to be supported

**Note**: VisionArena’s `dataset-name` should be set to `hf`

---
## Online Script Example

First start serving your model

```bash
MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
vllm serve ${MODEL_NAME}     --swap-space 16     --disable-log-requests
```

Then run the benchmarking script

```bash
MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
NUM_PROMPTS=10
BACKEND="openai-chat"
DATASET_NAME="sonnet"
DATASET_PATH="benchmarks/sonnet.txt"
python3 benchmarks/benchmark_serving.py --backend ${BACKEND} --model ${MODEL_NAME} --endpoint /v1/chat/completions --dataset-name ${DATASET_NAME} --dataset-path ${DATASET_PATH} --num-prompts ${NUM_PROMPTS}
```

For a successful run, you will see the following output:

```angular2html
============ Serving Benchmark Result ============
Successful requests:                     10  
Benchmark duration (s):                  1.37  
Total input tokens:                      5032  
Total generated tokens:                  1500  
Request throughput (req/s):              7.28  
Output token throughput (tok/s):         1092.22  
Total Token throughput (tok/s):          4756.24  
---------------Time to First Token----------------
Mean TTFT (ms):                          113.34  
Median TTFT (ms):                        115.14  
P99 TTFT (ms):                           141.10  
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          8.37  
Median TPOT (ms):                        8.35  
P99 TPOT (ms):                           8.62  
---------------Inter-token Latency----------------
Mean ITL (ms):                           8.31  
Median ITL (ms):                         8.19  
P99 ITL (ms):                            9.40  
```

---
## Offline Script Example

```bash
MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
NUM_PROMPTS=10
DATASET_NAME="sonnet"
DATASET_PATH="benchmarks/sonnet.txt"

python3 benchmarks/benchmark_throughput.py --model ${MODEL_NAME} --dataset-name ${DATASET_NAME} --dataset-name ${DATASET_NAME} --num-prompts ${NUM_PROMPTS}
```

For a successful run, you will see the following output:

```
Throughput: 7.11 requests/s, 8188.65 total tokens/s, 909.85 output tokens/s
```
