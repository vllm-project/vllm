# Benchmarking vLLM

This README guides you through running benchmark tests with the extensive
datasets supported on vLLM. It’s a living document, updated as new features and datasets
become available.

## Dataset Overview

<table style="width:100%; border-collapse: collapse;">
  <thead>
    <tr>
      <th style="width:15%; text-align: left;">Dataset</th>
      <th style="width:10%; text-align: center;">Online</th>
      <th style="width:10%; text-align: center;">Offline</th>
      <th style="width:65%; text-align: left;">Data Path</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>ShareGPT</strong></td>
      <td style="text-align: center;">✅</td>
      <td style="text-align: center;">✅</td>
      <td><code>wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json</code></td>
    </tr>
    <tr>
      <td><strong>BurstGPT</strong></td>
      <td style="text-align: center;">✅</td>
      <td style="text-align: center;">✅</td>
      <td><code>wget https://github.com/HPMLL/BurstGPT/releases/download/v1.1/BurstGPT_without_fails_2.csv</code></td>
    </tr>
    <tr>
      <td><strong>Sonnet</strong></td>
      <td style="text-align: center;">✅</td>
      <td style="text-align: center;">✅</td>
      <td>Local file: <code>benchmarks/sonnet.txt</code></td>
    </tr>
    <tr>
      <td><strong>Random</strong></td>
      <td style="text-align: center;">✅</td>
      <td style="text-align: center;">✅</td>
      <td><code>synthetic</code></td>
    </tr>
    <tr>
      <td><strong>HuggingFace</strong></td>
      <td style="text-align: center;">✅</td>
      <td style="text-align: center;">🟡</td>
      <td>Specify your dataset path on HuggingFace</td>
    </tr>
    <tr>
      <td><strong>VisionArena</strong></td>
      <td style="text-align: center;">✅</td>
      <td style="text-align: center;">✅</td>
      <td><code>lmarena-ai/vision-arena-bench-v0.1</code> (a HuggingFace dataset)</td>
    </tr>
  </tbody>
</table>

✅: supported

🚧: to be supported

🟡: Partial support. Currently, HuggingFaceDataset only supports dataset formats
similar to `lmms-lab/LLaVA-OneVision-Data`. If you need support for other dataset
formats, please consider contributing.

**Note**: VisionArena’s `dataset-name` should be set to `hf`

---
## Example - Online Benchmark

First start serving your model

```bash
MODEL_NAME="NousResearch/Hermes-3-Llama-3.1-8B"
vllm serve ${MODEL_NAME} --disable-log-requests
```

Then run the benchmarking script

```bash
# download dataset
# wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
MODEL_NAME="NousResearch/Hermes-3-Llama-3.1-8B"
NUM_PROMPTS=10
BACKEND="openai-chat"
DATASET_NAME="sharegpt"
DATASET_PATH="<your data path>/ShareGPT_V3_unfiltered_cleaned_split.json"
python3 vllm/benchmarks/benchmark_serving.py --backend ${BACKEND} --model ${MODEL_NAME} --endpoint /v1/chat/completions --dataset-name ${DATASET_NAME} --dataset-path ${DATASET_PATH} --num-prompts ${NUM_PROMPTS}
```

If successful, you will see the following output

```
============ Serving Benchmark Result ============
Successful requests:                     10        
Benchmark duration (s):                  5.78      
Total input tokens:                      1369      
Total generated tokens:                  2212      
Request throughput (req/s):              1.73      
Output token throughput (tok/s):         382.89    
Total Token throughput (tok/s):          619.85    
---------------Time to First Token----------------
Mean TTFT (ms):                          71.54     
Median TTFT (ms):                        73.88     
P99 TTFT (ms):                           79.49     
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          7.91      
Median TPOT (ms):                        7.96      
P99 TPOT (ms):                           8.03      
---------------Inter-token Latency----------------
Mean ITL (ms):                           7.74      
Median ITL (ms):                         7.70      
P99 ITL (ms):                            8.39      
==================================================
```

### VisionArena Benchmark for Vision Language Models

```bash
# need a model with vision capability here
vllm serve Qwen/Qwen2-VL-7B-Instruct --disable-log-requests
```

```bash
MODEL_NAME="Qwen/Qwen2-VL-7B-Instruct"
NUM_PROMPTS=10
BACKEND="openai-chat"
DATASET_NAME="hf"
DATASET_PATH="lmarena-ai/vision-arena-bench-v0.1"
DATASET_SPLIT='train'

python3 vllm/benchmarks/benchmark_serving.py \
  --backend "${BACKEND}" \
  --model "${MODEL_NAME}" \
  --endpoint "/v1/chat/completions" \
  --dataset-name "${DATASET_NAME}" \
  --dataset-path "${DATASET_PATH}" \
  --hf-split "${DATASET_SPLIT}" \
  --num-prompts "${NUM_PROMPTS}"
```

---
## Example - Offline Throughput Benchmark

```bash
MODEL_NAME="NousResearch/Hermes-3-Llama-3.1-8B"
NUM_PROMPTS=10
DATASET_NAME="sonnet"
DATASET_PATH="vllm/benchmarks/sonnet.txt"

python3 vllm/benchmarks/benchmark_throughput.py \
  --model "${MODEL_NAME}" \
  --dataset-name "${DATASET_NAME}" \
  --dataset-path "${DATASET_PATH}" \
  --num-prompts "${NUM_PROMPTS}"
```

If successful, you will see the following output

```
Throughput: 7.15 requests/s, 4656.00 total tokens/s, 1072.15 output tokens/s
Total num prompt tokens:  5014
Total num output tokens:  1500
```

### VisionArena Benchmark for Vision Language Models

``` bash
MODEL_NAME="Qwen/Qwen2-VL-7B-Instruct"
NUM_PROMPTS=10
DATASET_NAME="hf"
DATASET_PATH="lmarena-ai/vision-arena-bench-v0.1"
DATASET_SPLIT="train"

python3 vllm/benchmarks/benchmark_throughput.py \
  --model "${MODEL_NAME}" \
  --backend "vllm-chat" \
  --dataset-name "${DATASET_NAME}" \
  --dataset-path "${DATASET_PATH}" \
  --num-prompts "${NUM_PROMPTS}" \
  --hf-split "${DATASET_SPLIT}"
```

The `num prompt tokens` now includes image token counts

```
Throughput: 2.55 requests/s, 4036.92 total tokens/s, 326.90 output tokens/s
Total num prompt tokens:  14527
Total num output tokens:  1280
```

### Benchmark with LoRA Adapters

``` bash
# download dataset
# wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
MODEL_NAME="meta-llama/Llama-2-7b-hf"
BACKEND="vllm"
DATASET_NAME="sharegpt"
DATASET_PATH="<your data path>/ShareGPT_V3_unfiltered_cleaned_split.json"
NUM_PROMPTS=10
MAX_LORAS=2
MAX_LORA_RANK=8
ENABLE_LORA="--enable-lora"
LORA_PATH="yard1/llama-2-7b-sql-lora-test"

python3 vllm/benchmarks/benchmark_throughput.py \
  --model "${MODEL_NAME}" \
  --backend "${BACKEND}" \
  --dataset_path "${DATASET_PATH}" \
  --dataset_name "${DATASET_NAME}" \
  --num-prompts "${NUM_PROMPTS}" \
  --max-loras "${MAX_LORAS}" \
  --max-lora-rank "${MAX_LORA_RANK}" \
  ${ENABLE_LORA} \
  --lora-path "${LORA_PATH}"
  ```
