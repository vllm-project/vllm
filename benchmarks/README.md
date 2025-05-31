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
      <td><strong>HuggingFace-VisionArena</strong></td>
      <td style="text-align: center;">✅</td>
      <td style="text-align: center;">✅</td>
      <td><code>lmarena-ai/VisionArena-Chat</code></td>
    </tr>
    <tr>
      <td><strong>HuggingFace-InstructCoder</strong></td>
      <td style="text-align: center;">✅</td>
      <td style="text-align: center;">✅</td>
      <td><code>likaixin/InstructCoder</code></td>
    </tr>
      <tr>
      <td><strong>HuggingFace-AIMO</strong></td>
      <td style="text-align: center;">✅</td>
      <td style="text-align: center;">✅</td>
      <td><code>AI-MO/aimo-validation-aime</code> , <code>AI-MO/NuminaMath-1.5</code>, <code>AI-MO/NuminaMath-CoT</code></td>
    </tr>
    <tr>
      <td><strong>HuggingFace-Other</strong></td>
      <td style="text-align: center;">✅</td>
      <td style="text-align: center;">✅</td>
      <td><code>lmms-lab/LLaVA-OneVision-Data</code>, <code>Aeala/ShareGPT_Vicuna_unfiltered</code></td>
    </tr>
    <tr>
      <td><strong>Custom</strong></td>
      <td style="text-align: center;">✅</td>
      <td style="text-align: center;">✅</td>
      <td>Local file: <code>data.jsonl</code></td>
    </tr>
  </tbody>
</table>

✅: supported

🟡: Partial support

🚧: to be supported

**Note**: HuggingFace dataset's `dataset-name` should be set to `hf`

---
## Example - Online Benchmark

First start serving your model

```bash
vllm serve NousResearch/Hermes-3-Llama-3.1-8B --disable-log-requests
```

Then run the benchmarking script

```bash
# download dataset
# wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
python3 vllm/benchmarks/benchmark_serving.py \
  --backend vllm \
  --model NousResearch/Hermes-3-Llama-3.1-8B \
  --endpoint /v1/completions \
  --dataset-name sharegpt \
  --dataset-path <your data path>/ShareGPT_V3_unfiltered_cleaned_split.json \
  --num-prompts 10
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

### Custom Dataset
If the dataset you want to benchmark is not supported yet in vLLM, even then you can benchmark on it using `CustomDataset`. Your data needs to be in `.jsonl` format and needs to have "prompt" field per entry, e.g., data.jsonl

```
{"prompt": "What is the capital of India?"}
{"prompt": "What is the capital of Iran?"}
{"prompt": "What is the capital of China?"}
``` 

```bash
# start server
VLLM_USE_V1=1 vllm serve meta-llama/Llama-3.1-8B-Instruct --disable-log-requests
```

```bash
# run benchmarking script
python3 benchmarks/benchmark_serving.py --port 9001 --save-result --save-detailed \
  --backend vllm \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --endpoint /v1/completions \
  --dataset-name custom \
  --dataset-path <path-to-your-data-jsonl> \
  --custom-skip-chat-template \
  --num-prompts 80 \
  --max-concurrency 1 \
  --temperature=0.3 \
  --top-p=0.75 \
  --result-dir "./log/"
```

You can skip applying chat template if your data already has it by using `--custom-skip-chat-template`.

### VisionArena Benchmark for Vision Language Models

```bash
# need a model with vision capability here
vllm serve Qwen/Qwen2-VL-7B-Instruct --disable-log-requests
```

```bash
python3 vllm/benchmarks/benchmark_serving.py \
  --backend openai-chat \
  --model Qwen/Qwen2-VL-7B-Instruct \
  --endpoint /v1/chat/completions \
  --dataset-name hf \
  --dataset-path lmarena-ai/VisionArena-Chat \
  --hf-split train \
  --num-prompts 1000
```

### InstructCoder Benchmark with Speculative Decoding

``` bash
VLLM_USE_V1=1 vllm serve meta-llama/Meta-Llama-3-8B-Instruct \
    --speculative-config $'{"method": "ngram",
    "num_speculative_tokens": 5, "prompt_lookup_max": 5,
    "prompt_lookup_min": 2}'
```

``` bash
python3 benchmarks/benchmark_serving.py \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --dataset-name hf \
    --dataset-path likaixin/InstructCoder \
    --num-prompts 2048
```

### Other HuggingFaceDataset Examples

```bash
vllm serve Qwen/Qwen2-VL-7B-Instruct --disable-log-requests
```

**`lmms-lab/LLaVA-OneVision-Data`**

```bash
python3 vllm/benchmarks/benchmark_serving.py \
  --backend openai-chat \
  --model Qwen/Qwen2-VL-7B-Instruct \
  --endpoint /v1/chat/completions \
  --dataset-name hf \
  --dataset-path lmms-lab/LLaVA-OneVision-Data \
  --hf-split train \
  --hf-subset "chart2text(cauldron)" \
  --num-prompts 10
```

**`Aeala/ShareGPT_Vicuna_unfiltered`**

```bash
python3 vllm/benchmarks/benchmark_serving.py \
  --backend openai-chat \
  --model Qwen/Qwen2-VL-7B-Instruct \
  --endpoint /v1/chat/completions \
  --dataset-name hf \
  --dataset-path Aeala/ShareGPT_Vicuna_unfiltered \
  --hf-split train \
  --num-prompts 10
```

**`AI-MO/aimo-validation-aime`**

``` bash
python3 vllm/benchmarks/benchmark_serving.py \
    --model Qwen/QwQ-32B \
    --dataset-name hf \
    --dataset-path AI-MO/aimo-validation-aime \
    --num-prompts 10 \
    --seed 42
```

**`philschmid/mt-bench`**

``` bash
python3 vllm/benchmarks/benchmark_serving.py \
    --model Qwen/QwQ-32B \
    --dataset-name hf \
    --dataset-path philschmid/mt-bench \
    --num-prompts 80
```

### Running With Sampling Parameters

When using OpenAI-compatible backends such as `vllm`, optional sampling
parameters can be specified. Example client command:

```bash
python3 vllm/benchmarks/benchmark_serving.py \
  --backend vllm \
  --model NousResearch/Hermes-3-Llama-3.1-8B \
  --endpoint /v1/completions \
  --dataset-name sharegpt \
  --dataset-path <your data path>/ShareGPT_V3_unfiltered_cleaned_split.json \
  --top-k 10 \
  --top-p 0.9 \
  --temperature 0.5 \
  --num-prompts 10
```

---
## Example - Offline Throughput Benchmark

```bash
python3 vllm/benchmarks/benchmark_throughput.py \
  --model NousResearch/Hermes-3-Llama-3.1-8B \
  --dataset-name sonnet \
  --dataset-path vllm/benchmarks/sonnet.txt \
  --num-prompts 10
```

If successful, you will see the following output

```
Throughput: 7.15 requests/s, 4656.00 total tokens/s, 1072.15 output tokens/s
Total num prompt tokens:  5014
Total num output tokens:  1500
```

### VisionArena Benchmark for Vision Language Models

``` bash
python3 vllm/benchmarks/benchmark_throughput.py \
  --model Qwen/Qwen2-VL-7B-Instruct \
  --backend vllm-chat \
  --dataset-name hf \
  --dataset-path lmarena-ai/VisionArena-Chat \
  --num-prompts 1000 \
  --hf-split train
```

The `num prompt tokens` now includes image token counts

```
Throughput: 2.55 requests/s, 4036.92 total tokens/s, 326.90 output tokens/s
Total num prompt tokens:  14527
Total num output tokens:  1280
```

### InstructCoder Benchmark with Speculative Decoding

``` bash
VLLM_WORKER_MULTIPROC_METHOD=spawn \
VLLM_USE_V1=1 \
python3 vllm/benchmarks/benchmark_throughput.py \
    --dataset-name=hf \
    --dataset-path=likaixin/InstructCoder \
    --model=meta-llama/Meta-Llama-3-8B-Instruct \
    --input-len=1000 \
    --output-len=100 \
    --num-prompts=2048 \
    --async-engine \
    --speculative-config $'{"method": "ngram",
    "num_speculative_tokens": 5, "prompt_lookup_max": 5,
    "prompt_lookup_min": 2}'
```

```
Throughput: 104.77 requests/s, 23836.22 total tokens/s, 10477.10 output tokens/s
Total num prompt tokens:  261136
Total num output tokens:  204800
```

### Other HuggingFaceDataset Examples

**`lmms-lab/LLaVA-OneVision-Data`**

```bash
python3 vllm/benchmarks/benchmark_throughput.py \
  --model Qwen/Qwen2-VL-7B-Instruct \
  --backend vllm-chat \
  --dataset-name hf \
  --dataset-path lmms-lab/LLaVA-OneVision-Data \
  --hf-split train \
  --hf-subset "chart2text(cauldron)" \
  --num-prompts 10
```

**`Aeala/ShareGPT_Vicuna_unfiltered`**

```bash
python3 vllm/benchmarks/benchmark_throughput.py \
  --model Qwen/Qwen2-VL-7B-Instruct \
  --backend vllm-chat \
  --dataset-name hf \
  --dataset-path Aeala/ShareGPT_Vicuna_unfiltered \
  --hf-split train \
  --num-prompts 10
```

**`AI-MO/aimo-validation-aime`**

```bash
python3 benchmarks/benchmark_throughput.py \
  --model Qwen/QwQ-32B \
  --backend vllm \
  --dataset-name hf \
  --dataset-path AI-MO/aimo-validation-aime \
  --hf-split train \
  --num-prompts 10
```

### Benchmark with LoRA Adapters

``` bash
# download dataset
# wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
python3 vllm/benchmarks/benchmark_throughput.py \
  --model meta-llama/Llama-2-7b-hf \
  --backend vllm \
  --dataset_path <your data path>/ShareGPT_V3_unfiltered_cleaned_split.json \
  --dataset_name sharegpt \
  --num-prompts 10 \
  --max-loras 2 \
  --max-lora-rank 8 \
  --enable-lora \
  --lora-path yard1/llama-2-7b-sql-lora-test
  ```
