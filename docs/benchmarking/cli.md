# Benchmark CLI

This section guides you through running benchmark tests with the extensive datasets supported on vLLM.

It's a living document, updated as new features and datasets become available.

## Dataset Overview

<style>
th {
  min-width: 0 !important;
}
</style>

| Dataset | Online | Offline | Data Path |
|---------|--------|---------|-----------|
| ShareGPT | ✅ | ✅ | `wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json` |
| ShareGPT4V (Image) | ✅ | ✅ | `wget https://huggingface.co/datasets/Lin-Chen/ShareGPT4V/resolve/main/sharegpt4v_instruct_gpt4-vision_cap100k.json`<br>Note that the images need to be downloaded separately. For example, to download COCO's 2017 Train images:<br>`wget http://images.cocodataset.org/zips/train2017.zip` |
| ShareGPT4Video (Video) | ✅ | ✅ | `git clone https://huggingface.co/datasets/ShareGPT4Video/ShareGPT4Video` |
| BurstGPT | ✅ | ✅ | `wget https://github.com/HPMLL/BurstGPT/releases/download/v1.1/BurstGPT_without_fails_2.csv` |
| Sonnet (deprecated) | ✅ | ✅ | Local file: `benchmarks/sonnet.txt` |
| Random | ✅ | ✅ | `synthetic` |
| RandomMultiModal (Image/Video) | ✅ | ✅ | `synthetic` |
| RandomForReranking | ✅ | ✅ | `synthetic` |
| Prefix Repetition | ✅ | ✅ | `synthetic` |
| HuggingFace-VisionArena | ✅ | ✅ | `lmarena-ai/VisionArena-Chat` |
| HuggingFace-MMVU | ✅ | ✅ | `yale-nlp/MMVU` |
| HuggingFace-InstructCoder | ✅ | ✅ | `likaixin/InstructCoder` |
| HuggingFace-AIMO | ✅ | ✅ | `AI-MO/aimo-validation-aime`, `AI-MO/NuminaMath-1.5`, `AI-MO/NuminaMath-CoT` |
| HuggingFace-Other | ✅ | ✅ | `lmms-lab/LLaVA-OneVision-Data`, `Aeala/ShareGPT_Vicuna_unfiltered` |
| HuggingFace-MTBench | ✅ | ✅ | `philschmid/mt-bench` |
| HuggingFace-Blazedit | ✅ | ✅ | `vdaita/edit_5k_char`, `vdaita/edit_10k_char` |
| Spec Bench | ✅ | ✅ | `wget https://raw.githubusercontent.com/hemingkx/Spec-Bench/refs/heads/main/data/spec_bench/question.jsonl` |
| Custom | ✅ | ✅ | Local file: `data.jsonl` |
| Custom MM | ✅ | ✅ | Local file: `mm_data.jsonl` |

Legend:

- ✅ - supported

!!! note
    HuggingFace dataset's `dataset-name` should be set to `hf`.
    For local `dataset-path`, please set `hf-name` to its Hugging Face ID like

    ```bash
    --dataset-path /datasets/VisionArena-Chat/ --hf-name lmarena-ai/VisionArena-Chat
    ```

## Examples

### 🚀 Online Benchmark

<details class="admonition abstract" markdown="1">
<summary>Show more</summary>

First start serving your model:

```bash
vllm serve NousResearch/Hermes-3-Llama-3.1-8B
```

Then run the benchmarking script:

```bash
# download dataset
# wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
vllm bench serve \
  --backend vllm \
  --model NousResearch/Hermes-3-Llama-3.1-8B \
  --endpoint /v1/completions \
  --dataset-name sharegpt \
  --dataset-path <your data path>/ShareGPT_V3_unfiltered_cleaned_split.json \
  --num-prompts 10
```

If successful, you will see the following output:

```text
============ Serving Benchmark Result ============
Successful requests:                     10
Benchmark duration (s):                  5.78
Total input tokens:                      1369
Total generated tokens:                  2212
Request throughput (req/s):              1.73
Output token throughput (tok/s):         382.89
Total token throughput (tok/s):          619.85
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

#### Custom Dataset

If the dataset you want to benchmark is not supported yet in vLLM, even then you can benchmark on it using `CustomDataset`. Your data needs to be in `.jsonl` format and needs to have "prompt" field per entry, e.g., data.jsonl

```json
{"prompt": "What is the capital of India?"}
{"prompt": "What is the capital of Iran?"}
{"prompt": "What is the capital of China?"}
```

```bash
# start server
vllm serve meta-llama/Llama-3.1-8B-Instruct
```

```bash
# run benchmarking script
vllm bench serve --port 9001 --save-result --save-detailed \
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

#### Custom multimodal dataset

If the multimodal dataset you want to benchmark is not supported yet in vLLM, then you can benchmark on it using `CustomMMDataset`. Your data needs to be in `.jsonl` format and needs to have "prompt" and "image_files" field per entry, e.g., `mm_data.jsonl`:

```json
{"prompt": "How many animals are present in the given image?", "image_files": ["/path/to/image/folder/horsepony.jpg"]}
{"prompt": "What colour is the bird shown in the image?", "image_files": ["/path/to/image/folder/flycatcher.jpeg"]}
```

```bash
# need a model with vision capability here
vllm serve Qwen/Qwen2-VL-7B-Instruct
```

```bash
# run benchmarking script
vllm bench serve--save-result --save-detailed \
  --backend openai-chat \
  --model Qwen/Qwen2-VL-7B-Instruct \
  --endpoint /v1/chat/completions \
  --dataset-name custom_mm \
  --dataset-path <path-to-your-mm-data-jsonl> \
  --allowed-local-media-path /path/to/image/folder
```

Note that we need to use the `openai-chat` backend and `/v1/chat/completions` endpoint for multimodal inputs.

#### VisionArena Benchmark for Vision Language Models

```bash
# need a model with vision capability here
vllm serve Qwen/Qwen2-VL-7B-Instruct
```

```bash
vllm bench serve \
  --backend openai-chat \
  --model Qwen/Qwen2-VL-7B-Instruct \
  --endpoint /v1/chat/completions \
  --dataset-name hf \
  --dataset-path lmarena-ai/VisionArena-Chat \
  --hf-split train \
  --num-prompts 1000
```

#### InstructCoder Benchmark with Speculative Decoding

``` bash
vllm serve meta-llama/Meta-Llama-3-8B-Instruct \
    --speculative-config $'{"method": "ngram",
    "num_speculative_tokens": 5, "prompt_lookup_max": 5,
    "prompt_lookup_min": 2}'
```

``` bash
vllm bench serve \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --dataset-name hf \
    --dataset-path likaixin/InstructCoder \
    --num-prompts 2048
```

#### Spec Bench Benchmark with Speculative Decoding

``` bash
vllm serve meta-llama/Meta-Llama-3-8B-Instruct \
    --speculative-config $'{"method": "ngram",
    "num_speculative_tokens": 5, "prompt_lookup_max": 5,
    "prompt_lookup_min": 2}'
```

[SpecBench dataset](https://github.com/hemingkx/Spec-Bench)

Run all categories:

``` bash
# Download the dataset using:
# wget https://raw.githubusercontent.com/hemingkx/Spec-Bench/refs/heads/main/data/spec_bench/question.jsonl

vllm bench serve \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --dataset-name spec_bench \
    --dataset-path "<YOUR_DOWNLOADED_PATH>/data/spec_bench/question.jsonl" \
    --num-prompts -1
```

Available categories include `[writing, roleplay, reasoning, math, coding, extraction, stem, humanities, translation, summarization, qa, math_reasoning, rag]`.

Run only a specific category like "summarization":

``` bash
vllm bench serve \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --dataset-name spec_bench \
    --dataset-path "<YOUR_DOWNLOADED_PATH>/data/spec_bench/question.jsonl" \
    --num-prompts -1
    --spec-bench-category "summarization"
```

#### Other HuggingFaceDataset Examples

```bash
vllm serve Qwen/Qwen2-VL-7B-Instruct
```

`lmms-lab/LLaVA-OneVision-Data`:

```bash
vllm bench serve \
  --backend openai-chat \
  --model Qwen/Qwen2-VL-7B-Instruct \
  --endpoint /v1/chat/completions \
  --dataset-name hf \
  --dataset-path lmms-lab/LLaVA-OneVision-Data \
  --hf-split train \
  --hf-subset "chart2text(cauldron)" \
  --num-prompts 10
```

`Aeala/ShareGPT_Vicuna_unfiltered`:

```bash
vllm bench serve \
  --backend openai-chat \
  --model Qwen/Qwen2-VL-7B-Instruct \
  --endpoint /v1/chat/completions \
  --dataset-name hf \
  --dataset-path Aeala/ShareGPT_Vicuna_unfiltered \
  --hf-split train \
  --num-prompts 10
```

`AI-MO/aimo-validation-aime`:

``` bash
vllm bench serve \
    --model Qwen/QwQ-32B \
    --dataset-name hf \
    --dataset-path AI-MO/aimo-validation-aime \
    --num-prompts 10 \
    --seed 42
```

`philschmid/mt-bench`:

``` bash
vllm bench serve \
    --model Qwen/QwQ-32B \
    --dataset-name hf \
    --dataset-path philschmid/mt-bench \
    --num-prompts 80
```

`vdaita/edit_5k_char` or `vdaita/edit_10k_char`:

``` bash
vllm bench serve \
    --model Qwen/QwQ-32B \
    --dataset-name hf \
    --dataset-path vdaita/edit_5k_char \
    --num-prompts 90 \
    --blazedit-min-distance 0.01 \
    --blazedit-max-distance 0.99
```

#### Running With Sampling Parameters

When using OpenAI-compatible backends such as `vllm`, optional sampling
parameters can be specified. Example client command:

```bash
vllm bench serve \
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

#### Running With Ramp-Up Request Rate

The benchmark tool also supports ramping up the request rate over the
duration of the benchmark run. This can be useful for stress testing the
server or finding the maximum throughput that it can handle, given some latency budget.

Two ramp-up strategies are supported:

- `linear`: Increases the request rate linearly from a start value to an end value.
- `exponential`: Increases the request rate exponentially.

The following arguments can be used to control the ramp-up:

- `--ramp-up-strategy`: The ramp-up strategy to use (`linear` or `exponential`).
- `--ramp-up-start-rps`: The request rate at the beginning of the benchmark.
- `--ramp-up-end-rps`: The request rate at the end of the benchmark.

#### Load Pattern Configuration

vLLM's benchmark serving script provides sophisticated load pattern simulation capabilities through three key parameters that control request generation and concurrency behavior:

##### Load Pattern Control Parameters

- `--request-rate`: Controls the target request generation rate (requests per second). Set to `inf` for maximum throughput testing or finite values for controlled load simulation.
- `--burstiness`: Controls traffic variability using a Gamma distribution (range: > 0). Lower values create bursty traffic, higher values create uniform traffic.
- `--max-concurrency`: Limits concurrent outstanding requests. If this argument is not provided, concurrency is unlimited. Set a value to simulate backpressure.

These parameters work together to create realistic load patterns with carefully chosen defaults. The `--request-rate` parameter defaults to `inf` (infinite), which sends all requests immediately for maximum throughput testing. When set to finite values, it uses either a Poisson process (default `--burstiness=1.0`) or Gamma distribution for realistic request timing. The `--burstiness` parameter only takes effect when `--request-rate` is not infinite - a value of 1.0 creates natural Poisson traffic, while lower values (0.1-0.5) create bursty patterns and higher values (2.0-5.0) create uniform spacing. The `--max-concurrency` parameter defaults to `None` (unlimited) but can be set to simulate real-world constraints where a load balancer or API gateway limits concurrent connections. When combined, these parameters allow you to simulate everything from unrestricted stress testing (`--request-rate=inf`) to production-like scenarios with realistic arrival patterns and resource constraints.

The `--burstiness` parameter mathematically controls request arrival patterns using a Gamma distribution where:

- Shape parameter: `burstiness` value
- Coefficient of Variation (CV): $\frac{1}{\sqrt{burstiness}}$
- Traffic characteristics:
    - `burstiness = 0.1`: Highly bursty traffic (CV ≈ 3.16) - stress testing
    - `burstiness = 1.0`: Natural Poisson traffic (CV = 1.0) - realistic simulation  
    - `burstiness = 5.0`: Uniform traffic (CV ≈ 0.45) - controlled load testing

![Load Pattern Examples](../assets/contributing/load-pattern-examples.png)

*Figure: Load pattern examples for each use case. Top row: Request arrival timelines showing cumulative requests over time. Bottom row: Inter-arrival time distributions showing traffic variability patterns. Each column represents a different use case with its specific parameter settings and resulting traffic characteristics.*

Load Pattern Recommendations by Use Case:

| Use Case           | Burstiness   | Request Rate    | Max Concurrency | Description                                               |
| ---                | ---          | ---             | ---             | ---                                                       |
| Maximum Throughput | N/A          | Infinite        | Limited         | **Most common**: Simulates load balancer/gateway limits with unlimited user demand |
| Realistic Testing  | 1.0          | Moderate (5-20) | Infinite        | Natural Poisson traffic patterns for baseline performance |
| Stress Testing     | 0.1-0.5      | High (20-100)   | Infinite        | Challenging burst patterns to test resilience             |
| Latency Profiling  | 2.0-5.0      | Low (1-10)      | Infinite        | Uniform load for consistent timing analysis               |
| Capacity Planning  | 1.0          | Variable        | Limited         | Test resource limits with realistic constraints           |
| SLA Validation     | 1.0          | Target rate     | SLA limit       | Production-like constraints for compliance testing        |

These load patterns help evaluate different aspects of your vLLM deployment, from basic performance characteristics to resilience under challenging traffic conditions.

The **Maximum Throughput** pattern (`--request-rate=inf --max-concurrency=<limit>`) is the most commonly used configuration for production benchmarking. This simulates real-world deployment architectures where:

- Users send requests as fast as they can (infinite rate)
- A load balancer or API gateway controls the maximum concurrent connections
- The system operates at its concurrency limit, revealing true throughput capacity
- `--burstiness` has no effect since request timing is not controlled when rate is infinite

This pattern helps determine optimal concurrency settings for your production load balancer configuration.

To effectively configure load patterns, especially for **Capacity Planning** and **SLA Validation** use cases, you need to understand your system's resource limits. During startup, vLLM reports KV cache configuration that directly impacts your load testing parameters:

```text
GPU KV cache size: 15,728,640 tokens
Maximum concurrency for 8,192 tokens per request: 1920
```

Where:

- GPU KV cache size: Total tokens that can be cached across all concurrent requests
- Maximum concurrency: Theoretical maximum concurrent requests for the given `max_model_len`
- Calculation: `max_concurrency = kv_cache_size / max_model_len`

Using KV cache metrics for load pattern configuration:

- For Capacity Planning: Set `--max-concurrency` to 80-90% of the reported maximum to test realistic resource constraints
- For SLA Validation: Use the reported maximum as your SLA limit to ensure compliance testing matches production capacity
- For Realistic Testing: Monitor memory usage when approaching theoretical limits to understand sustainable request rates
- Request rate guidance: Use the KV cache size to estimate sustainable request rates for your specific workload and sequence lengths

</details>

### 📈 Offline Throughput Benchmark

<details class="admonition abstract" markdown="1">
<summary>Show more</summary>

```bash
vllm bench throughput \
  --model NousResearch/Hermes-3-Llama-3.1-8B \
  --dataset-name sonnet \
  --dataset-path vllm/benchmarks/sonnet.txt \
  --num-prompts 10
```

If successful, you will see the following output

```text
Throughput: 7.15 requests/s, 4656.00 total tokens/s, 1072.15 output tokens/s
Total num prompt tokens:  5014
Total num output tokens:  1500
```

#### VisionArena Benchmark for Vision Language Models

```bash
vllm bench throughput \
  --model Qwen/Qwen2-VL-7B-Instruct \
  --backend vllm-chat \
  --dataset-name hf \
  --dataset-path lmarena-ai/VisionArena-Chat \
  --num-prompts 1000 \
  --hf-split train
```

The `num prompt tokens` now includes image token counts

```text
Throughput: 2.55 requests/s, 4036.92 total tokens/s, 326.90 output tokens/s
Total num prompt tokens:  14527
Total num output tokens:  1280
```

#### InstructCoder Benchmark with Speculative Decoding

``` bash
VLLM_WORKER_MULTIPROC_METHOD=spawn \
vllm bench throughput \
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

```text
Throughput: 104.77 requests/s, 23836.22 total tokens/s, 10477.10 output tokens/s
Total num prompt tokens:  261136
Total num output tokens:  204800
```

#### Other HuggingFaceDataset Examples

`lmms-lab/LLaVA-OneVision-Data`:

```bash
vllm bench throughput \
  --model Qwen/Qwen2-VL-7B-Instruct \
  --backend vllm-chat \
  --dataset-name hf \
  --dataset-path lmms-lab/LLaVA-OneVision-Data \
  --hf-split train \
  --hf-subset "chart2text(cauldron)" \
  --num-prompts 10
```

`Aeala/ShareGPT_Vicuna_unfiltered`:

```bash
vllm bench throughput \
  --model Qwen/Qwen2-VL-7B-Instruct \
  --backend vllm-chat \
  --dataset-name hf \
  --dataset-path Aeala/ShareGPT_Vicuna_unfiltered \
  --hf-split train \
  --num-prompts 10
```

`AI-MO/aimo-validation-aime`:

```bash
vllm bench throughput \
  --model Qwen/QwQ-32B \
  --backend vllm \
  --dataset-name hf \
  --dataset-path AI-MO/aimo-validation-aime \
  --hf-split train \
  --num-prompts 10
```

Benchmark with LoRA adapters:

``` bash
# download dataset
# wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
vllm bench throughput \
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

#### Synthetic Random Multimodal (random-mm)

Generate synthetic multimodal inputs for offline throughput testing without external datasets.
Use `--backend vllm-chat` so that image tokens are counted correctly.

```bash
vllm bench throughput \
  --model Qwen/Qwen2-VL-7B-Instruct \
  --backend vllm-chat \
  --dataset-name random-mm \
  --num-prompts 100 \
  --random-input-len 300 \
  --random-output-len 40 \
  --random-mm-base-items-per-request 2 \
  --random-mm-limit-mm-per-prompt '{"image": 3, "video": 0}' \
  --random-mm-bucket-config '{(256, 256, 1): 0.7, (720, 1280, 1): 0.3}'
```

</details>

### 🛠️ Structured Output Benchmark

<details class="admonition abstract" markdown="1">
<summary>Show more</summary>

Benchmark the performance of structured output generation (JSON, grammar, regex).

#### Server Setup

```bash
vllm serve NousResearch/Hermes-3-Llama-3.1-8B
```

#### JSON Schema Benchmark

```bash
python3 benchmarks/benchmark_serving_structured_output.py \
  --backend vllm \
  --model NousResearch/Hermes-3-Llama-3.1-8B \
  --dataset json \
  --structured-output-ratio 1.0 \
  --request-rate 10 \
  --num-prompts 1000
```

#### Grammar-based Generation Benchmark

```bash
python3 benchmarks/benchmark_serving_structured_output.py \
  --backend vllm \
  --model NousResearch/Hermes-3-Llama-3.1-8B \
  --dataset grammar \
  --structure-type grammar \
  --request-rate 10 \
  --num-prompts 1000
```

#### Regex-based Generation Benchmark

```bash
python3 benchmarks/benchmark_serving_structured_output.py \
  --backend vllm \
  --model NousResearch/Hermes-3-Llama-3.1-8B \
  --dataset regex \
  --request-rate 10 \
  --num-prompts 1000
```

#### Choice-based Generation Benchmark

```bash
python3 benchmarks/benchmark_serving_structured_output.py \
  --backend vllm \
  --model NousResearch/Hermes-3-Llama-3.1-8B \
  --dataset choice \
  --request-rate 10 \
  --num-prompts 1000
```

#### XGrammar Benchmark Dataset

```bash
python3 benchmarks/benchmark_serving_structured_output.py \
  --backend vllm \
  --model NousResearch/Hermes-3-Llama-3.1-8B \
  --dataset xgrammar_bench \
  --request-rate 10 \
  --num-prompts 1000
```

</details>

### 📚 Long Document QA Benchmark

<details class="admonition abstract" markdown="1">
<summary>Show more</summary>

Benchmark the performance of long document question-answering with prefix caching.

#### Basic Long Document QA Test

```bash
python3 benchmarks/benchmark_long_document_qa_throughput.py \
  --model meta-llama/Llama-2-7b-chat-hf \
  --enable-prefix-caching \
  --num-documents 16 \
  --document-length 2000 \
  --output-len 50 \
  --repeat-count 5
```

#### Different Repeat Modes

```bash
# Random mode (default) - shuffle prompts randomly
python3 benchmarks/benchmark_long_document_qa_throughput.py \
  --model meta-llama/Llama-2-7b-chat-hf \
  --enable-prefix-caching \
  --num-documents 8 \
  --document-length 3000 \
  --repeat-count 3 \
  --repeat-mode random

# Tile mode - repeat entire prompt list in sequence
python3 benchmarks/benchmark_long_document_qa_throughput.py \
  --model meta-llama/Llama-2-7b-chat-hf \
  --enable-prefix-caching \
  --num-documents 8 \
  --document-length 3000 \
  --repeat-count 3 \
  --repeat-mode tile

# Interleave mode - repeat each prompt consecutively
python3 benchmarks/benchmark_long_document_qa_throughput.py \
  --model meta-llama/Llama-2-7b-chat-hf \
  --enable-prefix-caching \
  --num-documents 8 \
  --document-length 3000 \
  --repeat-count 3 \
  --repeat-mode interleave
```

</details>

### 🗂️ Prefix Caching Benchmark

<details class="admonition abstract" markdown="1">
<summary>Show more</summary>

Benchmark the efficiency of automatic prefix caching.

#### Fixed Prompt with Prefix Caching

```bash
python3 benchmarks/benchmark_prefix_caching.py \
  --model meta-llama/Llama-2-7b-chat-hf \
  --enable-prefix-caching \
  --num-prompts 1 \
  --repeat-count 100 \
  --input-length-range 128:256
```

#### ShareGPT Dataset with Prefix Caching

```bash
# download dataset
# wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json

python3 benchmarks/benchmark_prefix_caching.py \
  --model meta-llama/Llama-2-7b-chat-hf \
  --dataset-path /path/ShareGPT_V3_unfiltered_cleaned_split.json \
  --enable-prefix-caching \
  --num-prompts 20 \
  --repeat-count 5 \
  --input-length-range 128:256
```

##### Prefix Repetition Dataset

```bash
vllm bench serve \
  --backend openai \
  --model meta-llama/Llama-2-7b-chat-hf \
  --dataset-name prefix_repetition \
  --num-prompts 100 \
  --prefix-repetition-prefix-len 512 \
  --prefix-repetition-suffix-len 128 \
  --prefix-repetition-num-prefixes 5 \
  --prefix-repetition-output-len 128
```

</details>

### 🧪 Hashing Benchmarks

<details class="admonition abstract" markdown="1">
<summary>Show more</summary>

Two helper scripts live in `benchmarks/` to compare hashing options used by prefix caching and related utilities. They are standalone (no server required) and help choose a hash algorithm before enabling prefix caching in production.

- `benchmarks/benchmark_hash.py`: Micro-benchmark that measures per-call latency of three implementations on a representative `(bytes, tuple[int])` payload.

```bash
python benchmarks/benchmark_hash.py --iterations 20000 --seed 42
```

- `benchmarks/benchmark_prefix_block_hash.py`: End-to-end block hashing benchmark that runs the full prefix-cache hash pipeline (`hash_block_tokens`) across many fake blocks and reports throughput.

```bash
python benchmarks/benchmark_prefix_block_hash.py --num-blocks 20000 --block-size 32 --trials 5
```

Supported algorithms: `sha256`, `sha256_cbor`, `xxhash`, `xxhash_cbor`. Install optional deps to exercise all variants:

```bash
uv pip install xxhash cbor2
```

If an algorithm’s dependency is missing, the script will skip it and continue.

</details>

### ⚡ Request Prioritization Benchmark

<details class="admonition abstract" markdown="1">
<summary>Show more</summary>

Benchmark the performance of request prioritization in vLLM.

#### Basic Prioritization Test

```bash
python3 benchmarks/benchmark_prioritization.py \
  --model meta-llama/Llama-2-7b-chat-hf \
  --input-len 128 \
  --output-len 64 \
  --num-prompts 100 \
  --scheduling-policy priority
```

#### Multiple Sequences per Prompt

```bash
python3 benchmarks/benchmark_prioritization.py \
  --model meta-llama/Llama-2-7b-chat-hf \
  --input-len 128 \
  --output-len 64 \
  --num-prompts 100 \
  --scheduling-policy priority \
  --n 2
```

</details>

### 👁️ Multi-Modal Benchmark

<details class="admonition abstract" markdown="1">
<summary>Show more</summary>

Benchmark the performance of multi-modal requests in vLLM.

#### Images (ShareGPT4V)

Start vLLM:

```bash
vllm serve Qwen/Qwen2.5-VL-7B-Instruct \
  --dtype bfloat16 \
  --limit-mm-per-prompt '{"image": 1}' \
  --allowed-local-media-path /path/to/sharegpt4v/images
```

Send requests with images:

```bash
vllm bench serve \
  --backend openai-chat \
  --model Qwen/Qwen2.5-VL-7B-Instruct \
  --dataset-name sharegpt \
  --dataset-path /path/to/ShareGPT4V/sharegpt4v_instruct_gpt4-vision_cap100k.json \
  --num-prompts 100 \
  --save-result \
  --result-dir ~/vllm_benchmark_results \
  --save-detailed \
  --endpoint /v1/chat/completions
```

#### Videos (ShareGPT4Video)

Start vLLM:

```bash
vllm serve Qwen/Qwen2.5-VL-7B-Instruct \
  --dtype bfloat16 \
  --limit-mm-per-prompt '{"video": 1}' \
  --allowed-local-media-path /path/to/sharegpt4video/videos
```

Send requests with videos:

```bash
vllm bench serve \
  --backend openai-chat \
  --model Qwen/Qwen2.5-VL-7B-Instruct \
  --dataset-name sharegpt \
  --dataset-path /path/to/ShareGPT4Video/llava_v1_5_mix665k_with_video_chatgpt72k_share4video28k.json \
  --num-prompts 100 \
  --save-result \
  --result-dir ~/vllm_benchmark_results \
  --save-detailed \
  --endpoint /v1/chat/completions
```

#### Synthetic Random Images (random-mm)

Generate synthetic image inputs alongside random text prompts to stress-test vision models without external datasets.

Notes:

- For online benchmarks, use `--backend openai-chat` with endpoint `/v1/chat/completions`.
- For offline benchmarks, use `--backend vllm-chat` (see [Offline Throughput Benchmark](#offline-throughput-benchmark) for an example).

Start the server (example):

```bash
vllm serve Qwen/Qwen2.5-VL-3B-Instruct \
  --dtype bfloat16 \
  --max-model-len 16384 \
  --limit-mm-per-prompt '{"image": 3, "video": 0}' \
  --mm-processor-kwargs max_pixels=1003520
```

Benchmark. It is recommended to use the flag `--ignore-eos` to simulate real responses. You can set the size of the output via the arg `random-output-len`.

Ex.1: Fixed number of items and a single image resolution, enforcing generation of approx 40 tokens:

```bash
vllm bench serve \
  --backend openai-chat \
  --model Qwen/Qwen2.5-VL-3B-Instruct \
  --endpoint /v1/chat/completions \
  --dataset-name random-mm \
  --num-prompts 100 \
  --max-concurrency 10 \
  --random-prefix-len 25 \
  --random-input-len 300 \
  --random-output-len 40 \
  --random-range-ratio 0.2 \
  --random-mm-base-items-per-request 2 \
  --random-mm-limit-mm-per-prompt '{"image": 3, "video": 0}' \
  --random-mm-bucket-config '{(224, 224, 1): 1.0}' \
  --request-rate inf \
  --ignore-eos \
  --seed 42
```

The number of items per request can be controlled by passing multiple image buckets:

```bash
  --random-mm-base-items-per-request 2 \
  --random-mm-num-mm-items-range-ratio 0.5 \
  --random-mm-limit-mm-per-prompt '{"image": 4, "video": 0}' \
  --random-mm-bucket-config '{(256, 256, 1): 0.7, (720, 1280, 1): 0.3}' \
```

Flags specific to `random-mm`:

- `--random-mm-base-items-per-request`: base number of multimodal items per request.
- `--random-mm-num-mm-items-range-ratio`: vary item count uniformly in the closed integer range [floor(n·(1−r)), ceil(n·(1+r))]. Set r=0 to keep it fixed; r=1 allows 0 items.
- `--random-mm-limit-mm-per-prompt`: per-modality hard caps, e.g. '{"image": 3, "video": 0}'.
- `--random-mm-bucket-config`: dict mapping (H, W, T) → probability. Entries with probability 0 are removed; remaining probabilities are renormalized to sum to 1. Use T=1 for images. Set any T>1 for videos (video sampling not yet supported).

Behavioral notes:

- If the requested base item count cannot be satisfied under the provided per-prompt limits, the tool raises an error rather than silently clamping.

How sampling works:

- Determine per-request item count k by sampling uniformly from the integer range defined by `--random-mm-base-items-per-request` and `--random-mm-num-mm-items-range-ratio`, then clamp k to at most the sum of per-modality limits.
- For each of the k items, sample a bucket (H, W, T) according to the normalized probabilities in `--random-mm-bucket-config`, while tracking how many items of each modality have been added.
- If a modality (e.g., image) reaches its limit from `--random-mm-limit-mm-per-prompt`, all buckets of that modality are excluded and the remaining bucket probabilities are renormalized before continuing.
This should be seen as an edge case, and if this behavior can be avoided by setting `--random-mm-limit-mm-per-prompt` to a large number. Note that this might result in errors due to engine config `--limit-mm-per-prompt`.
- The resulting request contains synthetic image data in `multi_modal_data` (OpenAI Chat format). When `random-mm` is used with the OpenAI Chat backend, prompts remain text and MM content is attached via `multi_modal_data`.

</details>

### 🔬 Multimodal Processor Benchmark

Benchmark per-stage latency of the multimodal (MM) input processor pipeline, including the encoder forward pass. This is useful for profiling preprocessing bottlenecks in vision-language models.

<details class="admonition abstract" markdown="1">
<summary>Show more</summary>

The benchmark measures the following stages for each request:

| Stage | Description |
|-------|-------------|
| `hf_processor_time` | Time in the HuggingFace processor |
| `hashing_time` | Time hashing multimodal inputs |
| `cache_lookup_time` | Time looking up the processor cache |
| `prompt_update_time` | Time updating prompt tokens |
| `preprocessor_total_time` | Total preprocessing time |
| `encoder_forward_time` | Encoder model forward pass |

#### Basic Example with Synthetic Data (random-mm)

```bash
vllm bench mm-processor \
  --model Qwen/Qwen2-VL-7B-Instruct \
  --dataset-name random-mm \
  --num-prompts 50 \
  --random-input-len 300 \
  --random-output-len 40 \
  --random-mm-base-items-per-request 2 \
  --random-mm-limit-mm-per-prompt '{"image": 3, "video": 0}' \
  --random-mm-bucket-config '{(256, 256, 1): 0.7, (720, 1280, 1): 0.3}'
```

#### Using a HuggingFace Dataset

```bash
vllm bench mm-processor \
  --model Qwen/Qwen2-VL-7B-Instruct \
  --dataset-name hf \
  --dataset-path lmarena-ai/VisionArena-Chat \
  --hf-split train \
  --num-prompts 100
```

#### Warmup, Custom Percentiles, and JSON Output

```bash
vllm bench mm-processor \
  --model Qwen/Qwen2-VL-7B-Instruct \
  --dataset-name random-mm \
  --num-prompts 200 \
  --num-warmups 5 \
  --random-input-len 300 \
  --random-output-len 40 \
  --random-mm-base-items-per-request 1 \
  --metric-percentiles 50,90,95,99 \
  --output-json results.json
```

See [`vllm bench mm-processor`](../cli/bench/mm_processor.md) for the full argument reference.

</details>

### Embedding Benchmark

Benchmark the performance of embedding requests in vLLM.

<details class="admonition abstract" markdown="1">
<summary>Show more</summary>

#### Text Embeddings

Unlike generative models which use Completions API or Chat Completions API,
you should set `--backend openai-embeddings` and `--endpoint /v1/embeddings` to use the Embeddings API.

You can use any text dataset to benchmark the model, such as ShareGPT.

Start the server:

```bash
vllm serve jinaai/jina-embeddings-v3 --trust-remote-code
```

Run the benchmark:

```bash
# download dataset
# wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
vllm bench serve \
  --model jinaai/jina-embeddings-v3 \
  --backend openai-embeddings \
  --endpoint /v1/embeddings \
  --dataset-name sharegpt \
  --dataset-path <your data path>/ShareGPT_V3_unfiltered_cleaned_split.json
```

#### Multi-modal Embeddings

Unlike generative models which use Completions API or Chat Completions API,
you should set `--endpoint /v1/embeddings` to use the Embeddings API. The backend to use depends on the model:

- CLIP: `--backend openai-embeddings-clip`
- VLM2Vec: `--backend openai-embeddings-vlm2vec`

For other models, please add your own implementation inside [vllm/benchmarks/lib/endpoint_request_func.py](../../vllm/benchmarks/lib/endpoint_request_func.py) to match the expected instruction format.

You can use any text or multi-modal dataset to benchmark the model, as long as the model supports it.
For example, you can use ShareGPT and VisionArena to benchmark vision-language embeddings.

Serve and benchmark CLIP:

```bash
# Run this in another process
vllm serve openai/clip-vit-base-patch32

# Run these one by one after the server is up
# download dataset
# wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
vllm bench serve \
  --model openai/clip-vit-base-patch32 \
  --backend openai-embeddings-clip \
  --endpoint /v1/embeddings \
  --dataset-name sharegpt \
  --dataset-path <your data path>/ShareGPT_V3_unfiltered_cleaned_split.json

vllm bench serve \
  --model openai/clip-vit-base-patch32 \
  --backend openai-embeddings-clip \
  --endpoint /v1/embeddings \
  --dataset-name hf \
  --dataset-path lmarena-ai/VisionArena-Chat
```

Serve and benchmark VLM2Vec:

```bash
# Run this in another process
vllm serve TIGER-Lab/VLM2Vec-Full --runner pooling \
  --trust-remote-code \
  --chat-template examples/template_vlm2vec_phi3v.jinja

# Run these one by one after the server is up
# download dataset
# wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
vllm bench serve \
  --model TIGER-Lab/VLM2Vec-Full \
  --backend openai-embeddings-vlm2vec \
  --endpoint /v1/embeddings \
  --dataset-name sharegpt \
  --dataset-path <your data path>/ShareGPT_V3_unfiltered_cleaned_split.json

vllm bench serve \
  --model TIGER-Lab/VLM2Vec-Full \
  --backend openai-embeddings-vlm2vec \
  --endpoint /v1/embeddings \
  --dataset-name hf \
  --dataset-path lmarena-ai/VisionArena-Chat
```

</details>

### Reranker Benchmark

Benchmark the performance of rerank requests in vLLM.

<details class="admonition abstract" markdown="1">
<summary>Show more</summary>

Unlike generative models which use Completions API or Chat Completions API,
you should set `--backend vllm-rerank` and `--endpoint /v1/rerank` to use the Reranker API.

For reranking, the only supported dataset is `--dataset-name random-rerank`

Start the server:

```bash
vllm serve BAAI/bge-reranker-v2-m3
```

Run the benchmark:

```bash
vllm bench serve \
  --model BAAI/bge-reranker-v2-m3 \
  --backend vllm-rerank \
  --endpoint /v1/rerank \
  --dataset-name random-rerank \
  --tokenizer BAAI/bge-reranker-v2-m3 \
  --random-input-len 512 \
  --num-prompts 10 \
  --random-batch-size 5
```

For reranker models, this will create `num_prompts / random_batch_size` requests with
`random_batch_size` "documents" where each one has close to `random_input_len` tokens.
In the example above, this results in 2 rerank requests with 5 "documents" each where
each document has close to 512 tokens.

Please note that the `/v1/rerank` is also supported by embedding models. So if you're running
with an embedding model, also set `--no_reranker`. Because in this case the query is
treated as an individual prompt by the server, here we send `random_batch_size - 1` documents
to account for the extra prompt which is the query. The token accounting to report the
throughput numbers correctly is also adjusted.

</details>
