---
toc_depth: 4
---

# Benchmark Suites

vLLM provides comprehensive benchmarking tools for performance testing and evaluation:

- **[Benchmark CLI]**: `vllm bench` CLI tools and specialized benchmark scripts for interactive performance testing
- **[Performance benchmarks][performance-benchmarks]**: Automated CI benchmarks for development
- **[Nightly benchmarks][nightly-benchmarks]**: Comparative benchmarks against alternatives

[Benchmark CLI]: #benchmark-cli

## Benchmark CLI

This section guides you through running benchmark tests with the extensive
datasets supported on vLLM. It's a living document, updated as new features and datasets
become available.

### Dataset Overview

<style>
th {
  min-width: 0 !important;
}
</style>

| Dataset | Online | Offline | Data Path |
|---------|--------|---------|-----------|
| ShareGPT | ✅ | ✅ | `wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json` |
| ShareGPT4V (Image) | ✅ | ✅ | `wget https://huggingface.co/datasets/Lin-Chen/ShareGPT4V/blob/main/sharegpt4v_instruct_gpt4-vision_cap100k.json`<br>Note that the images need to be downloaded separately. For example, to download COCO's 2017 Train images:<br>`wget http://images.cocodataset.org/zips/train2017.zip` |
| ShareGPT4Video (Video) | ✅ | ✅ | `git clone https://huggingface.co/datasets/ShareGPT4Video/ShareGPT4Video` |
| BurstGPT | ✅ | ✅ | `wget https://github.com/HPMLL/BurstGPT/releases/download/v1.1/BurstGPT_without_fails_2.csv` |
| Sonnet (deprecated) | ✅ | ✅ | Local file: `benchmarks/sonnet.txt` |
| Random | ✅ | ✅ | `synthetic` |
| RandomMultiModal (Image/Video) | 🟡 | 🚧 | `synthetic` |
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

Legend:

- ✅ - supported
- 🟡 - Partial support
- 🚧 - to be supported

!!! note
    HuggingFace dataset's `dataset-name` should be set to `hf`.
    For local `dataset-path`, please set `hf-name` to its Hugging Face ID like

    ```bash
    --dataset-path /datasets/VisionArena-Chat/ --hf-name lmarena-ai/VisionArena-Chat
    ```

### Examples

#### 🚀 Online Benchmark

<details class="admonition abstract" markdown="1">
<summary>Show more</summary>

First start serving your model

```bash
vllm serve NousResearch/Hermes-3-Llama-3.1-8B
```

Then run the benchmarking script

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

If successful, you will see the following output

```text
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

##### Custom Dataset

If the dataset you want to benchmark is not supported yet in vLLM, even then you can benchmark on it using `CustomDataset`. Your data needs to be in `.jsonl` format and needs to have "prompt" field per entry, e.g., data.jsonl

```json
{"prompt": "What is the capital of India?"}
{"prompt": "What is the capital of Iran?"}
{"prompt": "What is the capital of China?"}
```

```bash
# start server
VLLM_USE_V1=1 vllm serve meta-llama/Llama-3.1-8B-Instruct
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

##### VisionArena Benchmark for Vision Language Models

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

##### InstructCoder Benchmark with Speculative Decoding

``` bash
VLLM_USE_V1=1 vllm serve meta-llama/Meta-Llama-3-8B-Instruct \
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

##### Spec Bench Benchmark with Speculative Decoding

``` bash
VLLM_USE_V1=1 vllm serve meta-llama/Meta-Llama-3-8B-Instruct \
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

##### Other HuggingFaceDataset Examples

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

##### Running With Sampling Parameters

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

##### Running With Ramp-Up Request Rate

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

</details>

#### 📈 Offline Throughput Benchmark

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

##### VisionArena Benchmark for Vision Language Models

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

##### InstructCoder Benchmark with Speculative Decoding

``` bash
VLLM_WORKER_MULTIPROC_METHOD=spawn \
VLLM_USE_V1=1 \
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

##### Other HuggingFaceDataset Examples

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

</details>

#### 🛠️ Structured Output Benchmark

<details class="admonition abstract" markdown="1">
<summary>Show more</summary>

Benchmark the performance of structured output generation (JSON, grammar, regex).

##### Server Setup

```bash
vllm serve NousResearch/Hermes-3-Llama-3.1-8B
```

##### JSON Schema Benchmark

```bash
python3 benchmarks/benchmark_serving_structured_output.py \
  --backend vllm \
  --model NousResearch/Hermes-3-Llama-3.1-8B \
  --dataset json \
  --structured-output-ratio 1.0 \
  --request-rate 10 \
  --num-prompts 1000
```

##### Grammar-based Generation Benchmark

```bash
python3 benchmarks/benchmark_serving_structured_output.py \
  --backend vllm \
  --model NousResearch/Hermes-3-Llama-3.1-8B \
  --dataset grammar \
  --structure-type grammar \
  --request-rate 10 \
  --num-prompts 1000
```

##### Regex-based Generation Benchmark

```bash
python3 benchmarks/benchmark_serving_structured_output.py \
  --backend vllm \
  --model NousResearch/Hermes-3-Llama-3.1-8B \
  --dataset regex \
  --request-rate 10 \
  --num-prompts 1000
```

##### Choice-based Generation Benchmark

```bash
python3 benchmarks/benchmark_serving_structured_output.py \
  --backend vllm \
  --model NousResearch/Hermes-3-Llama-3.1-8B \
  --dataset choice \
  --request-rate 10 \
  --num-prompts 1000
```

##### XGrammar Benchmark Dataset

```bash
python3 benchmarks/benchmark_serving_structured_output.py \
  --backend vllm \
  --model NousResearch/Hermes-3-Llama-3.1-8B \
  --dataset xgrammar_bench \
  --request-rate 10 \
  --num-prompts 1000
```

</details>

#### 📚 Long Document QA Benchmark

<details class="admonition abstract" markdown="1">
<summary>Show more</summary>

Benchmark the performance of long document question-answering with prefix caching.

##### Basic Long Document QA Test

```bash
python3 benchmarks/benchmark_long_document_qa_throughput.py \
  --model meta-llama/Llama-2-7b-chat-hf \
  --enable-prefix-caching \
  --num-documents 16 \
  --document-length 2000 \
  --output-len 50 \
  --repeat-count 5
```

##### Different Repeat Modes

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

#### 🗂️ Prefix Caching Benchmark

<details class="admonition abstract" markdown="1">
<summary>Show more</summary>

Benchmark the efficiency of automatic prefix caching.

##### Fixed Prompt with Prefix Caching

```bash
python3 benchmarks/benchmark_prefix_caching.py \
  --model meta-llama/Llama-2-7b-chat-hf \
  --enable-prefix-caching \
  --num-prompts 1 \
  --repeat-count 100 \
  --input-length-range 128:256
```

##### ShareGPT Dataset with Prefix Caching

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

#### ⚡ Request Prioritization Benchmark

<details class="admonition abstract" markdown="1">
<summary>Show more</summary>

Benchmark the performance of request prioritization in vLLM.

##### Basic Prioritization Test

```bash
python3 benchmarks/benchmark_prioritization.py \
  --model meta-llama/Llama-2-7b-chat-hf \
  --input-len 128 \
  --output-len 64 \
  --num-prompts 100 \
  --scheduling-policy priority
```

##### Multiple Sequences per Prompt

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

#### 👁️ Multi-Modal Benchmark

<details class="admonition abstract" markdown="1">
<summary>Show more</summary>

Benchmark the performance of multi-modal requests in vLLM.

##### Images (ShareGPT4V)

Start vLLM:

```bash
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-VL-7B-Instruct \
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
  --endpoint /v1/chat/completion
```

##### Videos (ShareGPT4Video)

Start vLLM:

```bash
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-VL-7B-Instruct \
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
  --endpoint /v1/chat/completion
```

##### Synthetic Random Images (random-mm)

Generate synthetic image inputs alongside random text prompts to stress-test vision models without external datasets.

Notes:

- Works only with online benchmark via the OpenAI  backend (`--backend openai-chat`) and endpoint `/v1/chat/completions`.
- Video sampling is not yet implemented.

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

[](){ #performance-benchmarks }

## Performance Benchmarks

The performance benchmarks are used for development to confirm whether new changes improve performance under various workloads. They are triggered on every commit with both the `perf-benchmarks` and `ready` labels, and when a PR is merged into vLLM.

### Manually Trigger the benchmark

Use [vllm-ci-test-repo images](https://gallery.ecr.aws/q9t5s3a7/vllm-ci-test-repo) with vLLM benchmark suite.
For CPU environment, please use the image with "-cpu" postfix.

Here is an example for docker run command for CPU.

```bash
docker run -it --entrypoint /bin/bash -v /data/huggingface:/root/.cache/huggingface  -e HF_TOKEN=''  --shm-size=16g --name vllm-cpu-ci  public.ecr.aws/q9t5s3a7/vllm-ci-test-repo:1da94e673c257373280026f75ceb4effac80e892-cpu
```

Then, run below command inside the docker instance.

```bash
bash .buildkite/nightly-benchmarks/scripts/run-performance-benchmarks.sh
```

When run, benchmark script generates results under **benchmark/results** folder, along with the benchmark_results.md and benchmark_results.json.

#### Runtime environment variables

- `ON_CPU`: set the value to '1' on Intel® Xeon® Processors. Default value is 0.
- `SERVING_JSON`: JSON file to use for the serving tests. Default value is empty string (use default file).
- `LATENCY_JSON`: JSON file to use for the latency tests. Default value is empty string (use default file).
- `THROUGHPUT_JSON`: JSON file to use for the throughout tests. Default value is empty string (use default file).
- `REMOTE_HOST`: IP for the remote vLLM service to benchmark. Default value is empty string.
- `REMOTE_PORT`: Port for the remote vLLM service to benchmark. Default value is empty string.

For more results visualization, check the [visualizing the results](https://github.com/intel-ai-tce/vllm/blob/more_cpu_models/.buildkite/nightly-benchmarks/README.md#visualizing-the-results).

The latest performance results are hosted on the public [vLLM Performance Dashboard](https://hud.pytorch.org/benchmark/llms?repoName=vllm-project%2Fvllm).

More information on the performance benchmarks and their parameters can be found in [Benchmark README](https://github.com/intel-ai-tce/vllm/blob/more_cpu_models/.buildkite/nightly-benchmarks/README.md) and [performance benchmark description](gh-file:.buildkite/nightly-benchmarks/performance-benchmarks-descriptions.md).

[](){ #nightly-benchmarks }

## Nightly Benchmarks

These compare vLLM's performance against alternatives (`tgi`, `trt-llm`, and `lmdeploy`) when there are major updates of vLLM (e.g., bumping up to a new version). They are primarily intended for consumers to evaluate when to choose vLLM over other options and are triggered on every commit with both the `perf-benchmarks` and `nightly-benchmarks` labels.

The latest nightly benchmark results are shared in major release blog posts such as [vLLM v0.6.0](https://blog.vllm.ai/2024/09/05/perf-update.html).

More information on the nightly benchmarks and their parameters can be found [here](gh-file:.buildkite/nightly-benchmarks/nightly-descriptions.md).
