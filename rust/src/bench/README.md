# vllm-bench

High-performance Rust benchmark client for vLLM serving endpoints. A drop-in replacement for `vllm bench serve` with near-instant startup, parallel dataset generation, and a fraction of the memory overhead — and no Python at runtime.

```bash
vllm-bench --backend vllm --base-url http://127.0.0.1:8000 \
  --model <model> --dataset-name random \
  --random-input-len 1024 --random-output-len 128 \
  --num-prompts 1000 --max-concurrency 200
```

## Highlights

- **Fast** — ~7 ms startup, single ~7 MB static binary, no Python imports.
- **Scales** — `Arc<str>` prompt sharing + mimalloc keep memory <100 MB at 1400+ concurrency.
- **Many datasets** — `random`, `random-mm` (VLM), `sharegpt`, `sonnet`, `speed-bench`, and any HuggingFace dataset.
- **Many backends** — completions, chat, embeddings, pooling, and rerank.
- **Beyond a single run** — concurrency/rate **sweeps**, **multi-run** stats, **multi-turn** conversations, **LoRA** multi-adapter, and result **comparison**.
- **Steady-state metrics** — throughput/latency measured over the saturated plateau, excluding ramp-up and drain.
- **Parity** — JSON output schema and timing semantics match Python `vllm bench serve` exactly.

### Performance vs. Python

| Metric | Python | Rust |
| -------- | -------- | ------ |
| Startup time | Multi-second (import vllm + numpy + aiohttp) | ~7 ms |
| 100k random prompts (input_len=8192) | Minutes | Seconds (rayon parallelism) |
| Binary size | — | ~7 MB |
| Peak memory at 1400 concurrency | High (GIL + per-object overhead) | <100 MB (`Arc<str>` prompt sharing) |

## Contents

- [Install](#install)
- [Quick Start](#quick-start)
- [Usage Examples](#usage-examples)
- [Supported Backends](#supported-backends)
- [Supported Datasets](#supported-datasets)
- [Metrics](#metrics)
- [CLI Reference](#cli-reference)
- [Tokenizer Support](#tokenizer-support)
- [Output Format](#output-format)
- [Architecture](#architecture)
- [Environment Variables](#environment-variables)

## Install

### Prebuilt binaries (Linux)

```bash
curl -fsSL https://github.com/vllm-project/vllm-bench/releases/latest/download/vllm-bench-$(uname -m)-linux-musl -o vllm-bench && chmod +x vllm-bench
```

### With Cargo

Install straight from the repository (builds from source; requires [Rust](https://rustup.rs/) stable and a C compiler for the native tokenizer dependency):

```bash
cargo install --git https://github.com/vllm-project/vllm-bench vllm-bench
```

The trailing `vllm-bench` selects the package — the repo also ships a `mock-llm-server` binary, so omitting it fails with `multiple packages with binaries found`. The binary is installed to `~/.cargo/bin/`.

### Build from source

Requires [Rust](https://rustup.rs/) (stable).

```bash
git clone https://github.com/vllm-project/vllm-bench.git
cd vllm-bench
./install.sh            # builds release and installs to ~/.local/bin
# or: ./install.sh --to ~/bin
```

## Quick Start

Point it at a running vLLM server and benchmark with synthetic prompts:

```bash
vllm-bench \
  --backend vllm \
  --base-url http://127.0.0.1:8000 \
  --model <model-name> \
  --dataset-name random \
  --random-input-len 1024 \
  --random-output-len 128 \
  --num-prompts 1000 \
  --max-concurrency 200
```

> **Tip:** prefer `127.0.0.1` over `localhost` — some systems resolve `localhost` to IPv6 `::1` while vLLM listens on IPv4 only.

Add `--save-result` to write a JSON file, or `--dry-run` to generate and inspect the dataset without sending any requests.

## Usage Examples

<details open>
<summary><b>Generation (completions / chat)</b></summary>

```bash
# Full production-style run with percentile metrics and result file
vllm-bench \
  --backend vllm \
  --base-url http://127.0.0.1:8000 \
  --model nvidia/Kimi-K2.5-NVFP4 \
  --dataset-name random \
  --random-input-len 8192 \
  --random-output-len 1024 \
  --ignore-eos \
  --num-prompts 4096 \
  --percentile-metrics "ttft,tpot,itl,e2el" \
  --save-result \
  --max-concurrency 1400

# Send token IDs instead of text (pure vLLM: skips server-side tokenization,
# exact token counts, faster). Random dataset only.
vllm-bench \
  --backend vllm \
  --base-url http://127.0.0.1:8000 \
  --model <model-name> \
  --dataset-name random \
  --random-input-len 1024 \
  --prompt-token-ids \
  --num-prompts 1000
```

</details>

<details>
<summary><b>Datasets (ShareGPT / Sonnet / HuggingFace / SPEED-Bench)</b></summary>

```bash
# ShareGPT (auto-downloads from HuggingFace on first run, cached afterwards)
vllm-bench \
  --backend openai-chat --base-url http://127.0.0.1:8000 --model <model-name> \
  --dataset-name sharegpt --num-prompts 500 --save-result

# ShareGPT with an explicit local file
vllm-bench \
  --backend openai-chat --base-url http://127.0.0.1:8000 --model <model-name> \
  --dataset-name sharegpt --dataset-path /path/to/ShareGPT_V3.json \
  --num-prompts 500 --save-result

# Sonnet — built-in Shakespeare sonnets, no dataset file needed.
# Generates prompts of a controllable token length with a shared prefix.
vllm-bench \
  --backend openai-chat --base-url http://127.0.0.1:8000 --model <model-name> \
  --dataset-name sonnet \
  --sonnet-input-len 550 --sonnet-output-len 150 --sonnet-prefix-len 200 \
  --num-prompts 500

# Any public HuggingFace dataset (auto-downloads, auto-detects columns)
vllm-bench \
  --backend openai-chat --base-url http://127.0.0.1:8000 --model <model-name> \
  --dataset-name hf --dataset-path allenai/WildChat-4.8M \
  --hf-split train --num-prompts 1000 --save-result

# HuggingFace dataset with subset + fixed output length (LongBench)
vllm-bench \
  --backend openai-chat --base-url http://127.0.0.1:8000 --model <model-name> \
  --dataset-name hf --dataset-path THUDM/LongBench \
  --hf-subset narrativeqa --hf-split test --hf-output-len 512 --num-prompts 200

# Gated HuggingFace dataset (requires HF_TOKEN)
HF_TOKEN=hf_xxx vllm-bench \
  --backend openai-chat --base-url http://127.0.0.1:8000 --model <model-name> \
  --dataset-name hf --dataset-path lmsys/lmsys-chat-1m \
  --hf-split train --hf-output-len 256 --num-prompts 1000

# SPEED-Bench for speculative decoding evaluation (auto-downloads, cached)
vllm-bench \
  --backend openai-chat --base-url http://127.0.0.1:8000 --model <model-name> \
  --dataset-name speed-bench --speed-bench-config qualitative \
  --num-prompts 200 --output-len 256 --save-result

# SPEED-Bench throughput split with entropy category filter + input truncation
vllm-bench \
  --backend openai-chat --base-url http://127.0.0.1:8000 --model <model-name> \
  --dataset-name speed-bench --speed-bench-config throughput_16k \
  --speed-bench-max-input-len 10240 --speed-bench-category low_entropy \
  --num-prompts 500 --output-len 256 --max-concurrency 200 --save-result
```

</details>

<details>
<summary><b>Multimodal (VLM with synthetic images)</b></summary>

```bash
# One synthetic image per request
vllm-bench \
  --backend openai-chat --base-url http://127.0.0.1:8000 \
  --model Qwen/Qwen2.5-VL-7B-Instruct \
  --dataset-name random-mm \
  --random-input-len 512 --random-output-len 128 --num-prompts 100 \
  --random-mm-base-items-per-request 1 \
  --random-mm-limit-mm-per-prompt '{"image": 1, "video": 0}' \
  --random-mm-bucket-config '{(1024, 800, 1): 1.0}'

# Multiple images per request, mixed resolutions
vllm-bench \
  --backend openai-chat --base-url http://127.0.0.1:8000 \
  --model Qwen/Qwen2.5-VL-7B-Instruct \
  --dataset-name random-mm \
  --random-input-len 256 --random-output-len 128 --num-prompts 50 \
  --random-mm-base-items-per-request 3 \
  --random-mm-limit-mm-per-prompt '{"image": 5, "video": 0}' \
  --random-mm-bucket-config '{(256,256,1): 0.5, (720,1280,1): 0.5}'
```

</details>

<details>
<summary><b>Embedding / Pooling / Rerank</b></summary>

```bash
# Text embedding
vllm-bench \
  --backend openai-embeddings --base-url http://127.0.0.1:8000 \
  --model BAAI/bge-large-en-v1.5 \
  --dataset-name random --random-input-len 512 --num-prompts 1000 \
  --max-concurrency 200 --save-result

# Chat-format embedding (supports multimodal content)
vllm-bench \
  --backend openai-embeddings-chat --base-url http://127.0.0.1:8000 \
  --model BAAI/bge-large-en-v1.5 \
  --dataset-name sharegpt --num-prompts 500 --save-result

# vLLM native pooling endpoint
vllm-bench \
  --backend vllm-pooling --base-url http://127.0.0.1:8000 \
  --model BAAI/bge-large-en-v1.5 \
  --dataset-name random --random-input-len 256 --num-prompts 1000 --save-result

# Rerank (query from dataset, documents via --extra-body)
vllm-bench \
  --backend vllm-rerank --base-url http://127.0.0.1:8000 \
  --model BAAI/bge-reranker-v2-m3 \
  --dataset-name sharegpt --num-prompts 500 \
  --extra-body '{"documents": ["document to rerank"]}' --save-result
```

</details>

<details>
<summary><b>Rate control, ramp-up &amp; goodput</b></summary>

```bash
# Ramp from 10 → 100 RPS with goodput SLO tracking
vllm-bench \
  --backend vllm --base-url http://127.0.0.1:8000 --model <model-name> \
  --num-prompts 2000 \
  --ramp-up-strategy linear --ramp-up-start-rps 10 --ramp-up-end-rps 100 \
  --goodput ttft:200 e2el:5000 \
  --save-result

# Fixed Poisson arrival rate at 50 RPS
vllm-bench \
  --backend vllm --base-url http://127.0.0.1:8000 --model <model-name> \
  --num-prompts 2000 --request-rate 50 --burstiness 1.0
```

</details>

<details>
<summary><b>Sweep — find the optimal concurrency / rate</b></summary>

```bash
# Sweep over concurrency values
vllm-bench \
  --backend vllm --base-url http://127.0.0.1:8000 --model <model-name> \
  --num-prompts 500 \
  --sweep-max-concurrency 1,10,50,100,200,500,1000

# Sweep over request rates
vllm-bench \
  --backend vllm --base-url http://127.0.0.1:8000 --model <model-name> \
  --num-prompts 500 \
  --sweep-request-rate 1,10,50,100,inf

# Scale work with concurrency and reset the prefix cache between points
# (--sweep-num-prompts-factor sets num_prompts = concurrency * factor;
#  --reset-prefix-cache requires VLLM_SERVER_DEV_MODE=1 on the server)
vllm-bench \
  --backend vllm --base-url http://127.0.0.1:8000 --model <model-name> \
  --sweep-max-concurrency 1,10,50,100 \
  --sweep-num-prompts-factor 20 \
  --reset-prefix-cache
```

</details>

<details>
<summary><b>Multi-run &amp; comparison</b></summary>

```bash
# Run 5 times, report mean/std/min/max with coefficient of variation
vllm-bench \
  --backend vllm --base-url http://127.0.0.1:8000 --model <model-name> \
  --num-prompts 1000 --max-concurrency 200 --num-runs 5

# Compare two saved result files side-by-side (no server needed)
vllm-bench --compare baseline.json optimized.json
```

</details>

<details>
<summary><b>Multi-turn conversations</b></summary>

```bash
# Synthetic multi-turn (controllable per-turn token lengths)
vllm-bench \
  --backend openai-chat --base-url http://127.0.0.1:8000 --model <model-name> \
  --dataset-name random --multi-turn --multi-turn-num-turns 5 \
  --random-input-len 512 --random-output-len 256 \
  --num-prompts 50 --multi-turn-concurrency 10 \
  --percentile-metrics "ttft,tpot,itl,e2el" --save-result

# Variable turn count per conversation + per-turn input length for turns 1+
vllm-bench \
  --backend openai-chat --base-url http://127.0.0.1:8000 --model <model-name> \
  --dataset-name random --multi-turn \
  --multi-turn-min-turns 2 --multi-turn-max-turns 8 \
  --random-input-len 2048 --per-turn-input-len 256 --random-output-len 128 \
  --num-prompts 100 --multi-turn-concurrency 20

# ShareGPT conversations (loads all turns, not just the first two)
vllm-bench \
  --backend openai-chat --base-url http://127.0.0.1:8000 --model <model-name> \
  --dataset-name sharegpt --multi-turn \
  --num-prompts 50 --multi-turn-concurrency 10 --save-result

# Think time between turns
vllm-bench \
  --backend openai-chat --base-url http://127.0.0.1:8000 --model <model-name> \
  --multi-turn --multi-turn-num-turns 3 --multi-turn-delay-ms 500 \
  --num-prompts 100 --multi-turn-concurrency 20
```

</details>

<details>
<summary><b>LoRA multi-adapter</b></summary>

```bash
# Distribute requests across N adapters registered on the server.
# --model stays the BASE model (tokenizer / readiness / /tokenize use it);
# the per-request `model` field is rewritten to one of --lora-modules.
vllm-bench \
  --backend openai-chat --base-url http://127.0.0.1:8000 \
  --model Qwen/Qwen3-30B-A3B \
  --lora-modules sql-lora-1 sql-lora-2 sql-lora-3 sql-lora-4 \
                 sql-lora-5 sql-lora-6 sql-lora-7 sql-lora-8 \
  --lora-assignment random \
  --dataset-name random --random-input-len 1024 --random-output-len 256 \
  --num-prompts 1000 --max-concurrency 64 --save-result

# Deterministic round-robin assignment (request i -> adapter[i % N])
vllm-bench \
  --backend openai-chat --base-url http://127.0.0.1:8000 \
  --model Qwen/Qwen3-30B-A3B \
  --lora-modules sql-lora-1 sql-lora-2 sql-lora-3 sql-lora-4 \
  --lora-assignment round-robin \
  --dataset-name random --num-prompts 1000
```

Server side — start vLLM with `--enable-lora` and one `name=path` pair per adapter:

```bash
vllm serve <base-model> \
  --enable-lora --max-loras 8 --max-lora-rank 16 \
  --lora-modules \
    sql-lora-1=jeeejeee/qwen3-moe-text2sql-spider \
    sql-lora-2=jeeejeee/qwen3-moe-text2sql-spider \
    ...
```

Set `--max-loras` ≥ number of adapter names to keep them all resident (clean steady-state numbers), or lower to stress the LoRA swap path.

</details>

<details>
<summary><b>Profiling &amp; dry-run</b></summary>

```bash
# Trigger vLLM server-side profiling (start before, stop after the benchmark)
vllm-bench \
  --backend vllm --base-url http://127.0.0.1:8000 --model <model-name> \
  --num-prompts 100 --profile

# Defer profiling until the server batch is full, then capture for 10s
vllm-bench \
  --backend vllm --base-url http://127.0.0.1:8000 --model <model-name> \
  --num-prompts 2000 --max-concurrency 256 \
  --profile --profile-batch-threshold 200 --profile-duration 10

# Dry run: generate dataset, print stats, send nothing
vllm-bench \
  --model <model-name> --num-prompts 100000 --random-input-len 8192 --dry-run
```

</details>

## Supported Backends

### Generation

| Backend | API Endpoint | Description |
| --------- | ------------- | ------------- |
| `vllm` / `openai` | `/v1/completions` | OpenAI-compatible completions (streaming) |
| `openai-chat` | `/v1/chat/completions` | OpenAI-compatible chat completions (streaming, multimodal) |

### Embedding / Pooling

| Backend | API Endpoint | Description |
| --------- | ------------- | ------------- |
| `openai-embeddings` | `/v1/embeddings` | Text embedding (accepts text or token IDs) |
| `openai-embeddings-chat` | `/v1/embeddings` | Chat-format embedding (supports multimodal content) |
| `vllm-pooling` | `/v1/pooling` | vLLM native pooling endpoint |
| `vllm-rerank` | `/v1/rerank` | vLLM reranking (query from prompt, documents via `--extra-body`) |

Pooling backends are non-streaming and report E2EL (end-to-end latency) only. Use `--dataset-name sharegpt`, `sonnet`, or `hf` for text-based embedding/rerank benchmarks, or `random` for token-ID-based embedding benchmarks.

## Supported Datasets

| Dataset | Description |
| --------- | ------------- |
| `random` | Synthetic prompts with exact token-length matching (default) |
| `random-mm` | Synthetic multimodal prompts with random JPEG images for VLM benchmarking (requires `openai-chat`) |
| `sharegpt` | Real conversations from ShareGPT (auto-downloads from HuggingFace, or use `--dataset-path`) |
| `sonnet` | Built-in Shakespeare sonnets; controllable token length + shared prefix, no dataset file needed |
| `speed-bench` | NVIDIA SPEED-Bench for speculative decoding evaluation (auto-downloads, 11 categories) |
| `hf` | Any HuggingFace dataset (auto-downloads via datasets-server API, auto-detects chat/text columns) |

## Metrics

### Generation backends

- **TTFT** (Time to First Token) — latency from request send to first token received
- **TPOT** (Time per Output Token) — average time between output tokens
- **ITL** (Inter-Token Latency) — per-token latency distribution
- **E2EL** (End-to-End Latency) — total request latency
- **Throughput** — requests/sec, output tokens/sec, peak output tokens/sec, total tokens/sec
- **Concurrency** — peak concurrent requests
- **Goodput** — requests/sec meeting all specified SLOs (with `--goodput`)

### Pooling / embedding backends

- **E2EL** — total request latency (mean, median, std, percentiles)
- **Throughput** — requests/sec, input tokens/sec
- **Concurrency** — peak concurrent requests

### Steady-state metrics

When `--max-concurrency` is set and `--request-rate` is `inf` (closed-loop mode), the benchmark automatically reports an additional **Steady-State Metrics** block. It measures throughput and latency only over the window during which in-flight concurrency stays at or above a fraction of `--max-concurrency`, excluding the ramp-up and drain phases. This sharply reduces run-to-run variance at very high concurrency.

The block reports request/input/output/total token throughput plus TTFT (mean, median, percentiles) and TPOT (mean, median, P90, P99) over the detected plateau, along with the window bounds and how many requests fell inside it. Tune it with `--steady-state-threshold` (default `0.95`) and `--steady-state-min-window`, or disable with `--no-steady-state`. The result JSON carries a `steady_state` object (null when not computed).

## CLI Reference

Run `vllm-bench --help` for the authoritative list. Grouped reference below.

<details>
<summary><b>Server connection</b></summary>

| Flag | Default | Description |
| ------ | --------- | ------------- |
| `--backend` | `openai` | Backend type (`vllm`, `openai`, `openai-chat`, `openai-embeddings`, `openai-embeddings-chat`, `vllm-pooling`, `vllm-rerank`) |
| `--base-url` | — | Server base URL (overrides `--host`/`--port`) |
| `--host` | `127.0.0.1` | Server host |
| `--port` | `8000` | Server port |
| `--endpoint` | Auto | API endpoint path (auto-selected per backend) |
| `--insecure` | `false` | Disable SSL certificate verification |

</details>

<details>
<summary><b>Model &amp; tokenizer</b></summary>

| Flag | Default | Description |
| ------ | --------- | ------------- |
| `--model` | Auto-detect | Model name (fetched from `/v1/models` if omitted) |
| `--served-model-name` | — | Model name used in API requests |
| `--tokenizer` | Same as model | Tokenizer name or path (supports HF, tiktoken, server fallback) |
| `--tokenizer-mode` | `auto` | Tokenizer mode (`auto`, `hf`, `slow`, `mistral`) |
| `--trust-remote-code` | `false` | Trust remote code for tokenizer |
| `--skip-tokenizer-init` | `false` | Skip tokenizer initialization |

</details>

<details>
<summary><b>Dataset</b></summary>

| Flag | Default | Description |
| ------ | --------- | ------------- |
| `--dataset-name` | `random` | Dataset type (`random`, `random-mm`, `sharegpt`, `sonnet`, `speed-bench`, `hf`) |
| `--dataset-path` | — | Path to dataset file (optional for `sharegpt`/`sonnet`, which auto-source) |
| `--num-prompts` | `1000` | Number of prompts to generate (conversations in multi-turn mode) |
| `--max-model-len` | — | Filter out requests where `prompt_len + output_len` exceeds this context length |
| `--input-len` | — | Override input length (general) |
| `--output-len` | — | Override output length (general) |
| `--no-oversample` | `false` | Don't oversample if dataset is smaller than `--num-prompts` |
| `--disable-shuffle` | `false` | Don't shuffle the dataset |
| `--seed` | `0` | Random seed for reproducibility |
| **Random** | | |
| `--random-input-len` | `1024` | Input token length |
| `--random-output-len` | `128` | Output token length |
| `--random-prefix-len` | `0` | Shared prefix length |
| `--random-range-ratio` | `1.0` | Length jitter, range `(0, 1]`. Lengths sampled from `[ratio × target, target]`; `1.0` = fixed length |
| `--prompt-token-ids` | `false` | Send prompts as token-ID arrays (skips server-side tokenization, exact counts). Random dataset only |
| **Random multimodal** | | |
| `--random-mm-base-items-per-request` | `1` | Base number of multimodal items (images) per request |
| `--random-mm-num-mm-items-range-ratio` | `0.0` | Range ratio for varying item count per request |
| `--random-mm-limit-mm-per-prompt` | `{"image": 255, "video": 1}` | Per-modality hard caps (JSON) |
| `--random-mm-bucket-config` | `{(256,256,1): 0.5, (720,1280,1): 0.5}` | `(height,width,frames)` → probability (Python tuple syntax; frames=1 = image) |
| **ShareGPT** | | |
| `--sharegpt-output-len` | — | Override output length |
| **Sonnet** | | |
| `--sonnet-input-len` | `550` | Input tokens per request |
| `--sonnet-output-len` | `150` | Output tokens per request |
| `--sonnet-prefix-len` | `200` | Prefix tokens shared across requests |
| **SPEED-Bench** | | |
| `--speed-bench-config` | `qualitative` | Split (`qualitative`, `throughput_1k`/`2k`/`8k`/`16k`/`32k`) |
| `--speed-bench-category` | — | Filter by category (`low_entropy`, `high_entropy`, `mixed_entropy`, `coding`, `math`, …) |
| `--speed-bench-max-input-len` | — | Truncate prompts to at most N tokens |
| **HuggingFace** | | |
| `--hf-split` | Auto | Split (`train`, `test`, `validation`); auto-detected if omitted |
| `--hf-subset` | — | Subset/config name (e.g. `narrativeqa` for LongBench) |
| `--hf-output-len` | — | Fixed output length for all requests (overrides dataset-derived length) |
| `--hf-text-column` | Auto | Column containing prompt text; auto-detected from common patterns |

</details>

<details>
<summary><b>Rate control</b></summary>

| Flag | Default | Description |
| ------ | --------- | ------------- |
| `--request-rate` | `inf` | Requests per second (`inf` = all at once) |
| `--burstiness` | `1.0` | Burstiness factor (1.0 = Poisson, >1 = bursty) |
| `--max-concurrency` | `num-prompts` | Maximum concurrent requests (semaphore) |
| `--ramp-up-strategy` | — | Ramp-up mode (`linear` or `exponential`) |
| `--ramp-up-start-rps` | — | Starting request rate for ramp-up |
| `--ramp-up-end-rps` | — | Ending request rate for ramp-up |

</details>

<details>
<summary><b>Sampling parameters</b></summary>

| Flag | Description |
| ------ | ------------- |
| `--temperature` | Temperature (server default if omitted) |
| `--top-p` | Top-p (nucleus) sampling |
| `--top-k` | Top-k sampling |
| `--min-p` | Min-p sampling |
| `--frequency-penalty` | Frequency penalty |
| `--presence-penalty` | Presence penalty |
| `--repetition-penalty` | Repetition penalty |

Merged into the request body. Only effective with generation backends (`vllm`, `openai`, `openai-chat`); ignored by pooling/embedding backends.

</details>

<details>
<summary><b>Output &amp; results</b></summary>

| Flag | Default | Description |
| ------ | --------- | ------------- |
| `--save-result` | `false` | Save results to JSON file |
| `--save-detailed` | `false` | Include per-request data in JSON (input/output lens, ITLs, texts) |
| `--append-result` | `false` | Append to existing JSON file (JSONL format) |
| `--result-dir` | — | Directory for result files |
| `--result-filename` | Auto | Custom result filename |
| `--percentile-metrics` | `ttft,tpot,itl,e2el` | Metrics for percentile reporting (pooling defaults to `e2el` only) |
| `--metric-percentiles` | `99` | Percentile values to compute |
| `--sweep-summary-percentiles` | — | Extra percentiles for sweep summary tables (auto-added to computed set) |
| `--goodput` | — | SLO pairs for goodput (`ttft:100 tpot:50 e2el:500`, values in ms) |
| `--disable-tqdm` | `false` | Disable progress bar |
| `--label` | — | Label prefix for result files |
| `--metadata` | — | Key-value metadata (`KEY=VALUE`, repeatable) |

</details>

<details>
<summary><b>Request options</b></summary>

| Flag | Default | Description |
| ------ | --------- | ------------- |
| `--ignore-eos` | `false` | Ignore EOS token (force full output length) |
| `--logprobs` | — | Number of logprobs per token |
| `--num-warmups` | `0` | Warmup requests before benchmarking |
| `--ready-check-timeout-sec` | `0` | Endpoint readiness timeout (0 = skip) |
| `--request-id-prefix` | Auto (UUID) | Prefix for request IDs |
| `--header` | — | Extra headers (`KEY=VALUE`, repeatable) |
| `--extra-body` | — | Extra JSON body parameters |
| `--dry-run` | `false` | Generate dataset only, skip benchmark |

</details>

<details>
<summary><b>Steady-state metrics</b></summary>

| Flag | Default | Description |
| ------ | --------- | ------------- |
| `--steady-state-threshold` | `0.95` | Fraction of `--max-concurrency` at which the steady-state window opens, range (0, 1] |
| `--steady-state-min-window` | Auto | Minimum window duration (s) below which a warning is attached. Default `max(10, 0.1 × run_duration)` |
| `--no-steady-state` | `false` | Disable steady-state metrics computation |

Computed only when `--max-concurrency` is set and `--request-rate` is `inf`.

</details>

<details>
<summary><b>Profiling</b></summary>

| Flag | Default | Description |
| ------ | --------- | ------------- |
| `--profile` | `false` | Trigger vLLM server-side profiling (`/start_profile` before, `/stop_profile` after) |
| `--profile-batch-threshold` | — | Defer profiling until `/metrics` reports ≥ N running requests, then capture. Requires `--profile` |
| `--profile-duration` | `5.0` | Seconds to capture once the batch threshold is reached. Requires `--profile-batch-threshold` |

</details>

<details>
<summary><b>Sweep mode</b></summary>

| Flag | Default | Description |
| ------ | --------- | ------------- |
| `--sweep-max-concurrency` | — | Comma-separated concurrency values to sweep (e.g. `1,10,50,100,500`) |
| `--sweep-request-rate` | — | Comma-separated rate values to sweep, supports `inf` (e.g. `1,10,100,inf`) |
| `--sweep-num-prompts-factor` | — | Set `num_prompts = concurrency × factor` per concurrency sweep point |
| `--reset-prefix-cache` | `false` | Reset the server's prefix cache before each sweep iteration (requires `VLLM_SERVER_DEV_MODE=1`) |

Runs the benchmark once per value, then prints a summary table comparing throughput and latency across all sweep points and identifies the best-throughput configuration. Works in multi-turn mode too. `--sweep-summary-percentiles` appends extra TTFT/TPOT/E2EL columns to the summary, auto-adding any missing percentiles to the computed set so they also appear in result JSON.

</details>

<details>
<summary><b>Multi-turn conversation benchmark</b></summary>

| Flag | Default | Description |
| ------ | --------- | ------------- |
| `--multi-turn` | `false` | Enable multi-turn conversation mode (requires `--backend openai-chat`) |
| `--multi-turn-num-turns` | `3` | Turns per conversation (synthetic mode) |
| `--multi-turn-min-turns` | `0` | Minimum turns per conversation (0 = use `--multi-turn-num-turns`) |
| `--multi-turn-max-turns` | `0` | Maximum turns per conversation (0 = `--multi-turn-num-turns` synthetic / uncapped ShareGPT) |
| `--multi-turn-concurrency` | — | Concurrent conversations (defaults to `--max-concurrency` or `--num-prompts`) |
| `--multi-turn-delay-ms` | `0` | Delay between turns in ms (simulates user think time) |
| `--per-turn-input-len` | `0` | Input token length for turns 1+ (0 = use `--random-input-len` for all turns) |
| `--multi-turn-prefix-global-ratio` | `0.0` | Fraction of per-turn input shared across all conversations (random dataset only) |
| `--multi-turn-prefix-conversation-ratio` | `0.0` | Fraction shared within each conversation (random dataset only) |

With `--multi-turn`, `--num-prompts` controls the number of **conversations**, not individual requests.

**How it works:**

- Turn 1: send `[user_1]`, get `assistant_1`
- Turn 2: send `[user_1, assistant_1, user_2]`, get `assistant_2`
- Turn N: send full history + `user_N` — measures growing-context performance

**Data sources:**

- `--dataset-name random` — synthetic conversations with controllable per-turn token lengths. Auto-sets `min_tokens` to enforce output length without `ignore_eos`.
- `--dataset-name sharegpt` — loads all turns (not just the first two); filters for entries with ≥ 2 real turns.

**Prefix sharing** (random dataset): when `--multi-turn-prefix-global-ratio` or `--multi-turn-prefix-conversation-ratio` is > 0, each turn sends a fixed-length message (no history accumulation) composed of a global prefix + per-conversation prefix + unique suffix. The two ratios must sum to < 1.0.

**Router affinity:** every turn sends `X-Session-ID: {conversation_id}` for KV-cache reuse behind a vLLM router.

**Output:** overall metrics plus a per-turn breakdown (TTFT/TPOT/ITL/E2EL by turn index). Expect TTFT to climb across turns due to growing context. JSON includes a `per_turn_metrics` array.

</details>

<details>
<summary><b>LoRA multi-adapter</b></summary>

| Flag | Default | Description |
| ------ | --------- | ------------- |
| `--lora-modules` | — | Adapter names registered on the server (`vllm serve --lora-modules name=path`). Each request's `model` field is rewritten to one of these. Repeatable |
| `--lora-assignment` | `random` | Distribution: `random` (uniform, seeded by `--seed`) or `round-robin` (deterministic `i % N`) |

`--model` must stay the **base** model — its tokenizer builds prompts, and `/v1/models`, `/tokenize`, ready check, and warmup all use it. Only the per-request `model` field in completions/chat payloads is rewritten to the assigned adapter (vLLM routes by name).

**Assignment scope:** per request in single-shot mode; **per conversation** (sticky across all turns) in multi-turn mode, to avoid breaking prefix-cache reuse mid-dialog.

**Reproducibility:** with `--lora-assignment random`, the same `--seed` + same `--lora-modules` list yields identical request-to-adapter mappings. Pooling/embedding backends are rejected — LoRA routing applies to generative paths only.

</details>

<details>
<summary><b>Multi-run &amp; comparison</b></summary>

| Flag | Default | Description |
| ------ | --------- | ------------- |
| `--num-runs` | `1` | Run benchmark N times; report mean/std/min/max with CV |
| `--compare` | — | Compare two result JSON files side-by-side (skips benchmarking) |

`--num-runs` aggregates metrics across runs and reports the coefficient of variation (CV) for throughput stability. `--compare` reads two previously-saved result files and prints a diff with delta, % change, and improvement/regression markers.

</details>

## Tokenizer Support

Tokenizers are loaded with a three-tier fallback chain:

1. **Local HuggingFace** — `tokenizer.json` from a local path or the Hub (fastest)
2. **Tiktoken** — `.tiktoken` / `.model` format for Kimi, Qwen, etc. (auto-extracts `pat_str` from Python source)
3. **Server-side** — falls back to vLLM's `/tokenize` + `/detokenize` endpoints

For the `random` dataset, prompt token lengths are verified against the server on the first run and cached; subsequent runs with the same model+server skip verification. Verification is also skipped when `--prompt-token-ids` is set (token counts are exact by construction).

Models without `tokenizer.json` (e.g. `nvidia/Kimi-K2.5-NVFP4`) fall back to server-side tokenization automatically; you can also point `--tokenizer` at a model that ships `tokenizer.json`.

## Output Format

JSON output is compatible with the `vllm bench serve` Python schema. Result files are named:

```text
{label}-{rate}qps-concurrency{max_concurrency}-{model}-{timestamp}.json
```

Use `--append-result` to append multiple runs to the same file in JSONL format. `--save-detailed` adds per-request arrays (input/output lengths, ITLs, generated text).

## Architecture

<details>
<summary><b>Source layout</b></summary>

```text
src/
├── main.rs                  # Entry point, mimalloc, tokio runtime, mode dispatch
├── cli.rs                   # clap CLI argument definitions
├── config.rs                # Validated config, goodput/ramp-up parsing
├── benchmark.rs             # Core orchestrator (schedule, spawn, collect, verify, profile)
├── multi_turn.rs            # Multi-turn conversation orchestrator (channel workers)
├── compare.rs               # Result diff (--compare file_a.json file_b.json)
├── sweep.rs                 # Parameter sweep (--sweep-max-concurrency, --sweep-request-rate)
├── multi_run.rs             # Multi-run statistics (--num-runs N)
├── rate_control.rs          # Gamma/Poisson scheduling + linear/exponential ramp-up
├── ready_checker.rs         # Endpoint readiness with retry
├── tokenizer.rs             # Tokenizer abstraction (HF, tiktoken, server)
├── tiktoken.rs              # Tiktoken BPE loader with pat_str extraction
├── error.rs                 # Error types
├── backends/
│   ├── mod.rs               # Backend enum dispatch, typed SSE structs
│   ├── streaming.rs         # SSE stream parser with speculative JSON parse
│   ├── openai_completions.rs # /v1/completions backend
│   ├── openai_chat.rs       # /v1/chat/completions backend
│   └── pooling.rs           # Embedding/pooling/rerank backends (non-streaming)
├── datasets/
│   ├── mod.rs               # SampleRequest, ConversationTurn, MultiTurnConversation types
│   ├── random.rs            # Random dataset with rayon parallelism
│   ├── random_mm.rs         # Random multimodal dataset (JPEG generation, bucket sampling)
│   ├── multi_turn.rs        # Multi-turn synthetic + ShareGPT conversation generators
│   ├── sharegpt.rs          # ShareGPT JSON dataset loader
│   ├── sonnet.rs            # Sonnet dataset (built-in Shakespeare sonnets)
│   ├── speed_bench.rs       # NVIDIA SPEED-Bench loader (auto-download + cache)
│   └── hf_dataset.rs        # Generic HuggingFace dataset (auto-download, column detection)
├── metrics/
│   ├── mod.rs               # BenchmarkMetrics, MultiTurnMetrics structs
│   ├── calculator.rs        # Percentile/throughput/goodput/peak/multi-turn computation
│   └── steady_state.rs      # Steady-state window detection + plateau metrics
└── output/
    ├── mod.rs
    ├── console.rs           # Terminal output (matches Python format)
    └── json.rs              # JSON result serialization (Python-compatible schema)
```

</details>

### Key design decisions

- **reqwest + tokio** — HTTP client with connection pooling, forced HTTP/1.1, TCP_NODELAY to match Python's aiohttp and avoid Nagle latency inflation on TTFT
- **mimalloc** — global allocator to reduce contention under high concurrency (1400+ tasks); page-agnostic, runs on aarch64 4K- and 64K-page kernels
- **`Arc<str>` prompts** — zero-copy prompt sharing across tokio tasks, eliminating ~3 GB peak memory at 100k requests with 8k-token prompts
- **Spawn-per-request** — `tokio::spawn` per request with a `Semaphore` for concurrency control (matches Python's asyncio pattern)
- **rayon** — parallel dataset generation across CPU cores (200–500× faster than Python for 100k+ prompts)
- **Enum dispatch** — backend variants instead of trait objects (avoids async trait-object limitations)
- **Typed SSE deserialization** — `CompletionChunk`/`ChatChunk` structs skip unused JSON fields (cheaper than `serde_json::Value`)
- **Speculative JSON parse** — SSE handler uses `serde_json::value::RawValue` to detect complete JSON before `\n\n` arrives, improving TTFT/ITL accuracy when TCP segments split
- **Connection error retry** — automatic retry with backoff on connection reset/timeout/refused (up to 3 attempts)
- **Tokenizer verification cache** — server-side token-length verification is cached per model+server pair

### Behavioral parity with Python

The Rust implementation matches Python `vllm bench serve` in:

- SSE streaming protocol handling (including speculative parse for split TCP segments)
- Timing semantics (monotonic `Instant` matching Python's `time.perf_counter()`)
- Chat vs. completions differences (`max_completion_tokens` vs. `max_tokens`, Content-Type, timestamp placement)
- JSON output schema (all fields, key naming, `request_rate` as the string `"inf"`)
- Rate control (Gamma distribution, normalization, burstiness, linear/exponential ramp-up)
- Metrics (TTFT/TPOT/ITL/E2EL percentiles, peak tokens/sec, peak concurrency, goodput)
- Sampling parameters merged into the request body via `extra_body` (same precedence rules)

## Environment Variables

| Variable | Description |
| ---------- | ------------- |
| `OPENAI_API_KEY` | API key for authenticated endpoints (cached, not read per-request) |
| `HF_TOKEN` | HuggingFace token for gated model tokenizers and gated datasets |
| `TOKIO_WORKER_THREADS` | Override tokio worker thread count (default: physical cores) |

## License

Apache-2.0
</content>
</invoke>
