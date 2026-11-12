# Expert pool: reproducing the generation-speed measurement

Sends a warmup request and then a measured request to Qwen3.8-Flash-Next
NVFP4. Each request is a software bug report; the model writes a fix
proposal and a verification plan.

## Files and requests

| file | content |
| --- | --- |
| [benchmark.py](benchmark.py) | HTTP requests, stream capture, decode-speed computation |
| [pair.json](pair.json) | the two prompts as measured: full text, frozen token ids, provenance |
| [prompts.md](prompts.md) | the prompt texts, readable |
| [provenance.json](provenance.json) | code versions measured, SHA256 of the original files, changes from the historical runner |
| [results/20261112.json](results/20261112.json) | November measurements, output hashes, memory observations and excluded runs |
| [LICENSE.prompts](LICENSE.prompts) | MIT license of the prompt source |

| order | role | bug report | input tokens |
| --- | --- | --- | --- |
| 1st | warmup (`role` = `warmup`) | rename the "Choose File" button to "Choose file" (task 28096_836) | 1070 |
| 2nd | measured (`role` = `measure`) | update the "Link sent!" message when the language setting changes (task 18827_741) | 753 |

The prompts come from SWE-Lancer in
[OpenAI frontier-evals](https://github.com/openai/frontier-evals/tree/51052cede8cc608f95bb00346635e03759013e5a),
two of its existing sanity tasks. This measures fix-proposal text
generation only; the code-editing and official-grading quality checks are a
separate procedure.

## Current-main validation configuration

The November performance measurements use production code at
`27e2470a154b790f67ad184d63a6796ba8371007`, based on main
`dac51063cc1295a2a8426998c4c372b19c42a0f9`. It uses current Engram disk
offloading instead of the historical deferred-PLE patch below.

The later commit `3abbddc9dfaf31149f505fce4d2d4edeeb5ee9cb` skips pool setup
when the loader finds no pooled layers. It changes only
`vllm/model_executor/model_loader/utils.py`; the pooled installation body
and inference kernels are unchanged. Its separate validation passed 89
pool, configuration and loader tests and two benchmark-client tests.
Performance was not remeasured on that commit.

The host is an RTX PRO 6000 Blackwell Max-Q with an external CUDA allocation
leaving approximately 48 GiB free before launching the server, and a
100 GiB container memory limit. This is not a measurement on RTX 6000 Ada.
The external allocation is fixed; desktop and driver allocations can change
the available memory slightly between launches.

KV cache is fixed at 768 MiB in both staging conditions. This disables the
extensible KV cache and takes precedence over `gpu-memory-utilization` for
cache sizing. An earlier automatic-sizing pilot completed four runs, but
its fifth launch failed after warmup: the extensible cache's safety reserve
left 0.53 GiB for KV, below the approximately 0.60 GiB needed for one
4096-token request. Pilot results are not mixed with fixed-KV measurements.

The utilization fraction below is specific to the physical 96 GiB-class
GPU used here; it is not a portable setting for other GPUs. Check available
memory before running these commands.

```bash
export CHECKPOINT=/data/models/Qwen3.8-Flash-Next-NVFP4-nvidia
export VLLM_DEBUG_WORKSPACE=1 VLLM_LOGGING_LEVEL=DEBUG
export PYTORCH_ALLOC_CONF=pinned_max_round_threshold_mb:1,pinned_max_cached_size_mb:1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export FLASHINFER_WORKSPACE_BASE=/path/to/persistent/flashinfer-workspace
unset VLLM_USE_BREAKABLE_CUDAGRAPH VLLM_USE_V2_MODEL_RUNNER
unset VLLM_PLE_MMAP VLLM_PLE_MMAP_DEFERRED VLLM_PLE_MMAP_PREWARM VLLM_PLE_MMAP_PINNED
.venv/bin/python -m vllm.entrypoints.openai.api_server \
  --model "$CHECKPOINT" --served-model-name flashnext \
  --host 127.0.0.1 --port 18098 --tensor-parallel-size 1 \
  --quantization modelopt --dtype bfloat16 --moe-backend marlin \
  --moe-expert-pool-rows 258 --language-model-only \
  --max-model-len 4096 --max-num-seqs 1 --max-num-batched-tokens 512 \
  --engram-config '{"disk_offload":true}' \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
  --kv-cache-memory-bytes 805306368 --gpu-memory-utilization 0.49 \
  --no-enable-flashinfer-autotune --safetensors-load-strategy lazy \
  --default-chat-template-kwargs '{"enable_thinking":false}' \
  --reasoning-parser qwen3 --generation-config vllm \
  --enable-auto-tool-choice --tool-call-parser qwen3_xml
```

Append `--no-moe-expert-pool-prefill-staging` for the comparison condition.
Run the [client](#client) against `http://127.0.0.1:18098` once per fresh
server. The two conditions use the same frozen requests and sampling
settings. Generated text may differ across fresh servers, including within
one condition; these are response-throughput measurements, not timings of
an identical generated continuation.

## November results

The following values are for the measured request (753 input tokens).
Staging ON and OFF each use three fresh servers, fixed 768 MiB KV capacity,
and the configuration above. The brackets show the observed range.

| Configuration | Fresh servers | Decode tok/s, median [range] | TTFT, median (s) | End-to-end tok/s, median |
| --- | --- | --- | --- | --- |
| Pool, staging ON | 3 | 64.9848 [64.1537–65.8998] | 9.3034 | 45.4134 |
| Pool, staging OFF | 3 | 64.8578 [62.1142–65.5667] | 6.9398 | 48.5435 |
| UVA expert offload, eager | 1 | 7.5227 | 13.0673 | 7.0312 |

For the UVA row, the pool is disabled with `--moe-expert-pool-rows 0`,
and `--offload-backend uva --cpu-offload-gb 40 --cpu-offload-params experts
--enforce-eager` is added. The source, checkpoint, fixed KV size, prompts
and sampling settings match the pool measurements. This single run changes
both offloading and graph execution; it does not isolate staging or measure
variation across restarts.

The pool decode ranges overlap. Staging ON had higher TTFT for these short
prompts and 512-token chunks; this experiment does not establish a benefit
for long-input prefill. Staging used another 2.64 GiB of GPU memory, while
both arms retained the same 4,864-token KV capacity. Generated text and
completion lengths differed even between launches of the same condition;
the JSON records both warmup and measured output hashes and token counts.

The original fixed-KV ON-2 run measured 62.1077 tok/s. CPU loader tests
overlapped its measured request from 06:03:05.516 to 06:03:36.221 UTC, with
the CPU test log finishing at 06:03:27.889 UTC. That run is retained but
excluded from the primary statistics. One fresh ON replacement was declared
before rerunning, after the six-run campaign completed; its 64.9848 tok/s
result is the third ON sample above. The exclusion is for concurrent work,
not a speed threshold. The earlier automatic-KV pilot is also retained
separately.

All included requests finished with `stop`, and all included containers
exited 0 with no OOM kill or restart. The four API checks run after the
first benchmark in each pool condition passed: exact JSON, a `read_file`
tool call, five-fact retrieval with a prompt over 512 tokens, and a bounded
JSON product list. These are lightweight checks, not official SWE-Lancer
grades or an accuracy benchmark. MTP is disabled for every speed result
above.

The maximum saved server NVML readings were 47,406 MiB with staging ON,
44,772 MiB with staging OFF, and 35,576 MiB for UVA. These are snapshots,
not continuous peak measurements. Cgroup-accounted memory, including file
cache, reached the 100 GiB limit in the pool runs without an OOM event.
Before launch, GPU free memory was
49,152–49,276 MiB under the nominal 48 GiB balloon budget. Shared model
files and compilation caches, including FlashInfer's persistent workspace,
were retained between fresh server processes.

The Python sources ran over the precompiled wheel at
`c9c36486c3f7c767c25bb37f087e38d101edf1ef`; native/build inputs were checked,
with two unrelated Rust MiniMax parser files differing from the wheel.
Exact source and wheel hashes, package versions, per-run values and
measurement limitations are in [the result file](results/20261112.json).
The historical measurements below used another vLLM and deferred-PLE
implementation and are not a controlled baseline for these results.

### MTP functional validation

The same four API checks also passed on
`3abbddc9dfaf31149f505fce4d2d4edeeb5ee9cb` with the checkpoint's FP8 draft.
This separate run used `VLLM_LOGGING_LEVEL=INFO`,
`--moe-expert-pool-rows 240`, `--no-moe-expert-pool-prefill-staging`,
`--kv-cache-memory-bytes 1610612736` (1.5 GiB), and
`--speculative-config '{"method":"mtp","num_speculative_tokens":3,"moe_backend":"triton"}'`.
The remaining model, context, batch and sampling settings matched those above.

Startup logs confirmed 40 decode staging rows (four verification tokens
times top-k 10), the FP8 draft experts remaining on the GPU with Triton,
and successful graph capture. These decode staging rows are separate from
the disabled prefill staging buffers. The server counters increased by
456 drafting steps, 1,368 draft tokens and 1,119 accepted tokens: 81.8%
acceptance across these four checks. All checks passed and the container
exited 0 without an OOM kill. This small set, dominated by the structured
output request, does not establish a representative acceptance rate, an
MTP speedup or an accuracy measurement.

Two earlier attempts are retained in the result JSON. With 768 MiB KV,
startup planning rejected the approximately 0.71 GiB available because
4096 context needed about 1.15 GiB. With 1.5 GiB KV and DEBUG logging,
upstream IR tensor formatting attempted a CUDA operation during draft
graph capture and invalidated capture. INFO logging passed on the same
production code; neither failed attempt was OOM-killed.

## Historical code and environment measured

| purpose | version |
| --- | --- |
| vLLM main used as the base | `a97dacb7106ee49f39f3d1fc6ae1800ff724e01d` |
| expert pool implementation (this PR, at the time of measurement) | `5fbc240ba5ddec82a10362340ac77339a1c24017` |
| deferred PLE rows ([01554/vllm#46](https://github.com/01554/vllm/pull/46), a diff on top of upstream PR [#54129](https://github.com/vllm-project/vllm/pull/54129)) | `4f859de9d0f55760b50358aee4834e6966e13bc8` |
| the combination of the above that produced the numbers below | `7dedc6d8d9b178b60f6a5b32f03d677145982441` |

The server ran the Python sources of the measured combination over a wheel
built from the base commit above, plus a separately built `_ple_memops`
extension. This directory holds client-side files added after the
measurement; `pair.json` is a byte copy of the file used.

The goal was a configuration that fits an RTX 6000 Ada (48 GB). Measurements
were taken on an RTX PRO 6000 Blackwell Max-Q (96 GiB) with a separate
process holding GPU memory so that the server had 48 GiB available; the
container's host memory limit was 100 GiB. The numbers below are from that GPU.

The expert pool holds 258 expert rows per layer for 48 layers, about
32 GiB. The memory limit is set by the external process; the
`gpu-memory-utilization` below is vLLM's budget as a fraction of the
physical GPU. Check the available memory and that the server is up before
running the client.

## Historical server launch

These commands describe the September measurement and its PLE patch;
they are not launch instructions for current vLLM. Current vLLM uses
`--engram-config '{"disk_offload":true}'` for this model's PLE table, with
different staging behavior from the deferred PLE patch measured here.
Use the checkpoint path appropriate to your host.
`VLLM_USE_BREAKABLE_CUDAGRAPH` was unset (automatic selection).
`VLLM_DEBUG_WORKSPACE`, `VLLM_LOGGING_LEVEL`, `PYTORCH_ALLOC_CONF` and the
thread counts are the values used during measurement, not requirements.

```bash
export CHECKPOINT=/data/models/Qwen3.8-Flash-Next-NVFP4-nvidia
unset VLLM_USE_BREAKABLE_CUDAGRAPH
export VLLM_DEBUG_WORKSPACE=1 VLLM_LOGGING_LEVEL=DEBUG
export PYTORCH_ALLOC_CONF=pinned_max_round_threshold_mb:1,pinned_max_cached_size_mb:1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_PLE_MMAP=1 VLLM_PLE_MMAP_DEFERRED=1
export VLLM_PLE_MMAP_PREWARM=0 VLLM_PLE_MMAP_PINNED=0
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
.venv/bin/python -m vllm.entrypoints.openai.api_server \
  --model "$CHECKPOINT" --served-model-name flashnext \
  --host 0.0.0.0 --port 8000 --tensor-parallel-size 1 \
  --quantization modelopt --dtype bfloat16 --moe-backend marlin \
  --moe-expert-pool-rows 258 --language-model-only \
  --max-model-len 4096 --max-num-seqs 1 --max-num-batched-tokens 512 \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
  --no-enable-flashinfer-autotune --gpu-memory-utilization 0.4548806288994517 \
  --safetensors-load-strategy lazy \
  --default-chat-template-kwargs '{"enable_thinking":false}' \
  --reasoning-parser qwen3 --generation-config vllm
```

The speed measurement uses a context limit of 4096. The separate quality
checks used 32768.

## Client

Run once after the server is ready. The client needs only the Python
standard library.

```bash
.venv/bin/python benchmarks/expert_pool/benchmark.py \
  --base-url http://127.0.0.1:8000 --model flashnext \
  --label fresh-0 --output results/fresh-0.jsonl
```

Requests go to `/v1/completions` with the frozen token ids from
`pair.json`, so confirm that the checkpoint's tokenizer matches the SHA256
recorded under `tokenization` in `pair.json`. Each request uses
`temperature=0`, `top_p=1`, `seed=0`, `max_tokens=2048` and the
chat-template token sequence with thinking disabled.
`tokenization.pair_sha256` is the hash of the material before the token ids
were added; the hash of the whole file is in `provenance.json`.

For the three-run measurement, **stop the server and start a new process
before each run, then send warmup -> measure once**, writing to
`fresh-0.jsonl`, `fresh-1.jsonl`, `fresh-2.jsonl`. The reported value is the
median of the three measured-request speeds. The client refuses to
overwrite an existing output file, and saves HTTP errors and incomplete
streams before stopping; keep failed runs as results too.

## Metric and measured values

Decode speed is computed from the usage completion-token count and the
client-side receive times:

```text
decode_tok_s = (completion_tokens - 1) / (last text event time - first text event time)
```

`first_token_s` is the time from request start to the first text event;
`e2e_tok_s` is completion tokens over the whole request time. A stream
event may carry more than one token, so both are client-observed values.
The raw SSE events, request body, usage, finish reason, text and its SHA256
are stored in the same JSONL.

| fresh server run | measured-request decode speed |
| --- | --- |
| 1 | 63.1882 tok/s |
| 2 | 62.7087 tok/s |
| 3 | 63.6519 tok/s |
| median | **63.1882 tok/s** |

Measured on the combination listed above. Every request finished with
`stop`. Memory was sampled during startup and generation; the process
exited 0 with OOMKilled=false.

## Client self-check

```bash
.venv/bin/python -m unittest discover -s benchmarks/expert_pool -p 'test_*.py'
```
