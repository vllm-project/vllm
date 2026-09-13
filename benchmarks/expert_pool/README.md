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

## Code and environment measured

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

## Server launch

Same features as the measured combination, checkpoint path as needed.
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
