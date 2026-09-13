# Validate decode CUDA Graph configuration

The Rust frontend forwards engine configuration to the Python vLLM engine.
An experiment that explicitly passes `--enforce-eager` disables CUDA Graphs;
its result should be distinguished from the upstream default configuration.
This guide shows how to compare that explicit control with decode-only Graphs
while keeping the frontend binary, model and request workload fixed.

## Start either configuration

Build the Rust frontend with `./build_rust.sh` and install the matching Python
checkout. Use the same local model snapshot and binary for both arms. Set
`MODEL_SNAPSHOT` to a downloaded, fixed revision of `Qwen/Qwen3-0.6B`.
Run one arm at a time on the same idle GPU and restart the entire server between
arms; stop the server you started before launching the other configuration.

```bash
export MODEL_SNAPSHOT=/path/to/Qwen3-0.6B
export VLLM_USE_RUST_FRONTEND=1
export VLLM_BATCH_INVARIANT=1
export TOKIO_WORKER_THREADS=2
export VLLM_RS_REQUEST_WORKER_THREADS=2
export RAYON_NUM_THREADS=2
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2

common_args=(
  "$MODEL_SNAPSHOT"
  --served-model-name Qwen/Qwen3-0.6B
  --host 127.0.0.1 --port 8000
  --api-server-count 1 --tensor-parallel-size 1
  --generation-config vllm --seed 0 --dtype bfloat16
  --max-num-seqs 8 --max-model-len 2048
  --no-enable-prefix-caching --kv-cache-memory-bytes 536870912
  --enable-auto-tool-choice --tool-call-parser hermes
  --reasoning-parser qwen3
)

# A: explicit eager control.
vllm serve "${common_args[@]}" --enforce-eager

# P: after stopping A, start a fresh server with the same common arguments.
vllm serve "${common_args[@]}" \
  --compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4,8]}'
```

`mode=0` disables compilation; `FULL_DECODE_ONLY` requests full Graphs for
uniform decode batches, with eager prefill. Capture sizes above the tested batch
range can change memory and startup costs. Do not pass `--enforce-eager` to the
Graph arm. Confirm the resolved configuration and successful capture in server
logs, then inspect an independent trace for actual Graph replay during decode.
A configuration string or capture-complete log alone does not prove replay.

## Compare a fixed workload

Preserve full request bodies, response bodies and process/configuration logs.
Use temperature zero, a fixed seed and fixed model/template files. For
completions, request `logprobs: 5`, `max_tokens: 128`, `ignore_eos: true` and
`stream: false`. Check exactly 128 completion tokens in each response.

Include real tool-call chat and a separate short-chat holdout. For the latter,
request 16 tokens, `ignore_eos: true`, `logprobs: true`, `top_logprobs: 5` and
`chat_template_kwargs: {"enable_thinking": false}`. Verify usage, finish reasons,
tool names and arguments, and all requested token/logprob fields. Normalize only
top-level response `id`/`created` and generated tool-call IDs. Keep every other
field in the comparison, including warmups and failed responses.

The confirmation protocol below separates tuning from measurement:

1. Freeze the request set and performance criteria before confirmation. Use
   concurrency two; each arm has four warmups per workload, then 32 completions,
   32 tool chats and 16 short-chat requests.
2. Run four APPA and four PAAP quartets in a shuffled order with a fixed seed.
   Each letter starts a fresh engine process; symmetric on-disk caches may stay
   warm. Exclude the independent replay trace from performance samples.
3. Compare all responses against the first arm by workload, phase and request
   index. A missing, duplicate, failed or differing response fails correctness.
4. Compute a geometric P/A ratio within each quartet. Bootstrap whole quartets,
   rather than treating correlated requests as independent samples. In the
   confirmation below, 20,000 samples with seed 20260912 were used.
5. Require all eight quartets to be valid, primary throughput improvement CI
   lower bound above 2%, and every workload's p95 regression CI upper bound at
   most 5%. Baseline throughput drift within any quartet must be at most 10%.
   Retain failed or interrupted runs instead of replacing them until a pass.

## Observed configuration comparison

On source `06e57f622cf53ed1c3f8d8295cdd48ea2c133bd1`, both arms used the same
unmodified Rust binary, fixed Qwen3-0.6B files, BF16, TP1, batch invariance,
prefix caching disabled and a 512 MiB KV budget.

| GPU | Completion throughput P/A | 95% CI | Valid quartets |
| --- | ---: | ---: | ---: |
| NVIDIA RTX 5060 Ti | 3.3194× | 3.2941–3.3551× | 8/8 |
| NVIDIA RTX PRO 4000 Blackwell | 4.0939× | 4.0722–4.1178× | 8/8 |

Each independent GPU confirmation completed 32 fresh engine processes,
2,944 HTTP requests and 2,852 strict comparisons with zero differences or
HTTP errors. All three workload p95 guards passed. Separate replay traces
were excluded from these timings. The hardware cohorts were not pooled.

These measurements compare an explicit eager test configuration with a
Graph configuration. They do not measure a Rust code change or an improvement
over upstream defaults. A 16-request short-chat cohort has a nearest-rank p95
at its maximum, so it is not a well-sampled production tail estimate.
Non-streaming responses do not provide TTFT, ITL or TPOT. Tool chat did not
request token logprobs in this confirmation. Recheck correctness, actual replay
and latency for other models, releases, hardware and workload shapes.
