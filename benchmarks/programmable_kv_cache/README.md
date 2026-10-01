# Programmable KV cache validation

`evaluate.py` establishes a model baseline using downloaded GSM8K data. It runs
the same seeded questions with forced prefix recomputation and with prefix-cache
reads enabled, recording per-question outputs, numeric correctness, cache hits,
truncation and output differences. The shared few-shot prefix must reach the
model's actual cache-group boundaries.

```bash
CUDA_VISIBLE_DEVICES=0,3 OMP_NUM_THREADS=4 \
VLLM_WORKER_MULTIPROC_METHOD=spawn HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
.venv/bin/python benchmarks/programmable_kv_cache/evaluate.py \
  --model /home/youwei/bzh/model/Qwen/Qwen3.8-27B \
  --dataset /home/youwei/bzh/dataset/openai/gsm8k \
  --num-samples 128 --num-fewshot 8 --max-tokens 1024 \
  --tensor-parallel-size 2 --kv-cache-memory-bytes 4294967296 \
  --output /tmp/programmable-kv-gsm8k-baseline-results.json
```

The local baseline uses model revision
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` and dataset revision
`740312add88f781978c0658806c59bc2815b9866`. Model and data preparation used the
user-supplied download functions. Other environments can pass their own local
paths.

## Recorded local baseline

The compressed JSON under `results/` contains the configuration and all 256
generation records. Both modes score 125/128 (97.66%); neither truncates an answer.
Forced recomputation has zero cached tokens; the cache-enabled run has 99,568.
Final numeric answers match on all questions, while token sequences differ on 27.

This run leaves the new programmable controller disconnected from serving.
It validates the existing model/cache baseline and does not demonstrate native
hint execution, backend transfers, policy benefits or a serving speedup.
The complete serving/backend/policy matrix remains in the OpenSpec task list.

## Native control runtime

Pass `--enable-kv-control` to the accuracy command to initialize the native control
runtime and exercise request-free association, following retention, replay, query,
events, stale-epoch rejection and explicit cleanup before the two 128-question
runs. The enabled run scores 125/128 in each mode, with no truncation or final
numeric-answer mismatches; token sequences differ on 33 questions. These are
accuracy/transport checks, with no cold-tier transfer or policy comparison.

`http_control.py` exercises a real serving process. Start it with:

```bash
CUDA_VISIBLE_DEVICES=0,3 .venv/bin/python -m vllm.entrypoints.openai.api_server \
  --model /home/youwei/bzh/model/Qwen/Qwen3.8-27B \
  --served-model-name qwen-validation --tensor-parallel-size 2 \
  --max-model-len 4096 --max-num-seqs 8 --max-num-batched-tokens 2048 \
  --kv-cache-memory-bytes 4294967296 --mamba-cache-mode align \
  --enforce-eager --disable-custom-all-reduce --dtype bfloat16 \
  --enable-prefix-caching --enable-prompt-tokens-details \
  --kv-control-config '{"engine_id":"http-validation"}' \
  --host 127.0.0.1 --port 18083
```

Then run:

```bash
CUDA_VISIBLE_DEVICES=0,3 .venv/bin/python \
  benchmarks/programmable_kv_cache/http_control.py \
  --base-url http://127.0.0.1:18083 \
  --dataset /home/youwei/bzh/dataset/openai/gsm8k \
  --output /tmp/qwen-http-kv-control.json
```

The workflow uses unique command IDs, the advertised hash/group profile and actual
server tokenization. It validates cold/warm cache hits, Retain/Release, two logical
branches, append-only rewrite rejection, physical Evict, completion/chat/streaming
attached hints, correlated queries/events, unsupported actions/versions/movement,
and engine/epoch/namespace errors. It checks identical greedy output token IDs
across cold, warm and evicted runs. Local results are **0 → 784 → 0 cached tokens**
and **4 GPU copies reclaimed**. Raw responses are saved under `results/`.

Current execution supports local GPU operations. CPU/Mooncake movement, automatic
request association/growth and policy workloads remain incomplete; the script
explicitly expects unsupported movement in this development stage. Update that
assertion when the adapters are connected. No serving speedup is inferred from
these runs.
