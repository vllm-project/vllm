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

Current execution supports local GPU operations and explicit request associations.
Add `--validate-request-associations` to check request snapshot scope, generated
growth, a resumed continuation with tool tokens, cache-hit membership and rejected
rewrites through real serving. Request growth requires `associate.observe_growth`;
completion alone never finishes a task. The recorded snapshot/hit prefixes contain
1,302 tokens, while the resumed live continuation contains 1,344. Each tracked
request hits 784 cached tokens. These checks use the common logical index and
actual engine observations, rather than application-invented residency.

Native Mooncake movement is validated below. CPU movement and policy workload
acceptance remain separate gates. No serving speedup is inferred from these runs.

## Native Mooncake movement and cross-engine preparation

Start an isolated Store and write its client configuration:

```bash
.venv/bin/mooncake_master --rpc_address=127.0.0.1 --rpc_port=19151 \
  --metrics_host=127.0.0.1 --metrics_port=19153 --enable_metric_reporting=false
```

```json
{
  "metadata_server": "P2PHANDSHAKE",
  "master_server_address": "127.0.0.1:19151",
  "protocol": "tcp",
  "device_name": "",
  "global_segment_size": "2GB",
  "local_buffer_size": "256MB"
}
```

Save the JSON as `/tmp/programmable-kv-mooncake-config.json`. Use the HTTP server
command above with `MOONCAKE_CONFIG_PATH` pointing to it, `OMP_NUM_THREADS=4`,
`VLLM_WORKER_MULTIPROC_METHOD=spawn`, and the following extra argument:

```bash
--kv-transfer-config '{"kv_connector":"MooncakeStoreConnector","kv_role":"kv_both","kv_connector_extra_config":{"enable_lookup":false,"save_decode_cache":false}}'
```

Disabling ordinary request lookup makes subsequent hits evidence of the explicit
Prefetch. Run the complete control workflow and movement accuracy comparison:

```bash
CUDA_VISIBLE_DEVICES=0,3 .venv/bin/python \
  benchmarks/programmable_kv_cache/http_control.py \
  --base-url http://127.0.0.1:18083 \
  --dataset /home/youwei/bzh/dataset/openai/gsm8k \
  --validate-movement mooncake --validate-request-associations \
  --evaluation-samples 128 --output /tmp/qwen-http-mooncake-evaluation.json
```

The accuracy comparison uses identical questions and model-visible prompts.
Unique per-question cache salts force recomputation in the baseline. The second
pass performs Demote and request-free Prefetch before each eight-question batch.
Both passes score **124/128 (96.875%)**, with no truncated answers and no numeric
answer mismatches. Cached tokens are **0 / 100,352**. Driver durations include
warmup and control operations; they are not a controlled speedup measurement.

For cross-engine validation, start a second server with the same model/layout,
Store configuration and TP=2, using `--port 18084`,
`--kv-control-config '{"engine_id":"http-destination"}'`, and
`--gpu-memory-utilization 0.45` when both instances share GPUs 0/3. Keep the fixed
4 GiB cache per rank. Add `--destination-url http://127.0.0.1:18084` to the workflow
command; `--evaluation-samples` can be omitted for this independent scenario.

The sender checks engine IDs, epochs and compatibility namespaces, submits Share
on A, prepares B through independent control, and then sends B its first inference
request. B hits **784 tokens**, its greedy output matches A, and A retains its own
warm prefix. The local Demote/Prefetch cycle moves and registers **four group
copies**. Share reports Store availability; only B's Prefetch reports B's readiness.
Compressed full responses and evaluation rows are saved under `results/`.

The current native adapter accepts full group boundaries and exact primary cache
entries with PCP=DCP=1; capability discovery reports these limits. Keys include
the control compatibility namespace and are separate from ordinary offload keys.
Paused/sleeping engines defer new physical work. In-process Python clients advance
idle connector work cooperatively during control polling.

## Native CPU movement

Use the same Qwen HTTP server configuration, replacing the connector argument with:

```bash
--kv-transfer-config '{"kv_connector":"OffloadingConnector","kv_role":"kv_both","kv_connector_extra_config":{"cpu_bytes_to_use":8589934592,"blocks_per_chunk":1}}'
```

Run `http_control.py` with `--validate-movement cpu`. The driver supplies the existing
`kv_transfer_params.max_load_tokens=0` on measured inference requests, so automatic
CPU lookup cannot substitute for the explicit Prefetch. The native adapter uses the
existing CPUOffloadingManager capacity and CPUOffloadingWorker copies. It accepts
full group boundaries and one block per CPU chunk, with PCP=DCP=1. Tiering/KVCR and
SimpleCPU do not advertise this native CPU adapter; CPU Share remains unsupported.

The recorded TP=2 workflow stores/reclaims and then restores **four group copies**,
and resumed inference hits **784 tokens** with the same greedy output. With
`--evaluation-samples 128`, recomputation scores **124/128 (96.875%)** and CPU
Prefetch scores **125/128 (97.65625%)**; neither truncates an answer. There is one
final numeric-answer difference. Cached tokens are **0 / 100,352**. Full outputs
and control results are in the `cpu-*.json.gz` result files. These are correctness
evaluations, not evidence that movement improves model accuracy or serving speed.

## Policy, rollout and recovery controllers

The native policy runtime is configured at engine startup. An empty `policies`
list keeps signal actions unconsumed while explicit Retain/Release/Prefetch
commands continue to work. A minimal configuration is:

```json
{
  "engine_id": "policy-validation",
  "policies": [
    {"name": "retention", "implementation": "retention", "options": {"ttl_seconds": 30}},
    {"name": "movement", "implementation": "tool_wait", "options": {"tier": "mooncake"}},
    {"name": "quota", "implementation": "quota", "options": {"max_wait_seconds": 1}}
  ],
  "composer": "priority"
}
```

Policies receive immutable pressure/request observations and submit proposals
through the same engine approval path as application commands. They cannot
rewrite explicit destinations or mutate scheduler queues; changing the policy
set requires an engine restart. The policy tests cover shared proposals,
cancellation, capacity competition, request scoring and prediction invalidation.

`rollout.py` generates a seeded multi-round, multi-branch plan with a finite
pressure budget. With `--base-url` it sends association, retention, prefetch
and release actions through HTTP and records every result:

```bash
.venv/bin/python benchmarks/programmable_kv_cache/rollout.py \
  --base-url http://127.0.0.1:18083 --rounds 3 --branches 4 \
  --pressure-blocks 8 --seed 7 --output /tmp/programmable-rollout.json
```

`recovery.py` exercises duplicate replay, unknown action versions, event gaps
and a stale restarted epoch. `verl_agent_loop.py` consumes a JSONL export from
the open-source verl Agent Loop, emits supported envelope actions, and records
the weight version so obsolete local or remote results can be rejected after a
version change. `coverage_check.py` verifies that each implemented coverage row
has source, tests and a runnable workload artifact.

These adapters provide reproducible control/workload evidence. They do not
replace the required real-transfer and external-model evaluations; those remain
listed as incomplete when their environment is unavailable.
