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

Native Mooncake and CPU movement are validated below. Policy workload
acceptance remains a separate gate. No serving speedup is inferred from these runs.

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

## Optional policy configuration

Policies load at startup; changing them requires restart. An empty `policies`
list leaves native signals unconsumed. Built-in implementations are `retention`,
`tool_wait` and `quota`. Each instance has a unique name, an implementation name
(or an importable `KVPolicy` subclass) and constructor `options`. `composer`
selects `priority` or an importable `KVPlanComposer` subclass.

```json
{
  "engine_id": "policy-validation",
  "policies": [
    {"name": "retain", "implementation": "retention", "options": {"ttl_seconds": 30}},
    {"name": "movement", "implementation": "tool_wait", "options": {"tier": "mooncake"}},
    {"name": "quota", "implementation": "quota", "options": {"max_wait_seconds": 1}}
  ],
  "composer": "priority"
}
```

Policies receive immutable observations and submit proposals through engine
approval. Signal completion reports consumption; each entry in
`derived_operations` must be queried to determine movement completion. The
configured composer chooses compatible proposals in configuration order. Explicit
retention or movement takes precedence over conflicting policy movement.

Quota observations/guidance refresh at `policy_interval_seconds` (default 0.05).
Signals invalidate the cached guidance. The scheduler bypasses optional deferral
at `max_policy_wait_seconds` even between observations; physical capacity checks
still apply. Request scores respect explicit scheduler priorities and use arrival
order to break ties. Observation overflow disables guidance until a complete
bounded observation is available. These are advisory estimates; selection still
scans the waiting queue and no end-to-end performance gain is established.

## Protocol probes and remaining workload coverage

```bash
.venv/bin/python benchmarks/programmable_kv_cache/recovery.py \
  --base-url http://127.0.0.1:18083 --engine-id http-validation \
  --output /tmp/kv-protocol-checks.json
```

The probe checks unknown-version rejection, exact duplicate replay, operation
query agreement and invalid epoch/cursor errors. It uses fresh IDs and explicitly
selects one engine. A stale epoch probe does not restart a server; a future cursor
probe does not reproduce lost event history. Real bounded association-history
loss and reconstruction are covered by the HTTP control tests.

Association query pages include `engine_epoch`, `namespace`, `revision` and
`event_sequence`. For recovery, paginate at the first page's revision and resume
association events from the first page's event sequence. Restart pagination on
`stale_revision`; take a fresh snapshot on `event_gap`. Association state does
not establish GPU residency. Physical residency queries/events remain incomplete.
Live weight-version changes are explicitly rejected with programmable control
enabled; restart with the new weights/version until task 7.2 is implemented.

The earlier `rollout.py`, `verl_agent_loop.py` and `coverage_check.py` were removed:
empty-target commands, an unverified trace format and file-existence checks could
not establish their claimed workload/coverage behavior. Multi-round pressure,
application-directed rollout, real verl integration, weight-update invalidation,
policy comparisons and restart/failure recovery remain in the OpenSpec task list.

## CPU control overhead

```bash
.venv/bin/python benchmarks/programmable_kv_cache/control_overhead.py \
  --output /tmp/control-overhead.json
```

This measures result polling and policy observations with a fixed clock, no
inference requests, and empty/full CPU cache metadata. The October 6 comparison
against `a41f867f69` records all five samples in
`results/control-overhead-2026-10-06.json.gz`. Median polling with 4,096 retained
results changes from 220.82 to 0.19 microseconds. Repeated guidance within one
observation interval changes from 10.99 milliseconds to 0.15 microseconds for
100,000 blocks. A fresh full-cache observation remains a scan (11.33 / 9.59 ms).
These are CPU component measurements, not throughput, fairness or disabled-path
serving measurements. No timing threshold is asserted in pytest.

The October 6 quality check additionally enables the built-in `quota` policy on
this CPU server. The 128-question recompute/prefetch runs both score 124/128 with
zero final numeric-answer mismatches, no truncation and 0/100,352 cached tokens.
`quality-quota-cpu.json.gz` records the long run; `quality-head-http.json.gz`
records a fresh-server run of the final source and actual restart/epoch checks
(both filenames carry the `qwen3.8-27b-tp2-` prefix). Process boundaries and exact
commands are recorded in the OpenSpec validation log. These do not establish
quota fairness or the complete policy/failure workload matrix.

## RFC #59708 primitive reference workflow

The `rfc-hint` branch adds `primitive_control.py`. Start the existing Qwen TP=2
Mooncake server above with engine ID `primitive-source`, ordinary lookup disabled
and no local policies. Use the isolated Store configuration described above.

```bash
CUDA_VISIBLE_DEVICES=0,3 .venv/bin/python \
  benchmarks/programmable_kv_cache/primitive_control.py \
  --base-url http://127.0.0.1:18083 --engine-id primitive-source \
  --dataset /home/youwei/bzh/dataset/openai/gsm8k \
  --output /tmp/rfc-hint-attached.json
```

R1 carries `vllm.move_to` and saves/reclaims the selected prefix at successful
completion. R2 carries `vllm.copy_to` and waits for Store→GPU loading at admission.
The driver uses a fresh cache salt, asserts every group was actually transferred
(with transfer IDs), checks cold/GPU milestones, then verifies cached tokens and
identical greedy outputs. Its request/range selector includes detached hybrid
checkpoints. This is a real attached-command workflow, not a request-free hint
sent after R1. COPY at R2 arrival has zero pre-arrival lead time.

To exercise the optional policy producer, start an engine with:

```json
{"engine_id":"primitive-destination","policies":[{"name":"movement","implementation":"tool_wait","options":{"tier":"mooncake"}}]}
```

Run the driver against that engine with `--policy`. After the attached workflow,
it submits explicit signals, follows each derived primitive result, then verifies
actual reuse. This does not infer tool calls from output tokens. `client.py`
provides shared envelope creation and queued-aware polling for both controllers.

The existing direct controller now accepts `--primitive-actions`, translating its
movement/priority recipes to the public primitive profile while retaining explicit
state commands as compatibility extensions. Add `--destination-url` for two-engine
COPY through the shared Store, and `--evaluation-samples 128` for the existing
seeded GSM8K recomputation/movement comparison. Old command names remain available
without the flag. New example failures preserve a partial JSON report.

Detailed payloads, copy-versus-logical priority lifetimes, ownership restrictions,
queued versus deferred status, and partial-scope reporting are documented in
[`vllm/v1/kv_hints/README.md`](../../vllm/v1/kv_hints/README.md).

The recorded `rfc59708-*.json.gz` results cover request-attached execution, the
optional movement policy, two-engine COPY and the 128-question accuracy comparison.
`rfc59708-composed.json.gz` additionally enables both `retention` and `tool_wait`:
the cold COPY finishes before its companion SET_PRIORITY applies, so TTL begins
on an actual cached copy. The resumed request again hits 784 tokens. These are
functional reference examples; broader policy/pressure comparisons remain open.
