# V4_l0_usdt_ebpf_probe_validation

## Goal

This engineering validation validates that vLLM high-level Python L0 request
lifecycle semantic events can be exposed as lightweight user-space tracepoints
and observed with `bpftrace`.

The target is USDT/eBPF reachability of L0 request lifecycle events. This is
not a final Primary collector validation, a JSONL trace validation, or a direct
uprobe of Python functions.

## Why Not Directly Uprobe Python Functions

The L0 lifecycle is mostly implemented in Python methods such as
`LLM.generate()`, `LLM._add_request()`, `LLM._run_engine()`,
`LLMEngine.add_request()`, `AsyncLLM.generate()`, `AsyncLLM.add_request()`,
`AsyncLLM._add_request()`, and `AsyncLLM.abort()`.

These Python methods are not stable ELF symbols. Direct uprobes can attach to
CPython interpreter symbols, but that does not provide a stable, low-friction
way to recover vLLM semantic fields such as `request_id`.

## Why USDT

USDT is a user-space static tracing mechanism. vLLM Python code can explicitly
fire a probe at the semantic point where `request_id`, lifecycle event name, and
extra metadata are already available. eBPF tools such as `bpftrace` can then
attach to those probes and collect the fields.

This experiment uses `stapsdt` / `python-stapsdt` as a POC provider backend. If
it is unavailable, the instrumentation is a no-op and vLLM continues to run.

## L0 Observed Events

Provider: `vllm_l0`

Each probe receives two arguments:

- `char *request_id`
- `char *extra`

`request_id` is passed as UTF-8 bytes. `extra` is a compact JSON string.

This validation covers these probes:

- `request_arrival`
- `request_id_mapping`
- `request_id_assigned`
- `request_engine_admitted`
- `request_first_output`
- `request_output`
- `request_finish`
- `request_abort`

The registered provider also exposes `first_token`, `output_token`,
`request_terminal`, `request_reject`, and `request_error`. They are outside the
scope of this focused validation.

## Implementation

The helper lives in `vllm/l0_usdt.py`.

Environment variables:

- `VLLM_L0_ENABLE_USDT=1`: enable USDT provider registration. The legacy
  compatibility name `VLLM_L0_USDT_ENABLE=1` is also accepted.
- `VLLM_L0_TRACE_PATH=/tmp/vllm_l0_sidecar.jsonl`: optional in-process JSONL
  sidecar path.
- `VLLM_L0_ENABLE_SIDECAR=0`: disable the sidecar explicitly when a trace path
  is present.

Default behavior:

- USDT is disabled by default.
- If `stapsdt` or `libstapsdt` is unavailable, USDT emission is a no-op and
  vLLM continues to run.
- Probe fire failures are caught and do not affect inference.
- JSONL is written only when `VLLM_L0_TRACE_PATH` is set and the sidecar is not
  disabled. It is an auxiliary same-hook transport oracle, not ground truth.

## Evidence Scope

`bpftrace` output in this document is labelled
`ENGINEERING_USDT_BPFTRACE_POC`. It is useful for bounded probe reachability
and transport debugging, but is not the final Primary
USDT → eBPF → libbpf-collector evidence path.

The optional Python JSONL sidecar is labelled
`AUXILIARY_SIDECAR_TRANSPORT_ORACLE`. It may reveal disagreement with the
captured USDT events, but must not fill missing USDT events or contribute rows
to a Primary result.

`stapsdt` installation for the vLLM development environment:

```bash
uv pip install stapsdt
```

`stapsdt` may also require the native `libstapsdt` runtime. If probes do not
appear in `bpftrace -l`, install `libstapsdt` for the host and restart vLLM.

## Instrumentation Points

Offline path:

- `vllm/entrypoints/llm.py::LLM._render_and_add_requests`
  - fires `request_arrival`
  - fields: `path="offline"`, `prompt_index`, `local_request_id`
- `vllm/entrypoints/llm.py::LLM._add_request`
  - fires `request_id_assigned`
  - fields: `path="offline"`, `prompt_index`, `local_request_id`,
    `params_type`, `output_kind`, `priority`
- `vllm/v1/engine/llm_engine.py::LLMEngine.add_request`
  - fires `request_engine_admitted` after `engine_core.add_request(...)`
  - fields: `path="offline_or_engine"`, `priority`, `arrival_time`
- `vllm/entrypoints/llm.py::LLM._run_engine`
  - fires `request_finish` when `output.finished`
  - fields: `path="offline"`, `finished`, `output_token_count`,
    `finish_reason`

Online / streaming path:

The online server path has two request ids:

- `external_request_id`: the OpenAI API / streaming-output visible id, for
  example `cmpl-...-0`.
- `internal_request_id`: the vLLM EngineCore id assigned by
  `InputProcessor.assign_request_id(...)`, for example `cmpl-...-0-acd3f4bd`.

Both ids are intentionally preserved. The `request_id_mapping` event records
the relationship so analysis can merge API/output-side events and engine-side
events into one canonical lifecycle.

- `vllm/v1/engine/async_llm.py::AsyncLLM.generate`
  - fires `request_arrival` at entry
  - fires `request_first_output` before the first server-side `yield out`
  - fires `request_output` before each server-side `yield out`
  - fires `request_finish` when the yielded output is finished, or when the
    streaming-input sentinel `STREAM_FINISHED` is observed
- `vllm/v1/engine/async_llm.py::AsyncLLM.add_request`
  - fires `request_id_mapping` after `assign_request_id(...)`
  - fields: `path="online"`, `external_request_id`, `internal_request_id`
  - fires `request_id_assigned` after `assign_request_id(...)`
  - uses `internal_request_id` as the probe `request_id`
- `vllm/v1/engine/async_llm.py::AsyncLLM._add_streaming_input_request`
  - fires `request_id_mapping` for streaming-input requests after the final
    internal request id is assigned
  - fires `request_id_assigned` for streaming-input requests after the final
    internal request id is assigned
- `vllm/v1/engine/async_llm.py::AsyncLLM._add_request`
  - fires `request_engine_admitted` after
    `engine_core.add_request_async(...)`
- `vllm/v1/engine/async_llm.py::AsyncLLM.abort`
  - fires `request_abort` after engine abort submission
  - fields: `path="online"`, `internal`, `abort_source`, `request_ids`,
    `engine_request_ids`

## Running the Offline Validation

Start the target process with USDT enabled:

```bash
export VLLM_L0_ENABLE_USDT=1
export VLLM_L0_TRACE_PATH=/tmp/vllm_l0_sidecar.jsonl
```

Example offline driver:

```bash
.venv/bin/python - <<'PY'
from vllm import LLM, SamplingParams

llm = LLM(model="facebook/opt-125m", enforce_eager=True)
outputs = llm.generate(
    ["Hello from L0 USDT"],
    SamplingParams(max_tokens=8),
    use_tqdm=False,
)
print(outputs[0].outputs[0].text)
PY
```

The provider is created in the target Python process. Use a long-lived driver
or pause before `generate()`, then attach `bpftrace` after the provider is
loaded and before the request is dispatched. In another terminal, find that
process:

```bash
pgrep -f vllm
pgrep -f python
```

List USDT probes:

```bash
sudo bpftrace -l 'usdt:*' -p "$PID" | grep vllm_l0
```

Expected offline probes include:

```text
usdt:/tmp/vllm_l0-XXXXXX.so:vllm_l0:request_arrival
usdt:/tmp/vllm_l0-XXXXXX.so:vllm_l0:request_id_assigned
usdt:/tmp/vllm_l0-XXXXXX.so:vllm_l0:request_engine_admitted
usdt:/tmp/vllm_l0-XXXXXX.so:vllm_l0:request_finish
```

Collect the minimal offline events:

```bash
sudo bpftrace -p "$PID" -e '
usdt:*:vllm_l0:request_id_assigned
{
  printf("event=request_id_assigned ts=%llu request_id=%s extra=%s\n",
         nsecs, str(arg0), str(arg1));
}

usdt:*:vllm_l0:request_engine_admitted
{
  printf("event=request_engine_admitted ts=%llu request_id=%s extra=%s\n",
         nsecs, str(arg0), str(arg1));
}

usdt:*:vllm_l0:request_finish
{
  printf("event=request_finish ts=%llu request_id=%s extra=%s\n",
         nsecs, str(arg0), str(arg1));
}
'
```

If the probe name format differs, copy the full probe name from
`bpftrace -l` and replace the wildcard forms above.

Example output:

```text
event=request_id_assigned ts=83946253100511 request_id=0 extra={"path":"offline","prompt_index":0,"local_request_id":"local-0","params_type":"SamplingParams","output_kind":"RequestOutputKind.FINAL_ONLY","priority":0}
event=request_engine_admitted ts=83946253611840 request_id=0 extra={"path":"offline_or_engine","priority":0,"arrival_time":null}
event=request_finish ts=83949122710219 request_id=0 extra={"path":"offline","finished":true,"output_token_count":8,"finish_reason":"length"}
```

## Running the Online Validation

Start the OpenAI-compatible server with the same environment variables:

```bash
export VLLM_L0_ENABLE_USDT=1
export VLLM_L0_TRACE_PATH=/tmp/vllm_l0_sidecar.jsonl
vllm serve facebook/opt-125m
```

Attach bpftrace to the server PID and add the online probes:

```bash
sudo bpftrace -p "$PID" -e '
usdt:*:vllm_l0:request_arrival
{
  printf("event=request_arrival ts=%llu request_id=%s extra=%s\n",
         nsecs, str(arg0), str(arg1));
}

usdt:*:vllm_l0:request_id_mapping
{
  printf("event=request_id_mapping ts=%llu request_id=%s extra=%s\n",
         nsecs, str(arg0), str(arg1));
}

usdt:*:vllm_l0:request_first_output
{
  printf("event=request_first_output ts=%llu request_id=%s extra=%s\n",
         nsecs, str(arg0), str(arg1));
}

usdt:*:vllm_l0:request_output
{
  printf("event=request_output ts=%llu request_id=%s extra=%s\n",
         nsecs, str(arg0), str(arg1));
}

usdt:*:vllm_l0:request_finish
{
  printf("event=request_finish ts=%llu request_id=%s extra=%s\n",
         nsecs, str(arg0), str(arg1));
}

usdt:*:vllm_l0:request_abort
{
  printf("event=request_abort ts=%llu request_id=%s extra=%s\n",
         nsecs, str(arg0), str(arg1));
}
'
```

The first-output timestamp is the server-side yield point. It is not client-side
TTFT.

## Analyzing Captured Output

Save bpftrace output to a text file and run:

```bash
.venv/bin/python scripts/analyze_l0_usdt_trace.py \
  /tmp/vllm_l0_bpftrace.txt \
  --sidecar-jsonl /tmp/vllm_l0_sidecar.jsonl
```

The script groups events by `request_id` and reports:

- `canonical_request_id`
- `raw_request_ids`
- `profile` (`offline` or `online`)
- `bpftrace_evidence_class=ENGINEERING_USDT_BPFTRACE_POC`
- `has_request_id_assigned`
- `has_engine_admitted`
- `has_first_output`
- `has_finish`
- `has_abort`
- `event_count`
- `first_event_ts`
- `last_event_ts`
- `lifecycle_duration_ms`
- `event_sequence`
- `usdt_event_sequence`
- `sidecar_event_sequence`
- `core_event_order_ok`

For online requests, the analyzer first reads `request_id_mapping` events and
builds aliases:

```text
alias_map[external_request_id] = external_request_id
alias_map[internal_request_id] = external_request_id
```

Grouping then uses the canonical request id rather than the raw probe
`request_id`. The output keeps both ids in `raw_request_ids`.

With sidecar JSONL, it also reports:

- `sidecar_event_count`
- `usdt_event_count`
- `missing_in_usdt`
- `extra_in_usdt`
- `timestamp_delta_ms_summary`
- `sidecar_evidence_class=AUXILIARY_SIDECAR_TRANSPORT_ORACLE`
- `timestamp_delta_pairing=same_canonical_request_and_event_order`

Use `--profile offline` or `--profile online` to force the applicable
lifecycle contract. The default `--profile auto` selects `online` when it
observes online-only events or `path="online"`; otherwise it selects
`offline`. The offline contract is:

```text
request_arrival → request_id_assigned → request_engine_admitted → request_finish
```

The online contract additionally requires request-ID mapping and the
server-side first-output boundary. The timestamp summary compares only events
within one canonical request and event order; it is not a cross-request latency
measurement.

## Success Criteria

The experiment succeeds when:

1. `bpftrace -l 'usdt:*' -p "$PID" | grep vllm_l0` lists vLLM L0 probes.
   Online validation should include `vllm_l0:request_id_mapping`.
2. One offline inference emits:
   - `request_arrival`
   - `request_id_assigned`
   - `request_engine_admitted`
   - `request_finish`
3. The three event classes have matching `request_id` values.
4. `request_finish_ts - request_id_assigned_ts` can be computed per
   `request_id`.
5. Online streaming additionally emits:
   - `request_id_mapping`
   - `request_first_output`
   - `request_output`
   - `request_finish`
   `request_id_mapping.extra` must include both `external_request_id` and
   `internal_request_id`.
6. With `VLLM_L0_ENABLE_USDT` (and its legacy compatibility name) unset, vLLM
   runs normally and no USDT provider is registered.
7. Without the optional USDT dependency, vLLM runs normally and simply produces
   no USDT events.
8. Analyzer output for one online request is not split into separate
   external/internal summaries. It should contain both ids in `raw_request_ids`,
   and the merged summary should have:
   - `has_request_id_assigned = true`
   - `has_engine_admitted = true`
   - `has_first_output = true`
   - `has_finish = true`
   - `missing_in_usdt = []` and `extra_in_usdt = []` for this bounded
     engineering comparison only
   - `core_event_order_ok = true`

## Current Limitations

- This is low-intrusion instrumentation, not a completely non-intrusive probe.
- This validates Python semantic events exposed through USDT; it is not direct
  uprobes on Python methods.
- Server-side `request_first_output` is not client-side TTFT.
- Offline `LLM.generate()` uses `FINAL_ONLY`, so this path does not provide
  per-token streaming output.
- The POC uses `stapsdt`. Some `stapsdt` versions expose only integer probe
  argument metadata, so the code falls back to pointer-sized arguments while
  still passing UTF-8 bytes and reading them with `str(arg0)` / `str(arg1)`.
- A future production version can replace the Python USDT POC with a C
  extension USDT provider and the project Primary path requires a dedicated
  libbpf collector.
