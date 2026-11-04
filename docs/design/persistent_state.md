# Persistent State

Status: experimental

Persistent state lets a model keep a fixed-size, per-request state bundle on
the GPU across scheduler steps and across the inputs of a resumable (streaming)
request. The scheduler owns allocation, lifetime, and release of each bundle; the
model only reads and writes the slot it is given for the current step. It
requires [Model Runner V2](model_runner_v2.md).

## State contract

A model declares its state by constructing a
`vllm.model_executor.layers.state.PersistentStateLayerBase` with a
`PersistentStateSpec`:

- A spec is one aggregate page made of named `PersistentStateDescriptor`
  components, each with a shape, dtype, byte offset, alignment, and an
  initializer callable. Components must not overlap and the page size must be the
  aligned extent of the layout. `block_size` is always 1.
- The KV cache layout must be state-only: exactly one cache group containing one
  persistent-state layer. Mixing persistent state with attention or Mamba cache
  groups is refused at startup.
- The KV cache memory budget is divided into state pages of `page_size_bytes`
  each, at most `max_num_seqs + 1`; one page is the reserved null block and the
  rest are slots. `--num-gpu-blocks-override` sets the page count directly,
  within the budget. Each request holds exactly one slot from first schedule
  until it is released; slots are never shared, prefix-cached, or swapped.
- Each allocation is identified by a `StateSlotHandle` (engine epoch, slot,
  generation). The generation increases on every reuse, so a stale handle can
  never address a recycled slot. A new generation starts from the descriptors'
  initializers.
- The worker exposes only the current generation's slot to the model, through
  `ModelState.state_slots`. Models that write state from a stream other than the
  runner's main stream must register that stream with
  `StateSlotBindings.register_writing_stream` before enqueueing the writes.
- The engine core selects `StateScheduler`
  (`vllm/v1/core/sched/state_scheduler.py`) when the KV cache layout holds
  persistent state. A configured `scheduler_cls` must subclass it and must not
  be an async scheduler.
- Preempting a request with persistent state finishes it with an error instead
  of recomputing it, because the state cannot be rebuilt from tokens.
- A resumable request parks for its next input only when a turn stops on EOS.
  Any other stop (length, stop token or string) ends the stream. Each new input
  replaces the request's prompt; the state carries the history. Multimodal
  positions are relative to the input's own prompt, and an input with an item
  outside it is refused before it is applied or queued.

[examples/features/persistent_state/state_slot_lifecycle.py](../../examples/features/persistent_state/state_slot_lifecycle.py)
walks one stream through admission, a scheduled step, parking, the release
fence, and slot reuse on the CPU, with a synthetic spec and no model weights.

## Model hooks

A persistent-state model uses these Model Runner V2 hooks:

- `ModelState.initialize_state_slots(bindings)` is called after the KV cache is
  allocated, with the `StateSlotBindings` that map a request's current handle
  to its state views. Override it to register auxiliary state-writing streams.
- `ModelState.prepare_step(scheduler_output, *, dummy_run, is_profile)` is
  called at the start of each `execute_model`, so the model can tell profiling
  and dummy runs from scheduled steps.
- A model class that sets `supports_persistent_state = True` also gets
  `prepare_step` on the dummy and profiling runs that happen before its state
  slots are bound. Without it, `prepare_step` runs only once slots exist.
- An optional `prepare_execution_memory()` method on the model is called once
  at the start of `profile_run`, before KV cache memory is measured. Allocate
  long-lived execution buffers there, such as workspaces or captured graphs, so
  they are counted before the remaining memory is divided into state pages.

## Release fence

A finished or aborted request does not free its slot immediately. The scheduler
retires the handle and sends it to the worker as a cleanup handle. The worker
joins the main stream with every registered state-writing stream, records an
event, and returns a `StateCompletion` with the model output only after that
event completes. The slot returns to the free pool only when the scheduler has
received a completion covering both the request's last scheduled step and its
cleanup. Until then, admission and scheduling treat the slot as occupied, so a
new request can never observe writes from the previous owner. Model failures are
reported through the same completion path, and a scheduled step without a
matching completion fails its request. A cleanup that is not acknowledged is
retried once; a second miss is a fatal engine error.

## Stream admission

With persistent state enabled, the engine core exposes the utility method
`admit_stream(request)`, which reserves a slot before any input arrives. It
atomically reserves a slot and parks an empty resumable request. The request
must be an empty, resumable sampling request (`n=1`, no stop strings, no
structured outputs, no transfer connectors). Async frontends call it through
`EngineCoreClient.admit_stream_async`, which routes the stream's outputs back
to the calling API server, as for an ordinary request.

The request ID names the stream, so the caller must give every session a
globally unique ID and never reuse it. vLLM's input processor adds a random
suffix to internal request IDs, but `admit_stream` does not go through it, and
`VLLM_DISABLE_REQUEST_ID_RANDOMIZATION` turns the suffix off. A retry must
resend the identical payload, including its `arrival_time` and calling
frontend; it then returns the live session's current result and never extends
its deadline. Any other use of an ID that is in use, including by a session
that is still retiring, is refused as `request_id_in_use`. Once a session
ends, the engine forgets it. After a timed-out or cancelled admission, the
caller aborts the request ID, abandons any retries under it, and admits with a
new ID; a late input or abort for the old ID cannot reach the new session.
Refusals return `StreamAdmissionResult(accepted=False, reason=...)` with one
of `request_id_in_use`, `capacity`, `active_limit`, or `engine_paused`, and
neither allocate a slot nor create a request.

Input is then sent as ordinary resumable updates for the same request ID with
`EngineCoreRequest.stream_admission` set to a `StreamAdmissionContext`;
setting `input_closed=True`, or sending a non-resumable update, closes input.
Inputs must keep the admission's restrictions and carry token IDs only (no
prompt embeddings); the engine refuses others before preprocessing. Multimodal
positions are relative to the input's own prompt, since each input replaces it;
an item that starts outside that prompt, or (for a decoder-only model) does
not end inside it, is refused as `validation_error`. Rejected
updates are reported on `EngineCoreOutput.stream_output` without ending the
session, and leave its slot and deadline unchanged. This holds for
`EngineCore` and `EngineCoreClient` consumers. The output processor that
`AsyncLLM` and `LLMEngine` use does not read `stream_output` yet: it treats the
rejection's `ERROR` finish as the end of the request and aborts it, which ends
the session and frees its slot. An admitted session holds
its slot from admission, so its inputs drain ahead of queued requests that
still need a slot, including under `PAUSED_NEW`. An
engine without persistent state refuses any update that carries
`stream_admission` as a validation error.

The engine expires sessions on its own monotonic clock, including while idle,
and aborts expired sessions with an error output. Configure the deadlines with
`--additional-config`:

```bash
vllm serve <model> \
  --additional-config '{"stream_admission": {"never_fed_timeout": 15.0, "parked_idle_timeout": 60.0}}'
```

| Key | Default | Meaning |
| --- | --- | --- |
| `never_fed_timeout` | `15.0` | Seconds an admitted stream may wait for its first input. |
| `parked_idle_timeout` | `60.0` | Seconds a parked stream may wait for its next input. |
| `draining_timeout` | `40.0` | Seconds a stream may take to finish after input closes. |

Timeouts must be finite and positive.

## Supported configurations

The engine refuses to start with a persistent-state model when any of the
following is configured:

- tensor, pipeline, data, decode-context, or prefill-context parallel size
  greater than 1
- async scheduling
- speculative decoding
- prefix caching
- KV transfer connectors
- dual batch overlap
- sleep mode or the CuMem allocator
- HiSparse

The extensible KV cache, on by default with the V2 model runner, is turned
off with a warning; the KV cache is then sized from profiling estimates.

## Observability

When persistent state is enabled, `SchedulerStats` carries:

- `stream_occupied`: slots by session state (`never_fed`, `live`, `parked`,
  `draining`, `retiring`). A `retiring` slot belongs to a finished session
  (normal finish, abort or deadline expiry). It keeps occupying capacity,
  including while quarantined, until the worker acknowledges the release
  fence for its last scheduled step; it is then freed and no longer counted.
- `stream_refusals`: admission refusals by reason.
- `stream_expiries`: expired sessions by deadline (`never_fed`, `parked_idle`,
  `draining`).

These fields are not yet exported as Prometheus metrics.
