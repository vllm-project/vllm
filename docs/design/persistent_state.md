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
event completes. The host waits on that event when the step's output is
collected, once per state step; with synchronous scheduling the wait runs on the
worker's main thread, at the output boundary. The slot returns to the free pool
only when the scheduler has received a completion covering both the request's
last scheduled step and its cleanup. Until then, admission and scheduling treat
the slot as occupied, so a new request can never observe writes from the
previous owner. Model failures are reported through the same completion path,
and a scheduled step without a matching completion fails its request. A cleanup
that is not acknowledged is retried once; a second miss is a fatal engine error.

### Retirement signal

The engine reports when an admitted stream's slot is back in the free pool.
When the last state manager releases a stream's retired slot, including a
never-scheduled slot released at once, the scheduler adds the stream's request
ID to `EngineCoreOutputs.retired_streams` for the client that admitted it. The
engine sends that batch even when it carries no other output, and the client
reads `retired_streams` before it filters out batches without outputs or stats.
The signal means only that the slot was released; it never exposes
`StateSlotHandle` or `StateCompletion`, since receiving a completion does not
imply release (the slot may be quarantined or still short of its fence).

- **R1.** `AsyncLLM.wait_stream_retired(request_id)` returns once the stream
  admitted under the caller's request ID has retired.
  `EngineCoreClient.wait_stream_retired_async(request_id)` does the same for an
  internal request ID.
- **R2.** The client starts tracking a stream before it sends the admission, so
  a retirement that arrives before the wait is not lost. A wait that starts
  while admission is pending keeps waiting until the admission reply arrives.
  If admission is refused, pending waits raise `StreamAdmissionError`; if the
  call is cancelled before the admission is sent, they raise `ValueError`.
  Either way the stream is no longer tracked.
- **R3.** A retirement is kept until the first wait for it returns. The client
  keeps at most 1024 retirements that nobody has waited for, dropping the
  oldest, and a new admission under the same caller request ID drops that ID's
  unwaited retirement. A wait for a stream that is unknown, already waited for
  or dropped raises `ValueError` at once.
- **R4.** Concurrent waits for the same stream all return. Cancelling one wait
  affects neither the others nor the tracking.
- **R5.** When the engine dies, including on a second cleanup miss, pending
  waits raise `EngineDeadError`; they never report a retirement.

## Stream admission

With persistent state enabled, the engine core exposes the utility method
`admit_stream(request)`, which reserves a slot before any input arrives. It
atomically reserves a slot and parks an empty resumable request. The request
must be a resumable sampling request with no prompt, `None` or empty (`n=1`,
no stop strings, no structured outputs, no transfer connectors). Async frontends call it through
`EngineCoreClient.admit_stream_async`, which routes the stream's outputs back
to the calling API server, as for an ordinary request, or through `AsyncLLM`
(see [Admitted streams in `AsyncLLM`](#admitted-streams-in-asyncllm)).

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
Accepted inputs always run in order, including a resumable input that sets
`input_closed=True`. Only a close that carries no input (a non-resumable
request) can end a session without running anything: sent while the session is
parked (including never fed) with no input queued, it ends the session at once
with a terminal output (finish reason `stop`, no tokens). Otherwise the session
ends after the last accepted input finishes. An output ends an admitted session
exactly when it has a finish reason and its context has `input_closed=True`.
Every session-ending output, whatever the reason, carries such a context: the
session's current input context, closed (`update_seq=0` if no input was
applied), except that a close ending a parked session at once carries the
close's own context. Outputs without a finish reason carry their input's
context unchanged.
Inputs must keep the admission's restrictions and carry token IDs only (no
prompt embeddings); the engine refuses others before preprocessing. Multimodal
positions are relative to the input's own prompt, since each input replaces it;
an item that starts outside that prompt, or (for a decoder-only model) does
not end inside it, is refused as `validation_error`. Rejected
updates are reported on `EngineCoreOutput.stream_output` without ending the
session, and leave its slot and deadline unchanged. Each input's
`StreamAdmissionContext.update_seq` numbers it within the stream; every output
of an admitted stream carries the context of the input it ran on, and a
rejection carries the rejected input's. An admitted session holds
its slot from admission, so its inputs drain ahead of queued requests that
still need a slot, including under `PAUSED_NEW`. An
engine without persistent state refuses any update that carries
`stream_admission` as a validation error.

The engine expires sessions on its own monotonic clock, including while idle,
and aborts expired sessions with a terminal output (finish reason `abort`).
Configure the deadlines with `--additional-config`:

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

## Admitted streams in `AsyncLLM`

`AsyncLLM.generate()` and `AsyncLLM.add_request()` admit a streaming-input
request when called with `stream_admission=True` and an async generator of
`StreamingInput` as the prompt. The engine must have persistent state, and the
sampling parameters must meet both the streaming-input and the admission
restrictions; otherwise the call raises `VLLMValidationError`, which also
carries the engine's message for any failure of the `admit_stream` utility.
`LLMEngine` has no streaming input, so it has no admitted streams.

The frontend sends three kinds of request for one stream, all under the internal
request ID that input processing assigns:

| Request | Prompt | `resumable` | `stream_admission` |
| --- | --- | --- | --- |
| Admission | none | `True` | none |
| Input *k* (*k* ≥ 1) | the item's prompt | `True` | `update_seq=k` |
| Close | ignored placeholder | `False` | `update_seq` = last input + 1, `input_closed=True` |

The engine stores the context of the input a session is running on. It copies
an input's context onto the session whenever it applies that input, either at
once to a parked session or later from the session's queue, and it captures the
context for an output before stop handling can apply the next queued input. An
admitted session with no input applied
carries `update_seq=0`. A rejection carries the rejected request's own context.

- **A1. Admission.** The frontend registers the stream's output state with no
  input applied, then sends the admission. It reads nothing from the generator
  until admission succeeds.
- **A2. Caller request ID.** The frontend reserves the caller's request ID
  from the start of the call until the stream has retired, or until admission
  is refused or cancelled before it was sent. A call that finds the ID reserved
  is refused as `request_id_in_use` without reaching the engine. The internal
  ID carries vLLM's random suffix, so a caller may reuse its ID once the
  earlier stream has retired. Waits already pending stay bound to the earlier
  stream; a wait that starts after the new call begins, even before its
  admission is sent, follows the new stream. With `VLLM_DISABLE_REQUEST_ID_RANDOMIZATION` set, the internal ID is
  the caller's, and the engine's rule applies: never reuse it.
- **A3. Refusal and cancellation.** A refusal raises `StreamAdmissionError`
  from the call, carrying the `StreamAdmissionResult` (served as HTTP 409 for
  `request_id_in_use` and 503 otherwise); the output state is removed and the
  engine has allocated nothing. If the call is cancelled while
  admission is pending, the frontend still collects the admission reply in the
  background and, if the stream was admitted, aborts it.
- **A4. Inputs.** Each generator item is sent as the next input. An item that
  fails frontend validation or preprocessing ends the stream as for other
  streaming input (the call raises, and the session is aborted) and is not
  numbered.
- **A5. Close.** When the generator ends, the frontend sends the close. The
  engine ends the session after its accepted inputs, or at once if it is
  parked with nothing queued. The frontend ends the caller's stream on the
  session-ending output (a finish reason with `input_closed=True` in its
  context), not when the close is sent; an EOS that ends an input is never
  mistaken for it.
- **A6. Rejected input.** A rejected input does not end the session. The
  caller receives one `RequestOutput` with `finished=False`, no new tokens,
  no finish reason, `error=None`, and `stream_rejection` set to a
  `StreamRejection` holding the item's 0-based position in the generator, the
  rejection reason and, if the engine attached one, its `RequestError` (as for
  `engine_paused`). Its completion outputs are those the output kind gives for
  a step with no new tokens. A rejection output is never merged with another
  output. The engine's rejection output itself keeps finish reason `error`, so
  `EngineCore` consumers see it unchanged; the frontend converts it.
- **A7. Matching.** The frontend matches outputs to inputs by `update_seq`, not
  by arrival order, so a rejection that arrives after later inputs were
  accepted is attributed correctly and later outputs report the prompt of the
  input that actually ran.
- **A8. Rejected close.** A rejected close produces no rejection output; the
  frontend aborts the session, because no further input can follow.
- **A9. Session end.** The caller sees exactly one end: `generate()` either
  yields one output with `finished=True` (after a close, abort, deadline
  expiry, preemption, a failed state step, or a stop other than EOS) or raises
  one exception (cancellation, a generator or input error, or a dead engine),
  never both; whichever reaches the stream's output collector first wins. The
  frontend's input task, not `generate()`, aborts the session on a generator or
  input error, so `add_request()` callers get the same cleanup. Outputs and
  rejections that arrive after the end are ignored.
- **A10. Unadmitted streams.** Streaming input without `stream_admission=True`
  is unchanged: a rejected input ends that session.

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
