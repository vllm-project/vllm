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
`PersistentStateSpec`, and its class sets `supports_persistent_state = True`.
The model registry reports that class attribute as `has_persistent_state`,
so configuration can apply the defaults of
[Supported configurations](#supported-configurations) before any layer exists:

- A spec is one aggregate page made of named `PersistentStateDescriptor`
  components, each with a shape, dtype, byte offset, alignment, and an
  initializer callable. Components must not overlap and the page size must be the
  aligned extent of the layout. `block_size` is always 1.
- The KV cache layout must be state-only: exactly one cache group containing one
  persistent-state layer. Mixing persistent state with attention or Mamba cache
  groups is refused at startup.
- The state pool has `max_num_seqs + 1` pages of `page_size_bytes` each: one
  page is the reserved null block and each other page is a slot.
  `--num-gpu-blocks-override` sets the page count instead. The pool is
  allocated in full at startup or startup fails; see
  [State capacity](#state-capacity). Each request holds exactly one slot from
  its first schedule, or from admission for an admitted stream (see
  [Stream admission](#stream-admission)), until it is released; slots are
  never shared, prefix-cached, or swapped.
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
  Any other stop (length, a stop token or repetition) ends the stream; stop
  strings are refused for streaming input, so they never end a turn. Each new
  input replaces the request's prompt; the state carries the history. Multimodal
  positions are relative to the input's own prompt, and an input with an item
  outside it is refused before it is applied or queued.

- **Scheduler selection.** When the KV cache layout holds persistent state, the engine core
  uses `StateScheduler`; a configured `scheduler_cls` that does not subclass it,
  or that is an async scheduler, fails at startup. A layout without persistent
  state keeps the configured scheduler.
- **State-only layout.** A layout with persistent state that is not exactly one cache group
  holding one persistent-state layer fails at startup, as does a spec whose
  components overlap, whose page size is not the aligned extent of its
  components, or whose `block_size` is not 1.
- **Parking at EOS.** A resumable persistent-state request whose turn stops on EOS keeps
  its slot and waits for its next input; a turn that stops on length, a stop
  token or repetition finishes the request. That turn's final output also
  finishes the client's stream: `AsyncLLM.generate` returns after it, and the
  frontend sends no later input of that stream. An input the frontend sent
  before it processed that output may still reach the engine. For a stream
  that was not admitted, it arrives as a new request with the same id; it
  waits for the retiring slot (fenced release), and the frontend's abort ends
  it. An admitted stream stops and ends on the same conditions, and its ending
  output is marked as E9 states; an input of an admitted stream that arrives
  after the stream ended creates no request and is rejected (E6): as
  `unknown_admission` when it reaches the scheduler, or with the reason of an
  earlier check in E6's table when one refuses it first.
- **One slot per request.** A request holds one slot from its first schedule, or from admission
  for an admitted stream (E1), until the slot is released (fenced release); no
  two live requests hold the same slot, and a request that was not admitted
  holds no slot before it is first scheduled.
- **Prompt replacement.** A parked request's next input replaces its prompt, and the next step
  binds the same handle, so the model sees the state the previous turn left.
- **Current handle only.** The worker binds a slot only for the scheduler's current handle; a
  handle from another engine epoch or an older generation is rejected, and
  state rows are available to the model only for the handles bound in the
  current step.
- **Refused configurations.** Each configuration in the refusal list of
  [Supported configurations](#supported-configurations) fails startup with a
  `ValueError` naming it.

- **Misplaced multimodal input.** When an input of a resumable persistent-state request has a
  multimodal item outside its own prompt, the scheduler applies and queues
  nothing from it. For a request that was not admitted through `admit_stream`,
  the request finishes with an error at once, even mid-turn, and its slot
  retires behind the release fence (a request that was never scheduled holds
  no slot). For an admitted stream, the input is rejected as
  `validation_error` without ending the session (E6).

## Model hooks

A persistent-state model uses these Model Runner V2 hooks:

- `ModelState.initialize_state_slots(bindings)` is called after the KV cache is
  allocated, with the `StateSlotBindings` that map a request's current handle
  to its state views. Override it to register auxiliary state-writing streams.
- `ModelState.prepare_step(scheduler_output, *, dummy_run, is_profile)` is
  called at the start of each `execute_model`, so the model can tell profiling
  and dummy runs from scheduled steps.
- **Early `prepare_step`.** A model class that sets `supports_persistent_state = True` also gets
  `prepare_step` on the dummy and profiling runs that happen before its state
  slots are bound.

`StateSlotBindings.bind_rows` authorizes the current step's handles; the
`StateSlotRows` it returns expose the complete component pools, but a model
may use them only for those rows. This is a contract with a trusted model, not
memory protection: tensors already returned are not revoked.

### State capacity

The pool's size is configured: `max_num_seqs + 1` pages, or
`num_gpu_blocks_override` when it is set. Planning computes how many pages fit
the KV cache budget. The budget comes from memory profiling or from
`kv_cache_memory_bytes`. When fewer pages fit than are configured, startup
fails and, without an override, reports the largest `max_num_seqs` that fits
the measured budget; the engine never runs with fewer slots than configured.
A profiled budget is measured again at the next startup, so the reported
value is a starting point for the next attempt, not a guarantee.

The serving pool is allocated once, in full, before the state bindings are
built, and its pages keep their addresses for the engine's lifetime, so the
state views that the bindings give the model stay valid. For that reason a
state-only layout does not use the extensible KV cache, which commits pages
after warmup, sized from measured memory: the state pool's size is already
fixed by configuration, and the bindings address every page from startup.

### Startup criteria

- **Slot initialization.** The first bind of a slot, and every bind to a newer generation,
  writes the descriptors' initial values to its page; binding the same handle
  again leaves the page as it is.
- **Planned capacity.** KV cache planning for a persistent-state layout plans
  `num_gpu_blocks_override` pages when it is set, and `max_num_seqs + 1`
  otherwise. When fewer pages fit the KV cache budget (the budget divided by the
  page size, rounded down) than planned, it raises `ValueError` before
  `max_model_len` is fitted automatically and before the generic capacity check.
  The message states the planned page count, the page count that fits and the
  page size; without the override it also states the largest `max_num_seqs` that
  fits (one less than the fitting page count), or that no slot fits when fewer
  than two pages fit. Under a persistent-state layout, a
  `num_gpu_blocks_override` below 2 raises `ValueError` stating that the pool
  would hold no slot, before memory profiling allocates any state pool. When
  the planned pages fit, the pool has exactly the planned page count; planning
  never reduces it.

## Release fence

A finished or aborted request does not free its slot immediately. The scheduler
retires the handle and sends it to the worker as a cleanup handle. The worker
joins the main stream with every registered state-writing stream, records an
event, and returns a `StateCompletion` with the model output only after that
event completes. The host waits on that event when the step's output is
collected, once per state step; with synchronous scheduling the wait runs on the
worker's main thread, at the output boundary. The slot returns to the free pool
only when the scheduler has received a completion covering both the request's
last scheduled step and its cleanup. Until then, admission and scheduling
treat the slot as occupied, so a new request can never observe writes from the
previous owner.
Model failures are reported through the same completion path, and a scheduled
step without a matching completion fails its request. A cleanup that is not
acknowledged is retried once; a second miss is a fatal engine error.

- **Fenced release.** A slot whose request finished, was aborted or, for an admitted
  stream, expired (E10) (preemption ends a request the same way but has no
  public path, because state requests never need new blocks and a reset is
  refused while a slot is bound or parked; see
  [Cache clearing and release](#cache-clearing-and-release)) returns to the
  free pool only after the scheduler has received a `StateCompletion`
  covering the request's last scheduled step and a later cleanup of its
  handle. Until then admission counts the slot as occupied and refuses its
  request id (E3), scheduling binds no other handle to it, and a new request
  with the same request id that is not an admission waits until that slot is
  released.
- **Unscheduled slot.** A slot that was never scheduled is released at once. A request that
  was not admitted gets its slot from a schedule, so only an admitted stream
  that ends before its first input is scheduled takes this path.
- **Completion event.** The worker records a completion's event on the main stream after
  joining every registered state-writing stream into it, and the completion
  reaches the scheduler only after that event has completed.
- **Failed step.** A scheduled state step without a matching completion finishes its
  request with an error, and so does a request that the matching completion
  reports as failed (the model reports it through
  `StateSlotBindings.fail_requests`). A completion for a handle of an older
  generation is ignored.
- **Unacknowledged cleanup.** A cleanup that is not acknowledged quarantines the slot, and the
  cleanup is sent once more; when that is not acknowledged either, the
  scheduler raises `RuntimeError`, which ends the engine. Clients of the
  multiprocess engine then see `EngineDeadError`; the in-process engine raises
  the `RuntimeError` from its step. Without a fault in the worker, every
  cleanup is acknowledged, so the engine-level outcome has no public path; the
  scheduler's raise is its testable part.

### Retirement notice

The engine tells the client that admitted a stream when the stream's slot is
back in the free pool, so a frontend can release what it holds for the stream
and can tell when capacity has returned. The notice means only that the slot
was released. It never exposes `StateSlotHandle` or `StateCompletion`:
receiving a completion does not imply release, because the slot may be
quarantined or still short of its fence.

- **R6.** When the last state manager releases the retired slot of an
  admitted stream, including a never-scheduled slot released at once (unscheduled slot),
  the scheduler adds the stream's request ID to
  `EngineCoreOutputs.retired_streams` of the next output batch that the
  engine delivers for the `client_index` of the admission request, exactly
  once per admitted session. In the in-process engine no client implements
  admission (R1), so this applies to a caller of `EngineCore.admit_stream`
  itself, which reads the notice from the outputs of the engine's steps. A
  batch of a step whose outputs the engine drops to drain a cache clear is
  not delivered, so a notice due in such a step goes out in the first
  externally delivered batch after the clear (see
  [Cache clearing and release](#cache-clearing-and-release)). The engine sends the
  batch even when it carries no other output. A request that was not
  admitted produces no notice.
- **R1.** `AsyncLLM.wait_stream_retired(request_id)` returns `None` once the
  stream admitted under the caller's request ID has retired.
  `EngineCoreClient.wait_stream_retired_async(request_id)` does the same for
  an internal request ID. Only the async multiprocess client implements
  `admit_stream_async` and `wait_stream_retired_async`; on every other client
  they raise `NotImplementedError`. A wait is not ordered against the
  stream's outputs: it may return before the caller has received the
  stream's last output.
- **R2.** The async multiprocess client starts tracking a stream before it
  sends the admission, and reads `retired_streams` (R6) from every batch
  before it drops batches that have no outputs or stats, so a retirement that
  arrives before the wait, or in a batch with nothing else, is not lost. A
  wait that starts while admission is pending keeps waiting until the
  admission settles. An admission settles in one of four ways:
    - accepted: the stream stays tracked until its retirement is consumed
      (R3);
    - refused: pending waits raise `StreamAdmissionError`;
    - failed with an exception: `admit_stream_async` raises the utility
      failure as a plain `Exception` whose message carries the engine's
      (for E2, `Call to admit_stream method failed: ...`, while
      `EngineCore.admit_stream` itself raises `ValueError`; for a call
      rejected during engine shutdown, `Server shutting down`), or
      `EngineDeadError` when the engine has died or the client is shutting
      down (R5); pending waits raise what the call raised;
    - cancelled before the admission is sent: pending waits raise
      `ValueError`.

    In the last three cases the call's own tracking is removed. Cancelling
    a call after its admission is sent does not discard the reply: the
    client keeps the admission's operation and settles the tracking from its
    reply, so an accepted stream stays tracked and its caller aborts it (A3
    does this for `AsyncLLM`), and a refusal or failure settles as above.
    The client never sends an admission again to recover a reply. While an
    admission under a request ID is pending, or its stream has not
    retired, the client refuses another admission under the same ID
    itself: `admit_stream_async` raises `StreamAdmissionError` with reason
    `request_id_in_use` from that call without sending anything, and the
    existing admission's tracking, waits and session are untouched. So one
    ID never has two admissions in flight, and an accepted reply always
    leaves its stream tracked. An admission after the stream has retired
    is tracked as any other; it drops that ID's unwaited retirement (R3). This holds for
    direct callers of `admit_stream_async` as well as for `AsyncLLM`.
- **R3.** A retirement is kept until the first wait for it returns, the
  engine dies or the client shuts down (R5). The async
  multiprocess client keeps at most 1024 retirements that nobody has waited
  for, dropping the oldest, and a new admission under the same caller request
  ID drops that ID's unwaited retirement. A wait for a stream that is unknown,
  already waited for or dropped raises `ValueError` at once.
- **R4.** Concurrent waits on the async multiprocess client for the same
  stream all return. Cancelling one wait affects neither the others nor the
  tracking.
- **R5.** When the engine dies, including on a second cleanup miss (unacknowledged cleanup), or
  the client shuts down, waits pending on the async multiprocess client raise
  `EngineDeadError`, and so does every later wait, including one for a
  retirement that arrived before; they never report a retirement. A shutdown
  takes effect when `shutdown()` is called, on any thread, before it stops
  the engine: a retirement that arrives while the engine is stopping is not
  reported.

## Cache clearing and release

A state page holds history that cannot be rebuilt from tokens, so clearing the
KV cache would end every stream that holds a slot, including parked ones,
which have no running request. A request to clear the cache that would end a
stream the caller did not end is therefore refused before anything changes;
the operator ends the streams first.

A slot that is retiring belongs to a stream that has already ended; it stops
blocking a clear once its fence is acknowledged, a step or two later. Bound and
parked slots belong to live streams.

- **Cache clears.** While any state slot is bound or parked:
  - `StateScheduler.reset_prefix_cache(reset_running_requests=True)` raises
    `RuntimeError` before any request is preempted, and every request keeps its
    status and slot. The message is `persistent state: {n} state slot(s) are
    held, and resetting running requests would end their streams. Abort the
    streams and wait until they have retired, then reset again.`
  - `pause_scheduler` with `clear_cache=True` and `mode="keep"` (either
    engine) or `mode="wait"` (multiprocess engine) raises `RuntimeError`
    before any request is preempted or finished and before any cache reset,
    and the pause state is unchanged. The message is `persistent state: {n}
    state slot(s) are held, and clearing the cache would end their streams.
    Pause with clear_cache=False; to clear the cache, abort the streams, wait
    until they have retired, then reset the prefix cache.` In the multiprocess
    engine, a keep pause whose finish was deferred until the engine is idle
    repeats the check when the finish runs, before the device sync and the
    cache reset; a failure there fails the pause's future with the same
    message, preempts nothing and leaves the pause state set at entry. A
    resume between the pause's entry and its finish lets a stream bind or
    park a slot, which is what this repeated check catches.

  When the only held slots are retiring:
  - In the in-process engine, when the scheduler has no unfinished request,
    `reset_prefix_cache` (whatever `reset_running_requests` is) and
    `pause_scheduler` with `clear_cache=True` step the engine until no slot is
    retiring, and then reset; such a step carries cleanups and no request's
    work. When a request is queued, `reset_prefix_cache` raises the retry
    message below without stepping (with `reset_running_requests=False` it
    returns `False` and changes nothing), and a pause that keeps or waits for
    requests raises it before any change; an abort pause has already finished
    every request, so it drains.
  - In the multiprocess engine, a pause with `clear_cache=True` finishes once
    the engine is idle, which is after every retiring slot has been released.
    `StateScheduler.reset_prefix_cache(reset_running_requests=True)` raises
    `RuntimeError` with `persistent state: {n} state slot(s) are still
    retiring; retry the reset once they have retired.`, preempting nothing;
    with `reset_running_requests=False` it returns `False` and changes
    nothing. The engine releases those slots within its next steps without any
    further call.

  `pause_scheduler(mode="abort", clear_cache=True)` ends every request, parked
  streams included, because the caller asked for that; the aborted streams'
  slots then retire, and the caches are cleared as above in both engines,
  whatever `reset_connector` is. With `reset_running_requests=False`, a reset
  while any slot is bound or parked returns `False` and changes nothing, as it
  does without persistent state while blocks are in use; with only retiring
  slots it follows the rules above. A stream parked
  across a pause with `mode="keep"` and `clear_cache=False` resumes under the
  same handle, with output equal to an unpaused run of a deterministic model
  whose outputs depend only on its state and inputs, with one stream and
  greedy sampling.

  Under persistent state, `release_kv_cache_memory` raises `RuntimeError`
  before its pause check, whether or not a slot is held, and releases nothing.
  The message is `persistent state: the KV cache holds per-stream state, so it
  cannot be released while the engine runs.` None of these refusals attributes
  the failure to remote KV transfer.

  Steps that the in-process engine runs only to drain a cache clear, whose
  outputs it drops, deliver no queued persistent-state error and no
  retirement notice (R6): an error or notice queued before or during those
  steps is delivered, once, by the next step after the clear.

  A slot reserved by admission counts as parked from admission, whether or
  not the stream has received an input. A stream that the engine has
  expired (E10) has ended, so its slot is retiring, not held; these checks do
  not evaluate deadlines themselves, so a session whose deadline has passed
  but that the engine has not yet expired still counts as held. An abort
  pause ends admitted streams as it ends other requests: each stream's slot
  retires, and its retirement notice follows (R6). In the multiprocess
  engine each stream also gets an output with finish reason `abort`, which
  ends its stream (E9); the in-process engine sends no output for the
  requests an abort pause ends, admitted or not.

## Stream admission

A resumable request gets its slot when its first input is scheduled, so a
frontend learns that the engine is full only after it has accepted a stream
and sent input. Stream admission reserves the slot before any input exists:
the engine either admits the stream, which then owns a slot until it ends, or
refuses it with a reason the frontend can report at once. An admitted stream
also has deadlines, so a stream whose client went away cannot hold a slot
forever, and its inputs can be refused one at a time without ending it.

With persistent state enabled, the engine core exposes the utility method
`admit_stream(request)`. Async frontends call it through
`EngineCoreClient.admit_stream_async(request)`, which routes the stream's
outputs back to the calling API server, as for an ordinary request, or through
`AsyncLLM` (see
[Admitted streams in `AsyncLLM`](#admitted-streams-in-asyncllm)). The
client sends the request in the wire form of an added request. The engine
decodes utility calls as plain values, and decodes the request of an
`admit_stream` call again as it decodes an added request, so its prompt and
tensors arrive as they do for `add_request`. A request that cannot be
decoded fails its own call as any failed utility call does (R2), and the
engine goes on reading requests.

`admit_stream_async` takes three optional keyword-only callbacks, which the
client calls on its event loop:

- `on_release(error)`: once, when the stream can no longer hold a slot. `error`
  is `None` when the stream has retired, and otherwise what pending waits
  raise: the refusal, the failure, the `ValueError` of an admission that was
  never sent, or `EngineDeadError`.
- `on_forget()`: when the stream's retirement can no longer be waited for,
  because a wait returned it, it was dropped (R3), or the admission settled
  without a stream.
- `on_late_accept()`: when the call was cancelled after its admission was
  sent and the reply then admits the stream. Cancelling the call cannot
  recall a sent admission, so this is how its caller learns that a stream
  exists that it must abort (R2); `AsyncLLM` uses it for A3. It is not called
  for a late refusal or failure, nor when the call was cancelled before the
  send.

None is called for an admission that the client refuses itself (R2).
Resumable requests that are not admitted keep working as the
[State contract](#state-contract) describes.

`vllm.v1.engine` defines the types:

- `StreamAdmissionContext(input_closed: bool = False, update_seq: int = 0)`
  marks an `EngineCoreRequest` as an input of an admitted stream, through
  `EngineCoreRequest.stream_admission`. `update_seq` numbers the input within
  its stream; `input_closed=True` closes the stream's input.
- `StreamAdmissionResult(accepted: bool, request_id: str, reason=None)` is
  the reply to an admission. `reason` is `None` when the stream is admitted
  and otherwise one of `request_id_in_use`, `capacity`, `active_limit` and
  `engine_paused`.
- `StreamAdmissionError(result)` is a `GracefulHTTPError` that carries a
  refusal in its `result` attribute. Its HTTP status is 409 for
  `request_id_in_use` and 503 otherwise.
- `StreamOutput(context, rejection=None)`, on
  `EngineCoreOutput.stream_output`, carries the context of the input an output
  belongs to and, for a refused input, the rejection reason.

An admitted session is in one of four states:

| Status | Meaning | Deadline |
| --- | --- | --- |
| `never_fed` | Admitted; no input accepted yet. | `never_fed_timeout`, from admission |
| `live` | An accepted input is running or queued, and input is open. | none |
| `parked` | The last turn stopped on EOS, nothing is queued, and input is open. | `parked_idle_timeout`, from the park |
| `draining` | Input is closed; accepted inputs are finishing. | `draining_timeout`, from the close |

The request ID names the stream, so the caller must give every session a
globally unique ID and never reuse it. vLLM's input processor adds a random
suffix to internal request IDs, but `admit_stream` does not go through it, and
`VLLM_DISABLE_REQUEST_ID_RANDOMIZATION` turns the suffix off. An admission
whose ID names a session is refused, even when it repeats the admission
that created it (E3). After a timed-out or cancelled admission, the caller
aborts the request ID and admits with a new ID; a late input or abort for
the old ID cannot reach the new session. Once a session has ended, its
slot has been released and its retirement notice has been delivered (R6),
the engine keeps no record of its ID.

- **E1. Admission.** When `EngineCore.admit_stream(request)` receives a valid
  admission request (E2) whose request ID names no session, and no refusal of
  E3 applies, the engine reserves one free slot under that ID, creates a
  resumable request with an empty prompt that waits for its first input,
  records the session as `never_fed` with the context `update_seq=0`, starts
  its `never_fed_timeout`, and returns
  `StreamAdmissionResult(accepted=True, request_id=request.request_id)`. The
  slot, the request and the session are created together or not at all.
  The stream's outputs go to the client whose
  `client_index` the request carries. Until an input is applied, the request
  waits for input as a parked request does: it is never scheduled and is not
  unfinished work, so it keeps no engine stepping and delays no pause that
  waits for requests to finish; it still counts toward `active_limit` (E3).
- **E2. Invalid admission.** If an admission request is not valid,
  `admit_stream` raises `ValueError`, changes nothing and counts no refusal. A
  valid admission request has `resumable=True`; a string request ID of 1 to
  256 characters; no prompt (`prompt_token_ids` is `None` or empty), no prompt
  embeddings, `prompt_is_token_ids` unset (`None`) and no multimodal features;
  sampling parameters, not pooling parameters, with `n=1`, an output kind
  other than `FINAL_ONLY`, no stop strings, no structured outputs and no KV or
  EC transfer parameters; no `stream_admission` context; `abort_immediately`
  unset; `data_parallel_rank` `None` or 0; and `current_wave` 0.
  `admit_stream` also raises `ValueError` when the model does not support
  resumable requests or a transfer connector is configured. On an engine
  without persistent state, it raises `ValueError` with `Stream admission
  requires persistent state`.
- **E3. Refusals.** If a valid admission request cannot be admitted,
  `admit_stream` returns `StreamAdmissionResult(accepted=False,
  request_id=request.request_id, reason=...)`, reserves no slot, creates no
  request and adds one to the refusal count of that reason (O1). It reports
  the first reason that applies, in this order:
    1. `request_id_in_use`: the ID names a session, whatever the request's
       other fields; or it belongs to a request in the scheduler that was
       not admitted, or to a slot that is still retiring (fenced release), or to an
       admitted stream whose retirement notice (R6) has not yet gone out
       in a delivered batch. An ID is reusable only after that notice.
    2. `engine_paused`: the scheduler's pause state is not `UNPAUSED`. Every
       pause sets it, including `pause_generation` in any mode.
    3. `capacity`: no slot is free. Retiring and quarantined slots are not
       free.
    4. `active_limit`: the scheduler holds at least `max_num_active_seqs`
       requests (`max_num_seqs` when it is unset), counting requests in every
       state. Scheduling compares the same option with the running requests
       plus the requests waiting for their next streaming input; admission
       also counts requests queued to run, including ones that still need a
       slot.
- **E5. Inputs.** An input of an admitted stream is an `EngineCoreRequest`
  with the session's request ID and a `stream_admission` context. When a
  resumable input is not refused (E6), the scheduler applies it at once to a
  session that is parked or never fed, or queues it behind the session's
  running turn; accepted inputs run in arrival order, and each replaces the
  prompt (prompt replacement). Unless the input also closes input (E7), the session becomes
  `live` and its deadline is cleared. The engine does not compare `update_seq`
  values; numbering is the caller's (A4). Because the session holds its slot
  from admission, its inputs are scheduled ahead of queued requests that still
  need a slot, including under `PAUSED_NEW`.
- **E6. Rejected input.** If the engine refuses an input that carries a
  `stream_admission` context, or the scheduler refuses a request without one
  whose ID names a session (`identity_mismatch`), it applies and queues
  nothing from it, creates no request, and leaves the session's status, slot,
  deadline and input state unchanged. It sends one rejection output to the
  client whose `client_index` the refused request carries: an
  `EngineCoreOutput` with no tokens, `ends_stream=False`, finish reason
  `error` (`abort` for `shutdown`), `stream_output=StreamOutput(context,
  rejection)` with the refused request's own context (`None` for
  `identity_mismatch`) and the reason below, and the `RequestError` or
  cache-miss hashes where the table says so. The request ID is not added to
  `finished_requests`.

    | `rejection` | The input is refused when |
    | --- | --- |
    | `validation_error` | it fails the engine's validation before preprocessing: it has pooling parameters, prompt embeddings, `prompt_is_token_ids` or sampling parameters that E2 excludes; a multimodal item lies outside its own prompt (misplaced multimodal input); or the engine has no persistent state. Carries a `RequestError` with code `validation_error`. |
    | `mm_cache_miss` | the multimodal processor cache misses one of its items. Carries the missing hashes. |
    | `preprocessing_error` | preprocessing raises an unexpected error. |
    | `shutdown` | the engine is shutting down. |
    | `engine_paused` | the engine is rejecting new requests and the scheduler holds no request with its ID, for example because its session has ended. Carries a retryable `RequestError` with code `engine_paused`. |
    | `unknown_admission` | it carries a context and its ID names no session: the stream was never admitted, or it has ended. |
    | `identity_mismatch` | its ID names a session and it carries no context. |
    | `unsupported_update` | it reaches the scheduler with pooling parameters, prompt embeddings, `prompt_is_token_ids` or sampling parameters that E2 excludes. Preprocessing refuses these first, so only a caller that adds requests to the scheduler directly sees this reason. |
    | `input_closed` | it is resumable and the session's input is closed (E7). |
    | `empty_update` | it is resumable and has no prompt tokens. |
    | `validation_error` | it is resumable and reaches the scheduler with a multimodal item outside its own prompt (misplaced multimodal input). Preprocessing refuses this first, so only a caller that adds requests to the scheduler directly sees it here; this rejection carries no `RequestError`. |

    When several reasons apply, the engine reports the first in table order:
    the first three come from preprocessing, the next two from the engine
    core's loop, and the rest from the scheduler.

    A request without a context is outside this contract until it reaches
    the scheduler. Preprocessing and the engine core's loop do not look up
    sessions, so when one of them refuses such a request (a validation
    error, a cache miss, a preprocessing error or a shutdown), the engine
    sends the ordinary error output of a request that was not admitted: no
    `stream_output`, and the request ID in `finished_requests`. The session
    in the scheduler is unchanged; a frontend that receives that output ends
    the caller's stream (E9) and aborts the session. `AsyncLLM` sends no
    request without a context for an admitted stream.

- **E7. Close.** Input closes when the scheduler accepts an input whose
  context has `input_closed=True` or that is not resumable. A resumable input
  with `input_closed=True` runs like any input, and the session ends when its
  turn finishes, whatever stops it. A request that is not resumable is a close
  that carries no input, and its prompt is ignored: when the session is
  parked or never fed with nothing queued, the session ends at once with a
  session-ending output (finish reason `stop`, no tokens) that carries the
  close's context; otherwise the session ends when the turn of its last
  accepted input finishes. The first accepted close makes the session
  `draining` and starts its `draining_timeout`. A later close is accepted and
  changes nothing; a later resumable input is rejected as `input_closed`.
- **E8. Output context.** Every output that the scheduler produces for an
  admitted stream and that is not a rejection carries
  `stream_output=StreamOutput(context)` with the context of the input it ran
  on, as that input sent it; an output produced before any input was applied
  carries `update_seq=0`. The one exception is the output of a close that ends
  a parked or never-fed session at once (E7): it carries the close's own
  context, also when no input was ever applied. The scheduler reads the
  context before stop handling can apply the next queued input, so the last
  output of one input never carries the next input's number. An output that
  the engine produces outside the scheduler, such as the ordinary error output
  of E6, carries no `stream_output`.
- **E9. Session end.** The scheduler sets `ends_stream=True` on the output
  it produces for an event that ends an admitted session: the final turn
  after a close, or a close that ends the session at once (`stop`); a turn
  that stops on length, a stop token or repetition (parking at EOS); deadline expiry
  (`abort`); and a failed state step or a preemption (`error`). A session has
  at most one such output. An abort output that the engine core sends
  outside the scheduler, for an abort pause in the multiprocess engine or a
  shutdown with `shutdown_timeout` 0, carries neither `ends_stream` nor
  `stream_output` (E8); the frontend ends the stream on its finish reason,
  as below. A shutdown that drains (`shutdown_timeout` above 0) waits only
  for unfinished work, which excludes never-fed and parked sessions (E1).
  Draining does not suspend deadlines: while the engine keeps running, a
  session can still expire with its output and notice (E10). For sessions
  still present when the engine exits, there is no output and no
  retirement notice; the client sees the engine's end (A9, R5). The output
  of a turn that stops on EOS while input is open has finish reason `stop` and
  `ends_stream=False`: it ends the turn, not the session. A frontend ends the
  caller's stream on an output that has a finish reason, is not a rejection
  (E6), and either has `ends_stream=True` or has finish reason `abort` or
  `error`; so an error output that reaches the frontend without the flag still
  ends the stream, and a rejection never does. A frontend that ends an
  admitted stream on an output without `ends_stream=True` also aborts the
  session, because the engine may still hold it and its slot. For a stream
  that was not admitted the engine sets the flag only as parking at EOS states, and the
  frontend applies the same rule.
- **E10. Deadlines.** While the scheduler's pause state is not `PAUSED_ALL`
  (E11), the engine reads its own monotonic clock before every admission,
  every input and every schedule; the multiprocess engine's idle loop also
  wakes at the earliest deadline, and the in-process engine checks on every
  step, including a step with no request to run. When a session's deadline
  has passed, the engine finishes its request as aborted, sends a
  session-ending output with finish reason `abort` and no tokens (E9),
  retires its slot (fenced release, or unscheduled slot when it was never
  scheduled) and adds one to
  the expiry count of that deadline (O1). A session has one deadline at a
  time, as the state table gives it: accepting a resumable input that does
  not close input clears it; a park starts `parked_idle_timeout`; and the
  first accepted close, whether a bare close or a resumable input that
  closes input (E7), replaces any deadline with `draining_timeout`, which
  then runs until the session ends. An input that reaches the scheduler after
  its session expired is rejected as `unknown_admission`; an earlier check of
  E6's table may refuse it first.
- **E11. Deadlines under a pause.** A pause that keeps requests must not end
  the streams it keeps, and a turn cannot run while nothing is scheduled. (A
  bare close that ends a parked session at once (E7) and a client abort still
  end a session during the pause.) So, while the scheduler's pause state is
  `PAUSED_ALL`, the engine evaluates no deadline, no session expires, and the
  idle loop does not wake for a deadline. A deadline that starts during the
  pause, such as the park of a turn that was running when the pause began or a
  close, is recorded but does not run. When the scheduler leaves `PAUSED_ALL`,
  for `UNPAUSED` or `PAUSED_NEW`, every session that has a deadline, including
  one that started during the pause, gets that deadline again in full from
  that moment. A stream whose client has gone therefore holds its slot for the
  length of the pause and one more full timeout.
- **E12. Deadline configuration.** The scheduler reads the deadlines from
  `SchedulerConfig.stream_admission_timeouts: dict[str, float]`, whose
  default is an empty mapping (`default_factory=dict`). On the command line,
  `--stream-admission-timeouts` takes a JSON object; a value that is not JSON
  exits with argparse's usage error. A key the table below lists sets that
  timeout, and an absent key keeps its default. A validator that runs before
  type coercion checks the field on every engine, because configuration
  builds `SchedulerConfig` whether or not the model declares persistent
  state: each value must be a finite, positive real number of seconds (an
  `int` or `float`, not a `bool` or a string). Any other value, or a key the
  table does not list, fails startup with `ValueError` naming the key, and a
  value that is not a mapping fails startup with `ValueError`. After
  validation, an engine without persistent state makes no use of the field.
  The field is excluded from the compile-cache key.

```bash
vllm serve <model> \
  --stream-admission-timeouts '{"never_fed_timeout": 15.0, "parked_idle_timeout": 60.0}'
```

| Key | Default | Meaning |
| --- | --- | --- |
| `never_fed_timeout` | `15.0` | Seconds an admitted stream may wait for its first input. |
| `parked_idle_timeout` | `60.0` | Seconds a parked stream may wait for its next input. |
| `draining_timeout` | `40.0` | Seconds a stream may take to finish after input closes. |

## Admitted streams in `AsyncLLM`

`AsyncLLM.generate()` and `AsyncLLM.add_request()` take the keyword-only
parameter `stream_admission: bool = False`. With `stream_admission=True` and
an async generator of `StreamingInput` as the prompt, the call admits the
stream before it reads any input, feeds each generator item as one input, and
closes input when the generator ends.
`AsyncLLM.wait_stream_retired(request_id)` waits for the slot's release
([Retirement notice](#retirement-notice)). These are `AsyncLLM` methods; the
`EngineClient` protocol does not declare them. `LLMEngine` has no streaming
input, so it has no admitted streams. The queue limits of
`EngineClient.check_admission` apply to an admitted stream as A1 states; a
rejection by them is counted in `vllm:admission_rejections` as for any
request, and a refusal that `admit_stream` returns is not (O1).

The frontend sends three kinds of request for one stream, all under the internal
request ID that input processing assigns:

| Request | Prompt | `resumable` | `stream_admission` |
| --- | --- | --- | --- |
| Admission | none | `True` | none |
| Input *k* (*k* ≥ 1) | the item's prompt | `True` | `update_seq=k` |
| Close | ignored placeholder | `False` | `update_seq` = last input + 1, `input_closed=True` |

- **A11. Call validation.** `AsyncLLM` checks itself every E2 condition
  that a caller or the configuration controls, and raises
  `VLLMValidationError` before it reserves the caller's request ID or sends
  anything when the prompt is not an async generator, or when one fails:
    - from the call: an internal request ID of 1 to 256 characters, which
      the caller's ID and the length of vLLM's random suffix decide before
      the internal ID exists;
      sampling parameters, not pooling parameters, with `n=1`, an output
      kind other than `FINAL_ONLY`, no stop strings, no structured outputs
      and no KV or EC transfer parameters; `data_parallel_rank` `None` or 0;
    - from the configuration: a model with persistent state that supports
      resumable requests, and no transfer connector.

    The remaining E2 conditions concern fields that `AsyncLLM` builds
    itself (`resumable=True`, no prompt, prompt embeddings, multimodal
    features or `prompt_is_token_ids`, no `stream_admission` context,
    `abort_immediately` unset, `current_wave` 0), so only a direct caller of
    `admit_stream` or `admit_stream_async` can violate them; E2's
    `ValueError` is that caller's guard. A failure of the admission itself
    reaches `add_request()` as `admit_stream_async` raised it (R2).
    `generate()` keeps its upstream error handling: it re-raises
    `VLLMValidationError`, `StreamAdmissionError`, the `check_admission`
    errors (all `VLLMClientError` or `GracefulHTTPError`) and
    `EngineDeadError` as they are, and raises `EngineGenerateError` from any
    other failure of the admission, such as the plain `Exception` of a failed
    utility call.
- **A1. Admission.** A call runs, in order: A11's checks, the reservation
  of the caller's request ID (A2), `check_admission`, the registration of
  the stream's output state, and the admission's send. Before the
  reservation, the first call on a frontend asks the engine for its
  supported tasks; if the engine dies meanwhile, the call raises
  `EngineDeadError` with nothing reserved. Nothing suspends between
  the reservation and the client's tracking of the admission (R2), so a
  failure there releases the reservation before any wait can attach. The
  send may suspend; a wait that starts meanwhile follows the tracked
  admission. The frontend
  applies `check_admission` once, for one sequence, at the point
  where an unadmitted stream's first input is checked. Its rejections,
  including the fault-tolerance health check's `EngineFaultedError`, raise
  from the call, which then sends nothing. The frontend then registers the
  stream's output state with no input applied and sends the admission. It
  reads nothing from the generator until admission succeeds. Later inputs
  and the close are not checked, since the output processor already holds
  the stream's ID. From registration until the stream ends, the stream
  counts as one unfinished request toward `max_num_queued_reqs` for later
  requests, whether or not it has input. When the call ends without an
  admitted stream (the admission is refused, fails with an exception or is
  cancelled before it is sent, or a reply collected after cancellation
  refuses or fails it, R2), the frontend removes that output state, so it
  no longer counts toward the queue limits.
- **A2. Caller request ID.** The frontend reserves the caller's request ID
  from its reservation (A1) until the stream has retired, or until the call
  ends without an admitted stream: a `check_admission` rejection (A1), or an
  admission that is refused, fails with an exception or is cancelled before
  it was sent (R2). A call that finds the ID reserved,
  or whose internal ID names a request the frontend already holds (possible
  only with `VLLM_DISABLE_REQUEST_ID_RANDOMIZATION`), raises
  `StreamAdmissionError` with reason `request_id_in_use` without reaching
  the engine or changing that request. The internal
  ID carries vLLM's random suffix, so a caller may reuse its ID once the
  earlier stream has retired. Waits already pending stay bound to the earlier
  stream; a wait that starts after the new call has reserved the ID, even
  before its admission is sent, follows the new stream. With
  `VLLM_DISABLE_REQUEST_ID_RANDOMIZATION` set, the internal ID is the
  caller's, and the engine's rule applies: never reuse it.
- **A3. Refusal and cancellation.** A refusal raises `StreamAdmissionError`
  from the call, carrying the `StreamAdmissionResult`; the output state is
  removed (A1) and the engine has allocated nothing. If the call is
  cancelled while admission is pending, the frontend still collects the
  admission reply in the background (R2) and, if the stream was admitted,
  aborts it.
- **A4. Inputs.** Each generator item is sent as the next input, numbered from
  1 in the order the generator yields them. An item that fails frontend
  validation or preprocessing ends the stream as for other streaming input
  (the call raises, and the session is aborted) and is not numbered.
- **A5. Close.** When the generator ends, the frontend sends the close. The
  engine ends the session after its accepted inputs, or at once if it is
  parked with nothing queued (E7). The frontend ends the caller's stream on
  the session-ending output (E9), not when the close is sent; an EOS that ends
  an input is never mistaken for it.
- **A6. Rejected input.** A rejected input does not end the session. The
  caller receives one `RequestOutput` with `finished=False`, no new tokens,
  no finish reason and `stream_rejection` set to a
  `vllm.outputs.StreamRejection(input_index, reason, error=None)` holding the
  item's 0-based position in the generator, the rejection reason and, if the
  engine attached one, its `RequestError` (as for `engine_paused`). Its
  completion outputs are those the output kind gives for a step with no new
  tokens. A rejection output is never merged with another output. The
  engine's rejection output itself keeps its finish reason (E6), so
  `EngineCore` consumers see it unchanged; the frontend converts it.
- **A7. Matching.** The frontend matches outputs to inputs by `update_seq`
  (E8), not by arrival order, so a rejection that arrives after later inputs
  were accepted is attributed correctly and later outputs report the prompt of
  the input that actually ran. An output without `stream_output`, or whose
  context is `update_seq=0` or the close's own (E8), matches no input; the
  frontend handles it under E9's rule alone. A rejection without a context
  (`identity_mismatch`, E6) refused a request that is no input of the
  stream: the frontend gives the caller no output for it and ends nothing.
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
- **A10. Unadmitted streams.** On an engine with persistent state, streaming
  input without `stream_admission=True` is not admitted: it has no deadline,
  no rejection output and no retirement notice, and a refused input ends its
  session (misplaced multimodal input). An engine without persistent state is outside this
  document.

## Supported configurations

The engine refuses to start with a persistent-state model when any of the
following is configured:

- tensor, pipeline, data, decode-context, or prefill-context parallel size
  greater than 1
- speculative decoding
- prefix caching
- KV transfer connectors
- dual batch overlap
- sleep mode or the CuMem allocator
- HiSparse

Four defaults differ for a model that declares persistent state. Prefix
caching is off by default, since a stream's state is not a function of a
shared token prefix. Asynchronous
scheduling is off by default: it schedules a step before the previous step's
output is processed, so a resumable request's next step can run on the EOS
that ended its turn and advance the stream's state on input it never
received; it is not yet validated with persistent state. The extensible KV
cache does not apply to a state-only layout (see
[State capacity](#state-capacity)), and the runner captures no decoder graphs.
As for other unsupported options, an
unset option resolves to off and an explicit request fails:

- **Async scheduling.** When the model declares persistent state and `async_scheduling` is
  unset, configuration resolves it to `False` and logs a warning that names
  persistent state, once each time it resolves the option. When
  `async_scheduling` is `True`, configuration raises `ValueError` with `Async
  scheduling is not yet validated with persistent state. Use
  --no-async-scheduling.` An explicit `False` is accepted without a warning.
  When another incompatibility also applies, configuration reports the first
  it checks, as it does for other options.
- **Extensible KV cache.** When the model declares persistent state and
  `enable_extensible_kv_cache` is unset, configuration resolves it to `False`
  without a warning. When it is `True`, configuration raises `ValueError`
  naming persistent state as the reason. When another reason that the
  extensible KV cache cannot be used also applies, configuration reports the
  first it checks.
- **Runner CUDA graphs.** The runner's own decoder CUDA graphs are not validated with
  persistent state. When the model declares persistent state and
  `cudagraph_mode` is unset, configuration resolves it to NONE, whatever the
  optimization level, and logs a warning that names persistent state, once
  each time it resolves the option. When `cudagraph_mode` is set to anything
  other than NONE, configuration raises `ValueError` with `CUDA graph mode
  {mode} is not yet validated with persistent state. Use cudagraph_mode NONE;
  the model can capture its own graphs.` An explicit NONE logs nothing.
  `enforce_eager` gives NONE without a warning and, as without persistent
  state, takes precedence over an explicit mode.
- **Prefix caching.** When the model declares persistent state and
  `enable_prefix_caching` is unset, configuration resolves it to `False`
  without a warning. When it is `True`, startup fails as a refused
  configuration.
- **Declaration.** The model registry reports `has_persistent_state` as `True` exactly
  for a model class that sets `supports_persistent_state = True`, including an
  out-of-tree class registered lazily, and `ModelConfig.has_persistent_state`
  returns it; it is `False` for every in-tree model. An engine whose model
  builds a `PersistentStateLayerBase` layer without that declaration fails at
  startup with `ValueError` naming the class
  attribute, before memory profiling.

## Observability

- **O1.** When the KV cache layout holds persistent state, every
  `SchedulerStats` the scheduler produces carries three fields, which are
  `None` on an engine without persistent state:
    - `stream_occupied`: the number of slots by session state, under the keys
      `never_fed`, `live`, `parked`, `draining` and `retiring`. A request that
      was not admitted counts as `parked` while it waits for its next input
      and as `live` otherwise, once it holds a slot. A `retiring` slot belongs
      to a session that has ended; it counts, including while quarantined,
      until the fence releases it (fenced release).
    - `stream_refusals`: the number of refusals that `admit_stream` has
      returned since startup, by reason (E3); no refusal by the client
      (R2) or the frontend (A2), and no `check_admission` rejection (A1),
      is counted.
    - `stream_expiries`: the number of expired sessions since startup, under
      the keys `never_fed`, `parked_idle` and `draining` (E10).

These fields are not yet exported as Prometheus metrics.
