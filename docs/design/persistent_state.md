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
  first schedule until it is released; slots are never shared, prefix-cached,
  or swapped.
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
  before it processed that output may still reach the engine as a new request
  with the same id; it waits for the retiring slot (fenced release), and the
  frontend's abort ends it.
- **One slot per request.** A request holds one slot from its first schedule until the slot is
  released (fenced release); no two live requests hold the same slot, and a request holds
  no slot before it is first scheduled.
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
  nothing from it. The request finishes with an error at once, even mid-turn,
  and its slot retires behind the release fence (a request that was never
  scheduled holds no slot).

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
last scheduled step and its cleanup. Until then, scheduling treats the slot as
occupied, so a new request can never observe writes from the previous owner.
Model failures are reported through the same completion path, and a scheduled
step without a matching completion fails its request. A cleanup that is not
acknowledged is retried once; a second miss is a fatal engine error.

- **Fenced release.** A slot whose request finished or was aborted (preemption ends a
  request the same way but has no public path, because state requests never
  need new blocks and a reset is refused while a slot is bound or parked; see
  [Cache clearing and release](#cache-clearing-and-release))
  returns to the free pool only after the scheduler has received a
  `StateCompletion` covering the request's last scheduled step and a later
  cleanup of its handle. Until then scheduling binds no other handle to it,
  and a new request with the same request id waits until that slot is
  released.
- **Unscheduled slot.** A slot that was never scheduled is released at once. Every slot is
  allocated by a schedule, so this is a property of the slot manager.
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
  outputs it drops, deliver no queued persistent-state error: an error queued
  before or during those steps is delivered, once, by the next step after the
  clear.

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
