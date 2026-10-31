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
- The KV cache memory budget is divided into state slots of `page_size_bytes`
  each. Each request holds exactly one slot from first schedule until it is
  released; slots are never shared, prefix-cached, or swapped.
- Each allocation is identified by a `StateSlotHandle` (engine epoch, slot,
  generation). The generation increases on every reuse, so a stale handle can
  never address a recycled slot. A new generation starts from the descriptors'
  initializers.
- The worker exposes only the current generation's slot to the model, through
  `ModelState.state_slots`. Models that write state from a stream other than the
  runner's main stream must register that stream with
  `StateSlotBindings.register_writing_stream` before enqueueing the writes.
- Preempting a request with persistent state finishes it with an error instead
  of recomputing it, because the state cannot be rebuilt from tokens.
- A resumable request parks for its next input only when a turn stops on EOS.
  Any other stop (length, stop token or string) ends the stream. Each new input
  replaces the request's prompt; the state carries the history.

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

With persistent state enabled, the engine core exposes two utility methods for
reserving a slot before any input arrives:

- `get_stream_admission_epoch(owner, renew=False)` returns a replay namespace
  (epoch) for an owner, such as one API server process. The epoch has a fixed
  lifetime; `renew=True` starts a new namespace and discards the old one.
- `admit_stream(owner, admission_id, request)` atomically reserves a slot and
  parks an empty resumable request. `admission_id` is a `StreamAdmissionId`
  (owner epoch, sequence). The request must be an empty, resumable sampling
  request (`n=1`, no stop strings, no structured outputs, no transfer
  connectors).

Admission is transactional and idempotent: retrying the same ID with the same
payload returns the recorded result and never extends a deadline; reusing an ID
with a different payload is an error. Refusals return
`StreamAdmissionResult(accepted=False, reason=...)` with one of `owner_epoch`,
`expired_id`, `request_id_in_use`, `capacity`, `active_limit`, or
`engine_paused`, and neither allocate a slot nor create a request.

Input is then sent as ordinary resumable updates for the same request ID with
`EngineCoreRequest.stream_admission` set to the admission identity; setting
`input_closed=True`, or sending a non-resumable update, closes input. Rejected
updates are reported on `EngineCoreOutput.stream_output` without ending the
session.

The engine expires sessions on its own monotonic clock, including while idle,
and aborts expired sessions with an error output. Configure the deadlines and
replay bounds with `--additional-config`:

```bash
vllm serve <model> \
  --additional-config '{"stream_admission": {"never_fed_timeout": 15.0, "parked_idle_timeout": 60.0}}'
```

| Key | Default | Meaning |
| --- | --- | --- |
| `never_fed_timeout` | `15.0` | Seconds an admitted stream may wait for its first input. |
| `parked_idle_timeout` | `60.0` | Seconds a parked stream may wait for its next input. |
| `draining_timeout` | `40.0` | Seconds a stream may take to finish after input closes. |
| `owner_epoch_timeout` | `300.0` | Lifetime of an owner epoch in seconds. |
| `max_records` | `4096` | Admission results remembered per owner for replay. |
| `max_owners` | `64` | Concurrent owner namespaces. |

Timeouts must be finite and positive; capacities must be positive integers.

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

## Observability

When persistent state is enabled, `SchedulerStats` carries:

- `stream_occupied`: slots by session state (`never_fed`, `live`, `parked`,
  `draining`, `awaiting_fence`).
- `stream_refusals`: admission refusals by reason.
- `stream_expiries`: expired sessions by deadline (`never_fed`, `parked_idle`,
  `draining`).

These fields are not yet exported as Prometheus metrics.
