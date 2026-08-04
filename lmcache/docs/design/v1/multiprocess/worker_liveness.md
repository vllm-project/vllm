# Worker Liveness Tracking and Reaping (Multiprocess Mode)

Covers the MP server (`lmcache/v1/multiprocess/`) and the worker heartbeat in
the vLLM multiprocess adapter.

## 1. Problem

When an engine worker dies without sending `UNREGISTER_KV_CACHE` (SIGKILL,
OOM-kill, node loss), its per-instance state leaks on the MP server forever: the
LMCache-driven `ContextEntry` (a `GPUCacheContext` holding CUDA IPC handles), the
`EngineDrivenContextEntry` + `TransferStrategy` pair, and any blend-mode
per-instance state (e.g. CB rope caches). Nothing observes worker death; on a
shared server, leaked contexts accumulate until device memory is exhausted.

Worse, `instance_id` is `os.getpid()`: containerized pods reuse small PIDs, so a
new worker can register with a dead worker's id, and the idempotent register
silently binds it to the stale context — wrong IPC handles, corrupted transfers.

## 2. Design Overview

The server already receives a periodic signal from every actively serving worker:
the heartbeat PING. The design adds the worker's `instance_id` to that message,
stamps `last_seen` on the per-instance entries that already exist, and runs one
periodic scan that reaps entries silent longer than a timeout via the same cleanup
as a client unregister. The heartbeat keeps its lazy start (first store/retrieve)
and starts healthy; a live worker pings every interval, refreshing its `last_seen`,
so it is never reaped while alive. Entries that never produced a liveness signal
fall under a generous registration grace. Recovery reuses existing client
machinery: on a genuine outage the heartbeat's pings fail, clearing `health_event`;
when the server returns the unhealthy-to-healthy edge fires the recover callback,
which re-registers — a noop when the entry survived.

```
engine worker adapter                            MP server
+---------------------------+                   +--------------------------------------+
| HeartbeatThread           |   PING [id]       | ManagementModule                     |
|  (instance_id, 10s)  -----+------------------>|  ping(id) -> touch_instance(id)      |
|  lazy start on first req, |   (NORMAL pool)   |  reaper thread (scan = timeout/4)    |
|   (starts healthy)        |                   |   -> reap_stale_instances(           |
|  unhealthy->healthy edge  |                   |        timeout, registration_grace)  |
|   -> re-register callback |   REGISTER        |   -> drop_instance_state(id) fan-out |
|                           +------------------>|        |                |            |
| register_kv_caches        |   STORE/RETRIEVE  |        v                v            |
|  (no pings until traffic) |   (refresh too)   | LMCacheDriven/          BlendV3      |
|                           |                   |  EngineDriven           (CB rope     |
|                           |                   |  TransferModule          dropped)    |
|                           |                   |  Entry{.., last_seen,               |
|                           |                   |   has_liveness_signal}               |
+---------------------------+                   |  _lock (leaf); pop -> cleanup        |
                                                +--------------------------------------+
```

## 3. Protocol Change

The PING payload changes in place from `[]` to `[int | None]`; the response stays
`bool`, always `True` (`None` marks an untracked prober such as the scheduler
adapter, which registers nothing and is never reapable). PING keeps its BLOCKING
dispatch on the NORMAL thread pool. SYNC dispatch was considered but rejected:
SYNC runs on the MQ main loop, where a slow `REGISTER_KV_CACHE` (also SYNC) would
block PING and make a live worker look dead. Sharing the NORMAL pool is in fact
desirable — if the pool cannot answer PING within the heartbeat timeout, the
worker *should* enter degraded mode (the same back-pressure signal). The payload
change is wire-visible, so both sides upgrade together (Section 7).

## 4. Instance ID Generation

The worker adapter replaces `os.getpid()` with
`uuid.uuid4().int & ((1 << 63) - 1)`, INFO-logged at construction so operators can
correlate reap warnings with a specific pod. `uuid4` reads OS entropy (identically
seeded processes cannot collide); the 63-bit mask keeps the value int64-safe.
Every id-carrying request reads the same field, so no other payloads change.

## 5. Server Side

### 5.1 Liveness state and two-tier windows

`ContextEntry` (LMCache-driven) and `EngineDrivenContextEntry` gain
`last_seen: float` (`time.monotonic()`) and `has_liveness_signal: bool`, latched
only by PING. The flag selects the staleness window: an entry that has pinged
provably runs the heartbeat protocol and is judged on the reap timeout; one that
never pinged (e.g. still warming under lazy start) is judged on the generous
registration grace. `last_seen` is refreshed by PING (`touch_instance`: refresh
if present, never insert), register (create and NOOP paths), and every transfer
path, so a worker mid-transfer is never reaped; traffic never latches the flag.

### 5.2 Locking

The reaper runs on its own thread, so the per-instance dicts are now mutated off
the MQ handler threads. Each transfer module gains one `threading.Lock` so the
reaper's scan-and-pop cannot race a concurrent register/unregister/transfer (which
would otherwise corrupt the dict or hand out a half-removed entry). In
`EngineDrivenTransferModule` the context and strategy dicts mutate as a pair under
that lock, so a reap racing a re-register can never strand a fresh context without its
strategy. It is a leaf lock — never held across context construction, storage
calls, or any other component — so no thread ever holds two locks. External
readers use the locked accessors `get_and_touch_context_entry` (get-and-refresh) and
`context_entries_snapshot` instead of touching the dict directly.

### 5.3 Reaper

`ManagementModule` (the PING owner) owns the reaper thread, scanning every
`reap_timeout / 4`. Each scan calls `reap_stale_instances` on every target: under
the module lock, collect ids whose staleness exceeds their window and pop them;
outside the lock, run the same cleanup as a client unregister and log a WARNING
per instance (repeated reaps of one id signal a too-small timeout). Reaped ids are
then passed to `drop_instance_state(id)` on every target; `BlendV3Module` drops
the reaped instance's per-instance CB state (e.g. rope state) there. (It no longer
mirrors the GPU cache context — that mirror was removed upstream, so reaping the
GPU entry now frees the context directly.) Collect+pop shares the module lock with
register's refresh, serializing every register-vs-reap race; on close, the reaper
is stopped and joined before any module clears state.

### 5.4 Public protocol and config

```python
class InstanceLivenessTarget(Protocol):
    # All methods default to a no-op; an implementer overrides only its role.
    def touch_instance(self, instance_id: int) -> None: ...
    def reap_stale_instances(
        self, reap_timeout_s: float, registration_grace_s: float
    ) -> list[int]: ...
    def tracked_instance_count(self) -> int: ...
    def drop_instance_state(self, instance_id: int) -> None: ...
```

One protocol covers both reaper-driven roles. The transfer modules override the
liveness methods (`touch`/`reap`/`count`); `BlendV3Module` overrides only
`drop_instance_state` to drop its mirrored CB state. `ManagementModule` receives
all targets in a single injected list. (Earlier a separate one-method
`InstanceReapListener` held `drop_instance_state`; it was folded in since only
`BlendV3Module` ever implemented it.)

Config: `worker_reap_timeout_seconds` (default `120.0`; `0` disables, otherwise
`>= 30.0`) and `worker_registration_grace_seconds` (default `3600.0`; `>=` the
reap timeout — a tighter grace would reap warming workers faster than crashed
ones), with matching CLI flags. Keep the timeout `>= 3 x` the client's
`lmcache.mp.heartbeat_interval` so a few missed pings never reap a live worker;
the worker adapter warns at startup when `3 x interval` exceeds the 30 s floor.

## 6. Adapter Side

### 6.1 Lazy start

The heartbeat keeps its lazy start on first store/retrieve — no pings during
warmup; the registration grace covers that window. It starts healthy (the event
is set at construction), so the first store/retrieve is not gated. A live worker
then pings every interval, refreshing its server-side `last_seen`, so it is never
reaped while alive — no re-registration is needed at start. The recover callback
re-registers only on a genuine recovery edge (Section 6.2). A retrieve dropped
while the server is unhealthy is still reported via `get_finished` so async loads
cannot hang.

### 6.2 Recovery after a reap

```
T0        outage begins; pings time out -> health_event cleared, traffic stops
T0+120s   server: entry stale -> reap pops it, frees GPUCacheContext/IPC,
          layout-desc refcount; blend rope state dropped via listener <- leak fixed
T1        connectivity back; next ping succeeds -> unhealthy->healthy edge
T1        recover callback re-registers (id absent -> fresh context) before
          health_event is set; traffic resumes             <- exactly one context
```

A shorter outage hits the NOOP register path, which refreshes `last_seen` and
builds nothing — the server never asks a worker to re-register.

### 6.3 Shutdown

`shutdown()` stops the heartbeat before sending UNREGISTER, so no stray ping
lands on a closing client. The heartbeat cycle skips the recover callback and
`health_event.set()` once stopped, and the callback skips re-registration when a
stop is already requested — a straggling cycle cannot re-create a ghost context.

## 7. Failure Modes

| Scenario | Behavior |
|---|---|
| Worker crash (SIGKILL, no UNREGISTER) after serving | Pings stop; reaped within ~`timeout + timeout/4`. Context, IPC handles, layout-desc refcount, and non-GPU strategy released via the same cleanup as a clean unregister; blend rope state dropped via the reap listener. The bug being fixed. |
| Worker crash during warmup (registered, never pinged) | Reaped on the registration grace. Bounded leak instead of a permanent one. |
| Worker alive but never pinged, idle past the grace | Reaped while alive only if it never pinged (heartbeat never started). Once the heartbeat is running, pings refresh `last_seen` every interval, so a live worker is never reaped regardless of traffic. |
| Heartbeat thread starved, worker transferring | Store/retrieve/prepare/commit refresh `last_seen`; never reaped. |
| Partition shorter than the reap window | No reap. On heal, the recover callback re-registers; the NOOP path refreshes `last_seen`; zero context churn. |
| Worker crash + restart | The new process gets a fresh uuid-derived id and a fresh entry; the dead id is reaped independently. No PID-reuse aliasing. |
| Mixed client/server versions | Every PING fails the payload-count check; the client sits permanently unhealthy. Loud, never silent corruption; upgrade both sides together. |
