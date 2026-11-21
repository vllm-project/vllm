# Shared-Memory Message Queue

`MessageQueue` in `vllm/distributed/device_communicators/shm_broadcast.py` is
vLLM's one-writer, many-reader channel between processes. It carries the
engine's per-step RPCs to its workers and the workers' replies back, and it
backs `GroupCoordinator.broadcast_object` for the TP, DCP and inner-DP world
groups. This page describes what uses it, how a message moves, the wake-up
protocol, the conditions under which a reader can miss a wake-up, and the
formal models that check the protocol.

## Producers and consumers

| Queue | Writer | Readers | Traffic |
| --- | --- | --- | --- |
| `rpc_broadcast_mq` (`MultiprocExecutor`, `RayExecutorV2`) | EngineCore executor | every worker of the DP replica | one message per `collective_rpc`, so one or more per scheduler step (`execute_model`, `sample_tokens`) |
| `worker_response_mq` (one per worker) | that worker | the executor | one reply per RPC from each replying rank; `execute_model` replies only from the output rank |
| `create_single_reader_mq_broadcasters` (inner DP world, multi-node) | every worker | the output-rank worker | cross-node replies relayed to the executor |
| `GroupCoordinator.mq_broadcaster` (`tp`, `dcp`, `inner_dp_world`) | group rank 0 | the other group ranks | `broadcast_object` and the metadata half of `broadcast_tensor_dict` |

The hot path is the executor loop: the executor enqueues an RPC, every worker
dequeues it, runs the step, and the output rank enqueues its reply, which the
executor dequeues. With async scheduling, several RPCs are outstanding at once.

## Data path

Local readers share an `ShmRingBuffer` of `max_chunks` slots of
`max_chunk_bytes` each. Each slot has a metadata row: one written flag and one
read flag per local reader. The writer waits until every reader has read a
slot, clears the written flag, copies the payload, clears the read flags, and
sets the written flag. A reader waits for written-and-unread, copies the
payload, and sets its read flag. Memory fences order the flag and payload
stores. After the metadata rows, the buffer holds one parked flag and one
fence-capability flag per local reader, then the wake-mode byte used by the
wake-up protocol below.

Two zmq XPUB data sockets carry what does not fit the ring:

- **Overflow.** A payload larger than a slot marks the slot as overflow and
  follows on `local_socket`. The reader releases the slot, then waits for the
  payload on the socket.
- **Remote readers.** Readers on other nodes receive every message on the
  `remote_socket` TCP XPUB. Nothing acknowledges these messages.

Both data sockets set `XPUB_NODROP`. A default XPUB silently drops a message
when a reader's pipe looks full, and neither a released overflow slot nor a
remote reader would ever see that message again. With `XPUB_NODROP`, a
blocking send waits for room instead, and the high water mark still bounds the
queued memory.

## Waiting

A reader that finds its slot empty spins for `busy_loop_s` (1 s) after its
last read, then parks in `zmq.Poller.poll` on its notification socket. After
a write, the writer sends a one-byte ping on that socket. Each reader's SUB
socket sets `CONFLATE`, so pings that arrive while the reader spins collapse
into one, and a parked reader consumes the single pending ping when it wakes.
A busy reader also drains its socket every `SHM_NOTIFY_POLL_INTERVAL_S` (2 s),
so libzmq's per-notification commands cannot pile up. Every park is capped at
`SHM_READER_RECHECK_INTERVAL_MS` (5 s), after which the reader re-reads the
flags. A writer waiting for readers to free a slot polls the flags, so it
never depends on a notification.

`shutdown()` sets `shutting_down` and sends the reader's cancel ping on an
inproc PAIR socket. Every reader wait polls that socket: the ring park, the
overflow-payload wait on `local_socket`, and a remote reader's wait on
`remote_socket`. The reader then raises `RuntimeError("cancelled")`.

### Hardware waits

With `VLLM_USE_SPINLOOP_EXT=1`, the spinning part of a wait runs in the
`vllm.spinloop` extension. It calls a Python predicate, and while the
predicate is false it waits in hardware on the metadata bytes the predicate
reads, so a store to them ends the wait:

| CPU | Armed wait | Condition | Otherwise |
| --- | --- | --- | --- |
| AMD with `MONITORX` | `MONITORX` / `MWAITX` | watched bytes in one monitor line | `MWAITX` with a 150 000-cycle timer, armed on the first line |
| x86-64 with `WAITPKG` and a usable invariant-TSC leaf | `UMONITOR` / `UMWAIT` in C0.1, at most 1 ms | watched bytes in one monitor line | `TPAUSE` in C0.1, at most 50 µs |
| AArch64 with `FEAT_WFxT` | exclusive load (`LDXR`) then `WFET`, at most 1 ms | watched bytes in one aligned 64-bit word | `WFET`, at most 50 µs |
| other | - | - | `pause` / `yield` |

Each armed wait closes the race between the predicate and the arm. On x86 the
predicate runs again after `MONITORX` or `UMONITOR` arms the line. On AArch64
the extension snapshots the watched word before the predicate and compares it
with the exclusive load: a store before the load changes the word and skips
the wait, a store after it clears the exclusive monitor and ends the `WFET`.
Nothing between the exclusive load and the `WFET` stores to memory.

An arm watches one monitor line or one aligned word, not an arbitrary byte
range. When the watched bytes do not fit in one, the extension uses a short
timed wait, because a store to the uncovered bytes would not end an armed
wait. The watched bytes are the written flag through the waiting reader's own
read flag: every byte the readability check reads. A writer's metadata row is
`1 + n_reader` bytes and can straddle a line, so it often takes the timed
wait.

`MessageQueue.spin_until_readable`, the executor's wait for a worker reply,
uses the same hardware wait in bounded steps, re-checking for new engine
inputs between steps.

### Wake-up invariant

A reader re-reads the flags after every wake-up and before every park, so the
protocol is correct as long as one property holds:

> A parked reader whose slot is ready always has a wake-up pending: in its
> `CONFLATE` slot, in flight, or about to be sent by the writer.

If a notification is lost, the parked reader sleeps until the recheck, up to
5 s. In the executor loop that stalls the whole step: the executor waits for
the reply while every worker waits in `dequeue`.

### How zmq can lose a notification

A zmq PUB socket drops a message, without an error, in two cases:

1. **Stale high water mark view.** The PUB drops a message when its view of
   the pipe says the pipe is full. The view is refreshed only when the socket
   processes its commands. A send does that at most once every ~3M CPU-counter
   ticks (~1 ms on x86, ~3 ms on aarch64), and the pipe reader returns credit
   only every `(hwm + 1) / 2` drained messages. With `SNDHWM=1`, a ping sent
   within that window of the previous one is dropped even after the reader has
   consumed the previous one. Any finite high water mark can drop a ping in the
   same way after enough sends without a refresh.
2. **Unknown subscription.** The PUB drops a message for a subscriber whose
   subscription it has not yet processed. The start-up handshake in
   `wait_until_ready` used to cover only `local_socket`, so the first pings
   could race the readers' notification subscriptions.

The fixed 1 s grace hides both most of the time: a reader rarely parks within
milliseconds of a read or of start-up. It still happens whenever a step takes
just over the grace, for example a burst of requests after an idle second.

#### History

The notification socket came with
[PR #28053](https://github.com/vllm-project/vllm/pull/28053), which removed
the idle busy loop for shared IBM Z hosts. It set `SNDHWM=1` so that "PUB
sockets will silently drop subsequent messages" while a reader is busy:
dropping was meant as rate limiting, on the assumption that a full pipe means
the reader already has an unread ping. The stale view above breaks that
assumption. [PR #63165](https://github.com/vllm-project/vllm/pull/63165) made
busy readers drain their socket, because that drop was also what capped
libzmq's per-notification command backlog. The parked-flag protocol below
recovers the saving `SNDHWM=1` was after, without dropping pings, and keeps
the busy-reader drain.

### Notification channel

The writer's notification socket is an `XPUB` with `SNDHWM=0` (no limit) and
`XPUB_VERBOSE`, and `wait_until_ready` waits for every local reader's
notification subscription before it sends `READY`:

- `SNDHWM=0` removes the drop condition for the first case. The queue stays
  bounded: `CONFLATE` collapses pings on the reader side, the writer's I/O
  thread drains the pipe, a reader that stops reading stops releasing ring
  slots, which blocks the writer after `max_chunks` writes, and the
  parked-flag protocol sends no ping while no reader is parked.
- The subscription handshake removes the second case. `XPUB_VERBOSE` is
  required because every reader sends the same empty subscription.

### Parked-flag protocol

A ping is useful only to a reader that is parked. When every local reader can
issue a full fence, the writer pings only if some reader's parked flag is set:

| Step | Reader (`MessageQueue._park`) | Writer (`MessageQueue._notify_readers`) |
| --- | --- | --- |
| 1 | store `parked = 1` | store `written = 1` (in `acquire_write`) |
| 2 | full fence | full fence |
| 3 | re-check the slot; if ready, clear the flag and read | load the parked flags |
| 4 | otherwise poll, then clear the flag | ping if any flag is set |

This is a Dekker handshake: with both fences, either the writer sees the
parked flag and pings, or the reader's re-check sees the written flag. Two
details carry the correctness:

- **Every poll is flagged.** `SpinCondition.wait` reads the busy-loop window
  once and then either yields or calls `_park`, which runs the four steps. A
  second read of the window could expire between the two reads and poll
  without the flag; the writer would skip the ping and the reader would sleep
  until the recheck. `message_queue.qnt` models that bug as
  `skipIdleFlaglessPark`.
- **Step 2 is a store-to-load fence.** Each step 3 load must not be satisfied
  before the other process's step 1 store is visible. The fence is the
  `vllm.spinloop` extension's `full_fence()`, a C11
  `__atomic_thread_fence(__ATOMIC_SEQ_CST)`. `memory_fence()`, which acquires
  and releases a `threading.Lock`, orders only acquire and release accesses
  and is not a store-load barrier on weakly ordered CPUs.

| Architecture | Instruction |
| --- | --- |
| x86-64 | `lock or` (or `mfence`) |
| AArch64 | `dmb ish` |
| POWER (ppc64le) | `sync` |
| IBM Z (s390x) | `bcr 15,0` |
| RISC-V | `fence rw,rw` |

Each reader publishes whether it has `full_fence` in the ring before it
subscribes. After the subscription handshake, the writer reads every flag and
picks the wake mode: parked-only pings if every reader and the writer have the
fence, otherwise a ping after every write. It stores the mode in the ring
before `READY`, and readers read it after `READY`. The extension is built for
every target except XPU, independently of `VLLM_USE_SPINLOOP_EXT`, so the
fallback applies only to builds without it.

## Formal models

`tools/formal/shm_broadcast/` holds three [Quint](https://quint-lang.org)
specifications:

- `message_queue.qnt` models one queue: the ring, the reader spin-or-park
  choice, the zmq channel with its high water mark, credits, command throttle
  and subscriptions, and the parked-flag protocol with store buffers on both
  flags, so a missing fence lets a store become visible late.
- `inference.qnt` composes four queues into one TP=2 executor loop: the RPC
  broadcast, two reply queues, and a TP `broadcast_object` inside every step.
  It abstracts the channel as lossless, losing a reader's first ping, or
  losing any ping.
- `paths.qnt` models one local and one remote reader with overflow payloads
  on `local_socket`, every message on `remote_socket`, and `shutdown()`
  stopping the writer and cancelling both readers.

The invariants:

- `noLostWakeup`: the wake-up invariant above.
- `ringSafe` / `inOrder`: every reader receives every message in order, and
  no slot is reused before it is read.
- `noLoss`: no data-socket message is dropped.
- `progress`: without timeouts, some process or I/O thread can always move
  until every message is read or, after shutdown, until every reader exits.

| Spec | Configuration | Result |
| --- | --- | --- |
| `message_queue` | `hwm1`, `hwm1OneReader`: `SNDHWM=1`, throttled commands (the code before this change) | `noLostWakeup` fails |
| `message_queue` | `hwm1NoThrottle`: `SNDHWM=1`, commands on every send | holds |
| `message_queue` | `hwm2Slow`, `hwm4Slow`, `hwm4Burst`: finite high water mark | `noLostWakeup` fails |
| `message_queue` | `pr58917`, `unlimitedNoHandshake`: no subscription handshake | `noLostWakeup` fails |
| `message_queue` | `unlimitedHandshake`: lossless channel, ping every write | holds |
| `message_queue` | `skipIdleFenced`: lossless channel, parked-only pings | holds |
| `message_queue` | `skipIdleUnfenced`: parked-only pings without fences | `noLostWakeup` fails |
| `message_queue` | `skipIdleFlaglessPark`: a reader may poll without its flag | `noLostWakeup` fails |
| `inference` | `fixedDecode`, `fixedAllReply`: lossless channel | holds |
| `inference` | `hwm1Decode`, `raceDecode`: lossy channel | `noLostWakeup` and `progress` fail; the executor and both workers end parked |
| `inference` | `backpressureAtBound`: 1-slot rings, 3 outstanding RPCs | holds |
| `inference` | `backpressureOver`: 1-slot rings, 4 outstanding RPCs | `progress` fails |
| `paths` | `pathsCurrent`: default XPUB data sockets, uncancellable socket waits (the code before this change) | `inOrder`, `noLoss` and `progress` fail |
| `paths` | `pathsNoDrop`: `XPUB_NODROP` only | `progress` fails: a reader blocked on a data socket never exits after shutdown |
| `paths` | `pathsCancel`: cancellable socket waits only | `inOrder`, `noLoss` and `progress` fail |
| `paths` | `pathsFixed`: `XPUB_NODROP` and cancellable socket waits | holds |

Run `tools/formal/shm_broadcast/check.sh` to check every invariant of every
configuration separately, with TLC and Apalache in parallel; it exits
non-zero when any result differs from the table. `check.sh tlc` or
`check.sh apalache` runs one backend. TLC covers the full state space of
every check. Apalache must find each expected violation within the length of
the TLC counterexample, and checks every behavior of up to 10 steps (6 for
`inference`) of each check expected to hold. The checks whose counterexamples
are too deep for Apalache run under `check.sh tlc` only: `noLostWakeup` of
`hwm1`, `hwm2Slow`, `hwm4Slow` and `hwm4Burst`, `progress` of
`backpressureOver`, `inOrder` of `pathsCurrent`, and `inOrder` and `progress`
of `pathsCancel`.
`hwm1OneReader` is the Apalache-sized copy of `hwm1`.

The models treat a notification as a message that cannot overtake the
writer's written flag, as zmq's lock-based pipes and the kernel socket
guarantee, and they assume the payload copy is ordered before the written
flag by `memory_fence()`. They do not model the multi-node composition of the
inner-DP queues or the `READY` handshake on the data sockets.

### Keeping the code and the model in step

Each action of `message_queue.qnt` maps to one step of `MessageQueue` or
`SpinCondition`, named in the code's docstrings:

| Model action | Code |
| --- | --- |
| `writerHandshake` | `wait_until_ready` (`SpinCondition.wait_for_readers`) |
| `writerAcquire`, `writerFlush` | `acquire_write` up to its `yield`, and its context exit |
| `writerNotify` | `_notify_readers` (`_should_ping`, then `SpinCondition.notify`) |
| `readerRead` | `acquire_read` on a readable slot |
| `readerFlushPark`, `readerRecheck` | `_set_parked(True)`, `_recheck_parked` |
| `readerPoll`, `readerTimeout` | `SpinCondition.poll_notify` returning, then `_set_parked(False)` |
| `readerDrain` | `SpinCondition.drain_notify` (from `record_read`) |
| `deliver` | libzmq's I/O thread |

A reader's notification socket can hold two messages (the `CONFLATE` pipe
and one prefetched by `poll()`); `_take_notifications` empties both, which is
what the model's single `conflated` slot assumes.

`tests/distributed/test_shm_broadcast_model.py` replays traces of the
`skipIdleFenced` configuration on a real writer and two readers, driving
each action through its method and comparing the written, read and parked
flags and each reader's pending notification after every step. The traces
live in `tools/formal/shm_broadcast/traces/` with the hash of the model they
came from; after any change to `message_queue.qnt`, run
`tools/formal/shm_broadcast/gen_traces.sh` and update the code until the
replay passes. The test fails while the hash is stale, and it fails when the
code takes a step the model does not: an unconditional ping, a park without
the re-check, or a drain that leaves a notification behind.

## Benchmark

`benchmarks/benchmark_shm_broadcast.py` runs one writer and N reader
processes in the executor's lock-step pattern, using only public
`MessageQueue` APIs, so it runs against any vLLM tree on `PYTHONPATH`:

```bash
TMPDIR=/dev/shm/vbench PYTHONPATH=<tree> VLLM_TARGET_DEVICE=cpu \
  python benchmarks/benchmark_shm_broadcast.py --variant <name> --repeat 1 \
  --readers 1,4 --gaps 0,1,10 --idle-seconds 10 --target-wall 6 \
  --json-out results.jsonl
```

`TMPDIR` must be short: zmq IPC paths are limited to about 107 characters.
