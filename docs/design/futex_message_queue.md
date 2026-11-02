# MessageQueue futex default and stock opt-out

The local shared-memory `MessageQueue` defaults to `auto`: it selects futex on
64-bit Linux x86_64/aarch64 CPython with the GIL enabled and the native extension
installed. Set `VLLM_ASA_SHM_WAIT=stock` to opt out; stock needs no native
extension. The futex backend changes how local participants wait for publication
and capacity, not the payload ring, serialization, tensor arena, overflow, or
remote ZMQ transport.

## Build and select the backend

This backend requires Linux on x86_64 or aarch64, lock-free process-shared atomics,
and the packaged `vllm._asa_shmwait` extension. Follow the
[incremental compilation workflow](../contributing/incremental_build.md) with
the same Python and build configuration as the editable vLLM installation:

```sh
cmake --preset release
cmake --build --preset release --target _asa_shmwait
cmake --install cmake-build-release --component _asa_shmwait
```

The extension uses Python's stable API from 3.11 on GIL-enabled builds. It
requires the GIL: releasing it while blocked does not make the process-local
endpoint state safe for concurrent free-threaded access. A free-threaded build
uses its actual interpreter extension suffix, but that is not a claim of
free-threaded runtime support or binary compatibility. A regular
source build includes the component on supported platforms. An older precompiled
wheel without it needs the targeted component build before selecting futex mode.

For example, with an already installed compatible model and environment:

```sh
VLLM_ASA_SHM_WAIT=futex .venv/bin/python \
  examples/basic/offline_inference/generate.py \
  --model /path/to/local/model --distributed-executor-backend mp \
  --enforce-eager --max-tokens 16
```

Explicitly select the multiprocessing executor when validating this path. A
single-GPU run that defaults to the `uni` executor does not exercise its queues.
For implicit `auto`, unsupported runtimes or an absent optional extension use
stock and emit an initialization message explaining the choice. Import failures
inside the extension, ABI/loader failures, invalid control configuration and
queue initialization errors are not hidden by this fallback. An explicit
`VLLM_ASA_SHM_WAIT=futex` request for local control fails on unsupported runtimes
or when the extension is absent. GIL-disabled execution is unsupported; the
runtime check uses CPython's
[`sys._is_gil_enabled()`](https://docs.python.org/3.13/library/sys.html#sys._is_gil_enabled)
where available. Python 3.11/3.12 CPython uses the GIL.

Remote-only queues do not select or create local control state and continue to
use ZMQ without loading the extension. Mixed queues select local control once
at the writer while retaining their remote ZMQ transport. Local readers follow
the writer's resolved handle even if their environment requests a different
backend; they still must run a supported GIL-enabled interpreter to attach to
futex control. This queue policy does not change weight-loading defaults.

## Configuration

Set these variables before constructing the writer's queue. Reader handles carry
the selected control geometry and wait policy to the local readers.
As with other vLLM environment settings, set them before service initialization
enables the environment cache; changing a worker's environment afterward does
not reconfigure an existing queue or override its writer's handle.

`VLLM_USE_SPINLOOP_EXT` continues to control the stock path. In `futex` mode,
the explicit `VLLM_ASA_SHM_SPIN_NS` interval is the alternative wait policy;
the stock spinloop extension and notification socket are not used for local
waits. Remote-only and mixed queues keep their existing ZMQ payload transport.

| Variable | Default | Accepted values |
| --- | --- | --- |
| `VLLM_ASA_SHM_WAIT` | `auto` | `auto`, `stock`, `futex` |
| `VLLM_ASA_SHM_READER_STRIDE` | `128` | `16`, `128` bytes |
| `VLLM_ASA_SHM_SPIN_NS` | `0` | Integer nanoseconds from `0` to `1000000` |

The stride separates per-reader control records. A larger stride can avoid false
sharing but uses more control memory. The optional spin interval trades CPU use
against wake latency before parking. There is no automatic adaptive tuning.

## Protocol and limits

There is one writer and one endpoint per local reader. All local participants
must use a compatible vLLM version and control ABI; mixed-version futex
participants are not supported. The payload storage remains in the existing
ring, with a separate shared-memory block for publication and consumption state.
Use one acquiring owner per endpoint. Shutdown must wake blocked operations and
wait for users to stop before dropping the endpoint's shared-memory mapping;
do not unmap storage while another thread is waiting or using an acquired slot.

Only the creator tracks and unlinks each shared-memory block. Non-owning ring,
arena, object-storage and control attachments use `SharedMemory(track=False)` on
Python 3.13 and newer. Python 3.11/3.12 retain the resource-tracker workaround;
they do not have that public attachment option. This changes lifetime
bookkeeping, not the queue's shared-memory protocol or wait policy.

The writer publishes only after a successful write. An aborted write does not
advance the sequence. A reader releases its slot even if its consumer raises.
Readers can drain committed messages after the writer closes; cancellation and
peer closure wake blocked participants. Waiter registration and a final predicate
check precede futex parking so publication and cancellation cannot miss a waiter.
Deadlines are monotonic, and the existing timeout/warning callbacks remain in use.
As with the existing queue lifecycle, executors must call `shutdown` when a
participant exits. Abrupt peer death without that lifecycle notification is not
detected by a futex alone; an unbounded acquire is not a process-liveness check.

This backend is not a GPU kernel or a change to model semantics. It still needs
serving-level validation for each intended deployment. Blocking wakeups can add
latency, and lower CPU use does not by itself establish higher throughput or
unchanged model quality. Compare against `stock` using the same multiprocessing
backend, model, requests, and hardware; do not use a `uni` run as the control.

The packaged native controls and actual queue integration tests are separate:

```sh
.venv/bin/python -m pytest \
  tests/distributed/test_asa_futex_source_contract.py
.venv/bin/python -m pytest tests/distributed/test_shm_broadcast.py
```
