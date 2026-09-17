# CPU multimodal input lifetime and local sharing

Model Runner V2 keeps the vision encoder's existing tensor-parallel computation.
This change addresses ownership of its CPU inputs, without changing TP, EP,
sequence parallelism, encoder batching, or GPU embedding-cache eviction.

## Request-local release

The scheduler sends `SchedulerOutput.free_encoder_input_ids`, a mapping from
request ID to input indices. An input is eligible once its placeholder end plus
prefill lookahead is at or before the confirmed computation boundary. That
boundary excludes both unconfirmed output placeholders and in-flight tokens,
including asynchronously scheduled prefill. Inputs scheduled for encoding in the
current step are excluded.

The worker applies the notification after installing new requests and computing
any initial model-specific positional metadata. Its `EncoderCache` drops the
selected feature's `data`, while retaining its identifier and placeholder
metadata. It does not clear all inputs with the same hash, and does not evict
GPU embeddings.

EngineCore retains its original feature objects for replay. The worker owns a
separate feature list and replaces entries rather than mutating those objects;
this also preserves correctness with the uniprocess executor. Preemption and
new/resumed/streaming request installation reset the scheduler's release tracking.
If a KV-load failure rewinds a running request without preempting it, a subsequent
encoder schedule reinstalls any previously released input from EngineCore through
`restore_encoder_inputs`. These tensors use the same optional local transport.
Prefix-cache hits are eligible even when no encoder computation was scheduled.
Model Runner V1 continues to use its existing lifetime policy. Address-only
`mm_processor_cache_type=shm` handles retain their existing ownership policy;
EngineCore does not hold replayable pixel tensors for that separate cache/ACK
protocol. The new lifetime and sharing paths target full tensor inputs (the
ordinary LRU processor cache or disabled processor caching).

## Optional local backing storage

Set `VLLM_MM_INPUT_SHARED_STORAGE_PATH` to an existing, writable, node-local
POSIX directory visible at the same absolute path to the multiprocess executor
and its local workers. Leave it unset for the ordinary transport.

```bash
export VLLM_MM_INPUT_SHARED_STORAGE_PATH=/dev/shm/vllm-mm-inputs
```

Provision that directory and its capacity before launching the service. A
memory-backed mount consumes the pod's memory budget; sharing does not make the
backing bytes free. Do not use a network filesystem. The existing 64 GiB `/dev/shm`
allocation must not be assumed sufficient for an unrestricted image workload.

Only CPU tensors of at least 1 MiB inside newly admitted multimodal features
use this path. Smaller fields and unsupported tensor types retain the ordinary
serialization. Other RPC tensors are not opted in. Each selected tensor gets a
single backing file; all local workers map that file. The writer still retains
its EngineCore input for replay, independently of this worker-shared allocation.

PyTorch has no read-only Tensor type. The implementation uses private,
copy-on-write file mappings with an immutable-input contract: reads share the
same physical pages, while an accidental in-place write cannot corrupt a peer.
Batching and device transfer can still allocate transient buffers. Do not sum
worker RSS to estimate shared physical bytes; use PSS and cgroup memory usage.

Each reader acknowledges only after mapping. The last acknowledgement unlinks
the file; kernel mapping lifetime then follows the surviving tensor storages.
Ring-buffer reuse cannot overwrite these inputs. The file API handles writes,
so storage exhaustion raises an allocation/write error before publishing the
input, instead of causing a mapped-write SIGBUS or silently restoring N copies.

Normal executor teardown removes files for readers that failed before mapping.
An advisory owner lease protects active stores. After a writer is killed, a new
store in the same directory reaps abandoned stores with released leases. This
recovery happens at startup, not continuously while the service is down.

Remote readers receive the ordinary tensor bytes, never local file paths. In the
TP8 x PP2 deployment with all eight first-stage TP workers on the driver node,
this removes those eight independent retained CPU allocations. This version
does not introduce a second shared-storage broadcaster on remote nodes, so it
does not provide node-wide sharing for TP groups spanning multiple nodes.

## Validation and rollout

CPU tests cover confirmed consumption, speculative lookahead, in-flight prefill,
repeated hashes, request-local release, prefix-style reinstalls, synchronous and
asynchronous preemption/resume, abort, and preservation of replay inputs.
Transport tests cover eight separate readers mapping one inode, BF16 and
noncontiguous inputs, a slow reader, ring and overflow paths, remote fallback,
copy-on-write isolation, unlink with live views, serialization failure, and
writer-crash recovery without removing another live store.

Before production rollout, compare the same model and deterministic request set
with ordinary transport and shared transport. Exercise chunked prefill, repeated
images, prefix hits, forced preemption, cancellation, and long decoding. Check
outputs and vision-encoder inputs, as well as CPU PSS, cgroup memory, worker live
input bytes, GPU memory, latency, and throughput. These CPU tests do not replace
a TP8 x PP2 model evaluation or a repeat of the original OOM workload.
