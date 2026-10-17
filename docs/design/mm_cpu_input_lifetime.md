# CPU multimodal input lifetime

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
`restore_encoder_inputs`. These tensors use the existing executor transport.
Prefix-cache hits are eligible even when no encoder computation was scheduled.
Model Runner V1 continues to use its existing lifetime policy. Address-only
`mm_processor_cache_type=shm` handles retain their existing ownership policy;
EngineCore does not hold replayable pixel tensors for that separate cache/ACK
protocol. The release protocol targets full tensor inputs (the
ordinary LRU processor cache or disabled processor caching).

## Validation

CPU tests cover confirmed consumption, speculative lookahead, in-flight prefill,
repeated hashes, request-local release, prefix-style reinstalls, synchronous and
asynchronous preemption/resume, abort, KV-load rollback restoration, and
preservation of EngineCore replay inputs.

The smallest regression selection uses CPU tensors and no model configuration
or weights:

```bash
VLLM_TARGET_DEVICE=cpu .venv/bin/python -m pytest --confcutdir=tests/v1/worker tests/v1/worker/test_encoder_runner.py -k consumed_cpu_input -q
VLLM_TARGET_DEVICE=cpu .venv/bin/python -m pytest --confcutdir=tests/v1/core tests/v1/core/test_scheduler.py -k 'cpu_encoder_input and not schedule_resume' -q
```

These seven cases cover storage lifetime, replay ownership, repeated occurrences,
the confirmed computation boundary, and rollback restoration. The two additional
`cpu_encoder_input_release_schedule_resume_and_abort` cases exercise real
synchronous/asynchronous scheduling with the LLaVA configuration, without loading
model weights. Use `-k cpu_encoder_input` to include them. `--confcutdir` avoids
unrelated root accelerator fixtures for these self-contained CPU tests.

Model validation should compare outputs and live worker input storage with
ordinary tensor transport. Preemption/cancellation and long decoding remain
necessary serving checks; CPU rollback coverage does not imply a model run
exercised KV-load rollback.
