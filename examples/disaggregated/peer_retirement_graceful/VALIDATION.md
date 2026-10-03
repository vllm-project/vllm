# Graceful Exit Validation

## What Was Tested

On 2026-09-20, an isolated eight-GPU H20 host ran two Prefill and two Decode
instances, TP=2 each, using DeepSeek-V4-Flash (non-0731), vLLM 0.26.0, NIXL 1.4.1
and UCX v1.22.x commit `5288e74cd40be622e109489de7bb80d821341bd5`.
The configuration included `UCX_TLS=tcp,cuda_ipc,cuda_copy`,
`UCX_CUDA_IPC_CACHE=n` and NIXL UCX `num_threads=8`. Actual loaded native
library paths and the cache setting were checked.

This directory packages the tested worker extension, middleware and exact-source
patch. License headers, formatting and test fixture paths are packaging changes;
the internal host-specific launcher/controller and raw internal logs are not
published. The README provides the API contract and an isolated exercise, not
the original controller as a portable one-command reproducer.

| Check | Observed result |
| --- | --- |
| Control: P exits without explicit D cleanup | 42,912 MiB remained on each former P GPU for approximately 120 seconds |
| Candidate: D clears old P references, then P exits | Both former P GPUs measured 0 MiB in all 60 samples over 119.502 seconds |
| Work admitted to old P at router fence | Two requests; both completed before cleanup |
| Fence to completed responses | 15.695 seconds |
| All-rank acknowledgement | Both TP ranks on each D returned prepared, then cleanup_returned |
| Unrelated peer | P1 remained registered on both D instances |
| Surviving processes | P1 and both D container identities and TP worker PIDs unchanged |
| Fresh P1 transfers after cleanup | 64K request to each D succeeded; 578,062,080 additional NIXL bytes per pair, zero failed-transfer increment |
| Drain / cleanup / memory observation traffic | 163/163 logical P-D requests succeeded |
| Replacement P | New generation became healthy on the same GPU pair |
| Background traffic during replacement | 107/107 succeeded; maximum completion gap 2.179 seconds |
| Final pair tests | Short and 64K requests across all four pairs: 8/8 succeeded |

Together with eight baseline requests, 286 logical P-D requests completed and
passed the experiment's checks. Native failed-transfer and failed-notification
counters on both D instances remained zero in the final repeated samples.
Experimental containers were stopped only after this acceptance; final full
teardown is not evidence for the earlier P-only memory-release result.

## Interpretation and Limits

The native binaries were the same in control and candidate. The Python candidate
includes generation fences, TTL safety guards and notification-only metadata
cleanup in addition to explicit retirement. This was not a one-line A/B.

Disabling the mapping cache alone did not eliminate retention. On the pinned
UCX commit, rkey release can call `uct_cuda_ipc_unmap_memhandle()`, which drops
a mapping reference. A cache region with zero references is destroyed when
caching is disabled. References still held by D must first be released.

The experiment did not directly trace each native unmap or CUDA close. The
evidence for reclamation is GPU memory returning to zero while D survives and
unrelated transfers succeed. `cleanup_returned` alone proves neither native
completion nor memory reclamation. No endpoint-wide forced cache purge was
introduced.

The D being cleaned temporarily admitted no new requests and had no active
HTTP responses. Python-visible connector queues were checked, not all native
thread queues. Background traffic went to the other D. Partial failures or
timeouts retain the admission freeze; the prototype cannot recover arbitrary
unknown native completion. This is not general lossless autoscaling, nor proof
that active generation on the target D is unaffected.

No claim is made for abrupt P death/OOM, in-flight DMA cancellation, force
unmap, multi-API, push/bidirectional transport, formal UCX v1.22.0 or other
revisions, B300 validation of this exact candidate, long-duration stability,
production SLA, or broad model accuracy. No native NIXL/UCX code changes are
part of this example.

## CPU Verification

Run the commands in [README.md](README.md#cpu-tests). The tests exercise actual
hash-pinned v0.26.0 methods with a mocked native boundary: cleanup ordering,
busy-state refusal, idempotence, generation fencing, TTL safety, notification-only
metadata, partial failure, all-rank acknowledgement, authentication and admission
races. They do not load the full vLLM package or exercise real CUDA/NIXL.

The separate pre-publication lab runtime/controller suite passed 51 tests.
Only the runtime tests are packaged here; do not attribute all 51 to this
directory. Publication-specific test counts and commands are recorded in the
PR description.

## Sources and Related Work

- [UCX rkey release](https://github.com/openucx/ucx/blob/5288e74cd40be622e109489de7bb80d821341bd5/src/uct/cuda/cuda_ipc/cuda_ipc_md.c)
- [UCX mapping reference count and cache](https://github.com/openucx/ucx/blob/5288e74cd40be622e109489de7bb80d821341bd5/src/uct/cuda/cuda_ipc/cuda_ipc_cache.c)
- [NIXL remote metadata invalidation](https://github.com/ai-dynamo/nixl/blob/v1.4.1/src/core/nixl_agent.cpp)
- [#50047: replacement-triggered peer cleanup](https://github.com/vllm-project/vllm/pull/50047)
- [#56341: P-side cleanup after D death](https://github.com/vllm-project/vllm/pull/56341)

The graceful scale-down signal here complements replacement-triggered cleanup:
no replacement may exist, or it may not be able to load until old memory is
released. An eventual current-main implementation should share the existing
lifecycle API and cleanup machinery instead of adding competing mechanisms.
