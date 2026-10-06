# [RFC][KV Hints] Execute COPY_TO/MOVE_TO through Mooncake Store

Draft reference implementation for RFC #59708, complementary to #59714's G1-only
SET_PRIORITY example. A real inference request can now queue GPU-to-Store MOVE
at successful completion; the next request can issue Store-to-GPU COPY at ingress,
wait for actual publication and demonstrate prefix reuse. No synthetic inference
request is needed for transfer progress after the producer finishes.

Public primitives use version 1.0 in the merged envelope. One dispatcher handles
both primitive and legacy profiles, with explicit whole-envelope backend ownership.
GPU resources remain owned by the existing KV manager; COPY/MOVE lower to existing
CPU/Mooncake workers with transfer holds, cancellation, all-rank publication and
per-target results. SET_PRIORITY uses revisioned copy-lifetime claims, with one
eviction queue shared with legacy content-level declarations. Optional policies
produce the same primitives and are disabled by default in the main example.

This adapts #59714's priority contract; it does not duplicate that PR's eviction
heap or claim its code as an independent contribution. Relative to #58198, the
reference includes envelope/lifecycle execution, correlated outcomes and real
Mooncake G2 work. Final duplication/reuse review remains required before publishing.
The local implementation also contains the broader #57103 experimental platform;
its incomplete requirements remain explicitly tracked rather than being described
as part of the validated example.

## Validation

- Final control/state/engine/cache/scheduler/HTTP regression: 638 passed.
- Store unit/real GPU buffer/KVCR suites: 396 passed.
- General CPU manager/worker/connector suites: 566 passed, 1 ROCm-specific skip.
- Real Qwen TP=2 R1 MOVE / R2 COPY: 784 cached tokens, identical short greedy outputs.
- Opt-in tool-wait policy: derived version-1.0 MOVE/COPY reach actual Mooncake and reuse.
- Two-engine COPY through Store: B actually reuses the prepared prefix; A is preserved.
- Primitive-profile GSM8K: recompute 124/128; movement 125/128; zero truncations,
  one final numeric-answer mismatch, cached tokens 0/100352. No accuracy/speedup claim.

Commands, intermediate failure/fix details, process boundaries and raw artifacts
are recorded in `validation.md` and the benchmark README. Remaining validation
includes full rollout/verl integration, live weight transitions, pressure/fairness,
physical residency reporting and the external model matrix. This is not a
completed Dynamo/llm-d integration or a final upstream COPY/MOVE wire specification.

## Contribution disclosure and publication gate

AI assistance was used for implementation, tests and documentation. The human
submitter must review every changed line, run relevant checks, understand the
resource/lifetime behavior and defend the change. Required GitHub CLI duplicate
checks have not succeeded in this unauthenticated environment; public REST review
identified #59714, #58198 and #38514. Complete the latest checks and coordination
before publication. No PR has been opened from this branch.
