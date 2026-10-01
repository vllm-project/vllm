# Programmable KV cache control

This change lets an application control supported KV retention, movement,
logical associations and task cleanup through the merged `KvHintsEnvelope`.
Native envelopes are interpreted once by vLLM and use the existing engine-owned
KV manager and connector paths; backend-selected envelopes remain opaque and
continue through KVCR's request-scoped hook. Request-free operations have
correlated bounded results, capability discovery, association events and state
queries. Optional startup policies propose work and scheduler guidance through
the same approval boundary without mutating queues or rewriting explicit
commands.

The delivery includes CPU offload and Mooncake Store adapters, a real local
Qwen TP=2 Mooncake cross-engine run, direct HTTP/Python controllers, seeded
rollout/recovery workloads, and a verl Agent Loop trace adapter. Mooncake source
content is preserved across Share and destination readiness is reported only
after registration.

## Validation

- `.venv/bin/python -m pytest tests/v1/test_kv_hint_control.py tests/v1/test_kv_hint_state.py tests/v1/engine/test_kv_control.py tests/v1/core/test_prefix_caching.py tests/v1/core/test_scheduler.py -q` — 574 passed.
- `.venv/bin/python -m pytest tests/entrypoints/serve/kv_control/test_api.py tests/v1/kv_offload/tiering/test_kvcr_tier.py -q` — 34 passed.
- CPU offload, Mooncake Store and real-buffer suites are recorded in `validation.md`; local Qwen movement and cross-engine results are saved under `benchmarks/programmable_kv_cache/results/`.
- External GPT-OSS-120B and DeepSeek-V4-Flash evaluation, policy pressure/fairness comparisons and live weight-update runs remain incomplete and are called out in the task list.

## Contribution disclosure

AI assistance was used for implementation and test drafting. The human
submitter must review every changed line, rerun the relevant tests, understand
the resource and backend behavior, and defend the change end to end. Duplicate
work checks are required before publication; no PR is opened by this change.
