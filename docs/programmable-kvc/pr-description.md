# Programmable KV cache PR description (draft)

## Purpose

Implement [RFC #57103](https://github.com/vllm-project/vllm/issues/57103) so an application can manage cache across model calls, tool pauses and branches through the existing `KvHintsEnvelope`. vLLM resolves the requested logical content, approves GPU resources, invokes CPU/Mooncake operations and reports actual completion.

Request-attached hints and independent Python/HTTP control use one native dispatcher. The public primitives are `SET_PRIORITY`, `COPY_TO` and `MOVE_TO`; session/continuation/range selectors, correlated results, bounded replay guards, ACK, cancellation and engine epochs support recovery. Logical associations remain separate from physical copies. Backend-owned KVCR envelopes retain their opaque interpretation path.

COPY publishes transferred content after all ranks complete. MOVE establishes the cold copy before reclaiming eligible source blocks. Shared operations share reservations and retain buffers until workers finish, including cancellation/reset races. CPU offload and Mooncake reuse their existing connector and allocation paths.

Optional retention, movement and quota policies submit proposals through that resource boundary. Quota rechecks current allocation demand after the scheduler's actual cache lookup and reserves running growth. Application pause estimates, observed movement cost, occupancy and sustained waiting guide movement; optional preparation waits for session admission and protects complete prefix groups. Explicit commands retain their requested semantics.

This updates PR #62382. Duplicate checks against RFC #57103 and the programmable-cache area found this PR and [#58198](https://github.com/vllm-project/vllm/pull/58198). That PR supplies a request-free CPU prefetch primitive; [#38514](https://github.com/vllm-project/vllm/pull/38514) supplies full-attention retention. This contribution covers the common native control/logical-state/result contract, hybrid multi-rank execution, CPU/Mooncake adaptation, application recovery and policy composition. Those narrower execution and retention overlaps remain explicit review/reuse boundaries.

## Test Plan

Reproduction commands and capability limits are in [the benchmark guide](../../benchmarks/programmable_kv_cache/README.md). All Python commands use the uv-managed `.venv`.

- Run the related control/state/engine/HTTP, scheduler/prefix-cache, Mooncake adapter, offloading and tiering suites using the complete command in [the testing report](rollout-report-2026-10-18.md).
- Run `tests/v1/kv_offload/cpu/{test_gpu_worker,test_canonical_layout,test_swap_blocks_batch,test_swap_blocks_triton}.py` and `tests/v1/kv_connector/mooncake_integration/test_store_buffers.py` on GPUs 0/3.
- Run `run_recovery` with `--evaluation-samples 128 --lifecycle --weight-update --cpu-evaluation-samples 128`, and separately with `--capacity-deferral-only`. This checks real cross-engine reuse, branches/TTLs, duplicate/racing/cancelled control, Store outage/restart, real IPC parameter update/rollback, capacity retry/fallback and paired model quality.
- Run deployment-budget native/quota/cooperative comparisons, cache-fit native/idle/noop, direct-command and local-policy ablations, separately labeled recorded-pause oracle and incorrect forecasts. Repeat actual verl tool loops separately from equal-work replay; audit GPU ownership in both.
- Run CPU metadata measurements and a separately instrumented CPU/GPU profile. Preserve regressions and failed validation attempts.
- Run default pre-commit hooks, `pre-commit run mypy-3.12 --all-files --hook-stage manual`, and `openspec validate programmable-kv-cache --strict`.

## Test Result

The following primary comparisons are complete. The remaining acceptance results below are still being collected; this description is not ready for publication until that evidence is incorporated.

Qwen3.8-27B BF16, TP=2, GPUs 0/3 (A100 80GB PCIe), `gpu_memory_utilization=0.5`, actual KV 12.12 GiB/card, concurrency 64, batch token budget 4096 and max context 16384. Compilation/CUDA graphs are enabled; no fixed KV-byte override or profiler is used for ordinary timing.

The pressure replay preserves 256 trajectories, 920 model calls, 60,136 output tokens and 8,781,870 prompt tokens. Three independently restarted runs per main configuration give:

Replay fixes each incoming token sequence and output length while regenerating outputs. Those regenerated tails are not fed into subsequent recorded inputs, so tail reuse can differ from a continuous live trajectory. Actual verl loops are evaluated separately and are not used for equal-work speedup percentages. The separately labeled oracle uses recorded tool-pause durations; it does not know future GPU admission times.

| Configuration | Median elapsed | Range | Throughput vs native |
| --- | ---: | ---: | ---: |
| Native APC | 1,997.26 s | 1,983.42–2,003.82 s | — |
| Quota | 1,386.21 s | 1,385.85–1,388.76 s | +44.1% |
| Cooperative | 1,188.53 s | 1,182.21–1,261.62 s | +68.0% |

Cooperative throughput is 16.6% above quota. Median TTFT P99 is 125.02/71.59/70.51 s; rollout P95 is 758.91/534.92/469.94 s. All 32 families have lower P95 medians with quota and cooperative settings, with eight trajectories per family per run. These small per-family samples describe the measured workload.

Cache-fit replay has four families and 256 trajectories. Two runs per configuration give native/quota/cooperative medians 170.87/168.97/169.61 s, identical cache hits, no preemption and no cooperative movement. These small differences do not establish stable gains or universal zero overhead. Pressure client CPU medians are 71.81/54.73/127.31 s; control cost remains measurable.

An additional cache-fit native/idle/noop comparison ran two full workloads per arm: medians 168.54/168.49/168.39 s, identical cache hits, no preemption or transfer, and six passing GPU ownership audits. The differences below 0.1% do not establish universal zero overhead.

Pressure forecast diagnostics preserve the same inputs, output lengths and actual pauses. Recorded-pause oracle, fixed 1 ms and deliberately inaccurate fixed 30 s predictions each ran twice, with medians 1,230.42/1,181.00/1,077.04 s. All six passed the workload/GPU checks with zero backend failure, missing or discarded transfers. The 30 s variant increases cache reuse in this pressure workload; its cache-fit behavior remains pending and it is not the selected default. This result does not establish that less accurate tool predictions are preferable or that tool-pause oracle knows future GPU admission.

Related offline CPU regression: 1,999 passed. The updated HTTP/benchmark regression file: 40 passed, overlapping the broad suite. Actual GPU buffer/CPU/Mooncake checks: 56 passed, one ROCm-specific skip. These overlapping counts are not summed.

The selected runtime is unchanged from the previous tested normal-cost MOVE implementation. Failed and incomplete harness attempts, forced-pressure cache-fit regressions, and the unselected COPY-backup candidate remain in the evidence index. Instrumented runs are excluded from ordinary throughput comparisons.

Final-runtime recovery, real parameter update/rollback, and capacity partial/retry/fallback passed. Both CPU and Mooncake paired 128-question evaluations score 124/128 for recomputation and preparation, with zero answer mismatches and zero length cuts; cached tokens are 0/784 per question. Final result records and pending transfers are empty. Functional runs use eager TP=2 and utilization 0.45 to accommodate two engines; they are excluded from performance ratios.

One direct-priority replay completed its model/control work but failed a predeclared process-log coverage limit (11.072 s gap, allowed maximum 10 s); its timing is excluded. Process ownership sampling is now one second, with the same rejection boundary and regression coverage. Remaining runs add same-sampling references and repeat the rejected arm.

Pending incorporation: complete direct/component/live/profile and cache-fit forecast matrix, final checks and consolidated report. Detailed executed evidence is in [the acceptance report](acceptance-report-2026-10-20.md).

Acceptance is scoped to this local model/hardware/layout. Native Mooncake uses complete primary boundaries and PCP=DCP=1; CPU control uses one block per chunk. CPU Share, cold Retain/delete, SimpleCPU control, KVCR request-free actions and mixed-owner envelopes are not advertised. Real Dynamo/llm-d and a full trainer/optimizer run are outside this delivery.

AI assistance was used: OpenAI Codex assisted implementation, experiment orchestration, testing and analysis. Before publishing, the human submitter must review every changed line, understand and defend the implementation, and run the relevant commands. Those human review/testing requirements are not represented as completed agent work.
