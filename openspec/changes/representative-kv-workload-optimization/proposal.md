# Representative KV Workloads and Core Optimization Proposal

## Why

Current performance evidence primarily comes from GSM8K rollouts. The net benefit
of application information, KV movement, and scheduler cooperation remains
unvalidated for real coding agents, long-document QA, and mixed long/short
requests. Small, fixed local samples must create reproducible cache pressure so
phase timings and operation results can identify the mechanisms limiting gains
and guide core changes whose benefits can be verified.

## What Changes

- Add three workloads: SWE-bench Verified + mini-swe-agent, long-doc-qa, and coding
  agents mixed with short chat. Fix tasks, inputs, seeds, arrivals, and resource
  budgets; create pressure through small samples and concurrency without running
  full datasets.
- Reuse existing control clients, server/Store management, replay, and analysis
  entry points. Reuse task lists and container adaptation experience from the
  adjacent TokenMoE project, disable unrelated MoE routing capture, and collect
  and tokenize new traces with the current model.
- Provide real model/tool execution and replay with fixed work. Live execution
  preserves task outcomes and failures. Replay fixes complete request inputs,
  output token counts, and tool intervals; report its performance separately
  without assigning new task-resolution scores.
- Compare Native APC, an identical connector enabled without hints, explicit
  application directives without local policies, quota-only, and full
  cooperation. Limit component ablations, oracle, prediction-error, and control
  latency experiments to diagnostic points.
- Calibrate cache-fit and pressure points using the native baseline, then freeze
  the measurement matrix. Use separate tuning and held-out tasks for analysis
  and validation, retaining tasks, runs, and configurations that do not improve.
- Translate analysis into verifiable engineering changes: measure transfer and
  recomputation costs, complete cache-group preparation, preparation-to-admission
  intervals, actual reuse, wasted movement, and interference between long and
  short requests. Use this evidence to change application control, optional
  policies, or their engine cooperation.
- Target full-cooperation throughput gains of at least 25% over Native APC, 10%
  over quota-only, and 5% over the frozen pre-change full-cooperation version at
  one or more held-out pressure points. Check short-chat tails and cache-fit
  overhead as well. Track performance targets, measurement validity, and
  functional checks separately; optimization acceptance remains incomplete when
  performance targets are unmet.
- Provide reproducible commands, machine-readable results, and analysis reports
  that retain negative results. Link evidence to the existing
  `programmable-kv-cache` change within the same PR.

## Capabilities

### New Capabilities

- `kv-representative-workloads`: Three bounded workloads, live execution and
  fixed-work replay, small-sample pressure calibration, comparable configurations,
  and performance/correctness evidence by task and workload class.
- `kv-experiment-driven-optimization`: Core changes based on experimental
  attribution, verification of application-information benefits, wasted-movement
  and fairness diagnostics, held-out validation, and explicit performance gates.

### Modified Capabilities

None. No canonical capability specifications have been archived under
`openspec/specs/`. This change depends on the ongoing `programmable-kv-cache`
change, preserves its control/state/execution contracts, and does not replace its
incomplete acceptance requirements.

## Impact

- Primarily affects `benchmarks/programmable_kv_cache/`, experiment documentation,
  and existing nearby tests. Extend shared entry points to avoid duplicating
  server management and result protocols for each workload.
- Evidence may justify changes to `vllm/v1/kv_hints/`, scheduler integration, or KV
  manager cooperation. The engine continues to approve resources, the scheduler
  owns queues, and existing connectors execute movement.
- Use local Qwen3.8-27B BF16, TP=2, GPUs 0/3, and
  `gpu_memory_utilization=0.5`. Record actual KV capacity, cache groups, CPU/Store
  resources, and source revisions. Primary performance runs do not use small
  fixed KV-byte overrides.
- Optional runtime dependencies include mini-swe-agent and an existing container
  runtime. Use the local Verified dataset and fixed small local inputs for
  documents and short chat. Manage dependencies through uv/venv.
- Align with [PR #62382](https://github.com/vllm-project/vllm/pull/62382), the
  [updated RFC #57103 boundary](https://github.com/vllm-project/vllm/issues/57103#issuecomment-5814964446),
  and [Agentic API #367](https://github.com/vllm-project/agentic-api/issues/367).
  Reuse `KvHintsEnvelope`, `SET_PRIORITY`/`COPY_TO`/`MOVE_TO`, and correlated
  completion results. Preserve the mechanism-only default, optional local
  policies, absence of public Pin, and opaque KVCR forwarding.
- Deliver experiments and evidence-supported optimization. New public protocols,
  context editing, scheduler replacement, changes to the Mooncake project, real
  Dynamo/llm-d integration, and a complete RL trainer are outside this scope. Do
  not publish a separate PR for this change.
