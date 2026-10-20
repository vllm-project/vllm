# Representative KV Experiment Design

## Context

This change extends the ongoing `programmable-kv-cache` change within
[PR #62382](https://github.com/vllm-project/vllm/pull/62382). Its interface follows
the [updated RFC #57103 discussion](https://github.com/vllm-project/vllm/issues/57103#issuecomment-5814964446)
and this branch's `vllm/v1/kv_hints/README.md`: the default executes explicit
commands, local policies are optional, and the engine owns resource approval and
physical operations. There is no public Pin. Earlier Pin descriptions in the RFC
body are outside the implementation target.

The local report dated 2026-10-19 already records V4 median throughput gains over
Native/quota in GSM8K pressure replay, and zero movement when the workload fits.
Short-pause cost checks, preparation on admission, shared-capacity revalidation,
and complete-group preparation protection already exist. Start from the actual
source frozen at implementation time, validate new workloads, and identify
remaining limitations without reimplementing those mechanisms. The V5 candidate
that preserved GPU backup sources was withdrawn; historical single runs do not
establish which alternative is faster.

The adjacent TokenMoE collection contains 20 tasks with real tools, whose median
duration is about 0.34 seconds and P95 about 1.65–1.75 seconds. This helps establish
workload shape. Its model, 32k context, MoE routing capture, and measurement
conditions differ. Reuse task selection and container adaptation, and collect
fresh raw messages and token inputs with the current model.

## Goals / Non-Goals

**Goals:**

- Establish measurable cache contention with three small local workloads, while
  retaining cache-fit controls at the same deployment budget.
- Quantify scheduling, explicit application control, and full cooperation
  separately, explaining actual reuse and long/short request waiting.
- Use observed limitations to guide core changes and verify net benefits with
  held-out samples and ordinary timing.
- Keep model work, completion semantics, and resources auditable across all
  comparisons; retain neutral, failed, and regressing results.

**Non-Goals:**

- Full SWE-bench runs, comprehensive model rankings, multi-machine deployment, or
  RL training throughput.
- New hint protocols, public Pin, context editing, modified model attention, or
  treating arbitrary document fragments as reusable prefixes.
- Porting the old TokenCake scheduler, changing the Mooncake project, bypassing
  engine allocation, or letting policies select preemption victims directly.

## Decisions

### 1. Reuse shared measurement entry points with thin workload adapters

Extend existing server management, `RolloutClient`, replay, and analysis under
`benchmarks/programmable_kv_cache/`, separating workload selection from resource
management. Add a shared workload module and mini-swe-agent adapter only where
needed. Reuse server startup, Store management, result queries, ACK, and cleanup
across workloads. Preserve the existing verl CLI and historical trace readers.

Pin mini-swe-agent to 2.4.6 or an explicitly verified commit and manage dependencies
with uv. Reuse the adjacent project's task lists, model/tool interaction patterns,
and container probes without copying its routing recorder, data platform, or
complete fork. Give each task an independent container workspace and save actual
commands, exit codes, tool outputs, and patches. Disable unrelated capture to
reduce CPU and I/O interference.

A shared HTTP workload entry point keeps control and timing consistent. Adding
another benchmark framework would increase behavioral differences and maintenance;
limit implementation to the smallest useful shared extraction.

### 2. Fix small samples and execution bounds

The following settings define the initial protocol. Write effective values to the
manifest after baseline calibration; changes in scale require a new protocol
version.

| Workload | Smoke / tuning | Held-out validation | Pressure controls and limits |
| --- | --- | --- | --- |
| SWE coding agent | 3 smoke tasks; 4 tuning tasks | 8 fixed tasks with real bash, file, and test operations | Session concurrency 4 or 8; add tasks in preregistered order if needed, up to 20 distinct tasks and concurrency 16 |
| long-doc-qa | 4 documents, 4 questions each | 16 documents from different sources, 4 questions each | About 4k/8k/12k tokens per document; up to 32 documents; concurrency 8, 16, or 32 |
| coding + short chat | Tuning coding tasks and 32 chats | Held-out coding tasks and 128 fixed chats | Coding concurrency 4 or 8; open-loop chat arrivals at 0.25, 0.5, or 1 req/s; freeze one primary pressure point after calibration |

SWE defaults are 20 model calls per task, a 900-second session budget, a 60-second
tool timeout, and at most 1,024 output tokens per call. Check the session budget
at turn boundaries and separately record actual wall time. Apply identical limits
to all configurations. Baseline checks may justify one common correction to an
unsuitable output budget before freezing it; do not change it by policy afterward.
Budget exhaustion, format errors, container errors, and normal submissions all
remain in the task denominator.

Start from the adjacent project's `collection/configs/runs/local-qwen3-full.yaml`.
Tuning tasks are `astropy__astropy-12907`, `django__django-10097`,
`matplotlib__matplotlib-13989`, and `psf__requests-1142`. Held-out tasks are
`mwaskom__seaborn-3069`, `pallets__flask-5014`, `pydata__xarray-2905`,
`pylint-dev__pylint-4551`, `pytest-dev__pytest-10051`,
`scikit-learn__scikit-learn-10297`, `sphinx-doc__sphinx-10323`, and
`sympy__sympy-11618`. Verify local dataset versions and task availability before
calibration. Assign roles to additional tasks before inspecting cooperation
results.

Use local natural text with provenance and hashes for long-doc-qa. Fix questions
as document suffixes and preserve identical document-prefix token IDs across
questions. Support random interleaving and round-by-round interleaving, with the
primary order frozen by seed. Consecutive questions for one document cannot be
the only pressure result. At least one question per document has a verifiable
answer. Synthetic documents and artificial waits belong to separate diagnostic
protocols and do not replace the primary natural-text results.

Select fixed small ShareGPT or existing multi-turn samples for short chat, with
128–512 input tokens and 64–128 output tokens. Fix selection rules, IDs, and context
budgets. Keep the coding subset, sampling, and limits identical between coding
alone and the mixed workload. Create mixed pressure through independent sessions
and arrivals without adding unrelated long shared prompts.

### 3. Calibrate pressure on Native, then freeze the protocol

Use local Qwen3.8-27B BF16, TP=2, GPUs 0/3, APC, PCP=DCP=1, and a batch token budget
of 4096. Set `gpu_memory_utilization=0.5` without a fixed KV-byte override. Start
with a 16,384-token context and check input plus output budget for every request;
do not truncate silently. If SWE needs 32k, first verify startup and capabilities,
then freeze that context and the resulting actual KV capacity for every arm.

Record block size, capacity, and hybrid-state layout for every cache group,
together with active/transfer references, reclaimable cache, and waiting demand.
Total prompt tokens or a single 100% occupancy sample cannot independently prove
pressure. A pressure point needs an independent working set larger than effective
reuse capacity, together with cache turnover causing recomputation/loads, or
actual capacity deferral, preemption, or sustained queueing. Report the affected
time fraction and observed concurrency.

Increase working sets and concurrency through the finite baseline candidates in
the table and select the smallest configuration that establishes pressure. Freeze
a low-contention point as well. Selection reads only Native resource and pressure
evidence, without ranking optimized configurations by speed. If the sample cap
still does not establish pressure, report the limiting resource and revise the
protocol; pressure acceptance remains incomplete.

Run GPU performance measurements serially with the existing process-ownership
audit. Schedule CPU regressions, model functional checks, and profiling in separate
windows too. Check GPU occupancy before launching; wait on resource conflicts
without terminating other tasks' processes. Record CPU/cgroup settings, container
limits, Store capacity, transport protocol, and other load. Disclose extra
cold-tier capacity used by movement arms at the same GPU budget.

### 4. Separate live execution from replay with fixed work

Execute the branch's model and real tools to collect small-sample task outcomes.
Convert traces for replay only after validating complete raw messages, templates,
tokenizers, tool results, and engine inputs. Do not reuse token IDs across models.

Replay records complete `prompt_ids`, output token counts, session/class,
dependencies, initial arrivals, and tool-batch durations for every request. Each
replay executes the real model. Later requests use recorded complete inputs,
without inserting newly generated content into fixed history or rerunning tools.
Replay does not produce new task-resolution scores. Model, template, inputs,
output budgets, dependencies, pauses, and arrivals jointly form the workload
fingerprint.

Preserve each session's dependency: previous turn completes → recorded tool
interval → next turn becomes ready. Independent chats become ready according to
the same pregenerated open-loop arrival table. Measure every planned request from
readiness, including waiting for client capacity. Continue timing until every
request, control terminal result, and device transfer has finished so queue tails
cannot disappear. Record offered, ready, sent, admitted, first-token, finished,
and cleanup stages; unknown stages are null with a reason.

Fixed replay preserves recorded normal-completion, budget-exhaustion, and failure
labels. Primary throughput is the same frozen work-unit count divided by time
through complete draining; also report output tokens/s. Do not improve scores by
reducing outputs or tool rounds. Live runs separately report submissions,
failures, exhausted budgets, patches, and small-sample scores; `submitted` does
not establish SWE-bench resolution.

### 5. Use a minimal primary matrix and targeted ablations

| Arm | Configuration and purpose |
| --- | --- |
| N: Native APC | Same source, control and cold-tier connector disabled; deployment baseline |
| B: idle/noop | Control and the same connector enabled, local policies off, no hints; isolate enablement overhead |
| D: explicit | Application issues `SET_PRIORITY`/`COPY_TO`/`MOVE_TO`, local policies off; engine may reject/defer, with bounded application retries |
| Q: quota-only | Same quota configuration, retention/movement and unnecessary signals off; quantify scheduling contribution |
| C: cooperative | Retention + movement + quota using application lifecycle and predictions through shared approval/execution |
| C0: frozen cooperative | Existing cooperative source/configuration frozen before this round's core changes; verify their incremental net benefit |

At each of the three primary pressure points, run N/D/Q/C with at least 3
independent server restarts per arm. Interleave arm order within repetitions and
record the ordering seed. Run N/B/Q/C at cache-fit points with at least 3
repetitions each. Start C from the existing normal cost checks, preparation on
admission, and pressure threshold 0.8. Keep quota waiting limits and aging
parameters identical for Q/C and record them, so quota parameter changes do not
masquerade as application-information benefits.

Preserve a C0 source snapshot before implementation. After selecting a core
candidate, run C0 at least three times at the target held-out pressure point with
the same inputs and parameters as C. Rerun affected N/Q baselines on final source
if changes touch their shared execution paths. Reuse historical baselines only
with evidence of equivalent paths and resource conditions.

Exclude compilation and graph-capture warmup from timing, then reset G1 and
isolate cold-tier namespaces/cache salts. Verify consistent initial cache state.
Time the first computation of actual workload inputs, association/declaration
setup, control preparation, and final task cleanup. If a separate prepopulated
cold-cache protocol is used, all relevant arms start with identical content and
report population cost separately. Replay must generate the agreed token counts;
early termination or omitted requests invalidate equal-work comparisons.

D/C share the same tool-duration predictor and available lifecycle facts. D
queries and issues directives at lifecycle boundaries, retrying `deferred` with
frozen bounded backoff. Include queries, retries, waits, and cleanup in elapsed
time. D/C use identical cold-tier budgets and transport.

At one tuning pressure point with a measurable movement opportunity, add
retention-only, movement-only, removal of application prediction, removal of
admission cooperation, and a simple prefetch-at-pause policy. Run targeted
oracle, 0.5×/2× duration predictions, wrong-target/cancellation, and 0/25/100 ms
control-delay diagnostics separately. Label oracle and exclude it from primary
acceptance. Normal policies cannot read actual future pauses, output lengths, or
request completion times. Recorded output lengths control replay only; they are
not predictor hints.

This staged matrix covers mechanism and policy contributions without taking the
Cartesian product of every parameter across all workloads. Retain one existing
GSM8K cache-fit point and one pressure regression point without rerunning the
entire historical matrix.

### 6. Select core changes from actual costs and failed reuse

For each round, establish which portions of end-to-end time are dominated by
tools, waiting after readiness, control, prefill, decode, transfer, and cleanup.
Show overlapping intervals separately. Summed thread or RPC durations are not
wall time.

Correlate prediction → save → load → publication of every group → admission →
hit/recompute → release using session/continuation, content range, cache group,
engine epoch, and operation/transfer ID. Add bounded diagnostic events that can
be disabled where existing observations are insufficient; do not scan every
session on each scheduler step.

| Mechanism limitation to test | Required evidence | Change direction if supported |
| --- | --- | --- |
| Movement costs exceed avoided recomputation | Save/load/recompute timings and overlap by actual bytes, prefix length, and group; separate tool time from readiness | Estimate recompute and non-overlapped transfer costs from completed samples, then choose retention, saving, loading, or recomputation within known application windows and current resources |
| Prepared content lacks a reusable complete-group boundary | Checkpoints/continuous ranges per group, partial success, repeated loads, actual prefix lookup | Select preparation ranges at common valid boundaries, avoiding transfers that cannot form a real hit while preserving all-rank/all-group publication and reference protection |
| Preparation is too early or holds scarce capacity | Preparation-to-admission delay, intervening eviction, transfer holds, and reservation occupancy | Combine resume windows, admission candidates, waiting demand, and in-flight budgets; stop preparations with no consumer or stale intent |
| Reuse scoring delays short chats | Ready/queue tails by class, task waiting, cold-prefix handling, aging fallback | Adjust bounded waiting and cost/reuse advice through the existing scheduler extension while retaining its queue and preemption ownership |
| Control/index costs consume the gains | Low-contention CPU/RPC/index growth and phase timings | Coalesce duplicate control, remove unnecessary queries, and use incremental state while retaining observable results and cancellation semantics |

Bytes saved do not establish additional allocatable capacity: unreferenced cache
blocks may already be on the free list. Distinguish active references,
reclaimable mappings, and transfer holds in the cost model. If an ordinary cache
hit cannot be attributed to a specific load, label its attribution unknown rather
than counting every existing GPU hit as a movement benefit.

For each core change, retain bottleneck evidence, a falsifiable hypothesis, a
minimal failing behavioral case, the change, expected effects, and comparable
before/after results. Change one main mechanism per round and select candidates
using tuning samples. Freeze the candidate, then validate on held-out tasks and
documents across all three workloads. If held-out results drive further tuning,
reclassify that set as tuning data and allocate a new held-out set from the
preregistered small reserve. Parameter search or increased pressure alone does
not complete the core-optimization deliverable.

### 7. Metrics and acceptance

Save per-task/request data and summaries for every run: complete-workload elapsed
time, fixed work units/s, output tokens/s, TTFT/TPOT/end-to-end P50/P95/P99,
application waiting and server queueing, recomputed/cached tokens, preemptions,
saved/loaded bytes, duplicate or unconsumed movement, control RPC/CPU/index costs,
failures, and cleanup results. Report sample counts and maxima with small-sample
P99; do not infer a population tail distribution from a few sessions.

Separate coding and chat metrics in mixed runs: offered/completed/failed counts,
requests/s meeting SLO, task waiting distributions, and maximum waiting. Freeze
SLOs during tuning and drain queues after arrivals stop. Complete at least one
small live-tool run per primary arm, retain commands and patches, and score
patches on the same fixed SWE subset. Different generated trajectories do not
support equal-work speedup claims. Verify long-doc answers and greedy
recompute/reuse behavior on identical inputs; performance cannot compensate for
new model errors.

Track three independent acceptance states:

1. **Valid experiment:** All three workloads establish pressure and cache-fit
   controls with frozen inputs/conditions, at least three repetitions, resource
   isolation, and complete results. Missing, timed-out, and invalid runs retain
   reasons and cannot be silently removed.
2. **Contract checks pass:** No new incorrect reuse, early publication, duplicate
   execution, unbounded waiting, or resource leaks. Failures/cancellations have
   explicit outcomes, and functional/regression checks pass.
3. **Performance targets met:** At one or more held-out pressure points, median
   fixed-work throughput for C improves at least 25% over N, 10% over Q, and 5%
   over C0, with gains in the same direction in at least two repetitions.
   Cache-fit median elapsed-time regression for C against N stays within 5%.
   Mixed short-chat P95 readiness-to-completion latency is at most 1.10 times N
   in that same workload, with equal completion counts and no new errors outside
   the task outcomes themselves. Report every scenario separately.

The 25% target carries forward the original change's performance objective.
The 10%, 5%, and 1.10 thresholds are local engineering acceptance criteria for
this proposal, not general performance guarantees. C0 verifies that the new core
change adds benefit; merely reproducing existing gains does not complete the
optimization task. If targets remain unmet, leave performance acceptance
incomplete and report the limiting evidence, rejected approaches, and concrete
next change. Selected speed figures cannot substitute for passing acceptance.

### 8. PR/RFC boundaries and regression coverage

| Design constraint | Experiment/change requirement |
| --- | --- |
| Mechanism-only default, optional local policies | D/B have no local policies; C/Q use explicit startup configuration; unconsumed signals have no effects |
| Stable explicit-command semantics | D/C use the same primitives; policies propose and the engine may defer/reject without rewriting caller commands |
| Shared resource approval and execution | Reuse KVCacheManager/BlockPool and CPU/Mooncake adapters; charge shared physical transfers once |
| Correlated completion and complete-group availability | Results report actual scope; verify real hits after completion, permit recomputation after failure, and preserve failures through ACK |
| Logical state independent of physical copies | Task completion preserves externally shared content; retention declarations and active references remain independent |
| Scheduler owns queues and preemption | Policies return bounded guidance without replacing the scheduler or selecting victims directly |
| Layout/weight compatibility | Compare within one weight version, report capability/alignment limits explicitly, and preserve opaque KVCR forwarding |

Reuse nearby `test_kv_hint_control.py`, `test_kv_hint_state.py`, engine/HTTP control,
prefix/scheduler, and connector suites. Define module I/O and failure behavior
before adding the cheapest effective check. Driver adapters use existing HTTP
control fixtures; replay fingerprints and missing-request checks need no GPU.
Actual bytes, complete-group publication, and model reuse use existing GPU and
real-server entry points. Specification scenarios define acceptance contracts
without requiring a new test file for each one.

This change does not replace incomplete cross-engine, recovery, or weight-version
acceptance in the original change. Rerun corresponding existing checks when
modifying those paths. Use `.venv/bin/python` for every Python command and
`uv pip` for dependencies. After implementation, run affected complete suites,
relevant GPU checks, pre-commit, Python 3.12 mypy, and strict OpenSpec validation.

## Risks / Trade-offs

- Small samples may not establish stable pressure → Select points using Native
  working-set/reference/turnover evidence, retain local scale caps, and report
  unmet pressure requirements.
- Coding trajectories/tool times vary with sampling and CPU contention → Use
  fixed replay for attribution, report live results separately, and fix
  container/CPU limits across arms.
- A 16k context may change task difficulty → Report over-limit requests, revise
  all configurations together, and recollect without dropping history or silently
  truncating inputs.
- Profiling/regressions can distort ordinary timing → Use separate serial
  windows, process audits, and source fingerprints; retain and replace invalid
  runs.
- Resource differences may be mistaken for policy gains → Disclose cold-tier
  resources and separate contributions through resource-matched D/C and
  quota-matched Q/C comparisons.
- Aggregate throughput may improve at the expense of task outcomes or chat tails
  → Enforce equal work, open-loop arrivals, full draining, and latency gates by
  class together.
- Another active change may edit shared files → Record working-tree ownership
  and frozen versions before implementation; include only clearly owned edits,
  and invalidate a measurement if source changes during it.

## Migration Plan

This proposal creates OpenSpec documents only. Implementation proceeds through
resource/workload preflight, thin adapters and smoke checks, baseline calibration,
diagnosis, core changes, frozen validation, reporting, and local commits. New
workload selection remains optional and existing verl commands continue working.
Withdraw final code candidates without confirmed benefits while retaining failed
experiments and raw data. Select runtime configuration at startup without adding
policy hot swapping.

## Open Questions

No additional user input is required to start implementation. Preflight and Native
calibration must determine the context needed for SWE trajectories, minimum
pressure scale, short-chat SLO, and dominant mechanism limitation. Freeze these
choices in the protocol without selecting them by optimized performance.
