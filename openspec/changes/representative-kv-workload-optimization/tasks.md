# Representative KV Workloads and Core Optimization Tasks

## 1. Freeze the protocol and inspect resources

- [ ] 1.1 Check the current PR/RFC boundaries, incomplete acceptance in the original change, and ownership of working-tree edits. Record the implementation baseline and C0 source/configuration snapshot without overwriting other active work.
- [ ] 1.2 Fix the 4 tuning and 8 held-out SWE tasks, reserve ordering, natural documents, and short-chat inputs. Verify local assets, task IDs, and versions; save hashes and generate bounded plans for all three workloads with dry-run support.
- [ ] 1.3 Verify GPUs 0/3, the current model, cache groups, 16k or necessary 32k context, CPU/container limits, and Mooncake capabilities. Record actual resources and prepare only missing, explicitly selected dependencies/data without full downloads.
- [ ] 1.4 Fix N/B/D/Q/C serving parameters, cold-tier budgets, sampling, task limits, initial cache state, and timing boundaries. Define three separate acceptance states and their report fields.

## 2. Add thin adapters for the three workloads

- [ ] 2.1 Add workload selection and a shared manifest to the existing matrix/client entry points. Reuse server/Store management, process ownership, control results, and cleanup while preserving verl commands and historical trace compatibility.
- [ ] 2.2 Integrate a pinned mini-swe-agent version using existing container experience. Execute real model/tool calls and retain messages, commands/exit codes, timings, patches, and failures; disable unrelated MoE routing capture.
- [ ] 2.3 Implement a small long-doc-qa workload with source/hash provenance, shared document prefixes, multiple questions, interleaved ordering, and verifiable questions. Record actual token lengths and reuse distances.
- [ ] 2.4 Implement mixed coding/chat execution with fixed open-loop arrivals and statistics by class. Measure latency from scheduled readiness, include client waiting, and drain every planned request.
- [ ] 2.5 Use minimal smoke runs for all three workloads to verify request/session association, primitive outcomes, error handling, and cleanup. Extend existing HTTP/control tests around module I/O and observed failures without duplicating the test framework.

## 3. Preserve equal work and valid comparisons

- [ ] 3.1 Convert real traces from the current model into fixed complete inputs, output token counts, dependencies, arrivals, and tool intervals. Retain rejection/failure provenance; do not reuse token IDs across models or feed new generation into recorded history.
- [ ] 3.2 Extend workload fingerprints and result audits to detect omitted requests, changed output lengths, context truncation, missing queue tails, and incomplete draining. Use existing lightweight fixtures to verify that these failures invalidate comparisons.
- [ ] 3.3 Implement D, the explicit-directive baseline without local policies, including bounded deferral retries. Share available application information and the predictor with C; normal prediction must not read recorded future outcomes.
- [ ] 3.4 Implement interleaved run ordering, independent restarts, post-warmup resets/cold-tier isolation, source identities, and GPU/CPU activity audits. Separate profiling and functional/regression windows from ordinary timing.

## 4. Calibrate baselines and diagnose mechanisms

- [ ] 4.1 Use only Native and the preregistered finite candidates to calibrate pressure and cache-fit points for all three workloads. Record actual capacity, references, working sets, cache turnover, and queueing evidence. Freeze scale and the short-chat SLO; keep acceptance incomplete when pressure is not established.
- [ ] 4.2 Run one N/D/Q/C comparison and one cache-fit N/B comparison on tuning samples. Break down tools, waiting after readiness, control, prefill/decode, transfer, and cleanup; identify the main limitation before expanding the final repeated matrix.
- [ ] 4.3 Correlate saving, publication, admission, and actual hits by session/content/cache group/operation. Quantify unused preparation, coverage gaps, recomputation, movement costs, and waiting by class; add bounded, disableable events where needed and retain unknown causes.
- [ ] 4.4 At one diagnostic point, run component policies, removal of prediction/admission cooperation, simple prefetch-at-pause, oracle, 0.5×/2× predictions, wrong-target/cancellation cases, and 0/25/100 ms control delays. Preserve positive and negative results.

## 5. Implement core changes supported by evidence

- [ ] 5.1 Select a main limitation from the diagnostics and record evidence, a falsifiable hypothesis, minimal failing behavior, and expected effects. Check existing V4/V5 results to avoid reimplementing existing mechanisms or adding unsupported switches.
- [ ] 5.2 Implement the selected core change within existing application control, optional policies, or engine cooperation. Address measured cost, complete-group coverage, admission timing, wasted movement, or long/short interference while preserving shared approval and execution contracts.
- [ ] 5.3 Add minimal behavioral regressions to nearby control/prefix/scheduler/connector suites. Run affected complete files and necessary real GPU/serving checks for references, group publication, shared capacity, cancellation/failure fallback, and output reuse.
- [ ] 5.4 Compare causal metrics and net elapsed time before and after the change on tuning samples; withdraw candidates without confirmed benefits. Continue from new evidence if the limitation remains. Freeze source, parameters, and held-out inputs once a candidate is selected, before performance validation.

## 6. Validate on held-out samples

- [ ] 6.1 Run at least three interleaved ordinary-timing repetitions of N/D/Q/C at each held-out pressure point. Rerun affected N/Q baselines when final source changes touch their paths; preserve valid and invalid runs.
- [ ] 6.2 Run at least three repetitions of N/B/Q/C at each cache-fit point. Check initial cache state, equal work, and low-contention costs; retain one existing GSM8K cache-fit/pressure regression pair.
- [ ] 6.3 Run C0 at least three times at the target held-out pressure point, using the same inputs/configuration as C to verify the net benefit of the new core change. Replace preregistered validation samples if the held-out set becomes tuning data.
- [ ] 6.4 Complete at least one live-tool small-sample run for each of N/D/Q/C, retain full task outcomes and patches, and score the same subset. Complete verifiable long-doc QA and greedy recompute/reuse checks; report live and replay separately.
- [ ] 6.5 Audit work performed, completions by class, short-chat P95/SLO, maximum waiting, actual hits, transfers, and resource draining for all three workloads. Report measurement validity, contract checks, and performance targets separately.
- [ ] 6.6 Verify median C throughput gains of at least +25%/+10%/+5% over N/Q/C0 at one or more held-out pressure points, with gains in the same direction in at least two repetitions. Also require cache-fit elapsed-time regression ≤5% and mixed short-chat P95 ≤1.10× Native. Keep this task incomplete if unmet and return to group 5 with a specific limitation.

## 7. Deliver reproducible evidence and checks

- [ ] 7.1 Update the shared experiment README and this round's report with commands, protocol/source/input hashes, raw result paths, mechanism analysis, measured scope, and negative conclusions. File presence or one fast run does not satisfy acceptance.
- [ ] 7.2 Link the original programmable-kv-cache coverage and validation records while preserving incomplete items. Explain alignment with the PR/RFC mechanism/policy boundaries and the scope of new evidence; do not create a separate PR.
- [ ] 7.3 Run final affected tests, necessary GPU/model checks, pre-commit, Python 3.12 mypy, and strict OpenSpec validation for this change. Record actual results and unexecuted coverage; after checks pass, commit only clearly owned changes locally with the required attribution.
