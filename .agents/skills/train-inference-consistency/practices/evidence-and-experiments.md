# Evidence and experiment rules

Rules for what counts as evidence, how experiments are designed and recorded, and when old evidence stops holding. Mechanisms live in the guides.

## 1. Evidence labels

PASS / FAIL / INVALID (did not enter the target path) / VACUOUS (condition had zero hits) / SUPERSEDED / HISTORICAL. These label evidence, not lifecycle states. The tables in [validation.md](../workflow/validation.md) and the [report template](../workflow/report-template.md) have only `pass / fail / inconclusive / not run`. PASS→`pass`, FAIL→`fail`; INVALID and VACUOUS become `inconclusive` with the reason in the notes; SUPERSEDED and HISTORICAL stay out of the current table and live only in the [candidate evidence ledger](#6-candidate-evidence-ledger) and historical reports. One SUPERSEDED piece of evidence makes the gate it covered stale. A successful run, train-inference consistency and a held-out score gain are different conclusions and cannot substitute for each other.

## 2. Evidence that must not be confused

- Enabling `VLLM_BATCH_INVARIANT` is not cross-framework bitwise alignment; the same text is not the same logprob.
- Zero difference in probability space, zero KL or identical printed numbers do not replace comparing valid logprobs in the original dtype.
- Replaying fixed inputs is a localization tool; statistics from old trajectories are not an unbiased KL under the current policy.
- The same function name, generated code or environment variables do not prove the same actual kernel, launch config and shared libraries.
- Output replacement, forced routing, disabling Graph and bypassing bad packing are valid causal experiments if you state what they change.
- Similar quality scores are not identical per-problem results; same-problem quality, bitwise logprob and 8K/1K performance are accounted separately. When the acceptance script fails, check the producer receipt first; do not resample or declare a pass automatically.
- Verify quantization rules on the deployed bytes, not dequantized weights ([precision pitfall P2](../guides/precision-quant.md#p2-verifying-a-rule-with-dequantized-weights-gives-a-false-positive)). QAT/low-precision training only narrows the weight gap; 0 diff comes only from training calling the same kernel as inference; see [How much QAT can shrink the mismatch](../guides/precision-quant.md#4-how-much-mismatch-qat-can-remove).
- Do not manufacture a pass by changing tokens, weights, model geometry, precision or tolerance. If the comparison baseline really must change, start a new experiment and re-verify quality.
- Do not carry DS4's EP/MoE config over to dense Qwen; do not manufacture a performance gain by weakening the BI=0 config, locally disguising BI as off, or removing existing correctness patches.
- Source tests passing is not the final image passing. Development overlay results belong only to their experiment; the delivered image digest, installed source and build artifacts must match the acceptance object, and the evidence chain is rebuilt after repackaging.

## 3. Inferences no longer carried forward

Historical numbers and criteria stay with the run that produced them; a new task does not inherit a past run's model, dataset, topology, performance statistics method or pass status. Lessons from retracted or narrowed claims:

- A number in a summary is not evidence: find the raw numerator and denominator for the same policy/workload/config, or retract the assertion.
- Execution completion is not numerical acceptance: finished steps, a started job, an existing adapter file or finite nonzero gradients do not show the target class ran or the comparison passed. Record the instantiated class and its import source, and accept non-empty original-dtype token comparison, module VJP against an independent reference, actual update, shutdown and long-term stability separately.
- Evidence does not transfer across candidates: a source pin, the same Docker tag, an earlier proxy, an experiment stack or a direct operator call does not prove the current candidate, image or compiled module dispatch. Verify branch, dirty patch, wheel, `.so`, dependencies and digest separately, and re-establish entry-gate evidence for the current candidate.
- A fixed split count or a pass on one path does not cover other batches, prefill/decode, the first round after resume or cross-node communication; each needs its own evidence.
- When an acceptance wrapper fails, find the failing layer before resampling or rerunning; a control-plane parsing error does not discard sealed numerical evidence.

Each new case records only "contract → minimal counterexample → discriminating experiment → root cause/minimal fix → regression on the original reproducer → scope/source", with the small raw receipt for disputed numbers. Merge status, recommended dependency versions and best-performing config are volatile; look them up in the target repo.

## 4. Experiment contract

### Define the objects of equality first

State the checkpoint/adapter/policy version, tokenizer and chat template, raw token IDs, position IDs, attention/response mask, padding/packing, and the logits processing on both training and inference sides.

For response token `y[t]`, compare `log p(y[t] | prompt, y[:t])`. Make explicit the prediction position of the first response token, whether EOS counts, and the exclusion rules for truncation and padding. Do not align logits rows of different positions.

Distinguish raw logits, raw logprobs, and logprobs after temperature/repetition penalty/top-k/top-p, and record which one the interface returns. The same seed does not guarantee the same scheduling or RNG consumption: freeze the tokens instead of resampling.

### Record these goals separately

| Goal | Valid evidence | Cannot infer |
| --- | --- | --- |
| Single-side repeatability | A/A with the same input and state | Consistency across batches or processes |
| batch/layout invariance | Same semantic sample identical under different positions, neighbors and packing | Training and inference are the same |
| prefill/decode parity | Two execution paths with the same forced tokens | backward or the training update is correct |
| Cross-stack forward parity | Real training forward against a frozen rollout | Full-vocabulary logits match, if only sampled tokens are compared |
| Effective training loop | Non-degenerate gradients, parameter change, post-update parity | Convergence or long-term stability |

When the user requires bitwise, compare original-dtype bytes and record mismatching/valid token counts; do not reach a pass through a float cast, probability-space rounding or a tolerance loosened afterwards. If the goal is an error budget, fix the metric and threshold in advance and measure each side's own noise; generic numerical alignment need not be forced to bitwise.

A finite logprob is necessary; an empty mask, all-zero placeholders, missed capture, NaN or the wrong weight version cannot pass. A zero may be legitimate; tell it from a placeholder by hook count, score source and a negative control.

### Recipe and coverage record

When building a train-inference recipe, also fix these inputs, recovering missing items from the existing config first:

| Item | What to record |
| --- | --- |
| Workload | Training context length, rollout input/output length distribution, concurrency, packing, training token batch |
| Hardware | GPU architecture/count, GPU memory, interconnect and target TP/CP/EP/PP/DP, trainer and rollout separately |
| Numerics | master/deployed weight, activation, KV, accumulation and gradient dtype; quantization granularity, scale rule and update method |
| Scope | Raw sampled logprob or full-vocabulary logits, applicable hardware/software, Graph/cache path |
| Budget | Throughput, peak GPU memory, quality and gradient criteria, experiment count and stop condition; with an unknown threshold, report the measurement and do not declare a pass |

Record the measured baseline of the current config, then build a strong performance baseline per the main flow, without exhaustive tuning first. Rust Bench 8192/1024, ignore EOS, c=1–64 is a fixed comparison and does not replace the target context and natural-length workload; a user-specified workload governs, with the difference reported.

A proxy records "full-model function → proxy position → test case → uncovered items" and keeps the hidden/head/expert/quantization geometry and layout that affect kernel selection. Reuse existing modules and tests, but after changing model, dtype or parallelism re-check the contract; changing the config does not carry over the old acceptance.

### Environment fingerprint

Keep the code commit, dirty patch hash, untracked runtime files, image digest, GPU architecture, Torch/Triton/CUDA, actual import path, and native `.so` path and hash. Record TP/PP/CP/EP/DP for trainer and rollout, dtype/quantization/scale, kernel backend, and Graph/compile/cache and autotune config.

Read the resolved config and actual modules/libraries inside the worker, not launcher settings. For weights, record content version and update status of derived weights/scales, not just the path. Keep immutable known-good and known-bad artifacts before diagnosing.

### Scope of conclusions

For each result, record model scale, valid sample count, independent token count, lengths and boundaries, process count, GPU/parallel config and number of updates. Four samples repeated into batch32 are still four independent samples; mixed scheduling a small model never triggers cannot cover the full model.

Record machine budget, maximum experiment count and stop condition. GPU or system faults give `inconclusive`; a failed numerical comparison gives `fail`; `pass` for a scope requires every declared acceptance item.

### Experiment discipline

Write these nine rules into the record before comparing two arms; if any is unmet, the conclusion is "pending re-check".

| Rule | How to do it | Mechanism and example |
| --- | --- | --- |
| One image per recipe | Both arms use the same image and recipe, differing only in the recipe switch; inject changes with an overlay and record the hash | [One recipe, one image](../guides/recipe-and-acceptance.md#one-recipe-one-image-deliver-true-on-policy-and-the-regular-baseline-with-the-bi-switch) |
| One seed per config | One seed by default; add more only when the user asks for variance, reported separately | Extra seeds spend budget on variance; most problems are systematic |
| Small-workload iteration | Iterate with small batch, short outputs and few steps; run the recipe default workload only for final acceptance | The default workload allows only one or two iterations a day |
| Freeze the criterion first and plant an error | Fix comparison window and tolerance in writing before the run; confirm with a known error that it can judge FAIL; never adjust the threshold after a failure | A threshold chosen afterwards can always pass; [paired acceptance](../guides/recipe-and-acceptance.md#paired-acceptance) |
| No conclusion from a single e2e run | Measure noise first; on the same hardware and image, report the median of multiple pairs and single pairs separately | [Performance and profiling](../guides/performance.md) |
| Same fabric | Before comparing communication bottlenecks, confirm both arms use the same transport and record the relevant environment variables | [Same-fabric fairness](../guides/debugging-mismatch.md#same-fabric-fairness) |
| Same-node comparison | Run paired arms sequentially on the same node or node group, with node names recorded | Memory fragmentation, resident processes and interconnect topology differ between nodes |
| JIT pollution: check `tl_after_bench=0` | After the run, confirm no new compilation after the bench; results with recompilation are void | [Performance and profiling](../guides/performance.md) |
| Take metrics by key name | Read numbers from raw metrics (jsonl/W&B) by full key name and cite the key in the report; never by column position in log lines | An `actor/grad_norm` column once read as entropy reversed a conclusion |

List confounders before comparing; see [paired acceptance](../guides/recipe-and-acceptance.md#paired-acceptance).

## 5. Execution state

For every comparison, record the actual execution state, and keep results from different execution states as separate evidence.

### Fields that must be recorded

- Resolved Graph mode, capture sizes, whether the step is prefill/decode/mixed, and kernel call evidence; a successful capture does not mean every step uses the same graph, nor numerics identical to eager.
- Whether the prefix cache hit, and whether it was disabled/isolated during diagnosis; passing with the cache off does not mean the native lifecycle is fixed.
- Weight lifecycle phase: cold load, sleep/wake, Graph replay, reload in a new process, one row each.
- Whether chunk boundary state and partial input buffers of stateful layers such as Mamba/SSM are saved.

### Keep separate evidence

Eager, compiled with Graph off, and production Graph each keep their own artifacts; hooks run at capture, not replay, so a capture dump does not represent replay. List separately: diagnostic runtime override vs formal source candidate, Graph vs no-Graph, forward-only vs real `train()`+autograd. A new process passing does not prove the old process's cache/Graph is fixed; two fixes that each pass go through the gate again when stacked. Once CUDA state is corrupted, end that experiment and collect no more numerical evidence from it.

### Where the mechanisms and methods are

| Question | Where to go |
| --- | --- |
| Graph capture vs replay, the Graph/eager 2×2 comparison with fixed composition, which bucket the filler falls into | [BI kernel appendix](../guides/bi-kernels.md) (cudagraph row) |
| prefix cache semantics, Mamba chunk boundary state, bitwise comparison with "cache off as the gold standard" | [rollout-engine](../guides/rollout-engine.md) |
| Localizing CUDA asynchronous errors | [debugging-mismatch](../guides/debugging-mismatch.md#isolate-the-side-first) |

## 6. Candidate evidence ledger

Goal: keep an old pass from becoming a new pass.

When an experiment changes model scale, quantization dispatch, attention, MoE, Graph or policy weights, create one ledger row per **candidate execution closure**, then decide what to rerun.

### What one row of evidence is bound to

| Field | What to record and check |
| --- | --- |
| Object | Original layer numbers and real adjacency, parameter/quantization geometry, tokenizer, checkpoint content, policy version; a proxy's manually spliced boundaries listed separately |
| Execution on both sides | Trainer/server image digest, source and overlay hashes, actually imported classes and files, resolved config, native/JIT artifacts |
| Path | `train()`/autograd/whether backward runs, actual dispatch of attention/norm/linear/MoE, compile/Graph/cache, TP/EP/PP layout |
| Workload | Content digest of prompt/response tokens, valid mask, length distribution, packing, concurrency, reward/advantage and actual accepted sample count |
| Criterion | Original dtype, finite valid token count, bitwise mismatch, sample count, first difference, and the independent states of gradient/update/normal exit |
| Artifacts | Inputs, score, worker/rank receipt, command, time, parent version, hash and `pass`/`fail`/`inconclusive`/`not run` |

`policy_manifest_sha256`, the exported `policy_content_id` and the rollout receipt hash answer three separate questions, **manifest version, weight content, run artifact**, and are not comparable with each other. If the exported content ID is the hash of the export receipt, follow the receipt to check shard hashes against the files the server actually loaded; one equal version string is not enough.

### Entry gates from cheap to expensive

1. **Module.** Use a real triggering input and a non-unit scale/real layout; A/A first, then the candidate's forward against an independent reference; for gradients, check dX/dW/other differentiable quantities separately with the same dY. A pass covers only the shapes, arithmetic and VJP tested.
2. **Minimal real-adjacency proxy.** Cover the actual order of layer types, not just the operator set. With a fixed 8192+1024 tokens, check inference BI/PD first, then per the [dual-path contract](../workflow/validation.md#old-logprob-and-training-forward-dual-path-contract) verify native old-logprob and the grad-enabled actor forward separately; for a fixed-token comparison only, do not add GSM8K/RL to the proxy. Record source closure, actual class, Graph, non-empty valid tokens and original-dtype mismatch. When any item changes, re-examine whether this step still transfers.
3. **Full-model zero-update.** Generate a real-workload rollout with the full model on the **current** serving candidate, saving tokens and raw scores; the actor replays the same policy and history before the optimizer update. Check the earliest failing sample, then expand to the agreed sample size; if unequal, stop here and narrow the boundary instead of working around it with five-step RL or a quality evaluation.
4. **Per-version update.** For each θk, compare on-policy first, then check reward/advantage, backward, master parameters and the optimizer step, then sample θk+1; "the last training step exited" is not "the last version's comparison passed".
5. **Independent acceptance.** Module gradients, whole-model gradient coverage, real RL, quality, performance, clean exit and the final image each get their own result. Numbers from a previous candidate or another execution mode stay as historical comparison and never fill the current candidate's pass cell.

A fixed-weight serving overlay without RL lifecycle operations such as sleep/wake is only a serving screening candidate, even if BI/PD and 8K/1K paired performance pass; once the lifecycle is restored, rerun the affected proxy, zero-update and post-update comparison. Do not splice historical throughput and another candidate's RL pass into one "completed" version.

If a phase fails midway, classify the failure as numerical, control, environment or resource plane; keep the sealed rollout and producer hash. A control-plane acceptance bug is handled by a separately versioned recovery acceptor reading the original artifacts; never edit the old receipt, or resample and call it the same trajectory.

### When evidence must be invalidated and rerun

| Change | Minimum regression; old conclusions that cannot be carried over |
| --- | --- |
| Proxy's original layer numbers/adjacency or hidden, head, expert geometry change | Coverage table, BI/PD, training forward; the old proxy's 0 diff no longer counts for promotion |
| Quantization algorithm, scale, actual dispatch or JIT/compiler change | Real module input/quantized bytes, proxy, full-model zero-update; a direct-call unit test does not replace the compiled module |
| Attention/MoE operator or its training adapter changes | Real class and source, same-input operator, relevant VJP, proxy and full-model zero-update; a pass on a substitute backend is diagnostic only |
| Graph/cache/packing/parallel layout changes | The corresponding BI/PD, replay and state lifecycle; an eager pass does not cover FULL Graph |
| Optimizer or master→deployment export changes | Gradient and one-step update, export byte hash, post-update comparison; a θ0 forward pass is not enough |
| Final image/overlay/native build changes | Verify the closure in the worker and rerun affected numerics and performance; development-mount evidence does not transfer |

If you can prove a change does not affect a gate, write the dependency closure and reason in the ledger to skip a mechanical rerun; equivalence cannot be decided from a file name, PR description, same tag or "looks like a cleanup".

### Separate cells for performance and quality

Tune BI=0 well under the agreed weight/KV precision and quality requirements, record the real dense GEMM, MoE expert, all-to-all and attention backends, and freeze the denominator after human audit. Pair BI=1 and BI=0 on the same full workload, nodes and candidate closure; Rust Bench 8192/1024, ignore EOS, c=1/8/32/64 is one fixed throughput workload, not natural DAPO lengths or GSM8K quality. Check effective tokens/error count first, then compute the loss per the [pre-reviewed statistical protocol](../guides/recipe-and-acceptance.md#pairing-and-computation); do not reach the budget by slowing BI=0, changing weight precision or changing prefix cache/Graph parameters.

Quality needs the same problems, prompt, chat template, seed, generation limit, scorer and no-retry policy; report per-problem pairing and the absolute score difference. A quality pass does not transfer to a candidate whose numerical path changed. If the user changes the threshold, the current one governs; keep the old threshold and change date for historical reports.

## 7. Failure records

Keep failure records as they are; append new results, with no overwriting, deletion or editing. For W&B, use a separate project per comparison and read back through the API to check no step is missing. For freezing the criterion, planting a known error and "no conclusion from a single e2e run", see [§4 Experiment discipline](#experiment-discipline).
