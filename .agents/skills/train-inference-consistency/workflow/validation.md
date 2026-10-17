# Tiered regression and completion criteria

## Full-model entry gate: pass proxy train-inference forward first

When backends, compile paths or delivery images have changed, check against the [candidate evidence ledger](../practices/evidence-and-experiments.md#6-candidate-evidence-ledger) whether old results still apply. A passing old candidate is history only and does not fill the current candidate's entry gate cells.

Before full-model correctness or training experiments, provide the evidence below. Missing or not executed counts as not passed; repeated full-model launches are no substitute. Proxy inference-internal BI/PD and proxy training forward↔inference logprob comparison are two independent results: the former passing does not mean the latter passes.

| Gate | Minimal evidence |
| --- | --- |
| Proxy coverage | Inventory of operator types, necessary adjacent-layer interactions, quantization geometry, state and parallel paths; uncovered items stated |
| Inference BI/PD | Batch, prefill/decode and mixed-scheduling comparison on the proxy with fixed weights and tokens; proves only consistency inside inference |
| Training forward↔inference | Same proxy, same policy weights, fixed tokens: verify native old-logprob recomputation and the grad-enabled actor forward used in training separately, each against the target production inference path; keep evidence per the dual-path contract below |
| Backward and quality | [Independent module gradient comparison](../guides/backward-validation.md#layer-1-module-level-vjp-comparison) with reference implementation, tolerances and raw results; controlled SFT per task scope, or an explicit statement that quality is not covered |
| Real RL update | At least two effective updates, θ0/θ1/θ2 same-policy comparison, actual parameter change, normal exit; sequence below |
| Performance feasibility | launch/sync/memory overhead on real shapes and the proxy profile, ruling out bottlenecks such as per-token launches in batchable prefill/training; the target Graph path works |

This is an escalation gate, not the completion criterion: two-step RL does not prove convergence, and the proxy does not prove full depth, capacity, long context or cross-node correctness. On the full model, extend coverage one item at a time. Inference-only tasks apply only the relevant gates, state that training is not accepted, and add no training work.

Before escalating, record per gate in the [report template](report-template.md#verification): `pass`/`fail`/`inconclusive`/`not run`, command, artifact path and the bound model geometry, weights, source/image, precision and parallelism. Accept only current-candidate evidence. For training forward↔inference, the bitwise target requires original-dtype mismatch of valid tokens to be 0; tasks with an agreed error budget use the predefined standard and cannot use it to report a bitwise failure as a pass. An unexecuted training forward, an empty mask, or passing only single-operator or inference BI/PD tests must not release to the full model. Keep the first failing sample on the proxy for layer-by-layer localization, and after the fix rerun the original failing case and the affected matrix.

### old-logprob and training forward dual-path contract

- **Native old-logprob recomputation:** follow the framework's actual path and its `no_grad`/`forward_only` semantics; do not force autograd on for acceptance. Its result proves only that this scoring path matches rollout.
- **Actual training forward:** with the same frozen θ, token history and valid mask, run the grad-enabled actor branch training actually uses, capture logprob before the loss transform, and compare with rollout independently. Keep the target packing, recompute, operators and parallelism; do not substitute a diagnostic forward. Passing does not replace the later independent backward validation.
- Record per path: model mode, grad mode, output `requires_grad`, operator class / backend / source hash, original dtype, non-empty valid token count and mismatch. Diagnostics (substitute attention/MoE, Graph off) prove only that path; `forward_only` alone is not grounds for rejection, and native old-logprob passing does not prove the grad-enabled branch.
- The comparison happens before the corresponding policy update, for both proxy and full-model θ0 entry. If the formal recipe has several minibatch updates per round, do not require 0 diff between the updated actor and the old-θ rollout, and do not modify the algorithm to manufacture equality; record the real parameter version and compare only within one version.

When an adapter / overlay is injected dynamically, the receipt must also record the actual object's class, source file and key config and assert that the target callsite was hit. A script directory placed before the overlay in `PYTHONPATH` can silently load an old adapter; a 0 diff from such a run cannot vouch for the new candidate.

If the user limits the minimal proxy to fixed-token logprob comparison (or forward only), do not require GSM8K, SFT or two-step RL; mark those gates `not run`, not passed. Scale up as agreed only after the proxy train-inference forward passes. Before any multi-step full-model RL, first complete a zero-update same-policy logprob comparison with **full-model θ0 real workload, real training mode, target production inference mode**, then the necessary gradient validation and effective-update checks. A fixed 8K/1K proxy pass does not replace the full-model variable-length workload. When full-model θ0 fails, stop optimizer updates, keep the failing trajectory and shrink to the first divergence. This scope adjustment does not waive the proxy forward gate.

### Limited exceptions that allow going to the full model early

- Establishing the BI=0 performance/quality baseline of the full target model is not approval for full-model BI debugging or training.
- When a fault depends on real depth, capacity, cross-layer state or cross-node topology and cannot be reproduced on a semantics-preserving proxy, targeted full-model diagnosis is allowed. Before the experiment, write down why the proxy is insufficient, the hypotheses to distinguish, the minimal scale and the stop condition.
- Scaling up does not scale up resource permissions: no extra GPUs, no stopping other jobs, no bypassing the model/runner scope the user set.

### Faster iteration and fallback on failure

Judge each change's impact first: minimal counterexample, single operator or fixed replay, then the relevant proxy gates, then the staged full model; do not repeat all ten steps per commit. Changes to shared arithmetic or gradients regress the affected forward/backward; final delivery still runs the full agreed acceptance.

On a full-model failure, first save the input before the first divergence, the weight version, metadata, cache/routing and necessary RNG state, and extract a module or few-layer reproducer. Iterate on it only after confirming the fault survives shrinking. If the fault disappears, a passing shrunk test is not a fix, and the full-model reproduction is kept.

Fixed rollout / module inputs may be cached and replayed in an isolated resident process, bound to fingerprints of weights, input, config and necessary state; a new implementation still reruns the comparison. Each scale-up states the existing evidence and the open question the new scale answers.

If a candidate fix changes the sampling trajectory or EOS stop point, save the new token history and replay training old logprob on it with the same weight version. The old trajectory remains a fixed-input diagnostic; comparing new inference against the old response length is not an end-to-end pass. List evidence separately for a diagnostic runtime override, a formal source candidate, Graph/no-Graph, forward-only and real `train()`+autograd. An isolation package may be used for localization; final acceptance returns to the delivered code and the default target execution mode.

## Start from the cheapest failure criterion

1. host metadata/shape arithmetic: minimal counterexample of a pure function and boundary parameterization; if needed, exhaustively check equivalence before and after the fix.
2. Single operator: same input, real dtype/layout, non-finite detection, repeat/cross-process and Graph.
3. frozen batch replay: the original failing A/C and A/A, full valid-token comparison; UID mapping passes.
4. engine: batch composition, prefill/decode, chunk boundaries, cache/Graph, actual parallelism scope.
5. Real training (when in scope): non-degenerate reward/advantage, finite and meaningful non-zero gradients, optimizer step, actual parameter change, frozen-token alignment after the update.
6. Production acceptance: full model, agreed multiple steps, update/restore lifecycle, quality evaluation and paired performance.

Not every task runs every tier. Report the tiers completed; when the user asks only for diagnosis, do not start full training. Extend existing test suites first; do not replace behavior tests with assertions on source strings.

## Preventing false greens

- Zero valid tokens, comparison not executed, missing rank/sample: invalid experiment.
- All-zero score, mask all false, the same sample repeated: check the producer and coverage; a passing comparison is not enough.
- probability/KL/logPPL is zero: still check the original logprob bytes.
- Zero gradient or identical rewards: maybe no effective update; check advantage, update count and parameter difference.
- Small model / short output passes: confirm the full model's mixed-step and long-context trigger conditions.
- A shrunk proxy's error magnitude is not the full-model baseline: in the external MiMo case the 12-layer proxy was `1.866e-12` and full-depth real weights were `1.143e-02`. The proxy only verifies the execution path.
- In-training mismatch metric is 0: confirm the micro-batch really ran the trainer forward. The framework may skip the forward for a zero-advantage batch and return 0, a trivial match; count only reports with non-zero advantage.
- A success marker written before the process was killed: separate a completed numerical experiment from normal exit / long-term stability.
- probs_diff is 0 but many sibling samples within a group are identical: a sampling diversity defect, independent of the consistency metric. Count distinct rollout samples per group; the mechanism and fix are in [rollout-engine §1](../guides/rollout-engine.md#1-sampling-seed-collision).
- The launcher shows `COMPLETED 0:0`: read the recipe's own exit code. When tests report passed, check the skipped count; GPU cases may be skipped wholesale by a marker or environment gate.

## Unified measurement basis and matrix for forward consistency

Fix policy, token history, position, mask and tensor semantics, and record per item the original dtype/shape, valid token count (greater than 0), non-finite count, bitwise mismatch count, maximum absolute difference and first failing position. A dtype difference is a contract inconsistency first; do not convert and then claim a bitwise pass. After aligning semantics, compare contiguous tensor bytes that preserve the bit pattern or an equal-width integer view; floating-point `==` does not distinguish positive and negative zero. NaN/Inf in a valid logprob is a failure even with identical bytes. Sampled-token comparison does not extrapolate to the full vocabulary.

Keep cases and artifacts per item for the target execution scope, and list unsupported / not run items:

| Dimension | Comparison under fixed semantics |
| --- | --- |
| A/A | Repeated execution with the same weights, input and state; in a separate process if needed |
| BI | Single request, different batch sizes, companion requests, permutation/position and packing |
| PD | Full prefill, chunked prefill and step-by-step decode on the same history, covering mixed scheduling |
| Length and branches | Real lengths, chunk/causal boundaries, and cases that trigger sparse selection, MoE and special modules; not only random tokens |
| Execution state | Target Graph capture/replay, cache reuse/reset and the actual parallel layout |

The bitwise target requires mismatch=0; error-budget tasks report against the predefined standard and are never recorded as a bitwise pass. Gradient tolerances are defined independently, see [module VJP](../guides/backward-validation.md#layer-1-module-level-vjp-comparison) and [SFT control](#proxy-sft-three-way-control).

## Two-step RL sequence

This sequence is the lower bound for entry; "container started / training script exited 0" is not a pass.

When the task includes the full train-inference loop, complete at least this sequence; every actor recomputation uses the same policy and token history as its rollout:

```text
θ0 → rollout0 → actor(θ0) recompute and compare
   → backward0 / optimizer update → θ1
θ1 → rollout1 → actor(θ1) recompute and compare
   → backward1 / optimizer update → θ2
θ2 → rollout2 → actor(θ2) recompute and compare
```

The θ2 comparison needs no third optimizer update. To compare outputs across policies, use the same frozen token history; do not rely on sampling happening to match.

For each stage save the policy content identifier, token/mask, non-empty valid comparison count and raw byte comparison result; for each update save finite loss/gradient, coverage of parameters that should have gradients, master update norm and optimizer step.

For a bitwise task the θ0, θ1 and θ2 comparisons must all have mismatch=0, with two effective updates and a normal exit actually completed. Skipped updates, both sides reading an old cache, an empty mask or comparing only probability summaries are not allowed.

A small master update may not cross a quantization bucket, so deployment bytes need not change every step. Judge an update by master update norm and optimizer step, not by step count or byte change.

## Proxy SFT three-way control

The three paths, controlled variables, recorded items and localization order are in [Backward validation layer 3](../guides/backward-validation.md#layer-3-proxy-sft-three-arm-comparison). If the module gradient comparison fails, fix the implementation before starting SFT.

## Performance and model quality

When forward arithmetic, precision, scorer or inference path changes, rerun the same quality evaluation within the human-reviewed scope and the target repository's requirements. Record model/data, problem count, prompt format, generation settings, statistical variation and unfinished parts; mark out-of-round items unverified, and add no SFT or long runs. A loss decrease or two-step reward change does not prove no quality regression.

Micro benchmarks, fixed-token parity, generation quality, normal exit and multi-step stability are separate. A run with four non-zero updates and 139,260 bitwise-equal tokens that ended in a DataLoader error supports four-step alignment, not clean shutdown or 50-step stability.

The full model covers the reviewed scope (real depth / weights, target context, concurrency, packing, parallelism/hardware, cache/Graph) and reruns affected correctness gates; proxy curves do not extrapolate to full-model quality. Final performance is accepted on 8K/1K per the [paired statistics protocol](../guides/recipe-and-acceptance.md#pairing-and-computation); training and full-step times may be listed as diagnostics and add no performance gate. Scope and stop condition of the two-step short run are in the [general two-step acceptance agreement](#general-two-step-acceptance-agreement).

## Failure classification and delivery

Use `pass`, `fail`, `inconclusive` with a reason. Keep source/patch/image/bundle hashes, commands, per-item counts, earliest difference, excluded hypotheses, and before/after results on the original reproducer. Cleaned-up code needs re-verification or explicit behavior-equivalence evidence; old tests do not vouch for new code.

Passed items stand despite unfinished ones, but the final conclusion is limited in scope. Use the [report template](report-template.md). Formal upstream contributions also follow the target repository's duplicate check, human line-by-line review, tests and AI assistance disclosure; historical cases do not mean the review was done on the author's behalf.

If the delivery includes an image, the gate object is the final immutable artifact, not a development image plus a source mount. Verify the closure per [runtime provenance](../guides/environment-provenance.md); accept save per [01 §3.2](../guides/checkpoint-resume.md#32-reproduce-and-discriminate), load per [§4](../guides/checkpoint-resume.md#4-load-enough-capacity-does-not-mean-enough-for-the-transient-peak), and the first resume per [§6](../guides/checkpoint-resume.md#6-how-to-prove-resume-is-correct). Passing environment checks proves only the preconditions for running, not GPU numerics or performance.

## Recipe review materials and two-step acceptance agreement

The integration point is in [training-side model support](../guides/training-forward.md); only process agreements are here.

### Recipe and PR delivery contract

Locate the existing integration layer in the selected version first. Megatron's `experimental/lite/examples/verl/verl_mlite/` provides the verl engine adapter; an external integration package installs normally into the unified image, and removing the runtime overlay does not require reimplementing the engine in verl. Check the actual engine registration, model factory, batch conversion, loss normalization and optimizer path, and list reusable interfaces and specific gaps.

- **verl PR** contains by default the model's DAPO recipe and the unified image Dockerfile. Public interface changes beyond these two: submit the gap and plan first, implement after human review; necessary regression tests may stay.
- **Megatron PR** contains the model, operators and necessary support in the existing `verl_mlite` integration layer; do not move generic code that does not belong in Megatron there just to keep the verl PR at two files.
- **vLLM PR** contains the necessary inference-side model, kernel, state and weight loading changes.
- Per the [general delivery target](../practices/code-and-delivery.md#8-three-repo-delivery-from-experimental-implementation-to-three-prs-and-one-image), submit complete PRs to the three main repositories, including the required model adaptation and not stacked on another unmerged feature PR; pin cross-repository dependency versions and state the build order. Consistency uses mLite and the training baseline uses the existing verl recipe's default MCore path (see [the workflow](README.md#gate-table)); do not merge them into one training scheme.

### DAPO reference and review materials

[verl #7688](https://github.com/verl-project/verl/pull/7688), at reviewed version `640c5f0a317307e99636924804c40624242fa19b`, uses Megatron's integration package through `model_engine=mlite` and `pkg://verl_mlite.config`, so the verl diff needs only the recipe and the Dockerfile. Follow its real `verl.trainer.main_ppo` execution chain; do not assume its config keys, DeepSeek precision/topology, mounts or overlay fit the current model.

The reference case's model, data and evaluation set are not defaults. Choose training data / reward by target model and human agreement, and keep the training set distinct from the quality evaluation set. Finish config resolution and the necessary proxy smoke, then submit for human review:

- Executable launch script, unified image digest, commit of each repository and full resolved config.
- Parameter differences from the reference recipe with reasons: data/reward, prompt/response length, rollout count, batch/microbatch, parallelism, quantization / KV precision, optimizer.
- Proxy check results, known uncovered items, and the acceptance scope of the full-model zero update and the two real updates.

Formal full-model E2E starts after a human confirms a specific recipe version and the proxy entry passes. Workload or numerical/update changes are reviewed again; path and log changes that leave input and execution semantics unchanged are only recorded. This gate does not block already authorized builds or proxy development.

### General two-step acceptance agreement

- **Freeze versions first.** Record the heads of the three delivery PRs, the unified image digest and the full recipe config. Changes that affect results require a version update and the related verification; old artifacts cannot be reused for acceptance.
- **Review the recipe first** (materials above). Confirming process or scope does not approve an unsubmitted recipe. The review checklist also marks which quality, restore and long-run checks are required or out of scope for this round, and the evidence the target repository requires; out of scope does not mean passed, and the short run ending does not hide missing required items.
- **Run two effective updates by default.** Use the formal recipe and shorten only the execution length to two steps; do not downgrade data, sequence length, rollout count, batch, optimizer, algorithm, parallelism or backend. If the stop step count drives the learning-rate schedule, keep the formal recipe's learning-rate/warmup semantics for the first two steps instead of compressing the full schedule into two; the implementation is reviewed with the recipe.
- **Keep the existing correctness gates.** After the proxy real training forward passes, run the full-model [two-step RL sequence](#two-step-rl-sequence) with actual parameter updates; a fixed-weight replay does not count instead.
- **Performance is accepted only on 8K/1K.** On the frozen version, use Rust Bench at 8192 input / 1024 output, ignore EOS, c=1/8/32/64, against the human-confirmed BI=0 baseline and performance budget. Training forward, backward and full RL step times may be recorded but add no performance gate.
- After the two updates, the θ2 comparison and the agreed 8K/1K performance pass, stop the short run and report. Overall acceptance is still judged item by item against the review checklist; if a required item is missing, state that it is not complete. Do not extend to 100 steps, add an MCore long-run comparison or new quality gates, and do not extrapolate the two-step result to long-run training quality.

### E2E comparison target

Run the [two-step RL sequence](#two-step-rl-sequence) through the real trainer entry point. 0 diff means equal original-dtype logprob for the same weight version, token history and valid mask; θ0 and θ1 outputs may differ. Verify the actual master update each round (not step count or byte change). Disk replay is a localization tool and does not replace acceptance through the real trainer entry point.

Finally re-verify with the image built from the delivery PRs and the reviewed recipe. Record standalone engine smoke, old image numerical results and historical performance separately; they do not replace full evidence from this execution entry point.
