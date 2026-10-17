# Fault localization and mismatch bisect

Use this when training and rollout do not match, or a stage of a multi-node run died. It finds the first divergence with the fewest observations and produces a reviewable root cause, without dumping the whole model and without changing precision first. Performance attribution is in [Performance](performance.md); OOM and staging peaks are in [Checkpoint, resume and memory budget](checkpoint-resume.md).

## Symptom → first step

| Symptom | Do first |
| --- | --- |
| Not bitwise identical already at θ0 | [01 Isolate the side first](#isolate-the-side-first): four control groups; whichever two are equal rule out that side |
| Only some sequences are wrong | 01 same-prefix sibling: two samples of the same prompt must be bitwise identical over the shared prefix, otherwise it is the engine side |
| Wrong starting from some token position | [02 Bisect](#bisect-and-first-divergence): single-request prefill, bisect along one axis of layer/module/position |
| The difference grows slowly with training (θ0 is basically right) | [01 Isolate the side first](#isolate-the-side-first): save a small batch of prompt/response, score repeatedly at the same version to get each side's noise floor, then compare early and late adapters |
| Same operator input, still different | [Numerical contracts](../workflow/README.md), [BI kernel appendix table](bi-kernels.md) |
| Multi-node will not start / NCCL errors | [05 Communication fault ladder](#gb200-first-tell-apart-topology-implementation-mode-and-build) |
| Not sure what to save to be able to reproduce | [06 Save and compare protocol](#save-and-compare-protocol) |

## Key rules

1. Find the earliest divergence, not the largest layer; all later differences propagate from earlier ones.
2. Isolate the side before bisecting; classify before changing precision: input mismatch → weight mismatch → discrete path mismatch → kernel numerics → precision contract. Precision comes last.
3. Align probes by semantics (LSE in ln or log2, before or after norm, before or after the residual add). Misaligned probes produce false divergences.
4. Stop bisecting at a discrete decision (top-k, index); fix the routing, then continue.
5. A single round does not count; hidden coupling (one switch also changed scheduling) cannot be attributed to the switch; the target path must actually be exercised, otherwise the result is VACUOUS.
6. A two-node test must prove cross-node transfer happened; successful initialization does not mean bandwidth is reasonable; one `max_abs=0` does not prove consistency for arbitrary batches.
7. Looks like precision but is something else: determinism only takes effect on the main thread (switch scope).

## Isolate the side first

Applies when the logprob difference between FSDP/LoRA training and vLLM rollout is abnormal or grows with training.

Let μ be the selected-token logprob saved by rollout and π the real trainer old-logprob. `μ != π` does not by itself show vLLM is wrong. The goal is not unconditional bit-exactness: rule out correctness errors, measure the noise floor, and explain growth beyond it. For quantization mechanisms see the [precision and quantization guide](precision-quant.md).

### Per-sequence observation patterns

Record valid/differing token counts, first divergence position, longest run of differences, prompt/response length, UID/group, and packing/rank mapping. Distinguish: a few sequences differ from the first response token; all drift slightly; differences only past a chunk/page; differences only after an update. The pattern chooses experiments; it does not prove root cause.

Confirm token shift, mask and EOS. ±1/±2 offset comparisons are fine for diagnosis, but a fix cannot be declared through a misaligned comparison. Check that no score from another sample or version was referenced.

### Existing rescoring tool

A verified prefill rescoring tool takes the full sample tokens as input with zero new tokens and the current adapter, returns scores from `prompt_len-1`, checks the response token IDs, rejects missing scores, and clears the prefix/KV cache before scoring.

### The cheapest control: do not dump all layers yet

Save a small batch of real `prompt_ids + response_ids`, the prompt length and the valid response mask. From then on only score: do not regenerate, and do not rewrite the sequence through decode/re-tokenize. Record the base revision, adapter digest/version, config, code SHA, hardware and kernel backend. Compare training and serving under the same weights, with parameters unchanged during scoring.

Score the same version repeatedly on each side for its noise floor, then compare early and late adapters. Rising ordinary mismatch (samples differ every round) is a signal, not proof that same-input error grows with training.

| Path | Method | What it can answer |
| --- | --- | --- |
| Prefill | vLLM full input token IDs + `prompt_logprobs`, slice out the response | Difference between training teacher-forcing and the serving full-sequence forward |
| Decode trace | Normal prompt, force fixed response IDs, keep the real logits/logprob | Additional difference from KV/recurrent state, step-by-step generation, and decode kernels |

`prompt_logprobs=0` means no extra top-k; target-token scores are still returned. If zero generation is not allowed, generate 1 token, discard it, and read only input scores. The first input token has no conditional score; the first response token is predicted by the logits at the last prompt position, so do not shift by one. Check trace switch, runner support and MTP limits against the actual vLLM version. Do not quietly turn off production MTP for the probe; an explicit no-MTP control is allowed but does not cover the MTP path. Prefill consistency does not imply decode or MTP consistency.

### Metric measurement basis

- Prefer raw logprob. When comparing the sampling distribution, unify temperature, penalty, top-k/top-p and similar processing. temperature=0 in the request does not prove which kind of logprob is returned; check the backend semantics.
- `logprob_mae` (token MAE): `sum(mask * abs(lp_train-lp_serve)) / sum(mask)` over valid tokens; also report p95/max and position.
- `probability_mae`: `mean(abs(exp(lp_train)-exp(lp_serve)))` over valid tokens. verl's `training/rollout_probs_diff_mean` is this, not logprob MAE.
- Per-token logprob difference in VERL: set `actor_rollout_ref.rollout.calculate_log_probs=true`; after the trainer computes `old_log_probs`, call `verl.utils.debug.metrics.calculate_log_prob_diff(old_log_probs, rollout_log_probs, response_mask.bool())`. It returns the 1-D absolute difference over valid tokens, not a mean; compute mean and quantiles yourself and keep mask indices for localization. The default debug metric applies `exp` first, so it yields a probability difference. Checked against VERL main at commit 8e4a572 (`verl/utils/debug/metrics.py`); on another version or fork, check return shape, mask and `exp` first, and record the VERL revision.
- Relative L2 and cosine are for tensors; do not mix them with logprob MAE/KL. Flag zero vectors separately. Report full-vocab log-softmax, probability total variation and argmax separately from the token metrics. An overall shift of raw logits is not a change in the probability distribution. There is no cross-model universal threshold.
- Schulman k3: with `d=lp_train-lp_rollout`, take statistics of `exp(d)-1-d`. It is a KL estimate only when samples come from the current rollout distribution; on a fixed old trajectory it is a diagnostic statistic, not the current policy KL. Check for overflow; do not silently clip and then claim the difference is small.

### Build four control groups

1. Inference alone: frozen tokens, repeated scoring, batch1/mixed batch, decode/full-prefill.
2. Training alone: A/A and A/C packing replay under the same weights.
3. Cross-stack: μ/π for tokens with the same semantics, plus full logits where needed.
4. Score layer: run the scorers separately on exactly the same logits; check dtype, temperature, logsumexp and the reduction over vocabulary shards.

If training A/C differ and inference alone is stable, look at the training layout first, but keep the inference verification scope. If both sides are stable alone but differ across stacks, check weights, arithmetic semantics and the scoring path first.

### Same-prefix sibling

Same policy, same prompt, same response prefix, same score processing rules: conditional probabilities must be equal. Equal sibling μ with different π is evidence that the training side is sensitive to layout/state. For processed logprobs the penalty history and processing config must also match.

Compare only the shared prefix, never past the generation fork. Use a clean sibling and repeated execution as negative controls. "One side looks more stable" is not proof of correctness over the full scope.

### Localizing CUDA async errors

An async illegal access may be reported after the faulty kernel. Save the first fault, every rank's logs, input/version and execution stage; localize with scoped synchronization, CUDA core dump/cuda-gdb, or a sanitizer. Once CUDA state is corrupted, end that experiment and collect no more numerical evidence from it. Disabling async globally or switching attention backend does not prove the stability problem solved.

### Validity of controls

Disabling the cache may execute a prefill that never ran before. Disabling Graph also changes speed and scheduling; to blame Graph itself, fix each step's composition offline and compare. A negative control must verify the target branch's execution count: zero difference on an unexercised path counts as not covered. For experiment design see [04 Ablation method](performance.md#ablation-method-make-ab-really-differ-in-only-one-variable).

After isolating the side, bisect along one axis ([02](#bisect-and-first-divergence)); for classification and handling order see [03](#classification-and-handling-order).

## Bisect and first divergence

Separate input/weight errors, operator implementation differences and quantization error with the smallest experiment, without a model-specific debug framework. Metrics: [Metric measurement basis](#metric-measurement-basis); classification: [03](#classification-and-handling-order).

### Choose the search axis first

- Version regression: good/bad revision on the same reproducer, compatible environment, fixed input.
- Config regression: paired single-factor intervention; a 2×2 matrix when factors couple.
- Data trigger: request subset, length, neighbors, boundary remainder.
- Compute path: layer → attention/MoE/norm → internal operator → metadata/launch/instruction order.

Do not change image, weights, kernel and parallel layout at once. For a dirty good/bad, save the full patch and run files; the same git SHA does not mean the same program. Build failure, dependency incompatibility and failure to reproduce are not a bad numerical revision; mark them skip/inconclusive.

### Single-request prefill: narrow step by step, do not dump the whole model directly

1. **Read existing training logs first.** Check the metric definition, policy version and sample source. Samples from different rounds only hint at an anomaly; they cannot prove same-input mismatch grows with training.
2. **Fixed-input scoring.** Save token IDs and position/mask from a single render. Both ends use same-source weights and the same-version adapter, and each scores repeatedly to measure its own noise. Rule out weight, input and score-semantics mismatches first ([01](#isolate-the-side-first)).
3. **Minimal forward.** Single request, one prefill, one output token. Keep the minimal prefix or layer set covering the divergent path and its upstream dependencies, with identical config on both ends and no training update. Confirm no cache hit skipped the prefill, and separate extra calls from warmup/draft/decode. An eager probe does not stand for graph/MTP acceptance.
4. **Find the earliest divergence.** Look first at the embedding, block outputs, and logits at the last valid prompt position. Align layout, valid tokens, shards, padding and deferred residuals by semantics, not by module names or tuple indices. Expand only the earliest block that exceeds its own noise.
5. **Same-input operator replay.** Give both implementations the same logical input and weights to separate this operator's difference from upstream propagation. If needed, replace only one operator's output and compare the final distribution; recurrent state must also satisfy the contract.
6. **Return to the original failing path.** After the fix, re-verify with the real adapter, input and run config. Passing with reduced layers/eager does not replace verifying the full model, decode, or CUDA Graph/MTP.

### Per-module dump: find the earliest divergence, not the largest layer

1. Build a semantic mapping table: layer, function, input/output role, module path on both sides, logical shape. FSDP wrapper names and fused vs separate Q/K/V need not match; combine before comparing.
2. Capture block boundaries and final logits first, on a few tokens/samples; refine the earliest significantly divergent block into norm, projection, router, expert, residual etc.
3. Align valid tokens, position, mask, TP/FSDP shards, expert ID, dimension order and padding. Never compare a physical shard with the full tensor.
4. A hook takes a snapshot, not an overwritable reference, and never changes the computation. For FSDP, sample at boundaries already materialized in this forward; do not all-gather the whole model or materialize a huge frozen PLE on GPU to observe. Export bounded data per layer/needed shard and release GPU staging promptly.
5. Record forward/recompute counts so backward recomputation does not overwrite the normal forward. Graph replay does not re-execute Python hooks, so an eager dump speaks only for eager. For graph-only problems use in-graph observation or snapshots of existing buffers; do not clear the graph or recapture to cover the problem.
6. If the operator input already differs, the operator is not yet proven buggy; give both ends the same logical input and check whether it amplifies the difference. When a fused kernel's inside is not visible, mark the observation boundary instead of guessing.

Minimal artifacts: manifest (run conditions) + bounded tensor dump + comparison table with module/role, shape/dtype, fraction of finite values, max abs, relative L2, cosine, first anomalous position. Save a block's detailed intermediate tensors only after the summary has localized it; do not write all tokens of the whole model by default. Artifact format: [06](#save-and-compare-protocol).

### Align probes by semantics

Build a cross-stack boundary table: embedding, residual stream, norm input/output, Q/K/V, compressed rows/index, attention output, router logits/indices, expert input/output, combine, lm_head and scorer. Record tensor meaning, dtype, global token/shard coordinates and valid mask, not just module-name pairs. EP dispatch may reorder, duplicate or pad tokens; align expert tensors by semantic token/expert/shard key, not local offset. After expert GEMMs match, still compare separately the communication reduction, the shared + routed expert combination, and its rounding after compilation.

If the saved rollout's first response token already differs, recompute it once from the original prompt alone. This rules out later-trajectory divergence as the only cause, but attributing it to prefill, attention or a kernel still needs controls on the same policy, position/mask/scorer and per-layer inputs.

Compare original-dtype digests per layer or at a few checkpoints first; then dump only the neighboring boundaries and target samples. Keep one clean sibling. Snapshot async buffers before reuse without changing the production stream order in a way that hides the problem.

If the difference does not move under any knob, suspect a too-coarse boundary first. In an external TPU MiMo case, attention core and o_proj were captured together and the difference stayed at `7.8125e-03` across CP 8/2, segment alignment 128/2048 and FP32 accumulation on/off; split apart, the core was 0 and the whole difference was in o_proj. Give adjacent or fused GEMMs separate boundaries by default.

### Minimal but credible observations

- Copy tensors before buffer reuse; save only necessary boundaries and tokens, with a byte cap. Model-specific mappings and artifacts stay in the model working directory.
- The manifest records code/image version, checkpoint and adapter identity, input digest, position/mask, dtype/quantization, score semantics, graph/MTP, rank/layout and intervention options. Same config does not mean same weights.
- One directory per call; write the manifest after completion. Refuse comparison on missing boundaries, repeated calls, or incompatible version/shape; do not silently take the intersection or truncate. Non-finite values fail explicitly; metrics on zero norm are marked undefined.
- Compare max abs, MAE, relative L2, cosine; for final full-vocab logits also log-softmax, probability total variation and argmax.

### Recommended logs: identity + metric measurement basis + evidence boundary

Reuse existing logs and JSON artifacts; no extra logging framework. One manifest per run, one detailed report per comparison. The terminal prints summaries, not full tensors. Mark unknown fields unknown; do not backfill them as verified.

```text
[mismatch] run=<id> step=<n> policy=<digest> input=<sha> phase=<prefill|decode> rank=<n> repeat=<n>
  scope=<fixed-input|independent-rollout> intervention=<none|name>
  metric=<name> value=<number> reference=<run/repeat/boundary>
  self_repeat=<noise> report=<relative-path> classification=<defect|numerical|quantization|unknown>
```

`metric` uses the names in [Metric measurement basis](#metric-measurement-basis) (`logprob_mae`, `probability_mae`, KL diagnostic statistic, tensor relative L2, vocab TV), never interchangeably.

Time load, rollout, score and update separately so cold start is not counted as training. A distributed job's outer console log is only a forwarded copy: when updates stop, check TaskRunner/worker raw logs, checkpoint times and completed steps first. One forward/backward stack does not prove a hang; do not restart a healthy job because log forwarding stopped.

### Limits of bisect

A floating-point difference may vanish in later rounding, masking or routing, so "this layer differs" is not monotonic in depth and O(log L) does not always find the first difference. Coarse-scan for a suspect interval, then bisect/fine-scan, verify the boundaries before it, and confirm responsibility with the same operator input.

A cut point must carry the complete state: residual, MHC mix, SSM/conv state, KV, positions, routing and necessary metadata. A fused upper-layer post and this layer's pre have no natural independent cut point. Prefix replacement that drops hidden state creates new differences.

Only a same-input operator A/B separates upstream propagation from this operator's error. Injecting reference activations, fixing the routing or replacing the kernel can confirm causality, but they are diagnostic interventions; accept the production fix only after restoring the original path.

### Form a reviewable root cause

The result must contain: the minimal input, the last identical/earliest different boundary, the same-input control, the source mechanism, the trigger condition, and the minimal change that makes the failure disappear. A config difference is only a lead: in Nemotron, running both reduction block sizes with the same kernel gave consistent results for both, which overturned the guess that the block size difference was the cause.

If after three rounds only the final score is known to differ, add observation points or re-check the sample mapping instead of rerunning in full. Keep the failing samples and the ruled-out hypotheses so the next round does not repeat them.

### Conclusion and stop conditions

The report answers only: what was observed, what the controls proved, what is not yet proven, and one minimal next step. Correctness errors must be fixed. Normal numerical differences are judged by noise and task impact. Quantization error needs same-input evidence; low-precision inference alone is not grounds to blame it. When improvement is needed, evaluate in order: supported high-precision rollout, QAT matching serving, real low-precision training. Do not cover wrong weights or adapters by changing precision.

**Replacing with the reference output is diagnosis, not a fix. Lower local error does not mean lower training mismatch; per-operator ablation gains do not add up.** Stop once this round's evidence goal is reached; do not change the training config or start new experiments along the way.

## Classification and handling order

After isolating the side ([01](#isolate-the-side-first)) and bisecting ([02](#bisect-and-first-divergence)), classify the difference, then decide what to touch. Precision comes last. Model-specific probes, runtime patches and measurements stay in the model working directory.

### Classification table

| Class | Evidence | Handling |
| --- | --- | --- |
| Correctness error | Missing weights, wrong mask/position/layout, wrong scale/block, wrong quantization range, operator semantics violate the contract | Must be fixed and the original failing path re-verified; for a diagnostic task, deliver the reproducer and evidence first, without widening modify permission |
| Normal floating-point difference | Explained by same-version repeatability/reduction/kernel noise, no contract error found | Report the noise floor and task impact; bit-exact is not the only pass condition |
| Error from correct quantization | Mapping and quantization rules are correct, and a same-input control localizes the divergence to the quantization path | Precision ablation in the order below |
| Unlocalized | Insufficient evidence or confounded variables | Keep unknown, continue with minimal probes, do not declare "it is just quantization" |

Precision ablation order: supported and affordable high-precision rollout → QAT matching the serving rules → real low-precision training (most backend requirements, last). A natively quantized model does not support BF16 just by changing the dtype. Fake quant must match the serving weight/activation rules but does not reproduce its GEMM. QAT is not a workaround for a wrong scale. QAT contract and route selection: [precision guide](precision-quant.md#1-choose-the-precision-route-first).

Keep a single checkpoint source; open a separate high-precision adapter branch when needed.

### Handling order

Input mismatch → weight mismatch → discrete path mismatch (top-k, index) → kernel numerics → precision contract. Rule out each item before moving to the next.

Acceptance: after the fix, the same failing probe passes, the original path is not broken, and error growth on fixed samples is explained/controlled; then look at training process metrics and the fixed task evaluation. Set tolerances from the model and the baseline; do not copy another model's 0.002. Statistics on a fixed old trajectory are diagnostics, not a KL estimate of the current policy.

This procedure does not authorize stopping jobs, switching dtype, adding GPUs or changing the production recipe. When the next diagnostic gate needs new permissions/resources, report the smallest next probe and its cost.

### Delivery

Follow [06 Save and compare protocol](#save-and-compare-protocol): manifest, bounded dump, comparison table, semantic boundary of the earliest divergence, classification and confidence, one next step; state which paths were actually tested. No separate framework, no cross-model universal threshold.

A case that looks like precision but is a switch-scope correctness error: [determinism takes effect only on the main thread](bi-kernels.md#determinism-takes-effect-only-on-the-main-thread).

## GB200: first tell apart topology, implementation, mode and build

Based on experiments on GB200 four-GPU nodes; future DeepEP/NVSHMEM releases may change the support matrix. Check the source of the target fork/commit and the installed artifacts before relying on it.

### Do not use one name for four things

| What to ask | Example record |
| --- | --- |
| Which implementation | AllToAll, some version of DeepEP, HybridEP; note the fork/revision |
| Which path of that implementation | The actual class, API and kernel behind high-throughput / low-latency / v2 |
| How it communicates | NVLink within a node; RDMA across nodes; or the MNNVL path actually enabled |
| What it was compiled into | 4-peer/8-peer macro, target GPU, NVSHMEM dynamic dependency |

`NVL4 + RDMA fallback` describes the communication topology. **It does not state the DeepEP mode**: do not answer low-latency or high-throughput unless the call site proves it. How `DeepEP v2`, `HybridEP` and HT/LL relate depends on the version; do not infer containment from the names.

MNNVL being off does not rule out cross-node EP: the verified path may be NVLink within a node and RDMA NICs across nodes. Nor does that path work on every GB200; devices, network, peer build, underlying libraries and the called API all need verification.

### Get the world size right first

Write the trainer and rollout process groups separately. Formulas for dense world and expert world, and grouping pitfalls such as "two DP4/EP4 are not one EP8 group", are in [training-side layout and boundaries](training-forward.md). The runtime decides the actual grouping; do not change the GPU count to fit a config string.

### Same-fabric fairness

Before any communication-bound A/B, prove both arms use the same transport. In one GB200 NVL case the BI=0 arm's NCCL AllGather/ReduceScatter went `via P2P/MNNVL` (same clique, cliqueSize 8) while the BI=1 arm's DeepEP low-latency a2a was pinned to RoCE (`VLLM_DEEPEP_LOW_LATENCY_USE_MNNVL=0`, `NVSHMEM_HCA_PREFIX=rocep`); a gap blamed on non-overlappable communication latency was mostly the wrong NIC, fixed by two environment variables.

Check steps:

1. `nvidia-smi -q | grep -A4 Fabric`: the ClusterUUID/CliqueId of the two nodes must match.
2. Run `NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET` once per arm and check whether the transport is P2P/MNNVL or NET/IB.
3. grep the arm environment files for DeepEP/NVSHMEM pins. A hard `export` overrides the value passed by the container `-e`; write it as `${VAR:-default}`.

Mechanism: `VLLM_DEEPEP_LOW_LATENCY_USE_MNNVL=1` → `deep_ep.Buffer(allow_mnnvl=True)`, DeepEP no longer exports `NVSHMEM_DISABLE_MNNVL=1`, and NVSHMEM 3.4.5 uses MNNVL for peers in the same clique automatically. `DEEPEP_MAX_NVL_PEERS` is not a runtime switch (it is DeepEP's compile-time `NUM_MAX_NVL_PEERS`); the peer-4/peer-8 build is selected by switching the overlay. MNNVL is not hybrid-only: the DeepEP v1 LL kernel branches per destination on `nvshmemi_get_p2p_ptr` (direct NVLink write vs IBGDA), and MNNVL only makes cross-tray peers P2P-mappable; "hybrid mode" is a v2/elastic concept. MNNVL was on by default before DeepEP #190 (2025-06-06) and turned off afterwards to save NVSHMEM memory.

## Save and compare protocol

Large tensors and checkpoints stay in `results/<run-id>/`; source and small summaries go into version control. Each framework dumps differently, so write these per project following the key points below: logprob comparison, per-layer trace, replay controls, a fixed-token oracle (logprob/argmax of a fixed token set before and after a weight update), GSM8K record comparison before and after an update, a `.meta.json` provenance sidecar for the results directory, and a pre-experiment snapshot commit + clean worktree of uncommitted changes with a check of the container's actual import path and file hashes.

The manifest records artifact path, size, hash, generating command, code/image/policy version, number of valid samples and hook execution count. Confirm the write succeeded before cleaning up compute processes. Do not auto-delete existing artifacts.

### Archive experiment material by step

Build the step index before the experiment and fill in status and evidence after each run, not at the end. The layout below is logical; link existing material by explicit path instead of moving or copying it:

```text
<workspace>/
  steps/
    INDEX.md                     # current step, evidence entry of each step, gates not passed
    01-baseline/
      INDEX.md                   # tuning comparison table, candidates and review records
      <run-id>/
        config/                  # resolved config, environment variables, version list
        scripts/                 # actual server/bench commands or pinned-version entry points
        results/                 # raw results/logs, or the list of persisted artifacts
        summary.md               # conclusion, limits, failure cause and next experiment
    05-trainer-forward/           # classified by C0-C10 gate number, created as needed
    10-final-acceptance/          # references the baseline ID and this candidate version
  results/<run-id>/               # large tensors/checkpoints, not in Git
```

- **Every run is traceable.** Save the actual launch command, full effective config, code/image/checkpoint identity, nodes and run time, evidence of the backend that actually ran (a CLI flag is not evidence), exit status and raw results path. A branch name is not provenance: record branch, snapshot commit, base SHA, patch, tree hash and image digest in a `.meta.json` sidecar. Replace secrets with references. Large files go to persistent storage with size and hash, never only a node-local path.
- **Baseline tuning: one row per run**: run ID, candidate config (MoE/communication), actual backend, load and concurrency, repeat rounds, valid input/output tokens, cache hits, throughput/latency, exit status, evidence link; N/A where inapplicable. Keep failed, unfinished, diagnostic and excluded runs with the reason. The summary lists the included run IDs; do not pick the best round or keep only the final mean.
- **A candidate is not a reviewed baseline.** The index marks the recommended candidate and the human-confirmed baseline ID separately; the baseline links to exact config/image/code, result set and review record, or reads "pending review". Later comparisons reference that ID; do not swap the denominator for a same-named script's new content or another backend's old result.
- **Reruns keep history.** A new run gets a new ID and overwrites nothing. Mark an invalidated conclusion with the reason and the replacing run. Continuing an experiment keeps separate attempt/segment and exit records. Reuse evidence across steps by reference, not by copy.
- **On handoff, read the index first.** The step owner updates its record. Reporting, recovery and acceptance check raw evidence through the index; a missing file, mismatched identity or chat-only conclusion is an evidence gap, not a pass. Index historical material; do not delete or bulk-move it.

### Replayable raw data

Keep the real framework batch: prompt/response IDs, attention/response masks, positions, lengths, UID/group/sibling, rollout logprobs, necessary reward and routing records, plus the packing mapping, parallel config and how to restore that step's rollout weights. The raw batch is the replay input; exported per-token logprobs are for comparison only.

Step N's rollout usually comes from the policy before update N, so the post-update checkpoint may not reproduce it. Under partial rollout or async staleness, segments of one sequence may come from different policies (re-prefill after interruption continues under the new policy); record the policy version per segment, or a single-checkpoint replay turns expected policy difference into false inconsistency. Freeze weights, mutable buffers and RNG/dropout state, and state whether the scorer uses eval or training mode.

### Compare per-token logprobs

Each side (rollout and training old-logprob, or two replays) exports `{request_id: (logprobs, response_mask)}` as original-dtype tensors with `torch.save`. Do not convert to JSON or rebuild from printed logs. Key points:

- **Align by logical request, not physical row.** Remove padding and restore the pre-packing UID order. Both request sets must match; a missing request raises an error rather than comparing the intersection.
- **Compare only valid response tokens.** Positions outside the mask are excluded. Every sequence needs at least one valid token; an empty batch is an invalid experiment.
- **Bitwise comparison looks at bytes, not values.** Value comparison treats `+0.0` and `-0.0` as equal and two NaNs as unequal. Compare integer views: `a.view(torch.int32)` (float32) or `a.view(torch.int16)` (bf16). Do not cast first and then claim bitwise identity in the original dtype.
- **Same contract on both sides.** Same policy version, same raw/processed basis (for processed, record temperature, truncation, penalty), same input. Compare input token hashes before scores; differences from different inputs are not numerical differences.
- **Report** valid token count, byte mismatch count, first mismatching (request, position), and max/mean absolute error. Matching sampled-token logprobs do not mean matching full logits.

```python
def compare_logprobs(a, b):
    assert a.keys() == b.keys(), "request sets differ"
    n = bad = 0; first = None; max_err = 0.0
    for uid in sorted(a):
        (x, mx), (y, my) = a[uid], b[uid]
        assert torch.equal(mx, my), f"{uid}: masks differ"
        x, y = x[mx], y[my]
        ix = x.view(torch.int16 if x.dtype == torch.bfloat16 else torch.int32)
        iy = y.view(torch.int16 if y.dtype == torch.bfloat16 else torch.int32)
        diff = (ix != iy).nonzero()
        if len(diff) and first is None:
            first = (uid, int(diff[0]))
        n += len(x); bad += len(diff)
        max_err = max(max_err, (x.double() - y.double()).abs().max().item())
    return {"tokens": n, "mismatched": bad, "first": first, "max_abs": max_err}
```

### Per-layer trace: find the boundary of the first divergence

When logprobs mismatch, record tensors with forward hooks at the same semantic boundaries on both sides (for example each layer's attention output, MoE routing, residual), find the first differing boundary in execution order, then do a bounded dump there. Key points:

- **Key by semantics, not rank.** Each record's key is `(request_id, boundary name, logical slice)`, for example `("step7-p2-s3", "layer2.attn.out", "tokens=0:128")`. Stitch TP/CP/EP shards back by global coordinates and remove padding before recording. Naming and order must match on both sides. A missing probe invalidates the whole experiment; it cannot be skipped.
- **Hashes first, tensors on demand.** Take SHA256 of the contiguous tensor's original-dtype bytes (`hashlib.sha256(t.contiguous().view(torch.uint8).cpu().numpy()).hexdigest()`), with shape, dtype and semantic execution sequence number. A hash shows where it differs, not by how much; save raw tensors only at the first differing boundary.
- **Clone before the buffer is reused.** Inference-side workspace and CUDA Graph output buffers are overwritten by the next call; `clone()` in the hook before hashing or saving.
- **The first difference is not the root cause.** To blame an operator, also prove its input is identical on both sides (record its input hash too).

### Replay controls

With fixed rollout and fixed weights, change one variable (packing order, batch composition, parallel layout, kernel switch), replay old-logprob, and use the comparison above to judge whether that variable affects the result. Key points:

- **One process per variant**, with a synchronous entry point that blocks until the replay finishes. Do not use background submission such as `sbatch`: on timeout or failure the job cannot be reclaimed. Variants that change compilation, libraries or initialization environment need a new process.
- **Each variant writes to a new directory** without overwriting old results; record the actual command, environment, exit code and elapsed time. Preview the command and budget before executing; on failure, stop at that variant.
- **Prove the replay is frozen**: record the weight fingerprint, confirm no generation and no optimizer update, and confirm the packing actually in effect, not just the command-line arguments.
- For the experiment record use [the report template](../workflow/report-template.md).
