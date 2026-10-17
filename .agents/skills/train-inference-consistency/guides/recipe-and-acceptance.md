# Recipe, evaluation and paired acceptance

Which recipe items must match the reference implementation mathematically (not by name), what puts two evaluations on the same measurement basis, and how to pair two arms so a conclusion can be drawn. The reference recipe for train-inference consistency is [verl#7688](https://github.com/verl-project/verl/pull/7688) (DS4 exact 0-diff true on-policy). Which reference a model aligned with, and the values it ended up using, are per-model facts and are not recorded here.

## Find by question

| Question | Where to go |
| --- | --- |
| Does a 5% gap between two arms count as a gap | [Paired acceptance](#paired-acceptance): noise floor, median over multiple pairs, confounders |
| The validation score only takes a few discrete values | [Evaluation contract](#evaluation-contract): under greedy the score is a multiple of 1/number of problems |
| How to design the BI=0 baseline arm | [How the two arms are defined](#how-the-two-arms-are-defined) |

## Key rules

1. Align the math, not the names; read the final CLI/resolved config, not the README or the model name.
2. Tabulate every deviation from the reference: item, reference value, our value, why, which conclusion it affects.
3. One recipe, one image, a switch selects the arm (following DS4 `EXACT_ALIGNMENT`, default BI=1; BI=0 goes through Bridge/mcore + R3 + QAT); deviations live only in the switch.
4. If any of the six evaluation items differs, it is not the same score: dataset version, prompt template, sampling parameters, number of repeats, scorer, length cap.
5. Rising training reward, probs_diff of 0, and a held-out score gain are three separate conclusions; none substitutes for another.
6. Before running, fix what is compared, how it is computed, the pass line and the noise floor; plant a known error to confirm the criterion can return FAIL.
7. One artifact answers one question: run and report performance, quality and consistency separately.
8. List confounders first, then compare; each is either aligned or quantified separately ([paired acceptance](#paired-acceptance)).
9. Keep failure records as they are and append new results; keep reruns and changed questions apart.

## One recipe, one image: deliver true on-policy and the regular baseline with the BI switch

Following DS4's EXACT_ALIGNMENT: one delivery image, one recipe, and `VLLM_BATCH_INVARIANT`, defined by the recipe, selects the whole arm. The arms may differ in training engine, precision and route replay; the switch is only the single entry point. The pitfalls below come from Nemotron-H Lightning NVFP4 (Mamba hybrid MoE); a new model must re-check them item by item, not copy the config.

### How the two arms are defined

BI=1 is true on-policy: the training forward calls the vLLM kernels and reads the deployed bytes, rollout takes the vLLM BI path, and the deployed bytes are sent down unchanged. BI=0 is the regular baseline: training takes the default MCore path (verl Megatron engine/Megatron-Bridge, without the alignment implementation of [C6](../workflow/README.md#gate-table)), and rollout uses vLLM's non-BI kernels with the configuration audited per [C1](../workflow/README.md#gate-table). For BI=0, decide the items below per model and record who approved each:

| Decision item | When needed | Nemotron's choice (example) |
| --- | --- | --- |
| Training precision: bf16 or weight QAT (STE) | QAT exists only when deployment uses quantized weights | QAT by default, bf16 optional |
| MoE route replay (R3) | MoE only | Always on |
| Requantize from master at export | Quantized deployment only | Requantize by the checkpoint's rules at every export |
| Rollout importance sampling correction (TIS etc.) | Arms whose train-inference mismatch is not 0 | Not enabled |
| rollout MoE backend | When the backend picked by `auto` does not run | Explicit `flashinfer_cutedsl` (`auto` picks Marlin, which reports a PTX version error), matching the BI0 arm of the BI=1 bench |

Do not build BI=0 as "the BI=1 actor with non-BI kernels swapped in". That requires changing the alignment implementation, and the result is neither the regular baseline nor true on-policy; a human rejected this design when it was tried.

### Rules

- **Same image, same recipe.** If the recipe lacks a switch, add it on the delivery branch first, then build one image shared by both arms; no separate image, recipe or source mount for a single experiment. Delete the old standalone baseline recipe.
- **Keep output and checkpoint paths separate.** Suffix only BI=0; BI=1 paths stay as before. With shared paths, `resume_mode=auto` resumes from the other arm's checkpoint. Renaming the run breaks an in-progress resume chain (it cannot find the first-segment checkpoint), so a running chain does not change images.
- **The BI=1 command is byte-for-byte unchanged.** The recipe must offer a mode that only prints the fully resolved training command without starting a job (Nemotron uses `DRY_RUN=1`) and a mode that only runs hydra compose. For each mode × optimizer, print the command from the recipe before and after the change and `diff` (or compare sha256): BI=1 must be identical; BI=0, against the original baseline recipe, may differ only in names, paths and newly added explicit switches. Every compose combination must return rc 0.
- **The header comment states only what was measured.** Reviews have caught untrue claims such as "the BI=0 actor uses vLLM's default kernels". Known rollout config differences between arms (such as R3 forcing prefix caching on) go into the comparison report.
- Delete untested switches (such as unverified mxfp8 and route replay variants) instead of leaving them for users to trip over.

### BI=0 arm integration pitfalls

| Problem | Consequence | Handling |
| --- | --- | --- |
| THD packing does not pass `total_tokens` into `PackedSeqParams` | Mamba gets no `seq_idx` and scans the whole pack as one sequence; logprob is silently corrupted | Pass it when CP=1; raise an error for unverified CP>1 |
| Actor needs the BF16 release, rollout needs the quantized checkpoint, both share `model.path` | Actor is built from quantized weights | Separate `hf_weights_path` |
| MTP does not turn off cleanly (the provider appends the pattern back) | Extra untrained layers | Clear both `mtp_num_layers` and the MTP pattern |
| Expert bias is updated during training by default | Routing drifts from inference | `moe_router_bias_update_rate=0` |
| Framework's built-in QAT changes only the GPT layer spec | Hybrid models read the hybrid stack spec, so QAT has no effect | Attach the quantizer after building the model from the TE spec, using the same `requantize` as the exporter |
| QAT library incompatible with the TE grouped linear signature | Startup error or wrong quantization | Quantize weights only; unit test that forward = deployed bytes and the gradient reaches master |
| param + optimizer offload, HF safetensors export | Host OOM on save | Chunked optimizer offload, save only dist ckpt, see [checkpoint and host memory](checkpoint-resume.md#memory-budget-host-and-gpu) |

R3 checks: R3 in vLLM requires prefix caching and Model Runner V2; the capture is indexed by **absolute layer number** (non-MoE layers are zero columns), and the training side selects the column by global `layer_number`, which must also line up under PP; exclude the MTP router. Known pitfalls: omitting `routed_scaling_factor` causes a drift of about 0.28; a monolithic backend that bypasses top-k capture replays all zeros.

### Smoke criteria

- BI=1 quick: `rollout_probs_diff` max = mean = 0 at every step.
- BI=0 quick: mean has the same magnitude as the independent reference (Nemotron QAT about 0.010, BF16 about 0.025), max may approach 1, grad norm is finite.
- When a step's grad norm is exactly 0, check within-group rewards first: if the whole group has the same reward, advantage is 0 and it is not a numerical problem.
- After changing the BI=0 QAT implementation, compare probs_diff against the old implementation on the same quick workload, and print the number of replaced modules (identical on every rank).

## Evaluation contract and paired acceptance

One artifact answers one question. Bitwise logprob on fixed tokens proves only the scoring of the same token history; natural generation tasks (GSM8K, AIME) answer quality; fixed-length workloads (Rust Bench) answer throughput. Token sets and timing differ, so none substitutes for another. After the numerical path, deployed weights or backend change, every in-scope quality and speed check needs fresh evidence from the **current candidate**; out-of-scope items are marked not verified, and no acceptance tasks are added on our own.

### Evaluation contract

If any of the six items differs, it is not the same score: dataset version (AIME24 ≠ AIME25); prompt/chat template and thinking mode; sampling parameters (temperature/top-p/top-k, stop tokens; greedy n=1, sampled average and pass@8 differ; pass@1 at n=8 is not greedy accuracy); number of repeats; scorer (answer region detection, boxed extraction, tail truncation length, reward extractor); length cap (training and eval caps may differ; 4K vs 8K results are not a pure capability difference).

- Under greedy, n repeats give the same result and the score is a multiple of 1/number of problems: with 30 problems one problem is 3.33 percentage points, and smaller differences are invisible.
- Rising training reward, probs_diff of 0, and a held-out score gain are three separate conclusions; dynamic sampling changes the sample composition of the training reward. Read the truncation ratio together with accuracy.
- A gain from relaxing the token budget is not a training gain. Charts come from raw metrics; do not fabricate a step 0.
- W&B: the run config behind a curve, the step basis (rollout or optimizer step) and the checkpoint version must be traceable; for cumulative metrics, take per-phase differences.
- Changing the scorer or migrating a reward handler: run both scorers offline on the same responses and inspect only the samples they judge differently; do not switch scorers for the sake of "alignment" and mark another model's valid output wrong over a format difference.

### Paired acceptance

**Freeze the criteria first, then run.** Fix the model/checkpoint content, image digest, source/overlay, hardware/nodes, runner/parallelism, weight and KV precision, the actual attention/dense GEMM/MoE/all-to-all backends, and the resolved Graph and cache config; fix what is compared, how it is computed, the pass line and the noise floor; plant a known error to confirm the criterion can return FAIL. Tune BI=0 into a reasonably strong baseline at the agreed precision and have a human audit it; do not weaken it to narrow the gap. When the candidate changes quantization dispatch, attention, MoE, exported weights or image, re-evaluate which old evidence is invalidated ([evidence ledger](../practices/evidence-and-experiments.md#6-candidate-evidence-ledger)).

**List confounders first, then compare.** Tabulate every difference between the arms; each is either aligned or quantified separately; a delta from several simultaneous changes is not attributed to one switch. Example: the two Nemotron-H arms differed at once in PC on/off, `gpu_memory_utilization`, training backend (Bridge vs mLite) and the R3 return path.

**Quality pairing**: same problem IDs, template, seed, sampling parameters, max length and scorer, no retries; record per problem whether each side is correct; `both_correct`/`only_bi0`/`only_bi1`/`both_wrong` must sum to the problem count; report each side's score, the absolute percentage-point difference and the agreed threshold. Close scores do not mean identical per-problem results, and passing quality does not mean bitwise-equal logprob.

**Fixed-length throughput**: the agreed workload is vLLM Rust Bench 8192 input/1024 output, ignore EOS, c=1/8/32/64; record actual input/output tokens, completed requests, error count, warmup, and per concurrency point the wall time, throughput and TTFT/TPOT. List any real rollout workload separately; do not mix fixed-length and natural-EOS results in one ratio.

### Pairing and computation

Before testing, a human freezes: number of repeats, pairing order, aggregation formula, the budget/pass rule for each target c, and how an invalid leg is identified and re-tested. The suggested statistic is the median per-round paired loss `median_r(1 - T_BI1[c,r] / T_BI0[c,r])`; if `1 - median_r(T_BI1[c,r]) / median_r(T_BI0[c,r])` is chosen instead, label it "ratio of median throughputs". The two are not equivalent and cannot be swapped after seeing results; without a reviewed statistical basis, report only measured values and do not declare the budget passed. Historical results keep their original formula and are not silently re-judged.

1. For each leg, record on every worker the image/package hash, actual kernel/backend, resolved config, Graph state, cache hits, and request and token statistics; verify client and server exit normally, separately.
2. Same node, same resources, no interfering load, crossed order (AB/BA/AB), each side warmed up separately every round; do not pair each side's fastest run picked across node pools.
3. Check workload and exit receipts item by item first. A mismatched leg is `inconclusive`; do not compute BI loss from throughputs with different request volumes.
4. Report raw throughput and loss `1 - T_BI1/T_BI0` for every round, then compute the formal result per c with the reviewed formula; keep failed/invalid rounds and the reasons for re-testing. Averaging across c must not hide a required c over the limit; keep single-round and aggregate overruns apart.

Optimization screening may test only the previously failing concurrency points, but it only decides whether to continue. Take the median over multiple pairs and report single pairs separately: if a candidate round sits right at the budget line, it cannot pass on the old baseline's historical median; re-pair both sides in crossed order on the same node, same image, same time window. When the candidate changes the model output path, the old candidate's quality and post-update alignment evidence do not carry over. Crossed pairing does not guarantee a strong denominator: if the same config and workload show clear fast/slow tiers, keep per-round raw numbers and check tuning receipts, CPU/GPU state, Graph, cache and actual operator selection; an optimization is not effective just because BI=0 landed in the slow tier. Mark unexplained fluctuation's causal attribution `inconclusive`; any formal target point over budget by the predefined rule is still `fail`, with no cherry-picking rounds or switching the aggregation formula afterwards.

Profile only outside the formal profiler-free E2E measurement; GPU kernel durations may overlap and cannot be summed to stand in for E2E; a kernel name is not a module identity, see [Performance and profiling](performance.md).

### Failure and recovery

When the acceptance checker errors, first tell apart producer, numerics, scorer and parser. If the original worker/server receipts and score are sealed and the wrapper parsed them incorrectly after a schema change, keep the original artifacts and read them with a recovery checker that is **separately versioned with its hash recorded**; it must verify all workers, EngineCore, API server, package identity and workload, and keep the original failure status. It can correct only the control-plane conclusion; it cannot fill in missing values, unfinished requests or a failed service exit.

Keep failure records as they are and append new results; keep reruns and changed questions apart. The final report lists at least, separately: train-inference forward, module backward, real update, quality, performance, clean exit, final immutable artifacts; each cell is `pass`/`fail`/`inconclusive`/`not run`. Example scope: "five steps pass bitwise, GSM8K pairing passes, first-round c=32/64 performance over budget".
