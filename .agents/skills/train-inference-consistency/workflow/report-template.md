# Train–inference consistency report

## Outcome and scope

State pass/fail/inconclusive for the requested scope. Record model/layers, independent samples, valid tokens, dtype, topology, Graph/cache paths, update count and untested conditions.

Link the human-approved acceptance checklist: required vs out-of-scope checks, recipe revision, two-update stop condition, quality/resume coverage and target-repository submission requirements. Out-of-scope checks are unverified, not passed; finishing the short run does not complete missing required checks.

## Immutable experiment

Record source commits and dirty patch hashes, loaded libraries, image digest, GPU, checkpoint/policy identity, input bundle digest, score semantics, resolved config, commands, resource budget and artifacts.

Separate source, installed wheel, loaded native libraries and JIT specializations. State whether the measured runtime and the final delivered image are identical; if not, name the missing verification instead of inheriting the result.

For recipe work, record training context, rollout length distribution, concurrency, packing/token batch, hardware/interconnect and master/deployment/activation/KV/accumulation/gradient dtypes, plus quantization granularity, scale/update rules and the quality/throughput/memory budgets defined before testing. Unknown budgets stay unaccepted even when measurements exist.

## Delivery manifest

For the three-PR / one-image delivery, fill this from the final clean build, not development overlays. Use full SHAs and immutable image identity; mark missing evidence as not run.

| Deliverable | Review / source identity | Build or acceptance evidence |
| --- | --- | --- |
| vLLM PR | URL; target owner/branch/base SHA; head SHA | Build command and wheel/native/JIT inputs |
| Megatron PR | URL; target owner/branch/base SHA; head SHA | Installed model / primitive / adapter identity |
| verl PR | URL; target owner/branch/base SHA; head SHA | Recipe and image build script paths |
| Unified image | Base and final platform digests; dependency pins | Build log; actual trainer and rollout worker imports/libraries; no development source mounts |
| Approved recipe | Script SHA, resolved config and human approval | Changes from reference recipe; dataset and topology |
| Final validation | Commands and artifact IDs tied to the above image | Proxy / full zero-update / θ0–θ2 / quality / performance status, each reported separately |

Each PR is complete against the agreed base, not stacked on an unmerged feature branch; cross-repository dependency pins are expected. Every required runtime file outside the PRs must be a pinned existing dependency; an uncommitted production patch is a delivery gap. A changed base, build input or recipe requires reassessing affected evidence.

## Proxy coverage and gradient contract (when training is in scope)

Map each full-model feature to its proxy layer/module and test artifact; name omitted depth, geometry, topology and execution paths. Per module, record visible forward semantics, trainable variables, master vs dequantized surrogate values, STE/clipping/scale rules, fixed discrete choices and differentiable scores, VJP accumulation/collectives, and saved/recomputed state with its memory cost.

Link the independent reference and explain why it does not call the tested backward. Record shape/dtype/layout and random, zero, sparse and captured upstream-gradient coverage; per input/parameter: missing/nonfinite gradients, max absolute error, relative L2, norms and cosine where defined. Include predefined thresholds with rationale, original forward-byte comparison and one-step master-parameter deltas.

## Symptom and localization

Record per-request/token differences, A/A and A/C results, side-isolation evidence, last equal / first unequal semantic boundary, identical-input operator comparison, competing explanations and counterexamples.

## Cause and change

Explain the violated arithmetic/metadata/state contract and the smallest triggering input. Separate the actual repair from diagnostic interventions. Link code changes and the preserved regression. Write unknown if evidence does not establish a cause.

## Verification

Record the proxy upgrade decision before escalating to full-model BI consistency or training. Do not merge inference-only and cross-stack parity into one result; a failed, missing, empty or stale forward comparison blocks escalation. If the user scoped the proxy to fixed-token forward parity, mark SFT/RL not run and list the later training gates.

| Proxy upgrade gate | Status (pass/fail/inconclusive/not run) | Bound command / artifact and valid-token count | Decision |
| --- | --- | --- | --- |
| Proxy feature/operator coverage | | | |
| Proxy inference BI and prefill/decode parity | | | |
| Proxy inference ↔ native old-logprob recomputation | | | |
| Proxy inference ↔ actual grad-enabled actor forward | | | |
| Backward / SFT / RL, if in proxy scope | | | |
| Upgrade to full-model BI work | | | |

For each cross-stack row, follow the [dual-path contract](validation.md#old-logprob-and-training-forward-dual-path-contract): model geometry, frozen policy/checkpoint identity, source/image digest, dtype, topology, model/grad modes, output requires_grad, actual backends, raw logprob mismatch count and first mismatch. Keep native no-grad scoring semantics; its pass does not prove grad-enabled forward parity. Capture scores before updates and loss transforms.

| Gate | Command / artifact | Result and count | Scope limitation |
| --- | --- | --- | --- |
| Original failing replay | | | |
| Operator / metadata regression | | | |
| Engine and production layouts | | | |
| Independent module VJP and one optimizer update | | | |
| Controlled proxy SFT (A/B/C paths) | | | |
| Real backward/update | | | |
| θ0 / θ1 / θ2 parity | | | |
| First post-checkpoint-resume export / parity | | | |
| Model quality | | | |
| Clean shutdown / sustained run | | | |
| Final immutable image, without development overlays | | | |

For two-update RL, keep a receipt per policy θ0, θ1, θ2: input/mask identity, valid token count, original-byte mismatches, actual trainer-forward execution and update norm/count. Record both forward-path results at matching frozen policy versions. Worker exit is checked at final shutdown, not at each policy transition.

## SFT controls and quality (when training is in scope)

Separate SFT-A (production), SFT-B (same visible forward at the same snapshot plus an independent surrogate) and SFT-C (verified native training); these labels are independent of replay packing variants. Link same-checkpoint/same-batch A/B gradients and optimizer deltas; once trajectories diverge, later forward values are not assumed identical. A/C differences include forward/quantization changes.

Record initial master and optimizer state, data/order, labels/masks, packing, effective-token normalization, batch/accumulation, learning-rate schedule, clipping, precision/loss scaling, RNG/dropout/recompute and intentional differences. Keep raw train, held-out and fixed-probe loss, gradient norm, update RMS, clipping/nonfinite/skipped-step counts, throughput and baseline/seed variation when available. Use one optimizer-step or effective-token axis; report predefined windows/thresholds, missing controls and whether evidence is only a short smoke run.

## Performance

Record matched work, warm/cold timing, node pool and paired order, repetitions, latency/throughput, kernel/rollout/full-step results and numerical receipts. Separate measured improvements, rejected candidates and estimates.

For 8K/1K acceptance, link the [approved aggregation protocol](../guides/recipe-and-acceptance.md#pairing-and-computation): formula, repetitions, per-concurrency thresholds and invalid-leg replacement rules. Include all raw pairs, exclusions with reasons and the approved aggregate per c; median paired loss and loss from median throughputs are different statistics. Other phase timings are diagnostics, not gates.

Where relevant, include training time, peak memory, launch count vs token batch, CPU gaps and repeated quantization/allocation. Separate early feasibility profiles from final full-model optimization; normal autoregressive decoding is not evidence of an avoidable per-token prefill/training loop.

State whether this is a BI-only causal comparison or each side's best valid stack, and list all differing flags/backends. Include prompt count vs expanded requests, actual input/output lengths, completed output tokens, wall-time denominator and whether the run is fixed-length synthetic or natural-EOS. Do not sum overlapping GPU ranges or percentages from separate ablations.

## Evidence corrections and retention

List superseded claims with the correcting artifact and reason. Record which original failure bundles, checkpoints and small receipts are retained, their storage budget/location, and any missing eval steps. Never fill a curve with a different policy lineage.

## Remaining work

List concrete gaps and the smallest next experiment. State whether production hooks/config were restored. Keep raw evidence; do not upgrade historical or partial results into current completion.

## Experiment record (YAML)

```yaml
# Train-inference consistency replay experiment record: each variant changes only one variable, everything else is fixed.
purpose: "frozen-policy old-logprob replay"
target: "bitwise selected-token raw logprob"
rollout: "/absolute/path/to/frozen-rollout"      # original framework batch, not exported scores
checkpoint: "content id + pre-update recovery path"
environment: "image digest, source+patch hashes, libraries, GPU"
topology:
  trainer: "TP/PP/CP/EP/DP"
  rollout: "TP/PP/CP/EP/DP"
state: "RNG, mutable buffers, Graph/cache and reset policy"
budget: {max_runs: 3, timeout_seconds: 600}
variants:
  - {name: A, change: "none (A/A baseline)", cmd: "", output: "", exit: null, result: ""}
  - {name: B, change: "none (A/A repeat)", cmd: "", output: "", exit: null, result: ""}
  - {name: C, change: "packing reversed", cmd: "", output: "", exit: null, result: ""}
```
