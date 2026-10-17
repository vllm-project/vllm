---
name: train-inference-consistency
description: Build and validate train-inference consistency (true on-policy RL), where vLLM rollout and the training forward give bitwise identical logprobs at high throughput. Use when aligning a training stack with vLLM batch-invariant inference, validating forward, backward and quantization contracts, or diagnosing logprob mismatch, layout and performance regressions between training and rollout.
---

# Train-inference consistency skill

Make the training side and the inference side in RL produce bitwise identical logprobs for the same token (diff = 0), and, while keeping 0 diff, bring performance back close to the high-performance baseline.

How to enable batch-invariant inference in vLLM is in [batch invariance](../../../docs/features/batch_invariance.md); this skill covers making the training side match it.

Start here: **[train-inference consistency workflow (C0–C10)](workflow/README.md)**. It answers "what to do next, and what counts as passing"; acceptance details are in [validation.md](workflow/validation.md), and the report template is [report-template.md](workflow/report-template.md).

Find a guide by problem:

| Guide | Questions it answers |
| --- | --- |
| [Fault localization and mismatch bisect](guides/debugging-mismatch.md) | Find the side first, bisect, classify before changing precision, save and compare protocol |
| [Training forward](guides/training-forward.md) | Module inventory and route (mLite / MCore / FSDP2), reproducing the inference forward by calling its kernels, layout and length, framework integration |
| [Backward validation](guides/backward-validation.md) | When to reuse backward, gradient contract, module-level VJP comparison, proxy SFT three-way control |
| [BI kernels](guides/bi-kernels.md) | Which kernel each operator uses in BI mode and what it costs |
| [Precision and quantization](guides/precision-quant.md) | The conversion chain master -> training forward weights -> exported bytes -> vLLM bytes; QAT/STE; known pitfalls |
| [Rollout engine and sampling](guides/rollout-engine.md) | Sampling RNG, prefix cache semantics, MTP constraints, R3 constraints, rollout replay |
| [Performance and profiling](guides/performance.md) | Workload and measurement basis, fair A/B, breakdown and ablation |
| [Checkpoint and resume](guides/checkpoint-resume.md) | Exact resume, derived state after restore, memory budget for colocated training and rollout |
| [Environment provenance](guides/environment-provenance.md) | Which version is actually running now; immutable delivery image |
| [Recipe and acceptance](guides/recipe-and-acceptance.md) | BI=1 / BI=0 arms in one recipe and one image; evaluation contract; paired acceptance |

Process rules: [evidence and experiments](practices/evidence-and-experiments.md), [code and delivery](practices/code-and-delivery.md).

Note: this skill comes from the DeepSeek-V4 and Nemotron-H train-inference consistency work and covers train-inference consistency only. Day-0 model support, per-model case records and cluster-specific notes are not included.
