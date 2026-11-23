# Kimi K3.1 BF16 alignment experiment

This branch preserves a coordinated four-layer experiment across yeyunu/verl,
yeyunu/vllm-ascend, yeyunu/vllm, and yeyunu/FSDPTurbo.
Use branch codex/kimi-k31-bf16-alignment-20261123 in all four repositories.

## Measured scope

- 4 layers, 16 experts, top16, BF16 on one Ascend 950DT 8-device server.
- Actor/reference FSDP8 and EP8, rollout TP8/EP8/DP1.
- Attempts 32, 33 and 34 each completed two steps. Every step had 1024 valid
  response tokens, zero nonfinite values, zero differing old/rollout logprobs,
  and zero mean/max absolute difference.
- Attempt 34 compared 32 actor old/grad response-score pairs and 32 actor/reference
  pairs. Every valid response comparison was exactly equal.
- Same-side attempt 34 covered one prompt/image and eight generated positions.
  A/A and actual batch1/batch2 prefill/decode logits and selected scores were
  equal. Full-prefill versus decode at the corresponding history was equal.
- Reversed submission did not change the actual target row in attempt 34.
  Actual row reordering, longer BI trajectories and batch16 remain pending.
- Chunked-prefill probe code is included as work in progress. Attempt 35 failed
  during HCCL initialization before that probe executed.
- Reward, advantage and actor grad norm were zero. Effective policy update and
  post-update synchronization acceptance remain pending.
- Full 93-layer, model-quality and performance acceptance remain pending.

## Source provenance and review

Inherited runtime overlays have their own commit before task changes. These
overlays came from the existing reference environment and must not be attributed
to the alignment fixes. The task copy had no Git metadata.

The vLLM base is ced6857afa0ea7b2e3f0846a62e1394e90f15607.
The verl base is wangdongleix/verl commit
d854d22c3a01359970e6b0fb05a59ce74555d4dd.
FSDPTurbo history comes from https://gitcode.com/Ascend/FSDPTurbo.git,
base c43ebadabb74f85ca131189c8d439ab2fae617b6.
The installed vLLM-Ascend reported 0.30.0rc2.dev252+g9a0b3fbd6.
That exact commit was unavailable in the public GitHub repository, so its changed
runtime files were captured in a separate inherited-overlay commit on the existing
fork. This is an experimental source export, not a fresh-install validation.

The shared helpers live in verl/examples/kimi_alignment/runtime. Add that
directory to PYTHONPATH alongside the four source checkouts. Scripts under
verl/examples/kimi_alignment/scripts preserve the task's existing workspace
layout and require the original model, environment and task paths.
model-code.patch applies to the original four-layer checkpoint's Python files.
It contains code only, with no model weights.

No rollout logprob reuse was used for the independent acceptance comparisons.
The diagnostic scheduler budget control is opt-in and belongs to the pending
chunked-prefill experiment. The BF16 alignment switches and extensive audit
captures need normal upstream review before production integration.

Code export checks cover syntax, file hashes and secret-pattern scanning.
The reported device results belong to the retained 423 runs. Upstream CI and
fresh-branch NPU validation have not been run.

The vLLM-Ascend format.sh ci command was attempted during export and exited
because pre-commit was unavailable. Python syntax and runtime-source byte checks
passed. Upstream lint and CI remain unverified.
