# Code, experiment snapshots and delivery

Code ownership across three repos for cross-stack train-inference consistency delivery and "three PRs, one image" (§8), and PRs and cleanup (§9). Lifecycle, cold-start and model acceptance details are not repeated.

## 8. Three-repo delivery: from experimental implementation to three PRs and one image

Cross-stack RL delivery for a model is one vLLM PR, one Megatron PR, one verl PR, and one unified image built from the versions the three pin. Default targets are `vllm-project/vllm:main`, `NVIDIA/Megatron-LM:dev` and `verl-project/verl:main`; a fork may host the head but does not replace the main repo's target branch. This section covers change ownership and delivery checks; it does not replace the [ten-step flow](../workflow/README.md), the [numerical gates](../workflow/validation.md) or human recipe review.

### Before changing code: give a change map first

Fix the inference-side change locations in step 3 and the training side and integration interfaces in step 5; step 1 need not design every interface. Submit a short table of "feature → existing module/interface → proposed change location → reason for reuse or addition", covering at least model construction, forward/backward, quantization, parameter registration, checkpoint and the build entry.

Read the implementation and callers in the selected checkout before choosing file names; the paths below are design references, not a requirement to copy DeepSeek's directories. Do not write a standalone engine first and discover at the end that the existing integration layer or the real trainer cannot use it.

### vLLM: implement in the formal execution path, do not deliver a patch loader

- Development environment and code sync follow [same-commit worktree, per-file sync and build verification](../guides/environment-provenance.md#boundary-between-overlay-and-native-wheel); development sync is not how the final image is installed and does not change the PR's target branch.
- Model-specific selection goes in the model construction/execution entry; changes to shared kernels, backends and metadata go in the existing owning module, stating which models and callers are affected.
- BI=0 keeps the reviewed native path; BI=1 explicitly selects the aligned implementation. Check the actual backend and fail early when a required BI capability is missing; never silently fall back while still claiming consistency.
- Kernel numerics, binding and build registration ship together; CUDA/C++ changes enter the normal build, and JIT changes such as CuTe DSL record the compiler and specialization. Do not submit only the Python call while depending on an uncommitted `.so`, source file or build script inside the experiment container.
- Verify the default path is unchanged and cover the modified shared interfaces; when only a certain model/runner/topology is supported, restrict it explicitly rather than implying other models support BI.

Reference points in [vLLM #54955](https://github.com/vllm-project/vllm/pull/54955): the formal kernel integration in `csrc/batch_invariant/`, the DeepGEMM BI capability check, and the stable tie-break/prefill-decode semantics of sparse selection. These address operator numerics, backend authenticity and execution shape change respectively; they are more than a BI switch.

### Megatron mLite: separate the model, the aligned operators and the runtime

The train-inference consistency implementation uses mLite; the training baseline uses the default MCore path of the existing verl recipe, without the aligned implementation. Record the training baseline and the vLLM BI=0 inference performance baseline separately; they are not the same comparison. When target-model support has gaps, state them and get them reviewed; do not change the training scheme on your own.

| Change | Preferred owner | Should not go in |
| --- | --- | --- |
| Model layer structure, parameters and checkpoint mapping | model/protocol/checkpoint modules of `experimental/lite/model/<model>/` | a temporary model implementation inside a verl recipe |
| forward aligned with vLLM, autograd/STE, model-specific quantization | that model's aligned implementation and primitive modules | miscellaneous top-level mLite scripts, global monkeypatch |
| Reusable training/communication/optimizer capabilities | existing shared primitive/runtime, changed only for a real interface gap | one model's recipe constants and diagnostic branches |
| verl engine adaptation, batch/loss bridging | existing `experimental/lite/examples/verl/verl_mlite/` interface | another equivalent engine inside verl |

- Select native vs aligned implementations through the existing protocol/factory/explicit config; do not scatter environment checks across low-level forwards or replace instance methods/global functions at runtime. Small-scope branches can stay; do not copy the whole model for a tidy directory; reuse shared math and backward, and implement independently only primitives whose numerical semantics really differ.
- Do not assemble production operators with `inspect.getsource`, AST rewriting, or string replacement plus `exec`, and do not register models via `sitecustomize` or temporary `PYTHONPATH` hijacking. Import stable interfaces from the real dependency, and commit necessary changes to the owning module.
- Keep the parameter registration, state_dict/checkpoint, distributed reduction and recomputation contracts; do not replace the training master parameters with an inference cache the optimizer does not manage just to align forward. Backward is accepted separately per the [gradient contract](../guides/backward-validation.md#write-down-the-gradient-contract-first); it need not be bitwise identical with inference, and a kernel without autograd does not automatically need STE.

[Megatron #7050](https://github.com/NVIDIA/Megatron-LM/pull/7050) separates DeepSeek-V4's `lite/` from `vllm/primitive/` and reuses the existing `verl_mlite`, illustrating these boundaries. `verl_mlite` exists in official Megatron dev, not only in a fork; still check the version this task pins and its actual capabilities.

### verl: a thin recipe, not a second training framework

What is delivered depends on the route:

- **mLite route**: for non-observability changes, deliver ideally only one recipe script and one image build script, reusing the real trainer of the current version and the installed `verl_mlite`.
- **FSDP2 Day-0 route**: verl lacks the model, so the deliverable is the self-written model implementation in verl mainline, its Auto class registration and the recipe, in one verl PR; the model implementation itself is the deliverable and needs no separate "generic interface gap" pre-review. See [Training-side model support §3.3](../guides/training-forward.md#33-fsdp2-verls-hf-interface-path).

On both routes, when there is a real generic interface gap, first give call-chain evidence, the interface to change and the scope of impact, **get human review before implementing**, and make the minimal fix in the owning repo; do not hide code in scripts or move it to the wrong repo to make up the file count.

The two-file form of [verl #7688](https://github.com/verl-project/verl/pull/7688) and the `verl.trainer.main_ppo`/`model_engine=mlite` chain are the reference; its development mounts and overlay are not how the final image is delivered. For data, config differences and review rules, use the [DAPO reference and review materials](../workflow/validation.md#dapo-reference-and-review-materials); do not invent an "approximate DAPO" script that bypasses the trainer.

### Step 10: from clean PRs to verifiable delivery

1. **Three complete diffs.** For each repo, record the target owner/branch/base SHA agreed with the user and the PR head; check the merge-base, commit history and actual diff. Required model support must be in the PR or already merged into the target branch; another unmerged feature branch cannot be the base. Do not carry over a preview PR's history drift or unrelated deletions; pin cross-repo dependencies by commit. After changing the base or rebasing, re-review the diff; evidence from the old head does not transfer.
2. **One rebuildable image.** Build from three clean checkouts, pinning the base image digest, dependencies and necessary native/JIT build inputs, and install all runtime packages normally, including external adapter packages. Training and inference workers use the same immutable platform digest and the same Python/Torch and runtime dependencies; different nodes/processes are fine, but not two independent environments inside the image to dodge compatibility. It must run without host source mounts, startup-time patches or uncommitted overlays.
3. **Re-accept from a new container.** Check the actual worker's interpreter, imports, libraries and versions; verify the proxy, full-model zero-update and the real RL loop of the reviewed recipe per the existing gates; then record performance/quality. A successful build, offline disk replay and historical 0 diff do not replace verifying the final artifact; on a difference, shrink back to a fixed failing replay first instead of relaunching full training to locate it.
4. **One mapping.** Use the [report template](../workflow/report-template.md) to list the three PRs, image digest, recipe review version and acceptance receipts, marking each item pass/fail/inconclusive/not run. Update it after code cleanup, build or dependency changes; an unverified clean implementation does not inherit the overlay prototype's pass.

When slimming the implementation PR, non-essential diagnostic tools, docs and experiment artifacts stay in the private repo/model workspace; necessary regression tests can go with the PR. Do not delete tests that exist on the target branch, and a slim PR still runs tests. Follow each target repo's contribution requirements, record the regressions and model evaluations actually run, and submit for human review; if the gates do not pass, keep the not-accepted status.

### Reference versions, not proof of passing

All three PRs were open when reviewed, and the first two were marked DO NOT MERGE:

| Reference | Reviewed head SHA |
| --- | --- |
| vLLM #54955 | `4b5c627c23e6690b8987100d1cbf78e34ac14bf9` |
| Megatron #7050 | `327a238239448de6771beea185dc6902a8de3eaa` |
| verl #7688 | `640c5f0a317307e99636924804c40624242fa19b` |

A PR body may describe a different historical pin; when citing the design, distinguish the reviewed head, the experiment version the body claims and the version measured now, and do not treat the reference as fully correct or merged. Its Triton implementation also does not override this workflow's constraints on added/replaced kernels.

## 9. PRs and cleanup

- Three-repo delivery follows the target branches in §8.
- The PR description records: minimal purpose, base and patch stack, related existing PRs/why it is not a duplicate, upstream/private PRs, RL-only content, test commands and results, model impact, rollback method, AI assistance and human review requirements.
- Split a large PR into small independently reviewable PRs and prove equivalence before and after the split with the tree hash; open PRs only from your own fork.
- Resolve conflicts on the head branch of the PR the user specified; do not open a similarly named new PR to pass the original off as fixed.
- Cross-repo order is usually kernel/runtime → framework integration → recipe/pins → final image acceptance, with each later stage locking the actual revisions of the earlier ones.

After a patch is merged and a new image is built, archive the worktree, delete the temporary overlay, and keep the PR, commit, manifest and verification report.

## Minimal changes, contribution and delivery

Change the config along the entry script the user specified, so multiple temporary launchers do not drift apart. End with one launch entry that actually runs, with defaults, necessary overrides and image/code pins traceable; exports that exist only in chat are not delivery.

A fix lives with the state/feature owner: fix vLLM backend/selector problems in vLLM, not with a global monkeypatch in the training framework; a numerically matched forward must keep the real backward. Call existing reusable functions instead of building synonymous logprob metrics and compatibility switches.

An authorized independent audit does not transfer responsibility to the auditor. Before declaring completion, check: are all of the user's goals covered? Do the results belong to the final code/image? Does the wording hide any untested item? If the evidence is insufficient, write down the gap; do not wrap it in a percentage or a success marker.
