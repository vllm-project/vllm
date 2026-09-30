---
name: pr-checklist
description: Prepare a vLLM change for human PR review or re-review, or analyze an open pull request. Use to check design fit, behavioral coverage, performance evidence, diff quality, and closure of previous feedback before requesting maintainer attention.
---

# PR Checklist

This SKILL is designed to help systematically enforce vLLM's code quality and contribution standards during the pull request process,
and to reduce the burden of manual code review on maintainers.

## Usage

The following checklist can be used to polish a change in preparation for a vLLM pull request, analyze an open pull request (either as the author or as a reviewer), or ensure that all contribution standards are met before requesting maintainer attention. To use this checklist effectively, go through each item and verify that it has been addressed in the diff / pull request.

This is intended to be used with a human in the loop: when running programmatically, prepare notes on every section to ensure a thorough report, with a clear header summarizing the findings and areas needing attention. When running interactively, present this report to the user with a clear list of actionable follow-up tasks and design decisions to make, then iterate.

### Re-Review

To update a PR following a review, read all comments and threads on the PR. For each concrete concern, work with your human operator to decide on a plan of action to address each either by a direct reply or by updating the change and acknowledging with a comment. Ensure the PR description remains fresh and all relevant documentation (including comments) in the change is up-to-date.

`gh pr view --comments` does not include inline review comments. Set `pr` to the PR number, then fetch all pages of conversation comments, review submissions, and inline comments (including replies):

```bash
gh api --paginate "repos/vllm-project/vllm/issues/$pr/comments"
gh api --paginate "repos/vllm-project/vllm/pulls/$pr/reviews"
gh api --paginate "repos/vllm-project/vllm/pulls/$pr/comments"
```

Use `in_reply_to_id` to group inline replies with their parent comments. Use GraphQL `reviewThreads` when resolved or outdated thread status is needed.

## Sections

### 1. Design Fit

Does the change align with the overall design and architecture of vLLM? Does it introduce any design inconsistencies or anti-patterns? Are there any potential long-term maintenance concerns related to this change?

#### 1.1: Impact on Core Components

vLLM's core components are shared across many different deployments. Changes to these files have high risk and changes should be structured in order to be easily maintainable and minimally invasive.

Such components include:

- The Model Runner (`vllm/v1/worker/*/model_runner.py`)
- The Scheduler (`vllm/v1/core/sched/*`)
- Core infrastructure (`vllm/v1/core/*`, including KV cache managers and utilities)

etc.

Changes should be structured to minimize complexity added to these files. Consider refactoring or isolating new functionality to reduce the impact on core components, both in terms of code maintainability and potential for introducing bugs.

Nontrivial logic associated with non-standard/non-default functionality should not be added to core/common code paths and files.

#### 1.2: Replication

Some changes may duplicate code that already exists elsewhere in the codebase. Does the change introduce unnecessary replication, or could existing functionality be reused instead? Consider refactoring to reduce duplication and improve maintainability.

Consult the affected area's agent instructions and contribution guides for its reuse conventions. See the [model implementation guide](../../../docs/contributing/model/basic.md) for model code and the [MRV2 development guidance](../../../docs/design/model_runner_v2.md#development-philosophy). Justify any duplication needed to isolate functionality and limit its impact on existing workflows.

#### 1.3: Complexity

Changes which increase the complexity of the codebase should be carefully scrutinized. Specifically those which add new abstractions, dependencies, or branching cases to existing components should be evaluated for their necessity and potential impact on maintainability.

Consider whether the added complexity could be mitigated through refactoring, modularization, or other design improvements. Aim to keep the codebase as simple and understandable as possible while still achieving the desired functionality.

As a rule of thumb, the complexity of a change can be estimated by the number of new lines of code added to the main codebase:

- Low complexity: < 50 lines. Generally easy to review and unlikely to introduce significant complexity.
- Medium complexity: 50-250 lines. Requires careful review and consideration of potential impact on maintainability.
- High complexity: > 250 lines. May introduce significant complexity and requires thorough review and justification.

You may exclude test cases and documentation updates from this line count, as they typically do not contribute significantly to the complexity of the main change. Note that line count is only a coarse indicator, meant to guide and flag possible overly-complex contributions. Use judgement when assessing the true semantic complexity and implied review burden.

Bug fixes should aim to remain below the "High complexity" bar. Those which exceed this threshold often warrant additional scrutiny and justification. Redesigning core components, modifying core interfaces, or otherwise introducing significant complexity in order to patch a bug is very often discouraged and should be carefully justified.

A PR that adds significant complexity or a substantial amount of new code must be well justified. A new feature should be relevant to, or requested by, enough users to warrant its size and complexity. A performance optimization should deliver end-to-end improvements on workloads that matter, and those gains should be large enough to outweigh the complexity it adds.

#### 1.4: Correctness and Compatibility

Ensure that all changes to vLLM are sound and do not violate existing contracts. Specifically, verify that all affected deployments which share an execution pathway with the changed code will remain correct. Trace related implementations, alternative backends sharing the same flow, and affected call sites of modified functions.

When adding features or changing contracts, ensure that relevant feature compatibility is assessed. Identify any potential incompatibilities or breaking changes and justify them explicitly in the PR notes.

When adding functionality, consider parallels with existing designs and identify opportunities to simplify and unify them rather than adding a separate implementation alongside them.

#### 1.5: Tradeoffs

When making a design change, adding a feature, or contributing a performance improvement, assess the tradeoffs and potential downsides. When changing a default backend to improve performance as measured on one configuration, could this cause a decrease in performance for another configuration? Could a new architecture change accelerate one workload but cause a feature incompatibility or performance regression in another? Ensure tradeoffs are clearly noted and validation is sufficient to catch potential lurking regressions or drawbacks.

Check that the change does not introduce CPU/GPU synchronization on the main model execution thread during steady-state inference. Inspect explicit synchronization, host reads of GPU tensors, boolean tensors as indices, and transfers that can block the host. Run the affected configuration with `VLLM_GPU_SYNC_CHECK=error` to confirm a concern; passing CI is insufficient when CI does not exercise that configuration. See the [MRV2 async execution design](../../../docs/design/model_runner_v2.md#2-async-first) for examples.

#### 1.6: Model Runner V1 Deprecation

Account for Model Runner V1 (MRV1) deprecation when reviewing model-runner changes. Target new GPU model-runner features at Model Runner V2 (MRV2) and follow its [design and development guidance](../../../docs/design/model_runner_v2.md#development-philosophy). For fixes needed by deployments still using MRV1, explain why the legacy path needs the change and check whether MRV2 is also affected.

### 2. Testing and Validation

All serving-engine changes contributed to vLLM must be _validated_: bug fixes must be tested on a reproducer, and new features or performance optimizations must be run locally and/or benchmarked to ensure correctness and effectiveness. It is unacceptable, for example, to contribute a bugfix patch without running vLLM end-to-end and verifying that it resolves the issue as intended.

Beyond _validation_, many (but not all) changes to vLLM should be _tested_: this includes writing new tests, updating existing tests, and ensuring that the test suite continues to provide reliable coverage of the affected functionality.

#### 2.1: Test Coverage and Effectiveness

When adding or modifying tests, consider both the coverage and the effectiveness of the test suite. Coverage refers to the extent to which the codebase is exercised by tests, while effectiveness refers to the ability of the tests to catch regressions and ensure correctness.

Assess whether the included test cases are meaningful, concise, and maintainable. Testing should not be exhaustive: local validation is typically sufficient to ensure a baseline correctness of the change. Contributed tests should focus exclusively on areas that are deemed prone to errors, regressions, or complex interactions within the codebase. Recommend deletion of redundant, trivial, or otherwise unnecessary tests: this is one of the most common complaints raised during code reviews.

Verify that testing actually exercises the new path, rather than passing through an unchanged path or fallback. Ensure added speculative decoding correctness tests assert nonzero acceptance rate, selection tests identify the desired backend, etc.

Aim to write tests that are meaningful, targeted, and maintainable. Avoid redundant or trivial tests that do not contribute to the overall reliability of the test suite. Avoid using mock objects, especially when highly coupled with internal logic.

#### 2.2: Test Reliability and Robustness

Some tests are considered "flaky": they may pass or fail intermittently without any changes to the codebase. Flaky tests can undermine confidence in the test suite and block successful CI runs. Analyze the added tests for potential flakiness and ensure they are reliable and robust.

Frequent sources of flaky tests include:

- Assertions requiring bit-exact output correctness when minor floating-point variation is expected. For example, a change in default kernel selection priority may cause exact-floating-point-correctness assertions to fail.
- Tests that depend on the order of execution or the state of shared resources, which can lead to intermittent failures when tests are run in parallel or in different environments.

#### 2.3: Test Automation and Integration

vLLM's CI runs in Buildkite, with the launch configuration under `.buildkite/`. Ensure new tests run in an appropriate CI workflow when the required resources are available; add or update a workflow where needed. If CI cannot exercise the changed path because the required hardware or checkpoint is unavailable, document the missing coverage, the resource constraint, and validation performed outside CI. A CI coverage gap does not waive validation: the human submitter must confirm the required validation before publishing the PR.

### 3. Code Quality and Style

Code quality is another of the primary areas that maintainer effort is spent to address. Carefully audit the changes for adherence to coding standards, readability, maintainability, and overall design principles.

#### 3.1: Comments

Keep comments concise and use them sparingly. Consider whether the same information can be conveyed in fewer lines, and prefer clear code with meaningful, concise variable names, and fewer comments overall. Use comments to explain the reasoning behind complex logic, non-obvious decisions, or important context that is not immediately apparent from the code itself. Context regarding specific user decisions made during the design phase are often not needed and can instead be included as notes in the PR description instead.

When writing new functions, consider when a docstring may be useful. Most short helper functions do not need a docstring, but more complex functions or those forming part of the public API should include one. In particular, docstrings are useful when adding custom kernels or other performance-optimized implementations of otherwise-straightforward logic, as they make clear the input and output shapes/dtypes/assumptions, and can shed light on hard-to-follow implementations.

For bug fixes, there is no need to include details of the original bug and explanation of why the pre-changed version of the code would be incorrect. Those details should be in the PR description instead.

#### 3.2: Documentation and Examples

Ensure that any new features, changes, or public APIs are properly documented. This includes updating relevant documentation files under `docs/` and providing usage examples where applicable. Refer to `supported_models.md` for new models and `examples/` for usage examples.

#### 3.3: Helper Functions

Identify and inline trivially-simple helper functions which have only one call site.

#### 3.4: Formatting

Avoid formatting or whitespace changes that are unrelated to the PR, as long as pre-commit still passes.

### 4. Pull Request Contents

This section pertains to the contents of the pull request itself, rather than the changes.

When acting as the author, prepare or update these contents. When acting as a reviewer, assess them and report missing context or evidence.

Start from `.github/PULL_REQUEST_TEMPLATE.md`, retaining its sections and checklist. When creating a PR interactively, use `gh pr create --template .github/PULL_REQUEST_TEMPLATE.md`. For noninteractive creation or editing, fill out a copy of that template and pass the completed file with `--body-file`.

#### 4.1: Description and Context

Ensure the PR description has a short 1-2 line summary of the changes to facilitate quick understanding by reviewers. This section should link relevant issues, predecessor/blocking PRs, and provide any necessary context for the changes.

Keep Claims to a few concrete bullets and explain motivation briefly. Keep explanations in Validation and Details concise too, while retaining the evidence, root cause, tradeoffs, and limitations needed to assess the change. Link full logs or use collapsible HTML details for lengthy supporting material rather than polluting the main narrative.

#### 4.2: Claims and Supporting Evidence

Ensure the PR makes clear, concise claims about the accomplishment:

- For bug fixes, what specific issue is being resolved and how can it be verified?
- For new features, what functionality is being added and how can it be utilized?
- For refactoring, what was the motivation and what improvements are expected?
- For performance optimizations, what specific model/hardware/serving configurations & workloads are being targeted and how can the performance gains be measured?

For the stated claims, ensure the PR provides sufficient evidence, examples, or references to support them, making it easy for reviewers to verify the correctness and impact of the changes.

Summarize model evaluations and benchmarks in compact tables with the configuration, baseline, changed result, and reproduction commands. For repeated benchmarks, report the run count and variability. For kernel performance changes, include before/after profile visualizations/timelines when they help explain the measured gains. Link relevant CI failures or regressing PRs when they motivate the change.

- For bug fixes, ensure the reproducer is clearly provided and was validated to confirm the issue and the resolution.
- For new features, ensure that usage examples are included and validated to confirm that the feature works as intended.
- For refactoring, ensure that changes are sufficiently validated at appropriate test levels (unit is sufficient for minor refactors, E2E serving is required for significant refactoring).
- For performance optimizations, ensure that benchmarks or tests are included to validate the claimed performance improvements. Ideally, an end-to-end evaluation should demonstrate speedup on a concrete serving example. Microbenchmark-only validation is often insufficient: make a note if end-to-end evaluation is not included (e.g. due to lack of access to sufficient hardware to fully validate).

#### 4.3: Root-Cause Analysis

When fixing bugs or issues, provide a clear root-cause analysis. Explain what caused the problem, how it was identified, and why the chosen solution effectively addresses it. A patch which resolves the issue without a proper root-cause analysis may lead to recurring problems and increased maintenance overhead.

#### 4.4: Implementation Details

Provide a detailed explanation of the implementation, including design decisions, trade-offs, and any relevant technical details. Highlight any non-obvious choices, tradeoffs, potential limitations, and areas where maintainer attention is required.

#### 4.5: Contributing Guide

Ensure that the PR follows the [contributing guide](https://docs.vllm.ai/en/latest/contributing), including adhering to coding standards, commit message conventions, and any other project-specific guidelines. This helps maintain consistency and quality across the codebase.
