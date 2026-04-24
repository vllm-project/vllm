---
name: add-test-docs
description: Plan or backfill concise test documentation entries under `docs/cohere/tests/`. Use `plan` mode to iteratively clarify `How it runs`, `Checks`, `Measurements`, `Compatibility`, and `Implementation` with the user; use `backfill` mode to update docs from an existing test implementation.
---

# Add Test Docs

Create or update test documentation in `docs/cohere/tests/` with a compact
multi-test-case format that works for TDD-first planning and later
implementation.

## When to Use

- When adding a new test entry to a doc under `docs/cohere/tests/`
- When the user wants test docs written before any implementation starts
- When the user wants to iteratively design a test case before writing code
- When the user wants to backfill docs from an existing test implementation
- When updating a feature doc to reflect new or changed test coverage

## Modes

Choose one of these two modes before writing or updating the doc:

### `plan`

- Use before writing the test implementation.
- Work iteratively with the user to clarify each bullet under:
  - `## How it runs` -- execution flow and invocation
  - `## Checks` -- assertions the test gates on
  - `## Measurements` -- artifacts uploaded by the CI workflow
  - `## Compatibility` -- feature-matrix classification
- Draft `## Implementation` with `### Setup` to anchor the test envelope.
- Draft an initial `## Compatibility` classification and review it with the
  user. Default unknown features to **N/A** and let the user promote them.
- Use planned repo paths in backticks until code exists.

### `backfill`

- Use when the test already exists and the doc needs to be updated to match it.
- Read the existing test, runtime, and workflow code first.
- Replace planned wording with concrete markdown links and actual names.
- Under each `## How it runs` bullet, add indented child bullets linking to the
  code that implements that step.
- Under each `## Checks` bullet, add indented child bullets naming the test
  entries (pytest functions) that verify that assertion.
- Under each `## Measurements` bullet, add indented child bullets naming the
  benchmark metric fields and the workflow step that uploads them.
- Infer `## Compatibility` from the test code (models, quantizations, hardware
  markers, engine features) and confirm the classification with the user.
- Fill `## Implementation` > `### Setup` from the implementation.

## Default Test Location

- By default, new Cohere test code should go under `tests/cohere/`.
- If the user does not specify a location, plan the test under `tests/cohere/`.
- Only choose a different test location when the user explicitly requests it or
  the existing test layout for that feature clearly requires it.

## Test Pipeline Integration

`docs/cohere/code_notes/ci-and-automation.md` contains a
**Test Pipeline Integration** section (section 7) that is the single source of
truth for:

- **JUnit XML reporting** -- which test groups must use `pytest`, how the
  `conftest.py` hook emits XML, and what breaks when XML is missing.
- **Hardware profiles** -- how `hardware_profiles.yaml` is applied, what
  `VLLM_HARDWARE_PROFILE_ARGS` does, and when tests may override profile
  values.
- **Nightly metric emission** -- how `nightly-benchmark.yaml` fans out jobs,
  which artifacts are uploaded, and which branch (`gh-pages` vs `ci_dump`)
  receives them.

Read that section before planning or backfilling any test doc that touches CI
reporting, GPU-specific config, or benchmark artifact upload.

## Observability Matrix Format

The numbering scheme, entry templates, and category headings used in
`docs/cohere/tests/observability_matrix.md` are documented in its
[Entry Format](docs/cohere/tests/observability_matrix.md#entry-format) section.
Read that section before adding or updating entries.

## Feature Matrix Integration

`docs/cohere/tests/feature_matrix.md` tracks cross-feature compatibility.

- The top of the file contains **empty template tables** listing every feature
  category (Input, Cohere Feature, Model Architecture, Quantization,
  Hardware, vLLM Feature).
- Under `## Features`, each documented feature has its own copy of those tables
  with cells filled in to record which test cases verify compatibility.
- Cell values: `T.<category>.<feature>.<seq>` (compatible, verified by that
  observability-matrix test case), `❌` (not compatible), or blank (not checked).
- When writing or updating a `## Compatibility` section in a feature doc,
  always propagate the classification to the matching per-feature tables in
  `feature_matrix.md` as a final step.
- If a feature section does not yet exist under `## Features`, create one by
  copying the template tables and filling in cells from the new test case's
  compatibility classification.

## Required Output Shape

Each feature doc may contain one or more numbered test cases.

Track per-feature status in `docs/cohere/tests/observability_matrix.md`
instead of adding a status table inside each feature doc.

For each test case, use these sections in this order:

1. `## How it runs`
2. `## Checks`
3. `## Measurements`
4. `## Compatibility`
5. `## Implementation`
   - `### Setup`

Wrap each test case in a toggle:

```markdown
<details>
<summary>Test case 1: Short label</summary>

...

</details>
```

Default toggle behavior:

- Use collapsed `<details>` blocks by default.
- Do not add the `open` attribute unless the user explicitly wants a test case
  expanded by default.

## Section Rules

### `## How it runs`

Describes the execution flow: invocation, runtime path, toggles, fixtures,
and CI routing.

- Use a short numbered list.
- Each bullet describes one step or aspect of the execution flow.
- **Plan mode**: describe the intended execution shape. Use planned repo paths
  in backticks when code does not exist yet.
- **Backfill mode**: under each bullet, add indented child bullets with markdown
  links to the code that implements that step. Example:

  ```markdown
  1. Runs two sequential generations with `VLLM_USE_LOGITS_FP32_COMPUTATION=0`
     then `=1`.
     - [`tests/cohere/unit/test_c5_fp32_logits.py`](...)
  ```

- When CI entry/reporting behavior matters, consult the
  **Test Pipeline Integration** section (section 7) of
  `docs/cohere/code_notes/ci-and-automation.md`.

### `## Checks`

This section is exclusively about **test assertions** -- the conditions the test
gates on to pass or fail.

- Prefer 1-3 short items.
- Use a short numbered list.
- Each item should describe one assertion or correctness condition the test
  enforces (e.g. token-match threshold, dtype equality, tolerance bound).
- Bold a few keywords when it improves scanning, such as the core assertion,
  dtype, tolerance, or failure condition.
- Do not include uploaded artifacts that the test emits but does not assert on
  -- those belong in `## Measurements`.
- Avoid restating setup or runtime flow unless it is the assertion itself.
- **Plan mode**: describe the intended assertions. Use planned repo paths in
  backticks when code does not exist yet.
- **Backfill mode**: under each bullet, add indented child bullets naming the
  pytest test entry (function name) that verifies that assertion. Example:

  ```markdown
  1. Fails on **token ID mismatch** when the shared prefix is shorter than
     `min_shared_prefix`.
     - `test_c5_fp32_logits_consistency`
  ```

### `## Measurements`

This section covers **artifacts that the CI workflow uploads** -- only outputs
that leave the runner via an upload step in `test-pipeline.yaml` (e.g.
`upload-results` action, `upload-artifact`, GCP upload script). Anything the
test already asserts on belongs in `## Checks`, not here. Files that are only
written locally (e.g. detailed debug logs, per-sample generation dumps) are
**not** measurements and should not be listed.

- Prefer 1-3 short items unless there are multiple distinct uploaded artifacts.
- Use a short numbered list.
- Describe each uploaded artifact: the file name, the upload destination
  (target branch path, GCP bucket, or artifact name), and the workflow step
  that performs the upload.
- Bold a few scan-worthy outputs when helpful, such as key metric names,
  artifact names, or CI-visible surfaces.
- If a test emits both an asserted metric and a non-asserted uploaded metric,
  put the asserted one in Checks and the uploaded one here.
- **Plan mode**: describe the intended upload artifacts. Use planned repo paths
  in backticks when code does not exist yet.
- **Backfill mode**: under each bullet, add indented child bullets naming the
  benchmark metric fields and their expected patterns, plus a link to the
  workflow step that uploads them. Prefer shared expectation codes from
  [Metric Pattern Codes](docs/cohere/tests/observability_matrix.md#metric-pattern-codes).
  Example:

  ```markdown
  1. Uploads **`unit_results_summary.json`** to `unit_data/summary` on
     `ci_dump` via `upload-results` action.
     - `bf16_median_ms` -- `LOWER-ANCHOR3+5%`
     - `fp32_median_ms` -- `LOWER-ANCHOR3+5%`
     - [`.github/workflows/test-pipeline.yaml`](...) -- "Upload model architecture results" step
  ```

- When CI/reporting behavior matters, consult the
  **Test Pipeline Integration** section (section 7) of
  `docs/cohere/code_notes/ci-and-automation.md`.
- If the test has no uploaded artifacts, this section may say `N/A`.

### `## Compatibility`

- Use a numbered bullet list with one bullet per category from
  [`docs/cohere/tests/feature_matrix.md`](docs/cohere/tests/feature_matrix.md):
  1. **Input**, 2. **Cohere Feature**, 3. **Model Architecture**,
  4. **Quantization**, 5. **Hardware**, 6. **vLLM Feature**.
- Under each category bullet, list only the features that are **compatible** or
  **not compatible** with this test case. Use `(compatible)` or
  `(not compatible)` labels after each feature name. Omit features that are N/A
  (irrelevant to the test).
- If every feature in a category is N/A, keep the category bullet but leave the
  body empty (no child bullets).
- Consult the **Compatibility Sources** table in
  [`docs/cohere/tests/feature_matrix.md`](docs/cohere/tests/feature_matrix.md)
  for where to check each category.
- In `plan` mode, list the numbered categories and leave the child bullets
  empty / TBD. Review with the user during planning.
- In `backfill` mode:
  - Infer compatibility from the test code, `runner_map.json` (Hardware), and
    `hardware_profiles.yaml` (vLLM Feature).
  - Add a child bullet for each compatible or not-compatible feature, linking to
    the source that confirms it (test file, `runner_map.json`, or
    `hardware_profiles.yaml`). Omit the link when the feature is obviously
    covered by the primary test and the link would be redundant.
  - Confirm the classification with the user before finalizing.
- After the compatibility section is settled, update the corresponding
  per-feature tables under `## Features` in
  `docs/cohere/tests/feature_matrix.md`:
  - **Compatible**: fill the cell with `T.<category>.<feature>.<seq>` using the
    test case's observability-matrix ID.
  - **Not compatible**: fill the cell with `❌`.
  - N/A or unchecked: leave the cell blank.
- If the feature does not yet have a section under `## Features` in
  `feature_matrix.md`, create one by copying the empty template tables from the
  top of the file and filling in the compatibility cells.

### `## Implementation`

Contains `### Setup` and optional implementation-choice notes. Code links,
test entries, and benchmark metrics are inlined under `## How it runs`,
`## Checks`, and `## Measurements` respectively (see those rules above).

- Open with the primary test file and relevant runtime/workflow file references
  (or planned paths in backticks when code does not exist yet).
- Once code exists, include concise notes about implementation choices.
- During planning, prefer repo paths in backticks instead of markdown links
  when jotting down provisional locations.

#### `### Setup`

- Keep this subsection concise.
- Use a short numbered list.
- Describe the execution assumptions and knobs for the test case.
- Include env vars, vLLM engine features/options, execution backend, fixtures,
  model/runtime mode, and the input cases or shapes the test should check
  against.
- By default, respect hardware profiles as described in the
  **Hardware Profiles** subsection of
  `docs/cohere/code_notes/ci-and-automation.md` (section 7).
- During planning, use this subsection to pin down the intended test envelope.
- If code exists, point to setup code or config with markdown links.
- If code does not exist yet, refer to planned files, flags, env vars, config,
  or tensor/input shapes in backticks.

## Plan Mode Workflow

Use `plan` mode as an iterative design loop before writing code.

1. Identify the feature doc to update under `docs/cohere/tests/`.
2. Ask the user enough questions to design one concrete test case at a time:
   execution flow, assertions, expected outputs, setup assumptions, and any
   special toggles or fixtures.
3. Ensure the feature is represented in
   `docs/cohere/tests/observability_matrix.md` under the matching `###`
   category heading. See
   [Entry Format](docs/cohere/tests/observability_matrix.md#entry-format) for
   the numbering scheme and entry templates.
4. Draft `How it runs` first to anchor the execution design, then draft
   `Checks` and `Measurements`. Keep all three terse in the first pass.
5. For `Compatibility`, list the six numbered categories from
   `docs/cohere/tests/feature_matrix.md` and draft compatible / not-compatible
   features under each. Leave categories empty when all features are N/A.
   Default unknown features to N/A and present the draft to the user for
   review.
6. Draft `Implementation` with `### Setup` to pin down the test envelope
   (env vars, engine flags, fixtures, input shapes).
7. Review the draft with the user and refine each bullet iteratively until
   all sections are clear enough to implement. Pay particular attention to the
   compatibility classification -- the user may promote features from N/A to
   Compatible or Not compatible as the design clarifies.
8. Default new test code to `tests/cohere/` unless the user or existing layout
   clearly requires a different location.
9. When the test will be surfaced in CI, consult the
   **Test Pipeline Integration** section (section 7) of
   `docs/cohere/code_notes/ci-and-automation.md` to anchor `How it runs` and
   `Measurements` to the correct workflow and upload path.
10. When planning benchmark expectations, prefer shared pattern codes from
    `docs/cohere/tests/observability_matrix.md` instead of repeating the full
    rule in prose.
11. Use planned paths in backticks when implementation does not exist yet.
12. Only after the user is satisfied with the planned test case should coding
    begin.
13. Update `docs/cohere/tests/observability_matrix.md` so it stays in sync
    with the feature doc. Add bullet-list entries under the matching category
    heading for each new test function (under `## Tests`) and each new metric
    (under `## Benchmarks`), following the
    [Entry Format](docs/cohere/tests/observability_matrix.md#entry-format).
14. As the final step, propagate the `Compatibility` classification to the
    per-feature tables under `## Features` in
    `docs/cohere/tests/feature_matrix.md`. Use
    `T.<category>.<feature>.<seq>` for compatible, `❌` for not compatible,
    and leave blank for N/A. Create a new feature section if one does not exist
    yet (see [Feature Matrix Integration](#feature-matrix-integration)).

## Backfill Mode Workflow

Use `backfill` mode when code already exists and the doc should be updated to
match the implementation.

1. Identify the feature doc to update under `docs/cohere/tests/`.
2. Identify whether the doc already has other test cases; if so, append the next
   numbered case instead of reshaping the whole page.
3. Read the existing test, runtime, and workflow files needed to understand the
   implementation.
4. Read `docs/cohere/tests/observability_matrix.md` to find the feature's
   existing entries (if any) and identify the next available
   `<category>.<seq>` IDs under the matching `###` category headings. See
   [Entry Format](docs/cohere/tests/observability_matrix.md#entry-format) for
   the numbering scheme and entry templates.
5. Add a collapsed `<details>` block with a summary like
   `Test case N: <short label>`.
6. Write `## How it runs` from the actual invocation and runtime flow. Under
   each bullet, add indented child bullets with markdown links to the code that
   implements that step.
7. Write `## Checks` from the actual assertions in code. Under each bullet,
   add indented child bullets naming the pytest test entry (function name) that
   verifies that assertion.
8. Write `## Measurements` from the artifacts that the CI workflow uploads
   (via `upload-results`, `upload-artifact`, or GCP upload scripts in
   `test-pipeline.yaml`). Under each bullet, add indented child bullets naming
   the metric fields and the workflow step that uploads them. Write `N/A` if
   the test has no CI-uploaded artifacts. Do not list files that are only
   written locally.
9. Prefer shared pattern codes from
   `docs/cohere/tests/observability_matrix.md` for benchmark expectations
   whenever one applies; use concise prose only for new or one-off contracts.
10. Write `## Compatibility` by inferring feature compatibility from the test
    code (model architectures loaded, quantization formats exercised, hardware
    skip markers, engine features enabled/disabled), `runner_map.json`
    (Hardware), and `hardware_profiles.yaml` (vLLM Feature). List the six
    numbered categories and place compatible / not-compatible features under
    each; leave categories empty when all features are N/A. Present the
    inferred classification to the user for confirmation before finalizing.
11. Write `## Implementation` with concrete markdown links to the primary test
    file and relevant runtime/workflow files, plus `### Setup` from the actual
    env vars, engine features, fixtures, and input cases/shapes.
12. When CI visibility matters, consult the
    **Test Pipeline Integration** section (section 7) of
    `docs/cohere/code_notes/ci-and-automation.md` to align `How it runs` and
    `Measurements` with the current workflow and upload paths.
13. Keep wording compact; avoid long rationale, background, or duplication.
14. Update `docs/cohere/tests/observability_matrix.md` so it stays in sync
    with the feature doc, following the
    [Entry Format](docs/cohere/tests/observability_matrix.md#entry-format):
    - Under `## Tests`, add one bullet per test entry name found as a child
      bullet under `## Checks` in the feature doc. Place the bullet under the
      matching `###` category heading and assign the next sequential
      `<category>.<seq>` ID.
    - Under `## Benchmarks`, add one bullet per metric name found as a child
      bullet under `## Measurements` in the feature doc, including the expected
      pattern code. Place the bullet under the matching `###` category heading
      and assign the next sequential `<category>.<seq>` ID.
15. As the final step, propagate the confirmed `Compatibility` classification to
    the per-feature tables under `## Features` in
    `docs/cohere/tests/feature_matrix.md`. Use
    `T.<category>.<feature>.<seq>` for compatible, `❌` for not compatible,
    and leave blank for N/A. Create a new feature section if one does not exist
    yet (see [Feature Matrix Integration](#feature-matrix-integration)).

## Reference Style

- Prefer relative markdown links that resolve from the current doc.
- Use repository paths as the link text, wrapped in backticks.
- When both are relevant, link to the test file and the runtime file.
- When CI behavior or benchmark expectations are part of the design, reference
  `docs/cohere/code_notes/ci-and-automation.md` (section 7) for Test Pipeline
  Integration (JUnit XML, hardware profiles, metric upload) and
  `docs/cohere/tests/observability_matrix.md` for Metric Pattern Codes.
- If code does not exist yet, do not create fake links.
- Reference intended repo paths and symbols in backticks and mark them as
  planned rather than inventing behavior.

## Output Template

Use this template for each test case:

Plan-mode template (no code yet):

```markdown
<details>
<summary>Test case 1: Short label</summary>

## How it runs

1. Planned execution step with intended entrypoints in `tests/...`.
2. Planned execution step with intended flags, env vars, or runtime
   touchpoints in `tests/...` and `vllm/...`.
3. Planned CI entry/reporting path -- see
   [Test Pipeline Integration](../../code_notes/ci-and-automation.md#7-test-pipeline-integration).

## Checks

1. Planned assertion the test gates on, with intended files or symbols in
   `tests/...` and `vllm/...`.
2. Planned assertion the test gates on, with intended files or symbols in
   `tests/...`.

## Measurements

1. Planned CI-uploaded artifact: summary file, benchmark JSON, or metric
   export, with intended upload destination and workflow step in
   `.github/workflows/...`.

N/A when the test has no CI-uploaded artifacts.

## Compatibility

Features from [Feature Matrix](../feature_matrix.md)
([Compatibility Sources](../feature_matrix.md#compatibility-sources)):

1. **Input**:
2. **Cohere Feature**:
3. **Model Architecture**:
4. **Quantization**:
5. **Hardware**:
6. **vLLM Feature**:

## Implementation

Planned test: `tests/cohere/...`
Planned runtime path: `vllm/...`

### Setup

1. Planned env vars, engine flags, backend choices -- respecting hardware
   profiles per
   [Hardware Profiles](../../code_notes/ci-and-automation.md#hardware-profiles).
2. Planned `torch.compile` / CUDA-graph usage.
3. Planned fixture shape, model mode, or input cases in `tests/...`.

Note: Backfill code links, test entries, metric names, and compatibility
classifications after the code lands. Update `feature_matrix.md` tables once
the observability-matrix ID is assigned.

</details>
```

Backfill-mode template (code exists):

```markdown
<details>
<summary>Test case 1: Short label</summary>

## How it runs

1. Description of execution step.
   - [`tests/cohere/unit/test_example.py`](...)
   - [`vllm/module/path.py`](...)
2. Description of another step.
   - [`tests/cohere/unit/test_example.py`](...)

## Checks

1. Description of **assertion** the test gates on.
   - `test_example_function_name`
2. Description of another **assertion**.
   - `test_example_function_name`

## Measurements

1. Uploads **`summary_file.json`** to `target/path` on `ci_dump` via
   `upload-results` action.
   - `metric_field_name` -- `PATTERN-CODE`
   - [`.github/workflows/test-pipeline.yaml`](...) -- "Upload step name" step
2. Uploads another artifact to GCP / artifact store.
   - `other_metric` -- `PATTERN-CODE`

N/A when the test has no CI-uploaded artifacts.

## Compatibility

Features from [Feature Matrix](../feature_matrix.md)
([Compatibility Sources](../feature_matrix.md#compatibility-sources)):

1. **Input**: Basic (compatible)
2. **Cohere Feature**:
3. **Model Architecture**: C5 Arch (compatible)
   - [`tests/cohere/unit/test_example.py`](...)
4. **Quantization**:
5. **Hardware**: H100, B200 (compatible); MI300x (not compatible)
   - [`tests/cohere/configs/runner_map.json`](...)
6. **vLLM Feature**: Torch Compile (not compatible)
   - [`tests/cohere/configs/hardware_profiles.yaml`](...)

## Implementation

Primary test: [`tests/cohere/unit/test_example.py`](...)
Runtime path: [`vllm/module/path.py`](...)

### Setup

1. Description of env vars, engine flags, or fixtures with links.
2. Description of input cases or shapes.

</details>
```

## TDD-First Notes

- The doc entry can be created before the full test implementation lands.
- Do not pretend unimplemented checks already exist.
- Keep per-feature status tracking in `docs/cohere/tests/observability_matrix.md`
  rather than inside each feature doc.
- Status cells in `observability_matrix.md` may be left blank during planning
  and backfilled later.
- For planned coverage, use wording like "Planned in `tests/cohere/...`" and
  keep claims scoped to the intended test shape.
- Draft `How it runs` first to anchor the execution design, then `Checks` and
  `Measurements`. Keep all three terse in the first pass; expand `### Setup`
  when extra detail is needed.
- `Checks` is strictly for assertions the test gates on. `Measurements` is
  strictly for artifacts that the CI workflow uploads (not local-only files). If
  a value is asserted, it belongs in Checks; if it is uploaded but not asserted,
  it belongs in Measurements. When there are no CI-uploaded artifacts,
  Measurements may say `N/A`.
- In backfill mode, every bullet in How it runs, Checks, and Measurements gets
  indented child items: code links under How it runs, test entry names under
  Checks, metric names and pattern codes under Measurements.
- In plan mode, bullets have no child items -- just the intended behavior.
- For `Compatibility`, default unknown features to **N/A** during planning and
  let the user promote them as design details clarify. Update
  `feature_matrix.md` only after the classification is confirmed.
- Use bold sparingly in `Checks` and `Measurements` to surface the most
  important words without turning whole sentences into emphasis.
- For benchmark contracts, prefer the shared expectation codes in
  `docs/cohere/tests/observability_matrix.md` over repeating long definitions in
  each feature doc.
- By default, planned new test files should live under `tests/cohere/`.
- Once the test exists, replace planned wording with concrete code references
  and add child items under each bullet.
- If multiple planned or implemented test cases exist, keep one toggle per case
  and number them in reading order.

## Collaboration Style

- Treat the doc as a planning artifact first, not just a record of existing
  code.
- If important design details are missing, ask targeted follow-up questions
  instead of guessing.
- Prefer converging on one test case at a time.
- When helpful, present a draft and explicitly ask what the user wants to
  change before moving on to implementation.

## Example Prompt Matches

```text
Add a test doc entry for this new feature before we write the test
```

```text
Document this test in docs/cohere/tests as the first step of TDD
```

```text
Update the feature doc with how it runs, checks, measurements, compatibility, and setup
```

```text
Backfill this feature doc from the existing test implementation
```
