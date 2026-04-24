---
name: check-docs-and-cursor-freshness
description: Validate docs/cohere, .cursor skills/agents, and test docs freshness against the current branch diff. Use during PR checks, code review, or when the user asks whether documentation and Cursor instructions are up to date.
---

# Check Docs and Cursor Freshness

Check whether `docs/cohere/` docs, `.cursor/` skills/agents, and test docs
still match the branch changes.

## When to Use

- As part of pre-PR validation
- During code review
- When a user asks if docs or Cursor instructions are stale

## Inputs

- Required: changed file list from `git diff --name-status <diff-ref> HEAD`
- Optional: explicit diff ref provided by the user

If you do not already have a changed file list, compute one first.

## Workflow

### Step 1: `docs/cohere/` Documentation Freshness

Goal: verify docs under `docs/cohere/` still accurately describe the current branch.

#### 1a. Identify doc-relevant changes

From the changed file list, flag changes that could make docs stale:

| Changed area | Potentially affected doc |
|-------------|------------------------|
| `.github/workflows/` or `.github/scripts/` | `docs/cohere/github/build-and-push.md`, `github-actions-scripts.md`, `wheel-uploads.md` |
| `tests/cohere/scripts/`, `tests/cohere/configs/`, test groups | `docs/cohere/tests/README.md` |
| Test fixtures (`tests/cohere/fixtures/`) | `docs/cohere/tests/fixtures/README.md` |
| Core runtime, model support, build config | `docs/cohere/code_notes/upstream-diff.md` |
| Docker files (`docker/Dockerfile*`, `Makefile`) | `docs/cohere/github/build-and-push.md`, `upstream-diff.md` |

If no files map to any row above, mark docs freshness as **SKIP**.

#### 1b. Cross-reference affected docs

For each potentially affected doc:

1. Read the doc.
2. Read changed files (or diffs) that map to it.
3. Compare whether the doc still matches reality.

Flag issues such as:
- Workflow inputs/steps added, removed, or renamed
- New test groups/models/scripts missing from docs
- Changed env vars, paths, or defaults
- New fixtures/images not documented
- New CI scripts missing from `github-actions-scripts.md`
- Upstream-diff changes not reflected in `upstream-diff.md`

#### 1c. Report

Per doc file, report:
- **PASS**: Up to date
- **STALE**: Specific gaps and suggested updates
- **SKIP**: Not affected

---

### Step 2: `.cursor/` Skills and Subagents Freshness

Goal: verify `.cursor/skills/` and `.cursor/agents/` still match current repo behavior.

#### 2a. Identify skill/agent-relevant changes

From the changed file list, flag changes that could make skills/agents stale:

| Changed area | Potentially affected skill/agent |
|-------------|--------------------------------|
| `.github/workflows/` | `trigger-github-actions`, `check-github-actions-status` |
| `tests/cohere/scripts/`, test group changes | `coder-ci-runner` |
| Cohere marker conventions, comment syntax | `check-cohere-markers` |
| Rebase workflow, upstream diff structure | `rebase-assistant` |
| `.cursor/rules/` | All skills (if rules changed affect behavior) |
| `docs/cohere/` | Skills that reference doc paths |
| Docker, build tooling | `coder-ci-runner`, `trigger-github-actions` |

If no files map to any row above, mark cursor freshness as **SKIP**.

#### 2b. Cross-reference affected skills/agents

For each potentially affected `.cursor/` file:

1. Read the skill/agent markdown.
2. Read changed source files (or diffs) that map to it.
3. Compare whether instructions still match reality.

Flag issues such as:
- Changed workflow names, inputs, or steps not reflected
- Renamed/added/removed test groups, models, script paths
- Docker image naming or registry path changes
- New env vars/config options not documented
- Path or directory structure changes
- Marker syntax/convention changes

#### 2c. Report

Per skill/agent file, report:
- **PASS**: Up to date
- **STALE**: Specific gaps and suggested updates
- **SKIP**: Not affected

---

### Step 3: Test Docs Freshness

Goal: ensure test documentation under `docs/cohere/tests/` stays in sync with
changed test files. Uses the `add-test-docs` skill in **backfill** mode — no
interactive prompting for new test cases.

#### 3a. Identify changed test files

From the changed file list, collect files matching:
- `tests/cohere/**` (added, modified, or renamed)

If no test files changed, mark test docs freshness as **SKIP**.

#### 3b. Map test files to feature docs

For each changed test file, determine the corresponding feature doc under
`docs/cohere/tests/`. Use these signals:
- Filename conventions (e.g. `test_c5_fp32_logits.py` maps to
  `docs/cohere/tests/features/fp32_logits.md`).
- Existing cross-references between test files and feature docs.
- The `# Tests` table in `docs/cohere/tests/observability_matrix.md`.

If a changed test file has no matching feature doc, note it as a gap but do not
block on it.

#### 3c. Backfill stale docs

For each feature doc that maps to a changed test file:

1. Read the `add-test-docs` skill at
   `.cursor/skills/add-test-docs/SKILL.md`.
2. Follow its **backfill** mode workflow to update the feature doc and
   `docs/cohere/tests/observability_matrix.md` from the current
   implementation.
3. Record what was updated.

#### 3d. Report

Per feature doc, report:
- **PASS**: Already up to date
- **STALE**: Updated — list the files changed
- **SKIP**: No test files changed in the diff

## Output Format

Use this compact structure:

```markdown
### Docs & Cursor Freshness

#### docs/cohere/ — PASS/STALE/SKIP
- <per-doc results>

#### .cursor/ Skills & Agents — PASS/STALE/SKIP
- <per-skill/agent results>

#### Test Docs (docs/cohere/tests/) — PASS/STALE/SKIP
- <per-feature-doc results>
```
