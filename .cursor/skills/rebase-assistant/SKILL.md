---
name: rebase-assistant
description: Rebase a custom branch onto a new upstream vllm tag by squashing custom commits, minimizing the upstream diff, and replaying with conflict resolution. Use when the user asks to rebase to a new upstream tag, update branch base version, or resolve rebase conflicts against upstream.
---

# Rebase Assistant

Rebase this repository's custom branch onto a newer upstream tag in a safe, repeatable way.

## Workflow

Follow these steps in order.

### Step 1: Find Merge Base and Squash Custom Commits

1. Get the current upstream tag the branch is based on (from user).
2. Ensure `upstream` is configured and fetched — follow Step A in
   [`../_shared/upstream-file-collection.md`](../_shared/upstream-file-collection.md).
3. Find merge base:
   - `git merge-base HEAD upstream/<old-tag-name>`
4. Record current custom commits:
   - `git log --oneline <merge-base>..HEAD`
   - `git rev-list --count <merge-base>..HEAD`
5. Create a backup branch:
   - `git branch backup-$(date +%Y%m%d-%H%M%S)`
6. Squash custom commits into one:
   - `git reset --soft <merge-base>`
   - `git commit -m "Squashed custom branch changes"`

### Step 2: Understand What the Branch Changes

1. Read `docs/cohere/code_notes/upstream-diff.md`.
2. Collect and classify changed files using Step C from
   [`../_shared/upstream-file-collection.md`](../_shared/upstream-file-collection.md),
   with the merge-base from Step 1 as `UPSTREAM_REF`:
   - Modified upstream files (`M`) -> require cohere markers
   - New files (`A`) -> no markers required
   - Deleted files (`D`)
3. Summarize branch intent (features, fixes, config, perf, refactors).
5. If needed, update `docs/cohere/code_notes/upstream-diff.md` for newly introduced differences.

### Step 3: Minimize Upstream Diff

Invoke the `minimize-upstream-diff` skill to audit and clean up the fork delta
before rebasing. Use the new upstream tag (the rebase target) as `LATEST_REF`
so convergence detection compares against where the branch is heading.

- Revert convergent, cosmetic, and stale hunks (Categories A, B, C).
- Review movable hunks (Category D) with the user.
- The skill's post-cleanup step runs `check-cohere-markers`, so marker
  validation is covered here.
- After cleanup, amend the squashed commit to include the diff reductions:
  `git add -A && git commit --amend --no-edit`

This step directly reduces conflict surface for the rebase that follows.

### Step 4: Validate Transformers Version Alignment

Before rebasing, verify that the test dependency pin and the checked-out
`cohere-transformers` version are aligned:

1. Extract `transformers` pin from `requirements/test.in`:
   - `TEST_TRANSFORMERS_VERSION=$(awk -F'==' '/^transformers==/{print $2; exit}' requirements/test.in)`
2. Extract the checkout ref from `.github/workflows/build-and-push.yaml`:
   - `COHERE_TRANSFORMERS_REF=$(awk '/git checkout v[0-9]+\\.[0-9]+\\.[0-9]+(\\.[0-9]+)?/{print $3; exit}' .github/workflows/build-and-push.yaml)`
3. Normalize checkout ref to upstream `transformers` version (drop leading `v`
   and trailing Cohere patch segment):
   - `COHERE_TRANSFORMERS_VERSION=${COHERE_TRANSFORMERS_REF#v}`
   - `COHERE_TRANSFORMERS_VERSION=${COHERE_TRANSFORMERS_VERSION%.*}`
4. Compare:
   - if `TEST_TRANSFORMERS_VERSION != COHERE_TRANSFORMERS_VERSION`, STOP and
     prompt the user to update the `cohere-transformers` checkout ref (repo
     tag/branch used in CI) to match `requirements/test.in`.
   - continue only after user confirms how to resolve mismatch.

### Step 5: Rebase Squashed Commit to New Upstream Tag

At this stage, there should be exactly one squashed custom commit on top of the old merge base.

1. Get target upstream tag (from user).
2. Create a new branch to preserve the original branch:
   - `<current-branch>-<upstream-tag>`
   - `git checkout -b <current-branch>-<upstream-tag>`
3. Fetch upstream:
   - `git fetch upstream`
4. Replay the squashed commit onto new tag:
   - `git rebase --onto upstream/<new-tag-name> <old-merge-base> HEAD`
5. Resolve conflicts methodically:
   - Read conflict markers in each file.
   - Compare upstream changes: `git log upstream/<old-tag>..upstream/<new-tag> --oneline`
   - If upstream deleted a file, prefer keeping the deletion (do not carry the file forward) unless the user explicitly asks to retain/reintroduce it.
   - Preserve custom behavior with correct cohere marker syntax.
   - Stage each resolved file: `git add <file>`
   - Continue: `git rebase --continue`
6. If uncertain, stop and ask the user before finalizing a risky resolution.

### Step 6: Verify Rebase Result

1. Confirm history shape:
   - Branch is based on `upstream/<new-tag>`
   - Exactly one custom squashed commit is on top
2. Re-run cohere marker verification for safety.
3. Run quick build/test checks when possible.
4. Provide summary:
   - Modified upstream files (with markers)
   - New files/folders
   - Conflicts resolved
   - Any follow-up review items

### Step 7: Optional Test Execution

Ask: `Would you like to run tests to verify the rebase? (local/remote/skip)`

- `local`: Use `coder-ci-runner` in iterative mode.
  - Reuse `vllm-tests` container if already running.
  - If stopped, remove and recreate.
  - Iterate fix -> rerun until pass or user stops.
  - Cleanup when done: `docker stop vllm-tests && docker rm vllm-tests`
- `remote`: Use `trigger-github-actions` with the workflow that matches the requested coverage:
  - `build-and-test` for feature validation
  - `build-and-eval` for eval validation (`lm_eval`, `bee_eval`)
  - `build-and-bench` for bench-only validation
- `skip`: Continue and note tests were not run.

## Conflict Resolution Rules

When resolving conflicts:

1. Understand both sides before editing.
2. Preserve custom branch intent.
3. Integrate upstream improvements that do not break custom behavior.
4. If upstream removed a file, keep it removed by default (avoid reviving removed upstream files unless explicitly required).
5. Keep cohere markers accurate and correctly paired.
6. Document non-obvious decisions.

Prompt the user when:
- Intent is unclear
- Upstream refactor significantly changed structure
- Multiple valid resolutions exist with trade-offs

## Command Reference

```bash
# Upstream setup — see ../_shared/upstream-file-collection.md Step A

# Merge base
git merge-base HEAD upstream/<old-tag>

# Backup + squash
git branch backup-<timestamp>
git reset --soft <merge-base>
git commit -m "Squashed custom branch changes"

# Transformers version alignment check
TEST_TRANSFORMERS_VERSION=$(awk -F'==' '/^transformers==/{print $2; exit}' requirements/test.in)
COHERE_TRANSFORMERS_REF=$(awk '/git checkout v[0-9]+\.[0-9]+\.[0-9]+(\.[0-9]+)?/{print $3; exit}' .github/workflows/build-and-push.yaml)
COHERE_TRANSFORMERS_VERSION=${COHERE_TRANSFORMERS_REF#v}
COHERE_TRANSFORMERS_VERSION=${COHERE_TRANSFORMERS_VERSION%.*}

# New branch and rebase to new tag
git checkout -b <current-branch>-<new-tag>
git rebase --onto upstream/<new-tag> <old-merge-base> HEAD

# Conflict loop
git status
git add <resolved-file>
git rebase --continue

# Abort if needed
git rebase --abort
```

## Usage Examples

```text
Rebase this branch from upstream/v0.15.1 to upstream/v0.16.0 using rebase-assistant.
```

```text
Use rebase-assistant to squash custom commits, rebase to upstream/v0.16.2, and run local tests.
```
