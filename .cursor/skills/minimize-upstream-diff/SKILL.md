---
name: minimize-upstream-diff
description: Investigate all file differences with upstream vllm-project/vllm and clean up unnecessary, redundant, or outdated diffs to minimize future merge conflicts. Focuses only on files that exist in upstream but are modified by the fork. Use before a planned upstream sync, during diff audits, or when the user asks to reduce fork divergence.
---

# Minimize Upstream Diff

Audit every fork modification to an upstream file, classify each hunk by
necessity, and clean up what can be safely reverted or restructured. The goal
is to shrink the diff surface against upstream so the next rebase/sync has
fewer conflicts.

**Scope**: only files that exist in upstream and are modified by the fork.
New files added by the fork are out of scope (they cause no merge conflicts).

## When to Use

- Before a planned rebase or upstream sync
- When the user asks to reduce fork divergence
- During periodic diff hygiene audits
- After upstream has released a new version that may have adopted fork changes

## Workflow

### Step 1: Collect Modified Upstream Files

Follow the upstream file collection workflow in
[`../_shared/upstream-file-collection.md`](../_shared/upstream-file-collection.md)
(Steps A through C) to determine `UPSTREAM_REF` (called `BASE_REF` below) and
collect the list of in-scope files.

### Step 2: Determine Latest Upstream Ref

In addition to `BASE_REF`, this skill needs a second ref to detect convergent
changes:

| Ref | Purpose |
|-----|---------|
| `BASE_REF` | `UPSTREAM_REF` from the shared workflow — the fork's divergence point |
| `LATEST_REF` | The latest upstream tag or `upstream/main` — used for convergence detection |

Suggest `LATEST_REF`:
1. Latest upstream tag: `git describe --tags --abbrev=0 --match="v*" upstream/main`
2. Or `upstream/main` if no tag is found.

Confirm `LATEST_REF` with the user. If `BASE_REF` and `LATEST_REF` are the
same tag, convergence detection (Category A) is skipped — there is no newer
upstream to compare against.

Count and display the in-scope file list. Ask the user whether to audit all
files or a subset (by directory prefix, file pattern, or explicit list).
Default: all.

### Step 3: Analyze Each File

For every in-scope file, generate two diffs:

| Diff | Command | Shows |
|------|---------|-------|
| **Fork diff** | `git diff "$BASE_REF" -- <file>` | What the fork changed relative to the original base |
| **Latest diff** | `git diff "$LATEST_REF" -- <file>` | What the fork still differs from the latest upstream |

Also read the current file content for full context.

Classify **each hunk** in the fork diff into one of these categories:

#### Category A — Convergent (upstream adopted the change)

A hunk appears in the fork diff against `BASE_REF` but is **absent or
substantially reduced** in the diff against `LATEST_REF`. This means upstream
has since made the same or a very similar change.

**Action**: the hunk is now redundant. It can be reverted to the `LATEST_REF`
version (or removed entirely if identical).

#### Category B — Cosmetic / Formatting-only

The hunk changes only whitespace, import ordering, comment wording, blank
lines, or string quoting with no functional effect.

**Action**: revert to upstream version to eliminate a needless conflict point.

#### Category C — Stale / Dead

The hunk modifies code that:
- is no longer called or reachable,
- guards a feature flag or path the fork no longer uses,
- was a workaround for a bug that upstream has since fixed.

**Action**: revert. Confirm with the user if uncertain.

#### Category D — Movable

The hunk adds new logic inline in an upstream file where it could instead live
in a separate fork-only file (e.g., a helper module under a cohere-specific
directory) and be imported or registered. Moving it out converts an upstream
file modification into a fork-only addition, eliminating a future conflict
point.

**Action**: suggest extraction. Do not auto-apply — present the refactor idea
to the user.

#### Category E — Necessary

The hunk is a functional fork-specific change that must stay. It has no
convergent counterpart in the latest upstream.

**Action**: keep. Verify cohere markers are present. Note it as irreducible
diff surface.

### Step 4: Produce the Audit Report

Group results by category, then by directory. For each file, show:

```
<file_path>
  Total hunks: N
  A (convergent): K — can revert
  B (cosmetic):   L — can revert
  C (stale):      M — can revert (confirm)
  D (movable):    P — suggest extraction
  E (necessary):  Q — keep
```

End with a summary:

```
=== Diff Audit Summary ===
Files audited:           XX
Total hunks:             YY
  Revertible (A+B+C):   ZZ  (estimated line reduction: ~NNN)
  Movable (D):           WW  (would eliminate file-level conflicts)
  Necessary (E):         VV  (irreducible)
```

### Step 5: Interactive Cleanup

Present the user with cleanup options:

1. **Auto-revert all A+B** — revert convergent and cosmetic hunks across all
   files. This is the safest bulk action.
2. **Review C (stale) one by one** — for each stale hunk, show context and ask
   keep/revert.
3. **Review D (movable) proposals** — for each movable hunk, present the
   extraction plan and ask proceed/skip.
4. **File-by-file** — walk through each file, showing all hunks with their
   category, and ask per-file whether to apply suggested changes.
5. **Skip** — produce report only, no changes.

When reverting a hunk:
- Use the `LATEST_REF` version of the code (not `BASE_REF`) so the file
  stays compatible with the newest upstream. If `LATEST_REF` and `BASE_REF`
  are the same for that hunk, either works.
- After reverting, remove any cohere markers that wrapped the now-reverted
  code. Orphaned markers are noise.
- If all hunks in a file are reverted, the file should match upstream exactly.
  Verify: `git diff "$LATEST_REF" -- <file>` should be empty.

### Step 6: Post-Cleanup Verification

After all changes:

1. **Re-diff against `LATEST_REF`** and report the new diff summary:
   ```bash
   git diff --stat "$LATEST_REF"
   ```
2. **Run `check-cohere-markers`** to ensure remaining diffs still have proper
   markers. Invoke the skill or replicate its Step 4 logic.
3. **Show before/after comparison**:
   ```
   Before cleanup:  XXX files modified, ~YYYY lines changed
   After cleanup:   XXX files modified, ~YYYY lines changed
   Reduction:       ZZ files, ~NNNN lines
   ```
4. Suggest running tests (point to `coder-ci-runner` skill) to verify nothing
   broke.

## Hunk Classification Heuristics

Use these signals when classifying hunks:

### Convergent detection

```bash
# If a hunk exists vs BASE_REF but not vs LATEST_REF, upstream converged
FORK_DIFF=$(git diff "$BASE_REF" -- "$FILE")
LATEST_DIFF=$(git diff "$LATEST_REF" -- "$FILE")
```

Compare hunk line ranges. If a fork-diff hunk's target lines are identical in
`LATEST_REF`, the change was adopted upstream.

### Cosmetic detection

Flag hunks where the only differences are:
- Leading/trailing whitespace changes
- Blank line additions/removals
- Import reordering (same set of imports, different order)
- Quote style changes (`'` vs `"`) with identical string content
- Comment-only edits (rewording, not adding cohere markers)
- Trailing comma additions/removals

### Stale detection

Flag hunks that:
- Modify functions/classes that no longer exist in `LATEST_REF`
- Reference feature flags, env vars, or config keys not used elsewhere in the
  fork
- Are surrounded by `# TODO`, `# HACK`, `# FIXME`, or `# WORKAROUND` comments
  indicating temporary intent

### Movable detection

Flag hunks that:
- Add a new function, class, or significant block (>15 lines) inside an
  upstream file
- Could be placed in a fork-only module and imported/registered
- Do not deeply interleave with upstream logic (i.e., the addition is
  self-contained)

## Batch vs. Incremental Mode

**Batch** (default): audit all in-scope files at once. Best for a full
pre-rebase sweep.

**Incremental**: the user specifies a file or directory subset. Useful for
targeted cleanup of a specific area (e.g., `vllm/model_executor/`).

When running incrementally, still show the full in-scope file count so the
user knows how much remains.

## Edge Cases

- **Large files (>1000 lines of diff)**: summarize hunk categories with counts
  instead of showing every hunk inline. Offer to drill into specific hunk
  ranges on request.
- **Files modified by both fork and upstream since `BASE_REF`**: the
  `LATEST_REF` diff is the authoritative view. A hunk that exists vs
  `BASE_REF` but not vs `LATEST_REF` is convergent even if upstream's
  implementation differs slightly — the functional intent was adopted.
- **Cohere markers themselves**: marker lines (`# cohere`, `# cohere start`,
  `# cohere end`) are always part of necessary hunks. Never classify a marker
  line as cosmetic.
- **Binary or generated files**: skip. Note them in the report.

## Tips for Effective Cleanup

- Start with Category A (convergent). These are zero-risk reverts.
- Category B (cosmetic) is low-risk but verify the file still passes linting
  after revert.
- Category C (stale) benefits from a quick grep to confirm the code path is
  truly dead before reverting.
- Category D (movable) changes the module structure. Discuss with the user and
  confirm the extraction plan before applying.
- After cleanup, the remaining Category E hunks are the true fork footprint.
  Document them in `docs/cohere/code_notes/` if not already covered.

## Usage Examples

```text
Use minimize-upstream-diff to audit all fork changes before our next rebase
```

```text
Use minimize-upstream-diff on vllm/model_executor/ to clean up unnecessary diffs
```

```text
Use minimize-upstream-diff — only show the report, don't make changes yet
```
