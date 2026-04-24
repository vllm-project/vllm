---
name: pr-checks
description: Run pre-PR quality checks including cohere marker validation, docs/cursor/test-docs freshness via a dedicated sub-skill, and pre-commit validation. Use before opening a pull request, during code review, or when the user asks to verify PR readiness.
---

# PR Checks

Run a suite of quality checks before opening or reviewing a pull request. Catches missing cohere markers, stale documentation, outdated Cursor skills/subagents, and stale test docs.

## When to Use

- Before opening a pull request
- During code review
- When the user asks "is this PR ready?" or "run PR checks"

## Checks Overview

| # | Check | What it verifies |
|---|-------|-----------------|
| 1 | Cohere markers | All modifications to upstream files have correct cohere annotations |
| 2 | Docs/Cursor/Test-docs freshness | Delegates to `check-docs-and-cursor-freshness` skill (includes general docs, .cursor skills, and test docs backfill) |
| 3 | Pre-commit validation | Local hooks pass or return actionable failures |

## Workflow

### Step 0: Determine Scope

Identify what changed in the current diff (same logic as `check-cohere-markers` Step 1):

```bash
# Diff ref: local unpushed commits against tracking branch
DIFF_REF=$(git rev-parse --abbrev-ref --symbolic-full-name @{u} 2>/dev/null)
# Fallback: prompt user or use upstream merge base

git diff --name-status "$DIFF_REF" HEAD
```

Store the changed file list — it drives the full check suite.

---

### Check 1: Cohere Markers

Invoke the `check-cohere-markers` skill (read its SKILL.md at `.cursor/skills/check-cohere-markers/SKILL.md` and follow its full workflow).

- Pass through any explicit upstream ref or diff ref provided by the user.
- If the skill reports failures, present them but **continue** to the remaining checks before asking about fixes. This lets the user see the full picture at once.
- Record result as PASS / FAIL.

---

### Check 2: Docs/Cursor/Test-docs Freshness (Delegated)

Invoke `check-docs-and-cursor-freshness` by reading
`.cursor/skills/check-docs-and-cursor-freshness/SKILL.md` and following its
full workflow (Steps 1-3: general docs, .cursor skills, and test docs backfill).

- Pass the changed file list from Step 0 into that workflow.
- If any sub-step reports issues, present them but continue to Check 3 so the
  user sees the full picture in one pass.
- Record the delegated check result as PASS / STALE / SKIP.

---

### Check 3: Pre-commit Validation

Goal: run local pre-commit hooks and either fix issues or report actionable failures before PR.

#### 3a. Ensure `pre-commit` is installed

If `pre-commit` is missing, prompt the user to install it first:

```bash
sudo apt install pipx
pipx install pre-commit
```

After installation, verify `pre-commit` is available and proceed.

#### 3b. Run pre-commit hooks

```bash
pre-commit run --show-diff-on-failure --color=always --hook-stage manual
```

#### 3c. Handle results

- If hooks pass, mark Check 3 as **PASS**.
- If hooks fail with auto-fixable changes, apply the fixes, re-stage as needed, and re-run:
  `pre-commit run --show-diff-on-failure --color=always --hook-stage manual`
- If hooks still fail, mark Check 3 as **FAIL** and list the exact hook failures with next actions.

---

### Step 4: Summary Report

Present a consolidated report:

```
## PR Checks Summary

### Check 1: Cohere Markers — PASS/FAIL
[Details or "All markers correct."]

### Check 2: Docs/Cursor/Test-docs Freshness (delegated) — PASS/STALE/SKIP
[Insert delegated output from check-docs-and-cursor-freshness, covering
 docs/cohere/, .cursor/ skills, and test docs backfill]

### Check 3: Pre-commit Validation — PASS/FAIL
[Hook results and actionable failures]

---
Overall: READY / NOT READY
[List of action items if any check failed or is stale]
```

---

### Step 5: Fix (Interactive)

If any checks reported issues:

1. **Show the full summary** first (so the user sees everything).
2. **Ask the user**: "Would you like me to fix the issues? (yes / no / review-each)"
   - **yes**: Fix all issues (add markers, update docs, update skills, backfill test docs).
   - **no**: Leave as-is.
   - **review-each**: Walk through each issue, asking for confirmation before fixing.
3. After fixes, **re-run only the failed checks** to confirm they now pass.

---

## Edge Cases

- **Only new files changed**: Check 2 may still apply if new files introduce features that should be documented.
- **Doc-only changes**: Check 1 likely skips (no code changes). Check 2 is the main focus.
- **Skill/agent-only changes**: Check 1 likely skips. Check 2 is the main focus.
- **Test-only changes**: Check 2's test docs step is the main focus — backfill docs from implementation.
- **No changes at all**: Report "Nothing to check" and exit.

## Usage Examples

**Full PR check:**
```
Run pr-checks on my local changes
```

**Check with auto-fix:**
```
Run pr-checks and fix any issues
```

**Check against specific base:**
```
Run pr-checks against upstream/v0.14.1
```
