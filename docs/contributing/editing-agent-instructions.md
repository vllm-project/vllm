# Editing Agent Instructions

> Read this before modifying `AGENTS.md` or any guide it links to.

## Token Budget Mindset

`AGENTS.md` loads on every agent request; domain guides load on entry to a relevant area.
Keep `AGENTS.md` under **200 lines** and each domain guide under **300 lines**.
When a file exceeds its budget, split or prune — do not compress prose to fit.

## When NOT to Add Content

Before writing a new rule, ask whether it is actually needed:

- **Agents already do it.** Test with a prompt first. If the agent behaves correctly without the rule, don't add it.
- **One-off incident.** Prefer a code-level fix (lint rule, CI check, test assertion) over a new doc rule.
- **Hardcoded paths.** File paths change; use "search for X" patterns instead.
- **Upstream docs.** Don't reproduce pytest, ruff, or other tool docs — link to them.
- **Contradicts an existing rule.** Search all linked guides before adding. If two rules conflict, consolidate into one.
- **Already covered elsewhere.** Search `AGENTS.md` and every linked guide for overlapping guidance.

If any of the above apply, **do not add the content**.

## Where Content Belongs

The goal is a lean `AGENTS.md` plus rich domain guides that teach agents what they can't learn from the code alone.

| Scope | File |
| ----- | ---- |
| Project-wide invariants (contribution policy, env setup, test/lint commands, commit conventions) | `AGENTS.md` |
| Area-specific knowledge (model patterns, format details, deprecation timelines) | Domain guide |

**Rules of thumb:**

- If it only matters for one area, put it in a domain guide.
- If it matters for all areas, consider `AGENTS.md` — but first verify agents don't already do it.
- Create a new domain guide when you have 5 or more non-obvious instructions sharing a coherent scope.

## What Makes a Good Domain Guide

Add what agents can't infer from the code or public docs: project-specific
conventions that differ from standard patterns, correct approaches that require
cross-file context, and fixes for repeated mistakes.
Each entry should be short, specific, and actionable — e.g., which files to
touch, what order to change them in, and which tests to run.

## Keeping Docs Lean

- Every addition should trigger review of surrounding content for stale or redundant items.
- Prefer examples over explanations — a 3-line snippet beats a paragraph of prose.
- Merge related bullets into one principle instead of listing variants.
- Use `search for X` instead of hardcoded file paths.
- PR references are fine in domain guides for traceability, but avoid them in `AGENTS.md`.

## Anti-Patterns

| Pattern | Problem |
| ------- | ------- |
| Reactive accumulation | Adding a rule per incident without pruning leads to bloat |
| Copy-paste between guides | Duplicated content drifts apart; keep in one place, link from the other |
| Imperative walls | Long DO NOT lists that agents skim past; consolidate into principles |
| Config snapshots | Show the command to get the value, not the value itself |

## Change Checklist

Before submitting changes to any agent instruction file:

- [ ] **Non-obvious?** Would an agent do the wrong thing without this rule?
- [ ] **No conflicts?** Searched all linked guides for contradictions?
- [ ] **Right file?** Project-wide goes in `AGENTS.md`, area-specific in a domain guide?
- [ ] **Offset the addition?** Removed or consolidated something to compensate?
- [ ] **Under budget?** `AGENTS.md` < 200 lines, domain guides < 300 lines?
- [ ] **No hardcoded paths?** Uses "search for X" where paths may change?
- [ ] **Tested?** Verified that an agent actually follows the new instruction?
