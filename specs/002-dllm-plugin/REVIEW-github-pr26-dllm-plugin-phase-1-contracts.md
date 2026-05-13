# Local code review: vllm-project/dllm-plugin PR #26

**PR:** [feat: Phase 1 shared contracts (config, RemaskingPolicy, CONTRACTS)](https://github.com/vllm-project/dllm-plugin/pull/26)  
**Base:** `main` · **Head:** `mvp/phase-1-contracts` (merge tip `c09d499104ee7f0d84c0f045a9b79d4c4a3fb3b1`)  
**Commits:** 2 (`100f6f9` feat, `c09d499` follow-up fixes)  
**Review date:** 2026-04-09  
**Method:** `gh pr view`, `gh pr diff`, `gh pr checks` (not posted to GitHub).

---

## Executive verdict

**Ready to merge** for what Phase 1 claims to be: shared constants, a remasking protocol/result type, contributor docs, and unit tests—**provided** the project’s normal gate is satisfied (at least one maintainer review if that is policy). GitHub reports **`mergeable: MERGEABLE`**. **DCO** and **ci (Python 3.10–3.13)** all **pass**.

This review is a **second opinion**: it agrees the slice is coherent and low-risk, but flags API/typing and process risks that are worth tracking after merge—not necessarily blocking for Phase 1.

---

## What changed (current tree)

| Path | Role |
|------|------|
| `vllm_dllm_plugin/config.py` | `DRAFT_SIZE` (32), architecture string placeholder, mock model id, strict-validation default flag |
| `vllm_dllm_plugin/remasking/base.py` | `RemaskStepResult`, `@runtime_checkable` `RemaskingPolicy`, `validate_remask_step_result()` |
| `vllm_dllm_plugin/remasking/__init__.py` | Public exports including **`validate_remask_step_result`** |
| `docs/CONTRACTS.md` | ASCII field-mapping / invariants; explicit **sync obligation** with `DESIGN_MVP.md` §6–§8 |
| `docs/DESIGN_MVP.md` | Pointers to `config`, `CONTRACTS`, and post-`apply` validation |
| `README.md` | Link to `CONTRACTS.md` (“section 7” wording, no section sign) |
| `tests/test_config.py` | Defaults / non-emptiness |
| `tests/test_remasking_base.py` | Protocol structural typing, validator happy/edge/sad paths |

---

## Strengths

1. **Scope discipline.** PR body separates HARD vs SOFT dependencies and in vs out of scope; aligns with milestone #19. Reviewers can reject scope creep mechanically.
2. **Single numeric source of truth.** `DRAFT_SIZE` in `config` avoids three divergent “32”s in scheduler/worker/policy later; test locks 32 to the MVP default.
3. **Validator is real and exported.** `validate_remask_step_result` encodes the two shape rules that are checkable without vocab context; exporting it from `remasking` matches how workers will consume it.
4. **Docs respond to drift risk.** The second commit adds an explicit **bi-directional** sync note between `CONTRACTS.md` and `DESIGN_MVP.md`—addressing the main “two sources of truth” objection from an earlier pass.
5. **Tests improved in follow-up.** `test_validate_accepts_committed_length_equal_draft_size` pins the **inclusive** upper bound on `committed_token_ids`; wrong `next_input_block` length and `DRAFT_SIZE+1` committed cases are covered.
6. **CI signal.** Full matrix green + DCO reduces “obvious breakage” risk for a doc-heavy PR.

---

## Critical and second-opinion concerns

### 1. `@runtime_checkable` + `Protocol` is a weak runtime guarantee

`isinstance(x, RemaskingPolicy)` only checks that **`apply` exists and is callable**—not parameter kinds, keyword-only contract, or return type. That is idiomatic for Protocols but **easy to misread**: contributors may think “structural subtype” means full API compliance.

**Recommendation (post-merge, doc-only or typing):** One line in `RemaskingPolicy` docstring or `CONTRACTS.md` that `isinstance` is **not** a substitute for tests or static checking. Optional later: `typing_extensions.assert_type` / pyright in CI for implementers.

### 2. Tuple vs list at the vLLM boundary

`RemaskStepResult` uses `tuple[int, ...]` while design prose often says `list[int]`. Tuples are a good fit for immutability and `frozen=True` slots; **call sites will copy or convert** when interfacing with vLLM APIs that expect lists. Not wrong—just ensure phase-2 worker code standardizes one convention (document “worker converts to list for engine” or similar).

### 3. `validate_remask_step_result` hard-wires module `DRAFT_SIZE`

The helper’s docstring already admits a future parameter if per-model block sizes appear. **Risk is low for MVP**; the critical second opinion is: if request-level block size ever differs from `config.DRAFT_SIZE`, this validator becomes a **footgun**. Track when introducing dynamic block sizes.

### 4. `LLADA2_ARCHITECTURE_NAME` may bake in a wrong string

The constant is honestly marked as tentative until registration (#5). **Merge is still reasonable**—the alternative is no constant and more sprawl. Watch for copy-paste of the string outside `config` before #5 lands.

### 5. No maintainer review on the PR (process)

At review time, `gh pr view` showed **no reviews**. “Ready to merge” here is **technical + scope** readiness, not **governance** readiness. If the repo expects maintainer ACK, that box is still empty.

### 6. CONTRACTS vs upstream vLLM renames

The table names vLLM fields (`Request.spec_token_ids`, `SchedulerOutput.*`, etc.). If vLLM moves faster than the plugin doc, **CONTRACTS can go stale without failing CI**. The sync obligation only ties two **plugin** docs together, not plugin docs to vLLM HEAD. Acceptable for Phase 1; later consider linking to a pinned vLLM version or a single “compatibility matrix” issue.

---

## Smaller nits (non-blocking)

- **`RemaskStepResult` allows invalid shapes until validation**—documented clearly in code and `CONTRACTS.md`; a stricter `__post_init__` could wait until #7 stabilizes policy behavior.
- **Stub policy’s `del input_block, ...`** is fine for a test double; no change needed.
- **Commit messages** reference `Closes vllm-project#3` style in one web view vs `#3` in JSON—GitHub typically resolves either on the same repo; worth no action.

---

## What the follow-up commit fixed (vs a stale first-pass review)

If comparing to an earlier local review of only `100f6f9`:

- Document sync obligation **added**.
- **`validate_remask_step_result`** re-exported from **`vllm_dllm_plugin.remasking`**.
- **Inclusive-bound** test for `len(committed_token_ids) == DRAFT_SIZE`.
- README wording avoids **§** in UI-sensitive places.

Those were reasonable review items; the author addressed them in-tree.

---

## Merge checklist (reviewer-style)

| Item | Status |
|------|--------|
| Scope matches Phase 1 (#3, #6, #15) | Yes |
| Tests meaningful for public API | Yes |
| Docs consistent with stated sync rules | Yes |
| CI + DCO | Pass |
| Merge conflict / mergeable | MERGEABLE |
| Maintainer / codeowner review | **None observed**—project-dependent |

---

## Bottom line

**Merge recommendation: yes** for Phase 1 contracts, assuming normal project review policy is satisfied elsewhere. The change set is additive, tested, documented, and the second commit shows responsiveness to doc/API surface feedback. Remaining concerns are **long-term API evolution** (runtime_checkable limits, tuple/list, validator parameterization) and **external doc drift** vs vLLM—not reasons to hold this PR absent maintainer objections.

---

*Generated with `gh` against `vllm-project/dllm-plugin` PR #26.*
