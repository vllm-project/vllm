# Maintainer-style review: dllm-plugin PR #29 (second opinion)

**PR:** [feat(remasking): connect forward logits to RemaskingPolicy (issue #13)](https://github.com/vllm-project/dllm-plugin/pull/29)  
**Repo:** `vllm-project/dllm-plugin`  
**Head (at review time):** `feat/phase-3-issue-13-forward-remask-handoff` — tip commit `5a99507` (*refactor(remasking): require policy in handoff; strengthen tests*)  
**Diff stat:** +365 / −8 across 7 files (handoff module, remasking exports, docs, mock docstring, new test module)  
**CI (`gh pr checks`):** DCO pass; `ci` (Python 3.10–3.13) pass; `vllm-extra` pass  
**Prior GitHub review:** AlonKellner-RedHat (commented review on `b55472c`, before the final `5a99507` refactor commit).  
**Review date:** 2026-04-16  
**Note:** Local second opinion only; **not** posted to GitHub.

---

## Verdict

**Ready to merge** from a technical standpoint: scope matches issue [#13](https://github.com/vllm-project/dllm-plugin/issues/13), the API is internally consistent, tests and full CI (including `vllm-extra` where `torch` paths run) are green, and the main concerns from the earlier inline review are addressed on the current tip.

Residual recommendations are **process and polish** (squash-merge for a cleaner history after the branch oscillated, and a quick human pass on the PR body at merge time), not correctness blockers.

---

## How this relates to the prior review

Alon’s review (on `b55472c`) raised four themes: optional default policy, weak “explicit policy” test, missing `remasking_config` coverage, and a stale PR description relative to README churn.

On **tip `5a99507`**, the diff and tests show:

| Earlier concern | Current state |
|-----------------|---------------|
| Optional `Llada2DefaultRemaskingPolicy` default | **Resolved:** `remask_after_block_forward(..., policy: RemaskingPolicy)` is mandatory; docs and `DESIGN_MVP` say the handoff does not choose a default. |
| Test did not prove `policy=` dispatch | **Resolved:** `_StubRemaskingPolicy` + `test_stub_policy_passed_through` returns a distinctive `RemaskStepResult`. |
| `remasking_config` not covered | **Resolved:** `test_remasking_config_forwarded_to_policy` compares handoff vs direct `apply` with `{"num_transfer": 1}`. |
| Stale PR description (README) | **Likely resolved:** current PR body (via `gh pr view`) describes required `policy` and does not list the README bullets that were removed; still worth confirming in the GitHub UI before merge in case the body was edited after this file was written. |

---

## What works well

1. **Thin adapter, right place.** `vllm_dllm_plugin.remasking.handoff` validates `input_draft` length and block logits outer shape, delegates to `RemaskingPolicy.apply`, then runs `validate_remask_step_result`. That is the correct seam for [#10](https://github.com/vllm-project/dllm-plugin/issues/10) without forking `GPUModelRunner`.

2. **Torch-free shape check where possible.** `assert_block_logits_shape` uses `hasattr(logits, "shape")` for tensors without importing `torch`, while documenting that per-row vocabulary width is the policy’s job. That matches the stated split of responsibilities.

3. **Actionable errors.** The `logits is None` path explains last-PP-rank behavior and points to `MOCK_STACK_MODEL.md`, which will save debugging time.

4. **Documentation coherence.** `DESIGN_MVP` forward→remasking subsection, `CONTRACTS` handoff section, and `MOCK_STACK_MODEL` alignment (`num_tokens == DRAFT_SIZE` for the block handoff) reduce drift between design and contributor-facing summaries.

5. **Tests match the dependency layout.** Module docstring + `importorskip("torch")` make it explicit why tensor cases may skip locally; `vllm-extra` passing shows they are not dead on CI.

6. **Parity tests.** `test_mock_shaped_logits_terminal_matches_direct_policy_apply` and the tensor-vs-list parity test give confidence the handoff is not subtly reshaping or dropping arguments before `apply`.

---

## Critical second-opinion notes (non-blocking)

### 1. Nested-sequence logits: outer shape only

The helper intentionally does **not** verify that each row has length `vocab_size`. That is documented and delegated to `Llada2DefaultRemaskingPolicy` / `_logits_to_rows`. The tradeoff is correct for a minimal helper but means **bad row lengths** fail later inside the policy with less obvious stack traces than if the handoff normalized validation. Acceptable for MVP; revisit if support tickets show confusion.

### 2. Duck typing on `logits.shape`

Any object with a `.shape` attribute enters the “tensor” branch. That is fine for PyTorch and NumPy-like types and is unlikely to bite in vLLM call sites. It is still a mild contract: exotic wrappers could theoretically misbehave.

### 3. No explicit test that `validate_remask_step_result` rejects a bad policy

The handoff always validates the policy output. There is no dedicated test that a deliberately invalid `RemaskStepResult` from a malicious or buggy policy surfaces as a validation error **through** `remask_after_block_forward`. Low priority if `validate_remask_step_result` is already covered elsewhere; adding one line in the stub test could lock the invariant.

### 4. Commit narrative noise

History includes an explicit oscillation (default policy → required → optional default → required again). Functionally the tip is good; **squash-merge** (or a single rebased commit) would keep `git blame` and archeology cleaner for future readers.

### 5. Cross-reference to issue #24

`CONTRACTS.md` points mock-path readers at issue #24. That is only a problem if #24 is stale or wrong; worth a glance when merging doc-heavy PRs.

---

## Minor nits

- **Unicode in `handoff.py` docstring** (`§5–8`): fine for Python source; avoid copying verbatim into `CONTRACTS.md`, which states ASCII-only for that file.
- **Naming:** “first dimension” wording for sequence-shaped logits is tensor-oriented but consistent with “outer length” in the docstring.

---

## Merge readiness checklist

| Criterion | Status |
|-----------|--------|
| Issue #13 scope (forward → remasking handoff) | Met |
| Required `policy` (multi-stack safety) | Met on tip |
| Tests: guards, parity, stub routing, config forwarding, torch paths | Met |
| CI: default matrix + `vllm-extra` | Green (`gh pr checks`) |
| DCO | Pass |
| Docs vs code | Aligned on reviewed diff |
| PR description vs tree | Verify at merge (likely OK after follow-up commits) |

**Bottom line:** **Approve / merge** as-is, preferably with **squash** and a final glance at the GitHub PR description. Optional follow-ups: invalid-result-through-handoff test; stricter row-width validation if real stacks produce ragged nested lists.
