# Second-opinion review: [vllm-project/dllm-plugin#30](https://github.com/vllm-project/dllm-plugin/pull/30)

**PR:** feat: implement Phase 4 scheduler/worker runtime path (#8 #9 #10)  
**Base:** `main` -> **Head:** `feat/phase4-runtime-8-9-10`  
**Commits reviewed:** `638b08f`, `3557e3e`, `5526937`, `53546ae`, `68d3ec2`, `3f9d185`, `cde86dc`  
**Review date:** 2026-04-28  
**Method:** `gh pr view --json ...`, `gh pr diff`, `gh issue view 8/9/10/19`, `gh pr checks`, plus local branch validation in `/Users/akellner/MyDir/Code/Open/dllm-plugin`:
- `uv run pytest tests/test_scheduler.py tests/test_worker.py tests/test_runtime_adapters.py` (23 passed)
- `VLLM_USE_V2_MODEL_RUNNER=1 uv run python -c "from vllm_dllm_plugin.runtime_scheduler import DllmRuntimeScheduler; from vllm_dllm_plugin.runtime_worker import DllmRuntimeWorker; print('runtime-adapter-import-ok')"` (passed)

**Not posted to GitHub** (local review only).

---

## Findings (ordered by severity)

### 1) Medium: runtime-worker remasking still uses synthesized logits, not model logits

`DllmRuntimeWorker.execute_model()` remask flow uses `run_block_contract_from_model_output()`, which builds deterministic one-hot mock logits from `sampled_token_ids` (`build_mock_block_logits`) instead of consuming real per-position model scores.

Why this matters:
- It satisfies current Phase 4 mock-contract goals, but it is semantically weaker than the `DESIGN_MVP.md` intent for worker one-block forward + remask coupling.
- It can hide bugs that only appear when remasking policy decisions depend on score distributions (not just argmax ids).

Assessment:
- Not a merge blocker for Phase 4 mock scope as defined in issue `#19`.
- Should be tracked as a follow-up before claiming production-grade policy behavior.

---

### 2) Low: limited adapter tests with real vLLM runtime objects

The new tests are strong for helper-level contracts and adapter utility functions, but runtime adapter tests mostly operate without full vLLM runtime fixtures.

Risk:
- API drift in upstream `DraftTokenIds`/`ModelRunnerOutput` internals could evade detection until integration testing.

Assessment:
- Acceptable for current milestone, since CI includes `vllm-extra` and imports pass.
- Recommend adding one end-to-end adapter smoke in Phase 6/`#17`.

---

## What is solid

- Scheduler-side responsibilities from `#8` are implemented cleanly:
  - first-block `spec_token_ids` initialization,
  - fixed `DRAFT_SIZE` scheduling,
  - commit-0 rollback and partial-commit accounting.
- Grammar-safety requirements from `#9` are explicit and fail-closed:
  - runtime scheduler rejects structured-output grammar rewriting in draft update paths,
  - helper tests cover grammar-constrained rejection behavior.
- Worker runtime path from `#10` is materially complete for Phase 4:
  - runtime worker subclasses vLLM `Worker`,
  - executes one-block helper flow,
  - validates missing/malformed scheduled draft blocks fail-fast,
  - validates draft handoff coverage (missing/duplicate/unexpected request IDs),
  - enforces `VLLM_USE_V2_MODEL_RUNNER=1` expectation.
- Contract hardening is good:
  - unknown/duplicate/missing request coverage checks on scheduler update,
  - centralized `DRAFT_SIZE` via `VLLM_DLLM_DRAFT_SIZE` across config/remasking/scheduler/worker.
- CI status is clean: DCO + py3.10-3.13 + `vllm-extra` all passing.

---

## Responsibility check vs issue #19 (Phase 4: #8/#9/#10)

- **#8 (scheduler semantics):** Met.
- **#9 (grammar safety):** Met for MVP non-support behavior.
- **#10 (worker one-block runtime path):** Met for mock-stack Phase 4 scope (including v2 expectation and handoff checks), with the caveat that remask currently derives from synthesized logits.

No missing core responsibilities were found relative to issue `#19` Phase 4 exit criteria.

---

## Ready to merge?

**Recommendation: _Ready to merge_ for Phase 4 mock-stack scope.**

Suggested follow-up (non-blocking):
1. Track synthesized-logit remask bridge as technical debt and replace with model-score-based handoff when Phase 6/7 integration deepens.
2. Add one integration smoke that exercises runtime adapters against concrete vLLM runtime objects in CI to reduce upstream API drift risk.
