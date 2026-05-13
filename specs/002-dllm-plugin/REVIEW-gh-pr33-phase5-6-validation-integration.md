# Second-opinion review: [vllm-project/dllm-plugin#33](https://github.com/vllm-project/dllm-plugin/pull/33)

**PR:** feat: implement Phase 5/6 validation + mock-stack integration confidence (#4 #14 #16 #17 #31 #32)  
**Base:** `main` -> **Head:** `feat/phase5-6-validation-integration-clean`  
**Commits reviewed:** `206f586`, `51449b6`, `56d3ea7`, `0ba7e26`, `f26b8d0`, `a4d7f69`  
**Review date:** 2026-04-28  
**Method:** `gh pr view --json ...`, `gh pr diff`, `gh issue view 4/14/16/17/19/31/32`, `gh pr checks`, and local branch validation in `/tmp/dllm-plugin-pr30`:
- `uv run pytest -q tests/test_validation.py tests/test_runtime_adapters.py tests/test_vllm_mock_integration.py` (passed locally in this environment)
- CI log inspection for `vllm-extra` job shows `tests/test_vllm_mock_integration.py` is skipped on GitHub-hosted runner (`s`), consistent with GPU gating.

**Not posted to GitHub** (local review only).

---

## Findings (ordered by severity)

### 1) High: mock-logit fallback still applies to `LLaDA2ForCausalLM`, not only explicit mock architecture

`resolve_runtime_block_logits()` falls back to synthetic deterministic logits when `dllm_block_logits` is absent and `_is_mock_stack_architecture()` is true. That helper currently treats both:
- `DLLM_MOCK_STACK_MODEL_ID`
- `LLADA2_ARCHITECTURE_NAME` (`LLaDA2ForCausalLM`)

as mock-safe architectures.

Why this is risky:
- Issue `#31` intent is to migrate runtime remask handoff to **model score/logit handoff**, with fallback semantics for **mock model path**.
- Allowing fallback for the canonical LLaDA2 architecture can silently mask missing `dllm_block_logits` payloads and regress score-driven remasking behavior once non-mock implementations are introduced.
- This increases the risk of false-positive "working" behavior in paths where logits wiring is actually broken.

Suggested fix:
- Restrict fallback to explicit mock ids only (or another explicit mock-mode signal).
- If architecture is `LLADA2_ARCHITECTURE_NAME`, require `dllm_block_logits` and fail fast when absent.
- Add regression test that `architectures=("LLaDA2ForCausalLM",)` without output logits raises.

---

### 2) Medium: Phase 6 CI "integration smoke" is passing while the dedicated test is skipped

The new workflow step runs:
- `uv run pytest -q tests/test_vllm_mock_integration.py`

but in GitHub Actions `vllm-extra` (`ubuntu-latest`) this test is skipped (`s`) because CUDA is unavailable.

Why this matters:
- Issue `#32` asks for confidence against runtime drift via concrete runtime-object smoke.
- Current CI status can look like integration coverage exists, but the dedicated integration test is not executed on default GitHub runners.
- This weakens the confidence claim for Phase 6 unless manual GPU evidence is treated as the primary gate.

Assessment:
- Not necessarily a blocker if the project accepts GPU-manual validation as the confidence source.
- It is a process/assurance gap that should be made explicit in PR and docs, or moved to a GPU-capable CI lane.

Suggested follow-up:
- Either wire this test to a real GPU job (self-hosted/GKE CI trigger), or rename CI step text to make skip expectation explicit and avoid false confidence.

---

### 3) Low: operator doc misses explicit minimum-version/compatibility note requested by issue #14

Issue `#14` acceptance asks to mention compatibility/minimum vLLM (`#2`).  
`docs/OPERATOR_LLaDA2.md` currently documents flags and wiring but does not specify minimum vLLM version or a pointer to the tracked pin/compat source.

Assessment:
- Small doc gap, easy to patch.
- Not a runtime correctness blocker.

Suggested fix:
- Add one line that links operator guidance to issue `#2`/`pyproject` optional dependency bounds for minimum tested vLLM.

---

### 4) Low: Helm smoke chart default branch appears stale for this phase

`tools/helm/dllm-plugin-gpu-test/values.yaml` defaults:
- `git.branch: feat/phase4-runtime-8-9-10`

Risk:
- Running the chart without override tests old branch content, not the latest Phase 5/6 state.
- Reproducibility for operators is weaker than intended unless branch override is always provided.

Assessment:
- Low severity (can be overridden at deploy time).
- Easy fix: default to `main` and override per validation run.

---

## What is strong

- Phase 5 strict stack validation exists and is wired in both runtime adapter constructors.
- Validation test coverage is meaningful for wrong scheduler/worker/arch combinations with actionable errors.
- Runtime remask path now consumes model-provided score rows when present (addressing core #31 direction).
- New operator runbook exists, is cross-linked, and includes concrete env/CLI guidance.
- Phase 6 integration artifact is present (`tests/test_vllm_mock_integration.py`) and a reproducible GPU Helm job scaffold is included.
- CI remains green across matrix + vllm-extra; no obvious regressions in existing test suite.

---

## Responsibility check vs issue #19 design contract

### Phase 5 (`#4`: strict validation)
- **Implemented:** `validation.assert_compatible_stack`, adapter init enforcement, pure unit tests.
- **Gap/caution:** architecture classification for fallback behavior in runtime worker should be stricter to avoid masking non-mock score wiring issues.
- **Status:** Mostly met, with one important correctness caveat.

### Phase 6 (`#14`, `#16`, `#17`, `#32`)
- **#14 operator doc:** Largely met (flags/CLI/first-block), but missing explicit min-version/compatibility note from `#2`.
- **#16 tests:** Met for contract-level tests and validation path; v2 expectations are reflected.
- **#17 integration confidence:** Artifacts exist (test + helm checklist path), but CI-run dedicated integration test is currently skipped on non-GPU runner.
- **#32 runtime-object smoke:** Test exists and instantiates concrete vLLM runtime objects, but CI execution is conditional on GPU availability and presently skipped in GH-hosted environment.

Overall against issue `#19`: responsibilities are largely implemented, but confidence claims are somewhat stronger than what automated CI currently demonstrates.

---

## Ready to merge?

**Recommendation: _Not ready to merge yet_ (one high-severity fix requested).**

Minimum changes before merge:
1. Restrict mock-logit fallback to explicit mock architecture/path only (do not silently fallback for `LLaDA2ForCausalLM`), with regression tests.

Strongly recommended (can be immediate follow-up if maintainers choose):
2. Clarify/upgrade Phase 6 CI integration confidence so skipped GPU test does not read as executed smoke.
3. Add explicit vLLM compatibility/min-version pointer in `docs/OPERATOR_LLaDA2.md`.
4. Update Helm chart default branch to `main` (or another non-stale default).
