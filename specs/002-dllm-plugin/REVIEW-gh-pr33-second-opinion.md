# PR Review: vllm-project/dllm-plugin#33 (second opinion)

## Verdict

Leaning **not ready to merge** until one responsibility-coverage gap is resolved or scope is explicitly narrowed.

Core Phase 5 work is solid, and most Phase 6 artifacts are good quality. The blocker is acceptance-level alignment: this PR claims to close integration-confidence responsibilities that are only partially exercised in CI.

## Findings (ordered by severity)

### 1) Medium-High (merge blocker): concrete runtime smoke exists, but CI signal is effectively skip-only

- **Where:** `tests/test_vllm_mock_integration.py`, `.github/workflows/ci.yml`, `.github/workflows/optional-vllm-smoke.yml`
- **What:** the new integration test is hard GPU-gated (`skipif(not torch.cuda.is_available())`), and both hosted workflows run on `ubuntu-latest` where the test is expected to skip.
- **Why this matters against issue #19 / #32 intent:** issue #19 emphasizes reproducible confidence for the mock stack and references concrete runtime integration evidence. A test that is always skipped in default CI does not materially protect against runtime API drift.
- **Required fix or scope clarification:**
  - add at least one non-skipping CI smoke that instantiates enough concrete runtime objects to catch integration drift on CPU, **or**
  - explicitly narrow closure language so this PR closes docs/manual-GPU confidence only, leaving CI-runtime drift gating to a follow-up.

### 2) Medium: v2-vs-v1 operator guidance is one-sided (v2-only enablement, no explicit fallback behavior matrix)

- **Where:** `docs/OPERATOR_LLaDA2.md`, workflow env settings
- **What:** docs and CI enforce `VLLM_USE_V2_MODEL_RUNNER=1` and disable v1 multiprocessing, but do not provide clear operator guidance for fallback/unsupported v1 behavior.
- **Why this matters against issue #19 text:** the Phase 6 deep-dive in #19 calls out v2 expectations and v1-vs-v2 confidence where feasible.
- **Recommendation:** add a short explicit section: "v1 runner status for mock-stack path" (supported/unsupported, expected failure mode, and why).

### 3) Medium (non-blocking now, risk for Phase 7): runtime logits normalization eagerly materializes Python floats

- **Where:** `vllm_dllm_plugin/runtime_worker.py` (`_normalize_block_logits_rows`)
- **What:** rows are converted element-by-element to Python `float`.
- **Why it matters:** acceptable for MVP mock-stack vocab sizes, but can become expensive if reused for real-model logits.
- **Recommendation:** keep as-is for Phase 6, but track a Phase 7 follow-up to avoid full Python materialization.

## Responsibility check against issue #19

### Phase 5 / issue #4 - strict stack validation

- Added `assert_compatible_stack` with explicit model/scheduler/worker checks and actionable errors: **Implemented**
- Enforced in runtime adapter constructors and mock model init path: **Implemented**
- Added targeted tests in `tests/test_validation.py`: **Implemented**
- Responsibility status: **Meets Phase 5 expectations**

### Phase 6 / issue #14 - operator doc

- Added `docs/OPERATOR_LLaDA2.md` with env, CLI, first-block notes, and GPU test invocation: **Implemented**
- README cross-link and dotted qualname guidance updated: **Implemented**
- Gap: v1 fallback/unsupported behavior is not clearly documented as an operator decision table: **Partially implemented**

### Phase 6 / issue #16 - unit/contract confidence

- Expanded runtime adapter tests for logits payload coverage and strict failure modes: **Implemented**
- Added validation unit tests for compatible/incompatible stack combinations: **Implemented**
- Responsibility status: **Substantially implemented**

### Phase 6 / issue #17 - integration confidence

- Added concrete vLLM runtime integration test and fixture config: **Implemented**
- Added reproducible external GPU path (`tools/helm/dllm-plugin-gpu-test`): **Implemented**
- Gap: default CI lane executes skip-only for that test, so routine drift protection is weak: **Partially implemented**

## Secondary notes

- Runtime fail-fast behavior is now wired at sensible entry points (scheduler, worker, model init).
- Restricting mock fallback to explicit mock architecture is a good safety improvement over sampled-token synthesis.
- Mock attention/KV-cache discovery adjustments are pragmatic and improve runtime bring-up reliability.

## Merge recommendation

- **Recommendation:** request one more revision before merge (or tighten claimed scope).
- **Blocking item to clear:** ensure issue #19/#32 integration-confidence responsibility is either truly CI-exercised or explicitly downgraded to manual GPU evidence in closure language.
