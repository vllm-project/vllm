# Local review: [PR #33](https://github.com/vllm-project/dllm-plugin/pull/33) — Phase 5/6 validation + mock-stack integration

**Repo:** `vllm-project/dllm-plugin`  
**Branch:** `feat/phase5-6-validation-integration-clean`  
**Reviewer:** local second opinion (not posted to GitHub)  
**Date:** 2026-04-29  

**References:** Milestone orchestration [issue #19](https://github.com/vllm-project/dllm-plugin/issues/19); PR targets [#4](https://github.com/vllm-project/dllm-plugin/issues/4), [#14](https://github.com/vllm-project/dllm-plugin/issues/14), [#16](https://github.com/vllm-project/dllm-plugin/issues/16), [#17](https://github.com/vllm-project/dllm-plugin/issues/17), [#31](https://github.com/vllm-project/dllm-plugin/issues/31), [#32](https://github.com/vllm-project/dllm-plugin/issues/32).

---

## Executive summary

| Question | Answer |
|----------|--------|
| **Ready to merge?** | **No.** The **`vllm-extra` CI job fails** on `ubuntu-latest` (CPU-only). Default `ci` matrix passes; DCO passes. |
| **Aligned with issue #19 (Phases 5–6)?** | **Mostly**, with gaps on PR hygiene (#19 checklist), partial fulfillment of [#31](https://github.com/vllm-project/dllm-plugin/issues/31) for mock E2E, and **#32 blocked** until the CPU smoke test is fixed in CI. |

Fix the failing test (see below), re-run Actions, then merge is plausible after maintainer review of the softer items.

---

## Blocking: CI failure (`vllm-extra`)

From GitHub Actions run `25108305687` (job `vllm-extra`), `tests/test_vllm_mock_integration_cpu_smoke.py::test_mock_stack_engine_args_resolve_paths_and_strict_validation_cpu` **fails** with:

```text
RuntimeError: Device string must not be empty
```

Stack: `EngineArgs.create_engine_config` → `DeviceConfig` → `current_platform.device_type` empty on a host **without** CUDA, so vLLM cannot infer a device.

**Impact:** The PR explicitly adds this test to validate [#32](https://github.com/vllm-project/dllm-plugin/issues/32) (“concrete vLLM objects”) on PR CI, but it breaks the **`vllm-extra` gate**. Until resolved (e.g. pass an explicit CPU device / `device="cpu"` or equivalent supported `EngineArgs` wiring for CPU-only platforms), this is not merge-ready.

**Suggested direction (for implementers, not verified here):** Mirror whatever vLLM recommends for CPU-only config construction in tests, or narrow the smoke test to APIs that do not require a resolvable accelerator device if full `VllmConfig` is impossible without GPU.

---

## Mapping to issue #19 (orchestration)

Issue [#19](https://github.com/vllm-project/dllm-plugin/issues/19) defines phase gates, dependencies, and a **PR checklist**. This PR implements work that corresponds to:

- **Phase 5** ([#4](https://github.com/vllm-project/dllm-plugin/issues/4)): strict stack validation — **implemented** (`dllm_plugin/validation.py`, wiring in runtime adapters + mock model `__init__`).
- **Phase 6** ([#16](https://github.com/vllm-project/dllm-plugin/issues/16), [#14](https://github.com/vllm-project/dllm-plugin/issues/14), [#17](https://github.com/vllm-project/dllm-plugin/issues/17)): tests, operator doc, integration evidence — **largely implemented**, modulo CI breakage above.

### Phase 5 exit (#19): “Invalid stack combinations fail fast with actionable errors”

**Strengths:**

- Validates dLLM-capable HF architectures (`LLADA2_ARCHITECTURE_NAME`, mock stack id).
- Scheduler resolved via `SchedulerConfig.get_scheduler_cls()` and compared to `DllmRuntimeScheduler` FQCN (normalized dotted form).
- Worker: resolves **`parallel_config.worker_cls` string** the same way vLLM would (`importlib` + attribute), addressing upstream asymmetry vs scheduler — **good design**.
- Explicit error when `worker_cls == "auto"` before platform resolution.
- `caller=` context on errors aids ops debugging.
- Env-controlled strictness (`resolve_strict_stack_validation`) documented in `docs/OPERATOR_LLaDA2.md`.

**Risks / nuance:**

- Validation is **exact FQCN match** (scheduler and worker). Legitimate **subclasses** of `DllmRuntimeWorker` / `DllmRuntimeScheduler` would fail even if behavior were correct; that may be intentional for MVP strictness, but it limits experimentation without code changes.
- CLI aliases `dllm_plugin.Scheduler` / `dllm_plugin.Worker` are plain bindings to runtime classes; **`Worker.__module__` remains `dllm_plugin.runtime_worker`**, so validation succeeds — verified by inspection (aliases do not create new class objects).

### Phase 6 exit (#19): “Unit/doc/integration evidence … mock plugin stack (v2 runner … where applicable)”

**Strengths:**

- `docs/OPERATOR_LLaDA2.md`: `VLLM_PLUGINS`, v2 runner, multiproc flag, strict validation toggle, dotted CLI class names, integration commands, Helm chart pointer — matches [#14](https://github.com/vllm-project/dllm-plugin/issues/14) / [#17](https://github.com/vllm-project/dllm-plugin/issues/17) intent.
- GPU integration test `tests/test_vllm_mock_integration.py`: real `LLM.generate` with mock HF fixture — strong signal when run on GPU (Helm path described in PR body).
- CPU smoke **intent** [#32](https://github.com/vllm-project/dllm-plugin/issues/32): `EngineArgs` + `create_engine_config` + `assert_compatible_stack` — **right idea**, currently **broken in CI** (see blocking section).

**Gaps vs #19 PR checklist**

The orchestration issue asks PRs to include: phase label, **HARD/SOFT dependencies**, scope, validation evidence, docs. The PR body lists issues and a test plan but **does not** use that checklist structure (e.g. “Phase 5–6”, explicit HARD/SOFT). Minor process gap, not a code defect.

---

## Issue-specific notes

### [#31](https://github.com/vllm-project/dllm-plugin/issues/31) — Model-score remask handoff vs synthesized logits

**Implemented:**

- `resolve_runtime_block_logits` prefers **`dllm_block_logits`** on model output (mapping with per-request coverage; fails fast on partial coverage — good hardening).
- Non-mock dLLM architecture **without** logits: explicit `ValueError` (no silent synthesis except documented mock path).
- Mock architecture: **fallback** to `build_mock_model_block_logits` when payload absent — scoped and documented.

**Gap (important for literal acceptance text):**

- Nothing in the plugin sets **`dllm_block_logits`** on `ModelRunnerOutput` from the mock forward (`grep` shows only tests + `runtime_worker.py`). So on real stacks without that field, mock-arch paths still use **deterministic synthesized rows**, not tensors routed from `compute_logits`. The **structural** move away from “worker always synthesizes” is real; **end-to-end score plumbing from the mock model into `dllm_block_logits`** is not present. That is acceptable as Phase 6 mock MVP **if** [#31](https://github.com/vllm-project/dllm-plugin/issues/31) is interpreted as “contract + worker resolution policy”; it is **not** literal completion of “always consume model scores” for the mock GPU run unless you close [#31](https://github.com/vllm-project/dllm-plugin/issues/31) with an explicit “mock may fallback until Phase 7” note.

### [#32](https://github.com/vllm-project/dllm-plugin/issues/32) — vLLM-extra integration smoke

Conceptually satisfied by `test_vllm_mock_integration_cpu_smoke.py`, but **CI currently fails** — so [#32](https://github.com/vllm-project/dllm-plugin/issues/32) is **not** actually delivered until the test passes on `ubuntu-latest` + `vllm` extra.

### Package rename (`dllm_plugin`, dotted CLI names)

Matches vLLM’s `resolve_obj_by_qualname` behavior; fixes colon-form breakage — **correct** and well documented.

---

## Other observations (non-blocking)

1. **Duplicate pytest step:** `.github/workflows/ci.yml` runs full `uv run pytest -q` and then `uv run pytest -q -rs tests/test_vllm_mock_integration.py`. Harmless but redundant cost.
2. **Optional workflow** `optional-vllm-smoke.yml` remains `workflow_dispatch`-only; CONTRIBUTING notes PR branches cannot run it until merged — fine for optional validation.
3. **Tests:** Large `tests/test_validation.py` addition improves determinism for [#4](https://github.com/vllm-project/dllm-plugin/issues/4); `tests/test_runtime_adapters.py` exercises logits edge cases — good.

---

## Verdict

- **Do not merge** until **`vllm-extra` is green** (fix CPU-only `EngineArgs` / device handling in the new smoke test or adjust test scope).
- After CI passes: **substantive review** suggests the Phase 5 validation work and Phase 6 documentation/GPU story are **strong**; treat [#31](https://github.com/vllm-project/dllm-plugin/issues/31) closure wording carefully relative to mock-stack fallback.
- Align PR description with **issue #19 milestone checklist** on a follow-up or amend for traceability.

---

## Commands used

- `gh pr view 33 --repo vllm-project/dllm-plugin --json …`
- `gh issue view 19 --repo vllm-project/dllm-plugin`
- `gh pr checks 33 --repo vllm-project/dllm-plugin`
- `gh run view 25108305687 --repo vllm-project/dllm-plugin --log-failed`
- Shallow clone of `feat/phase5-6-validation-integration-clean` under `/tmp` for file inspection

CI status reflects the run available at review time; re-check before merging.
