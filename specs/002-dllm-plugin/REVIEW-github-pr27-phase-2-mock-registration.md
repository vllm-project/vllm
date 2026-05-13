# Local review: [PR #27 — Phase 2 mock model registration](https://github.com/vllm-project/dllm-plugin/pull/27)

**Repo:** `vllm-project/dllm-plugin`  
**Branch:** `mvp/phase-2-mock-registration` → `main`  
**Commit reviewed:** `a932c9af38994f055c4c5f8720b73db0792c0d4a` (single commit)  
**Date:** 2026-04-09  
**Not posted to GitHub** (local second opinion only).

---

## Executive summary

The PR delivers what it claims: `register_dllm()` registers two `ModelRegistry` keys via lazy FQCN, both pointing at `DllmMockLlada2ForCausalLM`, with README/design docs and tests gated on an installable `vllm`. **CI on the default path (`uv sync --group dev` only) does not install the `vllm` extra**, so the tests that actually assert registration and mock import **are skipped** in the matrix that currently gates merge. That is the main reason a strict maintainer might hesitate.

**Verdict:** **Mergeable from a process standpoint** (checks green, DCO, mergeable state per `gh`), but **not fully “proven in CI”** for the new behavior. Whether that is acceptable depends on project policy: if Phase 2 is explicitly “registration hook + stub, validate with optional vLLM env,” this is fine; if every PR must exercise new code in default CI, add a job or install path.

**Recommendation:** **Approve with follow-ups** (below), or **merge** if the team already treats `vllm` extra as the integration tier.

---

## What looks solid

1. **Lazy registration** — Using `ModelRegistry.register_model(arch, "<module>:<Class>")` matches vLLM’s documented pattern (see in-repo dummy plugin `tests/plugins/vllm_add_dummy_model/…`: string lazy targets avoid eager CUDA/torch import at plugin load time).

2. **Idempotency** — Pre-checking `arch in ModelRegistry.get_supported_archs()` before registering avoids repeatedly hitting `register_model`’s overwrite **warning** path (vLLM warns when an architecture is already present and then overwrites).

3. **Failure mode** — `find_spec("vllm")` short-circuit, then `try: import ModelRegistry`, matches the README story: broken installs that pass `find_spec` but fail import get DEBUG + `exc_info`, no crash.

4. **Documentation** — `docs/MOCK_STACK_MODEL.md` is clear that this is not production inference; README/DESIGN updates reduce the earlier “skeleton no-op” confusion.

5. **Governance** — DCO sign-off, conventional title/body, links issues #5 / #24; **GitHub Actions** and **DCO** reported success for this PR.

6. **Mock shape (high level)** — `forward` / `IntermediateTensors` / `embed_input_ids` / `compute_logits` resemble patterns used in real vLLM causal stacks (e.g. first rank embeds, non-last returns `IntermediateTensors` with `hidden_states` and often `residual`). For **PP = 1**, this is plausibly enough for early wiring.

---

## Critical / high-priority concerns

### 1. Default CI does not run the meaningful tests

Workflow on `main` (and thus this PR unless changed) runs:

- `uv sync --locked --group dev`
- pre-commit
- `pytest`

There is **no** `uv sync … --extra vllm` step. Tests such as `test_register_dllm_registers_architectures_when_vllm_present` and `test_mock_model_class_importable_when_vllm_present` use `pytest.importorskip("vllm")`, so they **skip** in default CI. The PR description (“11 passed, 3 skipped without vllm extra”) is consistent with that.

**Impact:** Merge can be green while **registration and mock import are never executed** in automation.

**Second opinion:** For Phase 2, this may be intentional cost control (vLLM is heavy). For **readiness**, either:

- Add an optional scheduled workflow or a separate job with `extra vllm` (even `if: github.event_name == 'schedule'` or manual `workflow_dispatch`), or  
- Pin a minimal vLLM version in CI and install it on one Python version only.

Without one of these, “ready to merge” means “ready to merge **unverified** integration,” not “verified against vLLM.”

### 2. Architecture name `LLaDA2ForCausalLM` bound to a mock

Both registered keys map to the **same mock class**. The **explicit** id `DllmMockLlada2StackTest` is good. The **prod-style** name `LLaDA2ForCausalLM` is a deliberate placeholder (#5) but carries real risk:

- Any HF `config.json` that lists `LLaDA2ForCausalLM` will resolve to **deterministic fake logits**, not a real model, until Phase 7 replaces behavior.
- If upstream vLLM ever registers the same string for a real implementation, load order determines who wins; the plugin’s “already registered → skip” logic would **not** override core.

**Second opinion:** Acceptable for an MVP **if** docs and operator runbooks stress it (they mostly do). A stricter approach would register **only** the explicit mock id until the real class exists, and add the prod name in the same PR as the real model—at the cost of less HF-like configs in Phase 2–6.

### 3. `INFO` logging on every successful registration

Each new architecture logs at **INFO**. In multi-worker or repeated init paths, that can be noisy compared to the previous DEBUG-only skeleton message.

**Suggestion:** Log successful registration at DEBUG; keep WARNING/ERROR for real problems—or log once at INFO with both arch names in a single line.

---

## Medium concerns

### 4. Mock module excluded from `ty` entirely

`pyproject.toml` sets `exclude = ["vllm_dllm_plugin/models/mock_llada2.py"]` plus broad `allowed-unresolved-imports` for `torch.**` / `vllm.**`. That is pragmatic for optional deps but **removes static checking** from the file most coupled to vLLM APIs.

**Risk:** Drift when vLLM refactors model interfaces; failures move to runtime only.

### 5. Incomplete vLLM “full model” surface for pipeline parallel

The docstring admits focus on single-GPU / non-PP bring-up. The code still branches on `get_pp_group()` and returns `IntermediateTensors` with a **zero** residual on non-last ranks. Real models typically thread a real residual and often expose `make_empty_intermediate_tensors` (and related PP helpers). Missing pieces may surface as soon as someone turns **PP > 1** on the mock.

**Second opinion:** Not a blocker for “Phase 2 = registration + deterministic stub,” but the doc should avoid implying PP is fully supported, or the mock should grow `make_empty_intermediate_tensors` when PP integration starts.

### 6. Import style vs public API

The PR uses `from vllm.model_executor.models.registry import ModelRegistry`. vLLM also exposes `from vllm import ModelRegistry` (and `from vllm.model_executor.models import ModelRegistry`). Functionally equivalent today; using the **public** import would reduce coupling to internal module layout.

### 7. `load_weights` is a no-op returning `set()`

Fine for a stub; ensure no code path expects specific unloaded key reporting for this milestone.

---

## Low / nits

- **`**kwargs: object` on `forward`** — Reasonable for forward compatibility with the executor calling extra kwargs.
- **Deterministic logits** — Mass on token `0` is fine for shape tests; document that sampling will be degenerate (already implied).
- **PR body “AI-assisted: Cursor”** — Harmless; some projects prefer stripping marketing footers from merge commits (maintainer preference).

---

## Are these ready to merge?

**PR #27 only** (no other PRs were reviewed in this session).

| Criterion | Assessment |
|-----------|------------|
| CI / DCO / mergeable | Yes (per `gh pr view` for this PR). |
| Code matches stated scope (registration + mock + docs) | Yes. |
| Risk understood and documented | Mostly yes; mock vs prod architecture name remains the largest product risk. |
| Automated proof of vLLM integration | **No** on default CI (skipped tests). |

**Bottom line:** **Okay to merge** if the maintainers explicitly accept “vLLM-gated tests are optional tier.” If the bar is “every new integration path runs in PR CI,” **not ready** until the workflow installs `vllm` (at least once) or adds an optional job that does.

---

## Suggested follow-ups (non-blocking or small follow-on PR)

1. CI: one job with `--extra vllm` (or nightly) running the two `importorskip` tests.  
2. Logging: downgrade per-architecture success to DEBUG.  
3. Consider registering only `DllmMockLlada2StackTest` until real `LLaDA2ForCausalLM` ships, **or** gate the prod-style name behind an env flag.  
4. When PP work starts: add `make_empty_intermediate_tensors` (or equivalent) and a minimal PP test under `vllm` extra.  
5. Prefer `from vllm import ModelRegistry` in `__init__.py`.

---

*Generated from `gh pr view` / `gh pr diff` against `vllm-project/dllm-plugin` and cross-checks against the local `vllm` tree (`ModelRegistry.register_model`, `get_supported_archs`).*
