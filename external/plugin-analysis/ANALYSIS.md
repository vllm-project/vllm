# dllm-plugin: evidence-driven structure and integration analysis

**Analysis workspace path:** `external/plugin-analysis/` (relative to the vLLM repository root used as read-only context: `/Users/akellner/MyDir/Code/Open/vllm`).

**Document generated:** 2026-03-23 (local date when analysis was performed).

**Stale-data note:** This document is anchored to the **git SHAs** below. Any commit after those may invalidate specific file-line references; re-run inventory against new SHAs.

---

## 1. Executive summary

This report compares **`vllm-project/dllm-plugin`** (greenfield) with **`bart-plugin`** (model-only plugin) and **`vllm-metal`** (platform + v1 worker + native build), and maps them to the **closed RFC** [vllm#36155](https://github.com/vllm-project/vllm/issues/36155) (dLLM via spec-decode path reuse). The RFC requires a **custom scheduler**, **custom worker**, and **model**, plus a **small vLLM core change** (draft-token hook not gated only on speculative decoding). Neither reference plugin alone matches that scope: bart registers models only; metal replaces platform/worker for Apple Silicon. **Recommendations** in §8 are each tied to repository paths or vLLM symbols cited below.

---

## 2. Methodology

1. Created isolated directory [`external/plugin-analysis/README.md`](README.md).
2. Cloned via SSH into that directory (git operations **only** inside each clone):
   - `bart-plugin`
   - `vllm-metal`
3. **`vllm-project/dllm-plugin`:** recorded from **GitHub** `main` at the SHA in §3; the **editable package** for PRs lives at the vLLM checkout root [`dllm-plugin/`](../dllm-plugin/) and is **not** kept as a second clone under `plugin-analysis/` (avoids duplicate trees).
4. Recorded **branch**, **HEAD SHA**, **author date** per repo (§3).
5. Inventoried packaging, entry points, packages, CI, tests from clones (`find`, `rg`, read files).
6. Read **vLLM context** from the sibling checkout (no git writes): plugins loader, engine `post_step` / batch queue, `SchedulerConfig.get_scheduler_cls`, worker init, `take_draft_token_ids` on GPU model runner.
7. Pulled RFC thread via `gh issue view 36155` (comments summarized in §6).
8. Synthesized comparison tables, traceability, and recommendations.

**Excluded:** Full vLLM tree was not re-cloned inside `plugin-analysis/`; paths are `../..` from `plugin-analysis` to repo root or absolute paths as noted.

---

## 3. Repositories under analysis (evidence: SHAs)

| Repository | Branch | HEAD SHA | Commit date (author) |
|------------|--------|----------|----------------------|
| [dllm-plugin](https://github.com/vllm-project/dllm-plugin) | `main` | `39bb1cd7c76d849d343d70008c152cf26ee7a7b1` | 2026-03-23 |
| [bart-plugin](https://github.com/vllm-project/bart-plugin) | `master` | `5265385fa1753742917ecde31805b9eb75e999d6` | 2026-03-12 |
| [vllm-metal](https://github.com/vllm-project/vllm-metal) | `main` | `39f983b46dc49897f0ada03d24c731e968a2cb89` | 2026-03-23 |

---

## 4. Structural inventory and comparison

### 4.1 Top-level layout (summary)

- **dllm-plugin** (`39bb1cd`): only `LICENSE`, `README.md` — **no Python package**, no `pyproject.toml`, no tests.
- **bart-plugin** (`5265385`): `pyproject.toml`, `setup.py`, `vllm_bart_plugin/`, `tests/`, `verify_plugin.py`, examples, `scripts/`.
- **vllm-metal** (`39f983b`): `pyproject.toml` (maturin), `Cargo.toml`, `src/lib.rs`, `vllm_metal/` (large), `tests/`, `.github/workflows/ci.yml`, `scripts/`.

### 4.2 Comparison table (RFC-oriented)

Rows cite **representative paths** in the analyzed SHAs (or vLLM context paths).

| Concern | bart-plugin | vllm-metal | dllm-plugin (current) | RFC / dLLM need |
|--------|-------------|-----------|------------------------|-----------------|
| **Build backend** | setuptools (`pyproject.toml` L1–3) | maturin (`pyproject.toml` L1–4) | N/A | setuptools or maturin acceptable; **no Rust required** for baseline dLLM plugin unless Metal-like native ops |
| **Python floor** | `>=3.10` (`pyproject.toml` L10) | `>=3.12,<3.14` (`pyproject.toml` L12) | N/A | Align with **target vLLM** (bart uses `vllm>=0.14.0` L29) |
| **`vllm.general_plugins`** | `bart = vllm_bart_plugin:register_bart_model` (`pyproject.toml` L47–48) | `metal_ops = vllm_metal:register_ops` (`pyproject.toml` L70–71) | None | **Yes**: register model(s); optional extra registrations (configs) like metal `_register_ops` |
| **`vllm.platform_plugins`** | N/A | `metal = vllm_metal:register` (`pyproject.toml` L67–68) | N/A | **Likely N/A** for CUDA dLLM MVP; only if Apple/custom platform |
| **Model registration** | `ModelRegistry.register_model(...)` in `vllm_bart_plugin/__init__.py` L28–35 | STT HF config in `register_ops` → `stt/hf_config.py` (from `__init__.py` L105–107) | N/A | **Yes**: same pattern as bart for architecture name → qualified class |
| **Custom scheduler** | N/A | N/A (uses `MetalPlatform` + worker) | N/A | **Yes**: `--scheduler-cls` → `resolve_obj_by_qualname` (vLLM `scheduler.py` L160–180) |
| **Custom worker** | N/A | `vllm_metal/v1/worker.py` `MetalWorker(WorkerBase)` L66–71 | N/A | **Yes**: `--worker-cls` string → `resolve_obj_by_qualname` in `worker_base.py` L251–254 |
| **v1 `WorkerBase` subclass** | N/A | `MetalWorker` extends `WorkerBase` (`worker.py` L22, L66) | N/A | **Pattern to copy** for GPU dLLM worker (subclass `WorkerBase` / mirror `gpu_worker` behavior) |
| **CI** | Not present in tree at depth 3 | `.github/workflows/ci.yml` (lint matrix macOS/Ubuntu, test macOS) | N/A | **Recommend** GitHub Actions: lint + pytest (bart-style) on Linux when no Metal |
| **Tests** | `tests/test_model_initialization.py`, `test_model_inference.py` | Many `tests/test_*.py` (platform, worker, attention) | N/A | **Recommend** registration tests + shape/contract tests for scheduler/worker |
| **Entry-point filter** | Uses standard group `vllm.general_plugins` | Same + platform group | N/A | Document **`VLLM_PLUGINS`** (vLLM `plugins/__init__.py` L32–58) for selective load |

### 4.3 Evidence: bart-plugin (reference: model-only)

- **Registration API:** `vllm_bart_plugin/__init__.py` defines `register_bart_model()` and calls `ModelRegistry.register_model` with string targets `vllm_bart_plugin.bart:BartForConditionalGeneration` (L28–30) and Florence2 (L32–34).
- **Limits for dLLM:** No scheduler, no worker, no engine contract — **insufficient alone** for RFC which overloads spec-decode fields and commit-0.

### 4.4 Evidence: vllm-metal (reference: deep v1 integration)

- **Dual entry points:** `pyproject.toml` L67–71 — `vllm.platform_plugins` and `vllm.general_plugins`.
- **Platform plugin contract:** `vllm_metal/__init__.py` `_register()` returns FQCN `"vllm_metal.platform.MetalPlatform"` when available (L93–94).
- **General plugin side effects:** `_register_ops()` calls `register_qwen3_asr_config()` (L105–107) — pattern for **extra HF/config registration** without a new platform.
- **Worker:** `vllm_metal/v1/worker.py` imports `WorkerBase` from `vllm.v1.worker.worker_base` (L22) and defines `MetalWorker(WorkerBase)` (L66–71) — **template for “replace execution path”** while staying on v1 APIs.
- **CI:** `.github/workflows/ci.yml` runs `scripts/lint.sh` on multi-OS matrix and `scripts/test.sh` on macOS — **template for automation** (dllm-plugin could use Linux-only matrix).

### 4.5 Evidence: dllm-plugin (SHA in §3)

- **At analyzed SHA:** `README.md` is a one-line title; `LICENSE` only. **No implementation** in that snapshot.
- **Workspace note:** the live package for PRs is maintained at the vLLM checkout root [`dllm-plugin/`](../dllm-plugin/); re-inventory that tree for current files (it may diverge from the SHA above).

---

## 5. vLLM core integration points (read-only context)

### 5.1 Plugin loading

- **Group constants and loader:** [`vllm/plugins/__init__.py`](../../vllm/plugins/__init__.py) — `DEFAULT_PLUGINS_GROUP = "vllm.general_plugins"` (L14), `PLATFORM_PLUGINS_GROUP = "vllm.platform_plugins"` (L19), `load_general_plugins()` executes each loaded callable (L69–82).
- **Filtering:** `VLLM_PLUGINS` env limits which named plugins load (L32–58).

### 5.2 Speculative / draft-token hook (RFC “one engine change”)

- **`use_spec_decode`:** Set from `speculative_config is not None` in [`vllm/v1/engine/core.py`](../../vllm/v1/engine/core.py) L154.
- **`post_step` (sync path):** L410–418 — draft tokens taken only when `not async_scheduling and self.use_spec_decode and model_executed`.
- **`step_with_batch_queue` (deferred structured output):** L514–526 — `take_draft_token_ids` / `update_draft_token_ids_in_output` gated on `self.use_spec_decode`.

**RFC requirement:** Relax these guards so the hook runs whenever the model executed (and draft IDs exist), not only when speculative decoding is enabled — **change belongs in vLLM core**, not in `dllm-plugin`.

### 5.3 Worker draft token surface

- **GPU model runner:** [`vllm/v1/worker/gpu_model_runner.py`](../../vllm/v1/worker/gpu_model_runner.py) L3932–3936 — `take_draft_token_ids` returns `None` if `not self.num_spec_tokens or not self._draft_token_req_ids`. A dLLM worker must ensure **draft/spec path is populated** for the block semantics the RFC maps onto spec-decode fields (plugin responsibility).

### 5.4 Scheduler and worker class resolution

- **Scheduler:** [`vllm/config/scheduler.py`](../../vllm/config/scheduler.py) L123–128 documents default vs `"mod.custom_class"`; L160–180 `get_scheduler_cls()` uses `resolve_obj_by_qualname(self.scheduler_cls)` for strings, with **warning** that custom scheduler interface is not public (L173–176).
- **Worker:** [`vllm/v1/worker/worker_base.py`](../../vllm/v1/worker/worker_base.py) L251–254 — `worker_cls` must be a **string** FQCN resolved via `resolve_obj_by_qualname`.
- **CLI:** [`vllm/engine/arg_utils.py`](../../vllm/engine/arg_utils.py) exposes `--scheduler-cls` and `--worker-cls` (e.g. L933, L1199 area per grep).

---

## 6. RFC and discussion traceability ([issue #36155](https://github.com/vllm-project/vllm/issues/36155))

**State:** Closed (RFC); author noted project kickoff follow-up in thread.

### 6.1 RFC obligations → where implemented

| RFC obligation | Typical owner | Evidence / extension point |
|----------------|---------------|----------------------------|
| Reuse spec-decode fields for block in / Committed / next block | Plugin scheduler + worker + docs | vLLM fields consumed by default sched/worker; plugin replaces behavior |
| Engine: draft hook after every model execution | **vLLM core** | `core.py` `post_step` / `step_with_batch_queue` (§5.2) |
| Commit-0: rollback `num_computed_tokens` | **Plugin scheduler** `update_from_output` | No bart equivalent; must be new code in dllm-plugin |
| Grammar / structured output: do not apply AR grammar to dLLM next block | **Plugin scheduler** overrides `update_draft_token_ids` / `update_draft_token_ids_in_output` | RFC issue body “Grammar and structured output” section |
| Validate scheduler+worker pairing | **Plugin** at load or model load | RFC issue body “Validation” section |
| MVP: one architecture (e.g. LLaDA2.x) | **Plugin** `ModelRegistry` + model module | bart `__init__.py` pattern (§4.3) |

### 6.2 Maintainer thread → design consequences

| Commenter / topic | Summary | Implication for dllm-plugin |
|-------------------|---------|------------------------------|
| **benchislett** — structured outputs | Multi-token commits per step vs grammar masks designed for spec-decode | Document **worker/model** strategy (PDA validate before commit); may still need scheduler overrides for async/deferred path (`core.py` L514+) |
| **benchislett** — custom masks | Semi-causal / bidirectional attention not uniformly fast | Document **FlexAttention-first** baseline (as **robertgshaw2-redhat** suggested in thread); platform matrix in README |
| **robertgshaw2-redhat** | “start with FlexAttention for baseline impl” | Analysis aligns with vLLM `FlexAttentionBackend` discussion in maintainers; plugin doc should name attention backend expectations |
| **AlonKellner-RedHat** | Commit only PDA-valid tokens; remask on contradiction | Implementation note for **model + scheduler** contract doc |
| **mgoin** — block assumption / early-exit | Competitive dLLMs fit semi-causal; batching aligns steps | **Continuous batching** already matches “one schedule = one diffusion step”; document in `docs/` for operators |

---

## 7. Gap analysis (synthesis)

1. **dllm-plugin** is **empty** of code; all RFC work is ahead.
2. **bart-plugin** covers **~20%** of RFC needs (packaging + `ModelRegistry` + tests) but **0%** of scheduler/worker/spec-decode semantics.
3. **vllm-metal** covers **plugin packaging patterns**, **optional `general_plugins` side registrations**, **v1 `WorkerBase` subclass**, and **CI** — most transferable for process and worker skeleton; **platform plugin** is likely **out of scope** for a CUDA-first dLLM MVP.
4. **vLLM core** still gates draft-token updates on `use_spec_decode` — **blocking** for RFC as written until a small core patch lands; plugin should declare **minimum vLLM version** once known.

---

## 8. Recommendations (evidence-backed)

1. **Add `pyproject.toml` + package layout** mirroring bart (`bart-plugin/pyproject.toml`: setuptools, `vllm` pin, `[project.entry-points."vllm.general_plugins"]`). *Evidence:* bart L47–48; dllm-plugin has no build metadata.
2. **Single registration entrypoint** `register_dllm` calling `ModelRegistry.register_model` for each supported architecture (lazy import strings). *Evidence:* `vllm_bart_plugin/__init__.py` L28–35.
3. **Ship `DllmScheduler` and `DllmWorker` as importable FQCNs**; document `vllm serve ... --scheduler-cls ... --worker-cls ...`. *Evidence:* `scheduler.py` L160–180; `worker_base.py` L251–254; RFC field table.
4. **Implement startup validation** that dLLM-tagged models require plugin scheduler/worker strings (RFC “Validation”). *Evidence:* RFC issue body; absence in bart (N/A there).
5. **Override draft-token update methods** on scheduler for structured output / async compatibility. *Evidence:* `core.py` L514–526 + benchislett comment on thread.
6. **Document attention strategy** (FlexAttention / custom `logical_mask_mod`) and platform caveats. *Evidence:* robertgshaw2-redhat comment; vLLM FlexAttention backend (not duplicated here — see main vLLM tree).
7. **Add CI** modeled on bart (lightweight) or metal’s structure without macOS-only tests: `pytest` + ruff on Ubuntu. *Evidence:* `vllm-metal/.github/workflows/ci.yml` L1–45; bart tests directory.
8. **Version-gate on vLLM** after core draft-hook patch merges; until then, integration tests marked `skip` with reason. *Evidence:* §5.2 guards in `core.py`.
9. **Keep platform plugin out of MVP** unless targeting Apple Silicon dLLM path. *Evidence:* metal `pyproject.toml` L67–68 vs RFC CUDA-centric spec-decode reuse.

---

## 9. Suggested initial work breakdown (for implementation team)

| Phase | Deliverable | Depends on |
|-------|-------------|------------|
| A | `pyproject.toml`, `vllm_dllm_plugin/__init__.py`, README install + `VLLM_PLUGINS` + CLI example | None |
| B | Stub `DllmScheduler` / `DllmWorker` (importable, raise `NotImplementedError` or no-op with warnings) | A |
| C | Core vLLM patch: draft hook guard relaxation + release version bump | Upstream PR |
| D | Real scheduler: `spec_token_ids`, `scheduled_spec_decode_tokens`, `num_scheduled_tokens`, commit-0 rollback | C |
| E | Real worker: map to model forward, `sampled_token_ids`, `take_draft_token_ids` | C, D |
| F | First model (e.g. LLaDA2) + tests | E |
| G | Docs: structured output policy, attention backends, observability | Parallel to D–F |

---

## 10. Risks and open questions

- **Non-public scheduler API:** `get_scheduler_cls` logs compatibility warning (`scheduler.py` L173–176) — plugins may break on upstream refactors.
- **`take_draft_token_ids` preconditions:** GPU model runner returns `None` without spec-token state (`gpu_model_runner.py` L3932–3934) — dLLM worker must align internal flags with spec-decode expectations.
- **Structured output + async scheduling:** Interaction is concentrated in `step_with_batch_queue` deferred branch — highest complexity per benchislett.
- **Branch naming:** `bart-plugin` uses **`master`**; others use **`main`** — automation or docs must not assume one default.

---

## 11. Appendix A — Command log (reproducibility)

```bash
# Workspace (from vLLM repo root)
mkdir -p external/plugin-analysis && cd external/plugin-analysis
git clone git@github.com:vllm-project/dllm-plugin.git
git clone git@github.com:vllm-project/bart-plugin.git
git clone git@github.com:vllm-project/vllm-metal.git
# SHAs recorded in §3
```

---

## 12. Appendix B — `gh` reference

```bash
gh issue view 36155 --repo vllm-project/vllm --comments --json title,state,comments,url
```

Issue URL: https://github.com/vllm-project/vllm/issues/36155

---

*End of analysis document.*
