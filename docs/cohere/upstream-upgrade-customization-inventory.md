# Upstream upgrade customization inventory (v0.21.0 → Cohere fork)

Living document for the **v0.21.0** upstream rebase (`cohere-v0.21.0`). Use it to assign owners, plan merge work, and ask teams how each area can be **reduced** (upstreamed, deleted, or isolated).

**Comparison base:** `v0.21.0` (upstream tag) vs `HEAD` (fork branch).  
**Generated from:** `git diff v0.21.0...HEAD` on branch `cohere-v0.21.0`.

## Summary

| Category | Count | Rebase conflict risk | Typical reduction lever |
| --- | ---: | --- | --- |
| **Modified upstream files** | 46 | High — same paths as upstream | Shrink hunks; upstream features; plugins/hooks |
| **Fork-only additions** | 224 | Low — no path collision | Delete obsolete; move to separate packages/repos |
| **Files with `# cohere` markers** (in `vllm/`, `docker/`, tests touched) | ~20 | Marked hunks easier to audit | Keep markers; audit with `check-cohere-markers` skill |

**Modified upstream diff size:** ~945 insertions / 43 deletions across 46 files (rest of the 270-file diff is additions).

Deep technical notes already exist under [`code_notes/`](code_notes/upstream-diff.md). This doc is the **ownership and reduction** view.

---

## How to use this doc

1. **Tag owners** — Replace `OWNER_TBD:*` placeholders with GitHub teams or individuals.
2. **Per group, ask:**
   - Can this behavior move **upstream** (vLLM PR)?
   - Can it live in a **fork-only package** (`vllm/cohere/`, separate repo) instead of patching core files?
   - Can it be **configuration-only** (env, JSON configs, CI inputs) with no core code diff?
   - Can it be **deleted** (legacy model, unused benchmark, duplicate test)?
3. **Track decisions** in the “Reduction ideas” column during review meetings.

---

## Tier A — Modified upstream files (merge hotspots)

These exist in `vllm-project/vllm` and are edited in the fork. They are the primary source of rebase pain.

### A1. Models, registry, and HF config

**Suggested owner:** `OWNER_TBD:model-platform` (Command-R / C4 / C5 / reward / EAGLE)

| File | Δ (approx) | Purpose |
| --- | --- | --- |
| `vllm/model_executor/models/commandr.py` | +245 | Cohere2 causal + reward paths, MoE routing (`token_choice_with_bias`), pooler wiring |
| `vllm/model_executor/models/cohere2_moe.py` | +140 | C5 MoE module; heavy `# cohere` blocks |
| `vllm/model_executor/models/registry.py` | +11 | Register Cohere / reward / EAGLE / vision / ASR architectures |
| `vllm/transformers_utils/config.py` | +3 | `cohere2moe` / `cohere2_moe` → `Cohere2Config` aliases |
| `vllm/transformers_utils/configs/__init__.py` | small | Config exports |

#### A1 — reduction ideas

- Push `model_type` aliases and any generic MoE config fixes **upstream** if HF standard is stable.
- Keep **model-specific** logic in fork-only modules (`commandr.py`, `cohere2_moe.py`) rather than spreading into `registry.py` where possible.
- Reward models: evaluate upstream “pooling / classify” APIs vs custom `Cohere*ForRewardModel` classes.

**Related fork-only (Tier B):** `cohere_reward.py`, `commandr_eagle.py`, `gemma4_utils.py` (see A1b).

---

### A2. MoE kernels, quantization, and CUDA

**Suggested owner:** `OWNER_TBD:moe-kernels` (performance / TRT-LLM / FP8)

| File | Δ (approx) | Purpose |
| --- | --- | --- |
| `vllm/model_executor/layers/fused_moe/config.py` | +10 | `SigmoidRenorm`, `norm_topk_prob` for Cohere routing |
| `vllm/model_executor/layers/fused_moe/layer.py` | small | Kernel path selection |
| `vllm/model_executor/layers/fused_moe/modular_kernel.py` | small | Routing / modular kernel hooks |
| `vllm/model_executor/layers/fused_moe/unquantized_fused_moe_method.py` | small | Unquantized path |
| `vllm/model_executor/layers/fused_moe/experts/trtllm_*.py` (3 files) | small | TRT-LLM expert backends + `norm_topk_prob` threading |
| `vllm/model_executor/layers/logits_processor.py` | +40 | Cohere logits adjustments |
| `vllm/model_executor/layers/quantization/modelopt.py` | small | ModelOpt / FP8 paths |
| `vllm/model_executor/layers/quantization/utils/quant_utils.py` | small | Quant utilities |
| `benchmarks/kernels/benchmark_moe.py` | +13 | Cohere2 MoE in kernel benchmarks |

#### A2 — reduction ideas

- Upstream **sigmoid-renorm MoE** if vLLM adopts same routing as C4/C5; then delete enum/kernel forks.
- MoE **JSON tuning configs** (Tier B) stay fork-only; avoid editing `layer.py` for per-GPU JSON if upstream gains a plugin config dir.
- Consolidate TRT-LLM changes into one upstream extension point instead of three `trtllm_*.py` patches.

**Related fork-only:** `flashinfer_trtllm_moe.py`, `compressed_tensors_moe.py`, `fused_moe/configs/*.json` (40+ files), `csrc/quantization/w8a8/cutlass/scaled_mm_entry.cu`.

---

### A3. V1 runtime — scheduler, KV cache, worker

**Suggested owner:** `OWNER_TBD:runtime-v1` (scheduler, model runner, spec decode)

| File | Δ (approx) | Purpose |
| --- | --- | --- |
| `vllm/v1/core/sched/utils.py` | +92 | Token repetition detection (`# cohere`); env/logger hooks |
| `vllm/v1/worker/gpu_model_runner.py` | +55 | Reward model `token_classify` carve-out; null-block KV zeroing |
| `vllm/v1/core/kv_cache_utils.py` | +35 | KV layout / grouping (incl. draft layers) |
| `vllm/v1/core/sched/scheduler.py` | +1 | Spec-decode acceptance clamp (`max(0, …)`) |
| `vllm/v1/request.py` | +4 | Repetition streak state on `Request` |
| `vllm/v1/sample/rejection_sampler.py` | −1 | Minor sampling path |
| `vllm/v1/sample/metadata.py` | whitespace | Incidental |

#### A3 — reduction ideas

- **Repetition detection:** propose generic upstream scheduler hook vs Cohere-only function in `sched/utils.py`.
- **Reward pooling carve-out:** upstream supported-pooling API so `gpu_model_runner.py` does not special-case class names.
- **KV null-block fix:** upstream if bug affects all models; otherwise document as minimal hunk and keep.
- Note: **thinking-budget** stack is largely **upstream in v0.21**; fork adds tests under `tests/cohere/` but few core file deltas vs `v0.21.0`. Do not re-port old v0.14-era scheduler diffs blindly.

**Related tests:** `tests/cohere/test_thinking_budget.py`, `tests/v1/entrypoints/openai/test_thinking_token_budget.py`.

---

### A4. API, engine args, LoRA, tokenizers, datasets

**Suggested owner:** `OWNER_TBD:api-serving` (OpenAI API, LoRA, tokenizers)

| File | Purpose |
| --- | --- |
| `vllm/entrypoints/openai/chat_completion/protocol.py` | Chat completion protocol extensions (`# cohere`) |
| `vllm/engine/arg_utils.py` | CLI / engine flags (`# cohere`) |
| `vllm/config/lora.py` | LoRA config (`# cohere`) |
| `vllm/lora/worker_manager.py` | LoRA worker behavior (`# cohere`) |
| `vllm/transformers_utils/tokenizer.py` | Tokenizer behavior (`# cohere`) |
| `vllm/benchmarks/datasets/datasets.py` | `custom_mm`, `skip_chat_template` for benchmarks |
| `vllm/envs.py` | Env defaults (e.g. SHM / xgrammar-related) |

#### A4 — reduction ideas

- Prefer **OpenAI-compatible fields upstream** instead of protocol forks.
- LoRA: align with upstream PEFT story; drop fork patches if upstream covers C5 LoRA tests.
- Benchmark dataset changes: keep in **fork-only benchmark config** if possible, not `datasets.py`.

---

### A5. Build, packaging, and repo hygiene

**Suggested owner:** `OWNER_TBD:release-engineering` (Docker, CI images, wheels)

| File | Purpose |
| --- | --- |
| `docker/Dockerfile` | Depot **WebDAV-only** sccache; cohere overlay contract; build stages |
| `docker/Dockerfile.cpu` | Wheel handoff path for CPU image / upload |
| `docker/Dockerfile.rocm` | ROCm + sccache markers |
| `docker/Dockerfile.nightly_torch` | Nightly torch variant markers |
| `Makefile` | `build-vllm-*` / `build-cohere-*`, Depot build args |
| `setup.py`, `pyproject.toml`, `requirements/common.txt` | Package metadata / deps |
| `.gitignore`, `.pre-commit-config.yaml` | Repo hygiene |
| `docs/assets/contributing/dockerfile-stages-dependency.png` | Doc asset |

#### A5 — reduction ideas

- **sccache:** keep WebDAV-only in `csrc-build` (see `build-and-packaging.md`); never re-merge upstream S3 exports without AWS creds.
- **Two-stage image:** keep fork-only `Dockerfile.cohere` + Makefile targets; minimize edits to upstream `Dockerfile` (isolate cohere hunks between markers).
- Push generic wheel-handoff improvements **upstream** if accepted.

---

### A6. Upstream tests and Buildkite touched

**Suggested owner:** `OWNER_TBD:quality-ci`

| File | Purpose |
| --- | --- |
| `tests/utils.py` | Shared test helpers (`# cohere`) |
| `tests/conftest.py` | Fixtures |
| `tests/basic_correctness/test_basic_correctness.py` | Correctness adjustments |
| `tests/entrypoints/offline_mode/test_offline_mode.py` | Offline mode |
| `tests/quantization/test_compressed_tensors.py` | Quantization coverage |
| `.buildkite/performance-benchmarks/scripts/convert-results-json-to-markdown.py` | Result reporting |

#### A6 — reduction ideas

- Revert upstream test edits if **`tests/cohere/`** alone can cover the behavior.
- Keep Buildkite script changes only if GitHub Actions cannot own reporting.

---

## Tier B — Fork-only additions (no upstream path collision)

Grouped by function. Full list: `git diff --name-status v0.21.0...HEAD | awk '$1=="A"'`.

### B1. CI, GitHub Actions, and automation

**Suggested owner:** `OWNER_TBD:release-engineering`

| Area | Examples |
| --- | --- |
| Workflows | `.github/workflows/build-and-push.yaml`, `build-and-test.yaml`, `build-and-eval.yaml`, `build-and-bench.yaml`, `dispatcher.yaml`, `test-pipeline.yaml`, `auto-rebase-upstream.yaml`, … |
| Actions / scripts | `.github/actions/*`, `.github/scripts/dispatcher-set-matrix.js`, rebase/rerere scripts, wheel upload |
| Buildkite eval configs | `.buildkite/lm-eval-harness/configs/C4-*`, `Command-*` (24 YAMLs) |
| Tooling | `tools/cohere/extract_ci_dump_serving.py`, `Makefile`, `CODEOWNERS` |

#### B1 — reduction ideas

- No upstream merge conflict, but **maintenance cost** is high. Split “must-have” vs “nice-to-have” workflows.
- Consider one **dispatcher** workflow vs four near-duplicate build-and-* workflows.
- Document owner per workflow in `CODEOWNERS` (currently repo-wide `@cohere-ai/vllm-reviewers`).

---

### B2. Cohere Python package (`vllm/cohere/`)

**Suggested owner:** `OWNER_TBD:api-serving` + `OWNER_TBD:model-platform`

| Path | Purpose |
| --- | --- |
| `vllm/cohere/guided_decoding/*` | Structural tags, tool grammar, Command-A/R formats |
| `vllm/cohere/auto_config.py` | Hardware / model auto-config |
| `vllm/cohere/hardware_profiles.yaml` | Profile definitions |
| `vllm/cohere/utils/__init__.py` | Shared helpers (e.g. thinking token IDs) |
| `vllm/cohere/utils/eagle/*` | EAGLE benchmark/sweep scripts (mostly offline tooling) |

#### B2 — reduction ideas

- **Best isolation story:** grow `vllm/cohere/` and **stop patching** `entrypoints/` / `structured_output` where v0.21 already has hooks.
- Move EAGLE sweep scripts to **`benchmarks/`** or a separate tools repo to shrink runtime package.
- Upstream **guided decoding** APIs if structural tags become standard.

---

### B3. Model modules (new files)

**Suggested owner:** `OWNER_TBD:model-platform`

| File | Purpose |
| --- | --- |
| `vllm/model_executor/models/cohere_reward.py` | Legacy + vision reward |
| `vllm/model_executor/models/commandr_eagle.py` | EAGLE draft for Command-R |
| `vllm/model_executor/models/gemma4_utils.py` | Shared utils (verify still needed on v0.21) |
| `vllm/attention/ops/triton_reshape_and_cache_flash.py` | Attention op |

#### B3 — reduction ideas

- Merge reward paths into fewer modules if architectures converge.
- Delete `gemma4_utils.py` if unused after v0.21 port audit.

---

### B4. MoE config JSON + kernel modules (new)

**Suggested owner:** `OWNER_TBD:moe-kernels`

- `vllm/model_executor/layers/fused_moe/configs/*.json` (40+ tuning files)
- `vllm/model_executor/layers/fused_moe/flashinfer_trtllm_moe.py`
- `vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe.py`
- `benchmarks/kernels/run_moe_tune_and_bench.sh`

#### B4 — reduction ideas

- Store JSON in **object storage** or a config repo; load via env instead of bloating git.
- Generate JSON from tuning pipeline; don’t hand-merge on rebase.

---

### B5. Tests, fixtures, and eval harness (`tests/cohere/`)

**Suggested owner:** `OWNER_TBD:quality-ci` + feature owners per subdirectory

| Area | Scale |
| --- | --- |
| Feature tests | ~30+ `test_*.py` (vision, ASR, LoRA, guided gen, thinking, reward, …) |
| Config / maps | `tests/cohere/configs/*` (runner_map, model_eval_map, bee, …) |
| Scripts | `run_tests.sh`, `run-bee-eval.sh`, perf/benchmark scripts |
| Bee eval data | CSV/JSONL fixtures |

#### B5 — reduction ideas

- Map each test file to **Tier A** core change; delete tests for removed features.
- Collapse duplicate coverage (e.g. multiple guided-generation tests).
- Keep **`tests/cohere/`** as the only test delta where possible — avoid modifying upstream `tests/*` (Tier A6).

---

### B6. Documentation, Cursor, and agent tooling

**Suggested owner:** `OWNER_TBD:developer-experience`

- `docs/cohere/**` (code notes, test docs, GitHub docs, perf images)
- `.cursor/skills/**`, `.cursor/rules/**`, `.cursor/agents/**`
- `docker/Dockerfile.cohere`

#### B6 — reduction ideas

- Docs do not affect runtime; keep but assign a **docs maintainer** for rebase freshness (`check-docs-and-cursor-freshness` skill).
- Not a candidate for “reducing fork divergence” in vLLM core — separate concern.

---

## Cross-cutting reduction strategies

| Strategy | Applies to | Effort | Impact on rebase |
| --- | --- | --- | --- |
| **Marker discipline** | All Tier A edits | Low | High — use `# cohere start/end` on every upstream hunk |
| **Upstream PRs** | Generic features (repetition detection, pooling API, MoE routing) | High | Permanent reduction |
| **Isolate under `vllm/cohere/`** | API helpers, guided decoding, auto_config | Medium | Shrinks scattered patches |
| **Config/externalize** | MoE JSON, CI matrices, eval YAML | Medium | Removes large binary-ish diffs |
| **Delete legacy** | Old model paths, unused scripts, duplicate tests | Low | Immediate |
| **Revert incidental** | Whitespace (`metadata.py`), doc images | Low | Small |

---

## Suggested review agenda (by meeting)

| Session | Tier A groups | Key question |
| --- | --- | --- |
| 1 | A5 + B1 | Can CI/Docker stay fork-only with minimal `Dockerfile` hunks? |
| 2 | A1 + B2 + B3 | Which model code can move to `vllm/cohere/` or upstream? |
| 3 | A2 + B4 | MoE: upstream routing vs fork JSON + kernel patches? |
| 4 | A3 + A4 | Runtime/API: smallest hunks for reward, repetition, KV? |
| 5 | A6 + B5 | Test ownership and de-duplication vs upstream tests |

---

## References

| Doc | Content |
| --- | --- |
| [`code_notes/upstream-diff.md`](code_notes/upstream-diff.md) | Index to deep dives |
| [`code_notes/build-and-packaging.md`](code_notes/build-and-packaging.md) | Docker, sccache, wheels |
| [`code_notes/models-and-inference.md`](code_notes/models-and-inference.md) | Models, pooler, EAGLE |
| [`code_notes/runtime-and-scheduling.md`](code_notes/runtime-and-scheduling.md) | Scheduler / worker (incl. historical thinking-budget notes) |
| [`code_notes/ci-and-automation.md`](code_notes/ci-and-automation.md) | Workflows, dispatcher |
| [`code_notes/tests-benchmarks-and-data.md`](code_notes/tests-benchmarks-and-data.md) | Test matrix, benchmarks |
| `.cursor/skills/minimize-upstream-diff/` | Per-file hunk audit workflow |
| `.cursor/skills/check-cohere-markers/` | Marker validation vs upstream tag |

---

## Maintenance

- **Refresh file lists** after large merges:  
  `git diff --name-status v0.21.0...HEAD`
- **Refresh Tier A line counts:**  
  `git diff v0.21.0...HEAD --numstat -- $(git diff --name-status v0.21.0...HEAD | awk '$1=="M"{print $2}')`
- Update **owner placeholders** when teams are assigned in `CODEOWNERS`.
