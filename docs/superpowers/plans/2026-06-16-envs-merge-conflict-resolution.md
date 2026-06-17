# envs.py / test_envs.py Merge Conflict Resolution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Resolve the in-progress merge conflict in `vllm/envs.py` and `tests/test_envs.py` by taking the pydantic-refactor branch structurally and porting main's semantic delta, then amend the reusable runbook to cover deletions and modifications.

**Architecture:** Follow the runbook `docs/superpowers/specs/2026-05-14-envs-merge-conflict-resolution-design.md`: `git checkout --ours` each conflicted file (branch wins structurally), then apply targeted field-level edits for main's added / modified / deleted env vars. No structural blending.

**Tech Stack:** Python 3.12, `pydantic` / `pydantic_settings`, `uv` venv, `pytest`, `pre-commit`. All Python via `.venv/bin/python` (never system python — see `AGENTS.md`).

**Note on TDD:** This is a merge resolution, not feature work. The "test" for each task is the runbook's verification commands (parse-import, spot-check, the one ported pytest). Tasks are sequential by necessity: the structural `checkout --ours` (Task 1) must precede every port. Do NOT commit until the final task — a mid-merge `git commit` finalizes the merge commit, so all resolution lands in one signed-off commit.

**Reference facts (captured from branch HEAD `520828789` merge, base `ba94a3b99`):**
- After `git checkout --ours vllm/envs.py`, the file is byte-identical to `git show HEAD:vllm/envs.py`. All line numbers below refer to that post-checkout state.
- `_env_set` is used in many places — KEEP. `_warn_deprecated_env` (lines 55–63) and `import warnings` (line 10) are used ONLY by the two deprecation validators being removed in Task 5 — they become dead and must be removed.
- `use_flashinfer_moe_int4` (FlashInferSettings) and `_parse_triton_attn_use_td` (QuantSettings) sit among the deletion targets but are NOT deprecated — KEEP both.

---

## Task 1: Structural resolution — take ours for both files

**Files:**
- Modify: `vllm/envs.py` (overwrite conflicted with branch version)
- Modify: `tests/test_envs.py` (overwrite conflicted with branch version)

- [ ] **Step 1: Take ours for both conflicted files**

```bash
cd /Users/vidamoda/dev/os-help/vllm
git checkout --ours vllm/envs.py tests/test_envs.py
```

- [ ] **Step 2: Verify zero conflict markers remain**

```bash
grep -n "<<<<<<< \|>>>>>>> \|=======" vllm/envs.py tests/test_envs.py
```

Expected: no output (exit code 1 from grep).

- [ ] **Step 3: Stage both files**

```bash
git add vllm/envs.py tests/test_envs.py
```

- [ ] **Step 4: Sanity-check the module imports before porting**

```bash
.venv/bin/python -c "import vllm.envs; print('import OK')"
```

Expected: `import OK`. (Establishes a clean baseline; the branch version is already self-consistent.)

Do NOT commit yet.

---

## Task 2: Port additions — `MediaSettings` (2 audio vars)

**Files:**
- Modify: `vllm/envs.py` — `MediaSettings`, after `max_audio_clip_filesize_mb` (≈ line 1203–1210)

- [ ] **Step 1: Add the two audio fields after `max_audio_clip_filesize_mb`'s closing `)`**

Insert immediately after the `max_audio_clip_filesize_mb = Field(...)` block (before `video_loader_backend`):

```python
    max_audio_decode_duration_s: int = Field(
        default=600,
        description=(
            "Maximum decoded audio duration in seconds. Compressed audio "
            "files (e.g. OPUS at very low bitrate) can expand into gigabytes "
            "of float32 PCM. This limit is enforced during decoding so the "
            "memory is never allocated. Default is 600s (10 minutes)."
        ),
    )
    max_audio_preprocess_workers: int = Field(
        default_factory=lambda: max(1, min(os.cpu_count() or 1, 2)),
        description=(
            "Maximum number of worker threads used for STT preprocessing. "
            "The default intentionally caps at 2 because that performed best "
            "in profiling."
        ),
    )
```

- [ ] **Step 2: Verify both parse with correct defaults**

```bash
.venv/bin/python -c "import vllm.envs as e; print(e.VLLM_MAX_AUDIO_DECODE_DURATION_S, e.VLLM_MAX_AUDIO_PREPROCESS_WORKERS)"
```

Expected: `600 <N>` where `<N>` is 1 or 2.

---

## Task 3: Port additions — `CompilationSettings`, `ServerSettings`, `DistributedSettings`, `RocmSettings`, `ConnectorSettings` (9 vars)

**Files:**
- Modify: `vllm/envs.py` — five classes, anchors below.

- [ ] **Step 1: `CompilationSettings` — add 2 fields after `use_triton_awq` (≈ line 1981–1984), before the `@field_validator` for `float32_matmul_precision`**

```python
    fastsafetensors_queue_size: int = Field(
        default=0,
        description=(
            "Queue size for fastsafetensors ParallelLoader pipelined weight "
            "loading. Peak load-time VRAM is roughly model_weights + "
            "(1 + queue_size) * shard_size. Default 0 preserves the "
            "non-pipelined memory footprint. Set to 1 (or higher) to overlap "
            "producing the next shard's device buffer with the consumer "
            "copying the current shard into model params, at the cost of "
            "`queue_size` extra shard-sized buffers resident at peak."
        ),
    )
    triton_force_first_config: bool = Field(
        default=False,
        description=(
            "If set, monkey-patch triton.runtime.autotuner.Autotuner.run to "
            "skip benchmarking and select the first valid config. Used to "
            "eliminate autotuning variability when measuring kernel "
            "performance."
        ),
    )
```

- [ ] **Step 2: `ServerSettings` — add `regex_compilation_timeout_s` after `tool_parse_regex_timeout_seconds` (≈ line 551–554), before `tool_json_error_automatic_retry`**

```python
    regex_compilation_timeout_s: int = Field(
        default=5,
        description=(
            "Maximum time in seconds allowed for regex compilation in "
            "structured output backends (xgrammar, outlines). Prevents ReDoS "
            "attacks where adversarial patterns cause exponential DFA "
            "state-space explosion. Set to 0 to disable the timeout (not "
            "recommended in production)."
        ),
    )
```

- [ ] **Step 3: `ServerSettings` — add `worker_shutdown_timeout_seconds` after `execute_model_timeout_seconds` (≈ line 433–439), before `keep_alive_on_engine_death`**

```python
    worker_shutdown_timeout_seconds: int = Field(
        default=5,
        description=(
            "Timeout in seconds for engine and worker process shutdown."
        ),
    )
```

- [ ] **Step 4: `DistributedSettings` — add 3 DeepEP v2 fields after `deepep_low_latency_use_mnnvl` (≈ line 1801–1807), before `dbo_comm_sms`**

```python
    deepep_v2_allow_hybrid_mode: bool = Field(
        default=True,
        description="DeepEP v2: enable two-tier NVLink+RDMA hybrid mode.",
    )
    deepep_v2_prefer_overlap: bool = Field(
        default=False,
        description="DeepEP v2: use fewer SMs at slight throughput cost.",
    )
    deepep_v2_allow_multiple_reduction: bool = Field(
        default=False,
        description="DeepEP v2: trade precision for transfer size in combine.",
    )
```

- [ ] **Step 5: `RocmSettings` — add `mxfp8_emulation_dequant_at_load` after `rocm_use_aiter` (≈ line 1344–1351), before `rocm_use_aiter_paged_attn`**

```python
    mxfp8_emulation_dequant_at_load: bool = Field(
        default=True,
        description=(
            "On hardware without a native MXFP8 kernel (e.g. ROCm gfx942 / "
            "MI300), the MXFP8 emulation path dequantizes weights "
            "MXFP8->BF16 once at load time and runs as a BF16 checkpoint. "
            "Set to 0 to fall back to per-step dequant: keeps the 1-byte "
            "MXFP8 weights at the cost of dequantizing every forward step. "
            "Default on."
        ),
    )
```

- [ ] **Step 6: `ConnectorSettings` — add `wsl2_enable_pin_memory` after `weight_offloading_disable_uva` (≈ line 2159–2164), before `enable_cudagraph_gc`**

```python
    wsl2_enable_pin_memory: bool = Field(
        default=False,
        description=(
            "On WSL2 with a compatible kernel (>= 4.19.121), pinned memory "
            "is supported but disabled by default due to a small performance "
            "regression. Set to 1 when pinned memory or UVA is required "
            "(e.g. CPU offloading or v2 model runner)."
        ),
    )
```

- [ ] **Step 7: Verify all 9 added vars parse with correct defaults**

```bash
.venv/bin/python -c "import vllm.envs as e; print(\
e.VLLM_FASTSAFETENSORS_QUEUE_SIZE, \
e.VLLM_TRITON_FORCE_FIRST_CONFIG, \
e.VLLM_REGEX_COMPILATION_TIMEOUT_S, \
e.VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS, \
e.VLLM_DEEPEP_V2_ALLOW_HYBRID_MODE, \
e.VLLM_DEEPEP_V2_PREFER_OVERLAP, \
e.VLLM_DEEPEP_V2_ALLOW_MULTIPLE_REDUCTION, \
e.VLLM_MXFP8_EMULATION_DEQUANT_AT_LOAD, \
e.VLLM_WSL2_ENABLE_PIN_MEMORY)"
```

Expected: `0 False 5 5 True False False True False`

---

## Task 4: Port modification — `enforce_strict_tool_calling` default flip + `compile_factors` ignore-set

**Files:**
- Modify: `vllm/envs.py` — `ServerSettings` `enforce_strict_tool_calling` (≈ line 563–570); `compile_factors()` `ignored_factors` (anchor `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS` ≈ line 2669)

- [ ] **Step 1: Flip `enforce_strict_tool_calling` default `False → True`**

Replace the existing field:

```python
    enforce_strict_tool_calling: bool = Field(
        default=False,
        description=(
            "When 1, the model structural tags will be used to enforce the "
            "model output conforming to the model's tool-calling format and "
            "schema. Default 0 (off)."
        ),
    )
```

with:

```python
    enforce_strict_tool_calling: bool = Field(
        default=True,
        description=(
            "Enforce function parameter schemas in structural-tag based "
            "tool calling."
        ),
    )
```

- [ ] **Step 2: Add 3 new vars to `ignored_factors` in `compile_factors()`**

The branch uses an *exclude*-set (inverse of main's include-list). After the line `"VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS",` add:

```python
        "VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS",
```

After the line `"VLLM_MAX_AUDIO_CLIP_FILESIZE_MB",` add:

```python
        "VLLM_MAX_AUDIO_DECODE_DURATION_S",
        "VLLM_MAX_AUDIO_PREPROCESS_WORKERS",
```

- [ ] **Step 3: Verify default flip and that compile_factors runs**

```bash
.venv/bin/python -c "import vllm.envs as e; assert e.VLLM_ENFORCE_STRICT_TOOL_CALLING is True; \
f = e.compile_factors(); \
assert 'VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS' not in f, 'should be ignored'; \
assert 'VLLM_MAX_AUDIO_DECODE_DURATION_S' not in f, 'should be ignored'; \
print('strict=True, audio/shutdown ignored, factors OK', len(f))"
```

Expected: `strict=True, audio/shutdown ignored, factors OK <N>`

---

## Task 5: Port deletions — remove 11 deprecated vars + validators + dead helper (#44992)

**Files:**
- Modify: `vllm/envs.py` — `FlashInferSettings`, `QuantSettings`, module-level helper + import

- [ ] **Step 1: `FlashInferSettings` — delete 7 deprecated fields**

Delete these field blocks (keep `use_flashinfer_moe_int4` and `use_flashinfer_sampler` — they are NOT deprecated):
`use_flashinfer_moe_fp16`, `use_flashinfer_moe_fp8`, `use_flashinfer_moe_fp4`, `use_flashinfer_moe_mxfp4_mxfp8`, `use_flashinfer_moe_mxfp4_mxfp8_cutlass`, `use_flashinfer_moe_mxfp4_bf16`, `flashinfer_moe_backend`.

- [ ] **Step 2: `FlashInferSettings` — delete the `_warn_deprecated_moe_backend_envs` model_validator**

Delete the entire method (keep `_parse_json_thresholds`):

```python
    @model_validator(mode="after")
    def _warn_deprecated_moe_backend_envs(self) -> "FlashInferSettings":
        moe_backend_msg = (
            "Use --moe-backend (e.g. flashinfer_trtllm, flashinfer_cutlass)."
        )
        for var in (
            "VLLM_USE_FLASHINFER_MOE_FP16",
            "VLLM_USE_FLASHINFER_MOE_FP8",
            "VLLM_USE_FLASHINFER_MOE_FP4",
            "VLLM_USE_FLASHINFER_MOE_MXFP4_BF16",
        ):
            _warn_deprecated_env(var, "v0.23", moe_backend_msg)
        _warn_deprecated_env(
            "VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8",
            "v0.23",
            "Use --moe-backend flashinfer_trtllm with "
            "--quantization_config.moe.activation mxfp8.",
        )
        _warn_deprecated_env(
            "VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8_CUTLASS",
            "v0.23",
            "Use --moe-backend flashinfer_cutlass with "
            "--quantization_config.moe.activation mxfp8.",
        )
        _warn_deprecated_env(
            "VLLM_FLASHINFER_MOE_BACKEND",
            "v0.23",
            "Use --moe-backend flashinfer_trtllm, flashinfer_cutlass, or "
            "flashinfer_cutedsl.",
        )
        return self
```

- [ ] **Step 3: `QuantSettings` — delete 4 deprecated fields**

Delete: `mxfp4_use_marlin` (the `bool | None` field), `nvfp4_gemm_backend` (the multi-line `Literal[...] | None` field), `use_nvfp4_ct_emulations`, `use_fbgemm`. Keep neighbors `deepepll_nvfp4_dispatch`, `q_scale_constant`, `use_oink_ops`.

- [ ] **Step 4: `QuantSettings` — delete the `_parse_mxfp4` field_validator and the `_warn_deprecated_backend_envs` model_validator**

Delete `_parse_mxfp4` (bound to the now-removed `mxfp4_use_marlin`):

```python
    @field_validator("mxfp4_use_marlin", mode="before")
    @classmethod
    def _parse_mxfp4(cls, v: Any) -> Any:
        if v is None:
            return None
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            return bool(int(v))
        return v
```

KEEP `_parse_triton_attn_use_td` (unrelated). Then delete `_warn_deprecated_backend_envs`:

```python
    @model_validator(mode="after")
    def _warn_deprecated_backend_envs(self) -> "QuantSettings":
        _warn_deprecated_env(
            "VLLM_MXFP4_USE_MARLIN",
            "v0.23",
            "Use --moe-backend marlin or --linear-backend marlin.",
        )
        _warn_deprecated_env(
            "VLLM_USE_NVFP4_CT_EMULATIONS",
            "v0.23",
            "Use --linear-backend emulation.",
        )
        _warn_deprecated_env(
            "VLLM_NVFP4_GEMM_BACKEND",
            "v0.23",
            "Use --linear-backend.",
        )
        _warn_deprecated_env(
            "VLLM_USE_FBGEMM",
            "v0.23",
            "Use --linear-backend fbgemm.",
        )
        return self
```

- [ ] **Step 5: Remove the now-dead `_warn_deprecated_env` helper and `import warnings`**

Delete the helper (≈ lines 55–63):

```python
def _warn_deprecated_env(name: str, removal_version: str, replacement: str) -> None:
    """Emit a FutureWarning if an env var is explicitly set."""
    if _env_set(name):
        warnings.warn(
            f"{name} is deprecated and will be removed in "
            f"{removal_version}. {replacement}",
            FutureWarning,
            stacklevel=2,
        )
```

Delete `import warnings` (line 10). KEEP `_env_set`.

- [ ] **Step 6: Verify deletions — vars gone, helper gone, module still imports**

```bash
.venv/bin/python -c "import vllm.envs as e; print('import OK')"
.venv/bin/python -c "import vllm.envs as e; \
[print('FAIL present:', v) or exit(1) for v in ['VLLM_USE_FBGEMM','VLLM_MXFP4_USE_MARLIN','VLLM_FLASHINFER_MOE_BACKEND','VLLM_NVFP4_GEMM_BACKEND','VLLM_USE_NVFP4_CT_EMULATIONS','VLLM_USE_FLASHINFER_MOE_FP16','VLLM_USE_FLASHINFER_MOE_FP8','VLLM_USE_FLASHINFER_MOE_FP4','VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8','VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8_CUTLASS','VLLM_USE_FLASHINFER_MOE_MXFP4_BF16'] if hasattr(e, v)] or print('all 11 removed')"
.venv/bin/python -c "import vllm.envs as e; assert hasattr(e, 'VLLM_USE_FLASHINFER_MOE_INT4'); print('moe_int4 kept OK')"
grep -n "_warn_deprecated_env\|^import warnings" vllm/envs.py || echo "helper + warnings import removed"
```

Expected: `import OK`, `all 11 removed`, `moe_int4 kept OK`, `helper + warnings import removed`.

---

## Task 6: Port test delta — expand `test_precompiled_install_flags_are_orthogonal`

**Files:**
- Modify: `tests/test_envs.py` — `test_precompiled_install_flags_are_orthogonal` (≈ line 86)

- [ ] **Step 1: Replace the branch's single-block test with main's expanded version**

Replace:

```python
def test_precompiled_install_flags_are_orthogonal() -> None:
    with patch.dict(
        os.environ,
        {
            "VLLM_PRECOMPILED_WHEEL_LOCATION": "/tmp/vllm.whl",
            "VLLM_USE_PRECOMPILED_RUST": "1",
        },
        clear=False,
    ):
        assert environment_variables["VLLM_USE_PRECOMPILED"]() is False
        assert environment_variables["VLLM_USE_PRECOMPILED_RUST"]() is True
```

with:

```python
def test_precompiled_install_flags_are_orthogonal() -> None:
    # The Rust frontend flag is independent of the C-extension precompiled
    # flag: requesting the precompiled Rust frontend must not implicitly
    # enable the precompiled C extensions.
    with patch.dict(os.environ, {"VLLM_USE_PRECOMPILED_RUST": "1"}, clear=True):
        assert environment_variables["VLLM_USE_PRECOMPILED"]() is False
        assert environment_variables["VLLM_USE_PRECOMPILED_RUST"]() is True

    # ...and the reverse: requesting precompiled C extensions (here via a
    # wheel location, which enables VLLM_USE_PRECOMPILED) must not flip the
    # Rust frontend flag.
    with patch.dict(
        os.environ, {"VLLM_PRECOMPILED_WHEEL_LOCATION": "/tmp/vllm.whl"}, clear=True
    ):
        assert environment_variables["VLLM_USE_PRECOMPILED"]() is True
        assert environment_variables["VLLM_USE_PRECOMPILED_RUST"]() is False

    # ...and with both set together, each flag is still parsed independently.
    with patch.dict(
        os.environ,
        {
            "VLLM_PRECOMPILED_WHEEL_LOCATION": "/tmp/vllm.whl",
            "VLLM_USE_PRECOMPILED_RUST": "1",
        },
        clear=True,
    ):
        assert environment_variables["VLLM_USE_PRECOMPILED"]() is True
        assert environment_variables["VLLM_USE_PRECOMPILED_RUST"]() is True
```

- [ ] **Step 2: Run the ported test (must PASS — branch shim + validator already satisfy it)**

```bash
.venv/bin/python -m pytest tests/test_envs.py -k test_precompiled_install_flags_are_orthogonal -v
```

Expected: PASS. (If it fails on the `clear=True` cases, the branch's `_force_use_precompiled_when_wheel_set` validator is the thing under test — investigate before forcing.)

- [ ] **Step 3: Run the full test_envs.py to confirm no regressions**

```bash
.venv/bin/python -m pytest tests/test_envs.py -v
```

Expected: all PASS.

---

## Task 7: Amend the reusable runbook

**Files:**
- Modify: `docs/superpowers/specs/2026-05-14-envs-merge-conflict-resolution-design.md`

- [ ] **Step 1: Add a "deletion" bullet to the classification list (section "Enumerate the semantic delta from main")**

After the "Pure rename or comment change" bullet, add:

```markdown
- **Existing env var deleted:** a main-side removal is a semantic delta
  too. If the branch *deleted* the same var, taking ours already agrees —
  nothing to do. But if the branch *refactored* the var (kept it as a
  pydantic `Field`, possibly with deprecation scaffolding), "take ours"
  silently re-introduces a var main intentionally removed. Port the
  deletion: remove the `Field`, any `field_validator` bound only to it, any
  deprecation `model_validator` / helper that becomes dead, and any import
  that becomes unused. Removing the field also drops it from generated
  structures (`_VAR_TO_PATH`, the `environment_variables` back-compat shim)
  automatically — no separate edit needed.
```

- [ ] **Step 2: Strengthen the modification guidance (same list)**

Append to the "Existing env var modified" bullet:

```markdown
  A pure default change (e.g. a bool default flipped `False`->`True`) is a
  one-line edit to the `Field(default=...)`; pydantic parses
  `"1"`/`"0"`/`"true"`/`"false"` natively, so no validator is needed.
```

- [ ] **Step 3: Add a `compile_factors()` polarity note to step 3 ("Port the semantic delta")**

Add as a new bullet:

```markdown
- `compile_factors()` polarity: if the branch represents compile factors as
  an `ignored_factors` *exclude*-set (the inverse of main's explicit
  include-list), then a main-side "add var to the include-list" ports to the
  *opposite* edit on the branch — "add var to the ignore set". Confirm the
  polarity against an existing neighbor before editing.
```

- [ ] **Step 4: Add a dated appendix entry for this run**

Append to the end of the file:

```markdown
## Appendix B: 2026-06-16 execution

Merge of `origin/main` (`520828789`) into the branch, base `ba94a3b99`.
13 main-side commits touched `vllm/envs.py`. Resolved take-ours + ported:

- **Additions (11 new vars):** `VLLM_MAX_AUDIO_DECODE_DURATION_S`,
  `VLLM_MAX_AUDIO_PREPROCESS_WORKERS` (MediaSettings);
  `VLLM_FASTSAFETENSORS_QUEUE_SIZE`, `VLLM_TRITON_FORCE_FIRST_CONFIG`
  (CompilationSettings); `VLLM_REGEX_COMPILATION_TIMEOUT_S`,
  `VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS` (ServerSettings);
  `VLLM_DEEPEP_V2_ALLOW_HYBRID_MODE`, `VLLM_DEEPEP_V2_PREFER_OVERLAP`,
  `VLLM_DEEPEP_V2_ALLOW_MULTIPLE_REDUCTION` (DistributedSettings);
  `VLLM_MXFP8_EMULATION_DEQUANT_AT_LOAD` (RocmSettings);
  `VLLM_WSL2_ENABLE_PIN_MEMORY` (ConnectorSettings). The 3 timeout/limit
  vars were added to the `ignored_factors` exclude-set.
- **Modification:** `VLLM_ENFORCE_STRICT_TOOL_CALLING` default `False -> True`
  (#45003).
- **Deletions (#44992):** removed 11 deprecated vars
  (`VLLM_MXFP4_USE_MARLIN`, `VLLM_USE_FLASHINFER_MOE_FP16/FP8/FP4`,
  `VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8(_CUTLASS)`,
  `VLLM_USE_FLASHINFER_MOE_MXFP4_BF16`, `VLLM_FLASHINFER_MOE_BACKEND`,
  `VLLM_USE_NVFP4_CT_EMULATIONS`, `VLLM_NVFP4_GEMM_BACKEND`,
  `VLLM_USE_FBGEMM`) plus the `_warn_deprecated_moe_backend_envs` /
  `_warn_deprecated_backend_envs` validators, the `_parse_mxfp4`
  field_validator, the now-dead `_warn_deprecated_env` helper, and
  `import warnings`. Kept `use_flashinfer_moe_int4`,
  `_parse_triton_attn_use_td`, `_env_set`.
- **Not ported:** `VLLM_RPC_TIMEOUT` dead-env removal (#45777) — already
  absent on the branch.
- **test_envs.py:** ported only the expanded
  `test_precompiled_install_flags_are_orthogonal` (main did not add test
  classes; the branch had already removed the helper-function test classes).
```

- [ ] **Step 5: Stage the runbook**

```bash
git add docs/superpowers/specs/2026-05-14-envs-merge-conflict-resolution-design.md
```

---

## Task 8: Final verification + sign-off commit

**Files:** none (verification + commit only)

- [ ] **Step 1: No conflict markers anywhere**

```bash
grep -rn "<<<<<<< \|>>>>>>> \|=======" vllm/envs.py tests/test_envs.py || echo "no markers"
```

- [ ] **Step 2: Lint both code files as CI does**

```bash
pre-commit run --files vllm/envs.py tests/test_envs.py
```

Expected: all hooks pass (ruff, mypy, etc.). Fix any reported issue (e.g. unused import) before committing.

- [ ] **Step 3: Final import + full test_envs run**

```bash
.venv/bin/python -c "import vllm.envs; print(vllm.envs.envs)"
.venv/bin/python -m pytest tests/test_envs.py -v
```

Expected: prints the Settings repr; all tests PASS.

- [ ] **Step 4: Stage the design + plan docs and any remaining resolved files**

```bash
git add docs/superpowers/specs/2026-06-16-envs-merge-conflict-resolution-execution-design.md
git add docs/superpowers/plans/2026-06-16-envs-merge-conflict-resolution.md
git add vllm/envs.py tests/test_envs.py
git status
```

Expected: `vllm/envs.py` and `tests/test_envs.py` no longer listed as unmerged (UU).

- [ ] **Step 5: Commit the merge resolution with sign-off**

Use `gcsm` (sign-off) so the DCO trailer is added. The message must enumerate the ported delta and note the deletion-port and AI assistance:

```bash
gcsm "Merge origin/main into pydantic-settings refactor: resolve envs.py conflicts

Dropped the legacy TYPE_CHECKING block and environment_variables dict
wholesale (already superseded by pydantic *Settings models on this branch),
then ported main's semantic delta to envs.py:

Additions (11 new Fields): VLLM_MAX_AUDIO_DECODE_DURATION_S,
VLLM_MAX_AUDIO_PREPROCESS_WORKERS, VLLM_FASTSAFETENSORS_QUEUE_SIZE,
VLLM_TRITON_FORCE_FIRST_CONFIG, VLLM_REGEX_COMPILATION_TIMEOUT_S,
VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS, VLLM_DEEPEP_V2_ALLOW_HYBRID_MODE,
VLLM_DEEPEP_V2_PREFER_OVERLAP, VLLM_DEEPEP_V2_ALLOW_MULTIPLE_REDUCTION,
VLLM_MXFP8_EMULATION_DEQUANT_AT_LOAD, VLLM_WSL2_ENABLE_PIN_MEMORY.
Modification: VLLM_ENFORCE_STRICT_TOOL_CALLING default False->True (#45003).
Deletions (#44992): removed 11 deprecated vars + their deprecation
validators and the now-dead _warn_deprecated_env helper / import warnings;
the branch had refactored these into Fields, so take-ours would have
re-introduced vars main removed.
Not ported: VLLM_RPC_TIMEOUT (#45777) already absent on branch.
test_envs.py: ported the expanded test_precompiled_install_flags_are_orthogonal.
Also amended the reusable runbook to cover deletions and modifications.

AI assistance (Claude) was used for this merge resolution.

Co-authored-by: Claude"
```

- [ ] **Step 6: Confirm the merge is complete**

```bash
git status
git log --oneline -1
```

Expected: clean working tree w.r.t. the merge (no unmerged paths); HEAD is the merge commit.

---

## Self-Review Notes

- **Spec coverage:** Task 1 = structural take-ours (spec Component 1 + File-2 take-ours). Task 2–3 = additions (spec delta A, all 11 vars, exact home classes). Task 4 = modification + compile_factors polarity (spec delta B). Task 5 = deletions (spec delta C, the user-approved port). Task 6 = test delta (spec Component 3). Task 7 = runbook amendment (spec File 3, user-requested). Task 8 = verification + commit (spec Components 4, lightweight step-4 + step-5). All spec sections mapped.
- **Deferred intentionally:** full parity replay (harness baselines absent) — out of scope per spec and user's "lightweight" choice.
- **Type/name consistency:** field names match the `VLLM_FOO → foo` convention and the back-compat shim auto-exposes `VLLM_*` accessors; verification commands reference the `VLLM_*` public names consistently.
