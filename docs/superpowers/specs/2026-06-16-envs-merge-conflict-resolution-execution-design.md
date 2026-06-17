# envs.py merge conflict resolution — 2026-06-16 execution

## Context

The branch `vrdn-23/refactor-envs-to-use-pydantic-settings` is mid-merge with
`origin/main` (`520828789`, merge base `ba94a3b99`). Two files are conflicted:
`vllm/envs.py` and `tests/test_envs.py`. The reusable playbook for resolving
these is
[`2026-05-14-envs-merge-conflict-resolution-design.md`](2026-05-14-envs-merge-conflict-resolution-design.md);
this spec is the dated record of one execution of that playbook plus a
playbook amendment.

The strategy is unchanged: **take ours structurally, then port main's
semantic delta** as targeted field-level edits to the pydantic models. No
structural blending.

This execution surfaced a gap in the playbook: it classifies the delta as
*additions / modifications / renames* but does not cover **deletions**. Main
PR #44992 deleted 11 deprecated env vars, while the branch had independently
refactored those same vars into pydantic `Field`s with `_warn_deprecated_env`
validators — so a naive "take ours" silently re-introduces vars main removed.
We therefore also amend the playbook to add a deletion classification and
sharpen the modification guidance.

## Semantic delta (enumerated 2026-06-16)

13 main-side commits touched `vllm/envs.py` since the merge base. They reduce
to three kinds of change.

### A. Additions — 11 new env vars (none present on the branch)

Each becomes a `Field(default=..., description=...)` on the topically-correct
`*Settings` class. The legacy lambdas all use the `os.getenv(...) in
("1","true")` / `bool(int(...))` idioms, which pydantic parses natively — **no
`field_validator` needed**.

| Env var | Type / default | Home class | Extra |
| --- | --- | --- | --- |
| `VLLM_MAX_AUDIO_DECODE_DURATION_S` | `int = 600` | `MediaSettings` | + ignore set |
| `VLLM_MAX_AUDIO_PREPROCESS_WORKERS` | `int`, `default_factory=lambda: max(1, min(os.cpu_count() or 1, 2))` | `MediaSettings` | + ignore set |
| `VLLM_FASTSAFETENSORS_QUEUE_SIZE` | `int = 0` | `CompilationSettings` | |
| `VLLM_TRITON_FORCE_FIRST_CONFIG` | `bool = False` | `CompilationSettings` | |
| `VLLM_MXFP8_EMULATION_DEQUANT_AT_LOAD` | `bool = True` | `RocmSettings` (next to `rocm_use_aiter`) | |
| `VLLM_REGEX_COMPILATION_TIMEOUT_S` | `int = 5` | `ServerSettings` | |
| `VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS` | `int = 5` | `ServerSettings` | + ignore set |
| `VLLM_DEEPEP_V2_ALLOW_HYBRID_MODE` | `bool = True` | `DistributedSettings` | |
| `VLLM_DEEPEP_V2_PREFER_OVERLAP` | `bool = False` | `DistributedSettings` | |
| `VLLM_DEEPEP_V2_ALLOW_MULTIPLE_REDUCTION` | `bool = False` | `DistributedSettings` | |
| `VLLM_WSL2_ENABLE_PIN_MEMORY` | `bool = False` | `ConnectorSettings` (next to `weight_offloading_disable_*`) | |

**`compile_factors()` polarity.** The branch inverted main's explicit
*include-list* into an `ignored_factors` *exclude-set*. Main added three vars
to its include-list; on the branch that maps to the **opposite edit** — adding
them to `ignored_factors`: `VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS`,
`VLLM_MAX_AUDIO_DECODE_DURATION_S`, `VLLM_MAX_AUDIO_PREPROCESS_WORKERS`.
Confirm polarity against the existing neighbor
`VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS`, which is already in `ignored_factors`.

### B. Modification — 1 default change

`VLLM_ENFORCE_STRICT_TOOL_CALLING` default flips `False → True` (#45003). The
branch field already exists at default `False`; change it to `True`. Pydantic
parses `"True"/"1"` natively, so no validator change.

### C. Deletions — 11 deprecated vars + helper (#44992)

Main deleted: `VLLM_MXFP4_USE_MARLIN`, `VLLM_USE_FLASHINFER_MOE_FP16`,
`VLLM_USE_FLASHINFER_MOE_FP8`, `VLLM_USE_FLASHINFER_MOE_FP4`,
`VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8`,
`VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8_CUTLASS`,
`VLLM_USE_FLASHINFER_MOE_MXFP4_BF16`, `VLLM_FLASHINFER_MOE_BACKEND`,
`VLLM_USE_NVFP4_CT_EMULATIONS`, `VLLM_NVFP4_GEMM_BACKEND`, `VLLM_USE_FBGEMM`,
plus the legacy `deprecated_env` helper. The branch still defines all 11 as
pydantic `Field`s plus two `_warn_deprecated_env` `model_validator`s. None are
referenced anywhere else in the working tree.

**Decision: port the deletions.** Remove from the branch:

- the 11 deprecated `Field` declarations,
- any `field_validator` that existed *only* to parse one of them,
- the two `model_validator`s `_warn_deprecated_moe_backend_envs`
  (`FlashInferSettings`) and `_warn_deprecated_backend_envs` (`QuantSettings`),
- the `_warn_deprecated_env` helper and the `import warnings` **iff** they
  become unused after the above.

Keep `_env_set` (used elsewhere). `_VAR_TO_PATH` and the `environment_variables`
back-compat shim are generated from the fields, so they drop the removed vars
automatically.

`VLLM_RPC_TIMEOUT` (dead-env removal in #45777) is already absent on the
branch — nothing to do.

## File 1 — `vllm/envs.py`

1. `git checkout --ours vllm/envs.py && git add vllm/envs.py`; verify zero
   conflict markers.
2. Apply delta A (add 11 `Field`s in the home classes named in the delta-A
   table + 3 `ignored_factors` entries), B (one default flip), C (remove 11
   deprecated fields + their validators + dead helper/import).

## File 2 — `tests/test_envs.py`

This conflict is **not** the appendix's pattern. Main did *not* add new test
classes since the merge base. The conflict is:

- the branch deleted ~377 lines — the `TestEnvWithChoices` /
  `TestEnvListWithChoices` / `TestEnvSetWithChoices` / `TestVllmConfigureLogging`
  classes — because the `env_with_choices`-style helpers they tested no longer
  exist after the refactor; and
- main independently expanded `test_precompiled_install_flags_are_orthogonal`
  (two new `clear=True` sub-cases and a flipped assertion:
  `VLLM_USE_PRECOMPILED` `False → True` when both flags are set).

Resolution: `git checkout --ours tests/test_envs.py`, then port **only** main's
expanded body of `test_precompiled_install_flags_are_orthogonal`. The branch's
`environment_variables[...]()` shim and the
`_force_use_precompiled_when_wheel_set` validator already satisfy main's new
assertions, so the test passes unchanged in semantics. "Apply the same logic"
here means the same take-ours-then-port-delta discipline, not identical line
edits.

## File 3 — `docs/.../2026-05-14-...-design.md` (playbook amendment)

Amend the reusable playbook so future re-runs handle deletions and
modifications correctly:

- In "For each commit, classify the change", add a **deletion** bullet: a
  main-side removal of an env var is a semantic delta too. If the branch
  refactored (rather than deleted) the same var, "take ours" re-introduces it;
  the var and any branch-side deprecation scaffolding must be removed to
  converge. Removing a var also drops it from generated structures
  (`_VAR_TO_PATH`, the `environment_variables` shim) automatically.
- Strengthen the modification bullet with the default-flip case and the note
  that pydantic parses `"True"/"1"`/`"0"` natively (no validator needed for a
  pure default change).
- Add a note about `compile_factors()` polarity: the branch uses an
  `ignored_factors` exclude-set, so a main-side "add var to the include-list"
  ports to "add var to the ignore set" — the inverse edit.
- Keep the body generic (no commit hashes in the prose); record this run's
  specifics in a new dated appendix entry.

## Verification (playbook step 4 — lightweight)

```bash
# No markers in either file.
grep -n "<<<<<<< \|>>>>>>> \|=======" vllm/envs.py tests/test_envs.py

# Module imports and the settings model instantiates.
.venv/bin/python -c "import vllm.envs; print(vllm.envs.envs)"

# Spot-check ported additions + the modified default.
.venv/bin/python -c "import vllm.envs as e; \
print(e.VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS, \
e.VLLM_TRITON_FORCE_FIRST_CONFIG, e.VLLM_ENFORCE_STRICT_TOOL_CALLING)"

# Confirm a deleted var is gone (must raise AttributeError).
.venv/bin/python -c "import vllm.envs as e; e.VLLM_USE_FBGEMM" \
  && echo "FAIL: still present" || echo "OK: removed"

# The one ported test.
.venv/bin/python -m pytest tests/test_envs.py \
  -k test_precompiled_install_flags_are_orthogonal -v

# Lint.
pre-commit run --files vllm/envs.py tests/test_envs.py
```

## Commit (playbook step 5)

`gcsm` (sign-off) with a message that: states the legacy `TYPE_CHECKING` block
and `environment_variables` dict were dropped wholesale; enumerates every
main-side PR touching `envs.py` since the merge base with what was ported for
each (additions, the strict-tool-calling default flip, the 11 deletions); and
notes that the test resolution ported only the expanded
`test_precompiled_install_flags_are_orthogonal`. Mentions AI assistance per
`AGENTS.md`.

## Out of scope

- Other merge conflicts (none flagged beyond these two files).
- The parity harness in `scripts/_envs_refactor/` is currently inert (baseline
  JSONs absent); a full parity replay was explicitly deferred in favor of the
  lightweight step-4 checks.
- The full vLLM test matrix — human submitter's responsibility per `AGENTS.md`.
