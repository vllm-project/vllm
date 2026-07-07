# envs.py merge conflict resolution playbook

## Context

Branch `vrdn-23/refactor-envs-to-use-pydantic-settings` rewrote
`vllm/envs.py` from a legacy `if TYPE_CHECKING:` declaration block plus an
`environment_variables: dict[str, Callable]` runtime dict into a tree of
`pydantic_settings.BaseSettings` subclasses. Until that branch lands, every
merge from `origin/main` produces structural conflicts whenever a main-side
PR adds or modifies a `VLLM_*` env var: main edits the legacy block, the
branch deleted the block, and git records the disappeared context as a
conflict.

This document is a **reusable playbook** for resolving those conflicts. It
is intentionally generic — do not encode specific commit hashes or PR
numbers into the body, because the merge will be re-run multiple times as
main advances. A dated snapshot of the conflict found on first execution is
preserved in the appendix for historical reference.

## Why these conflicts are mechanical, not semantic

The branch and main are not disagreeing about behavior — they're
disagreeing about *where* env vars live. On main, a new env var is added by
editing two locations:

1. The `if TYPE_CHECKING:` block near the top (type hints + literal
   defaults).
2. The `environment_variables: dict[str, Callable[[], Any]]` mapping near
   the bottom (runtime parsing lambdas, including string-to-bool
   coercions, defaults, and casts).

On the branch, the same env var would be added by:

1. A `Field(default=..., description=...)` declaration on the appropriate
   `BaseSettings` subclass (e.g. `CompilationSettings`,
   `ConnectorSettings`, `ServerSettings`).
2. Optionally, a `@field_validator(mode="before")` if the legacy lambda
   did non-trivial parsing — most commonly the `os.environ.get("X", "0")
   == "1"` idiom for booleans, which pydantic does not need (it accepts
   `"1"`/`"0"`/`"true"`/`"false"` natively for `bool`).
3. Optionally, a `validation_alias=AliasChoices(...)` if the env var is
   accessed under multiple names (legacy fallbacks like
   `DO_NOT_TRACK`).

Resolving the conflict therefore means: take the branch's structure
wholesale, then **port the semantic delta** — the new env vars that main
added — into the pydantic model.

## Resolution strategy — mass-take-ours, then port deltas

> Note on aborting: if this spec file was staged into an in-progress
> merge, `git merge --abort` unstages it but leaves the file on disk
> (where it appears as untracked). Re-running the merge then re-staging
> the spec is sufficient.

### 1. Enumerate the semantic delta from main

Find every main-side commit that touched `vllm/envs.py` since the merge
base:

```bash
MERGE_BASE=$(git merge-base HEAD origin/main)
git log "$MERGE_BASE"..origin/main --oneline -- vllm/envs.py
git log "$MERGE_BASE"..origin/main -p -- vllm/envs.py
```

For each commit, classify the change:

- **New env var added:** must be ported to a pydantic `Field`. Capture
  the name, type, default, and any non-trivial parsing behavior from the
  lambda. Note which `*Settings` subclass should own it (group by topic
  — compilation flags go to `CompilationSettings`, KV connector flags go
  to `ConnectorSettings`, etc.).
- **Existing env var modified (e.g. type widened, default changed,
  validator added):** must be ported as a targeted edit to the
  corresponding `Field` and/or `field_validator`. Tri-state booleans
  (`bool | None` with `None` meaning "fall back to config default") are
  the typical pattern. A pure default change (e.g. a bool default
  flipped `False` -> `True`) is a one-line edit to the
  `Field(default=...)`; pydantic parses `"1"`/`"0"`/`"true"`/`"false"`
  natively, so no validator is needed.
- **Existing env var deleted:** a main-side removal is a semantic delta
  too. If the branch *deleted* the same var, taking ours already agrees
  — nothing to do. But if the branch *refactored* the var (kept it as a
  pydantic `Field`, possibly with deprecation scaffolding), "take ours"
  silently re-introduces a var main intentionally removed. Port the
  deletion: remove the `Field`, any `field_validator` bound only to it,
  any deprecation `model_validator` / helper that becomes dead, and any
  import that becomes unused. Removing the field also drops it from
  generated structures (`_VAR_TO_PATH`, the `environment_variables`
  back-compat shim) automatically — no separate edit needed.
- **Pure rename or comment change:** can usually be ignored on the
  branch side; the pydantic field's `description=` is the new home for
  prose.

The output of this enumeration is the **port list** — a flat list of
field-level edits to make to the pydantic model after the conflict is
resolved.

### 2. Resolve the conflict structurally — take ours

Re-run the merge taking the branch wholesale for `vllm/envs.py`:

```bash
git checkout --ours vllm/envs.py
git add vllm/envs.py
```

(If the merge is already in progress with conflict markers in the file,
this overwrites them with the branch version.)

Verify zero markers remain:

```bash
grep -n "<<<<<<< \|>>>>>>> \|=======" vllm/envs.py
```

### 3. Port the semantic delta

For each item on the port list from step 1:

- Locate the appropriate `class *Settings(BaseSettings)` block.
- Add a `Field(default=..., description=...)` declaration matching main's
  type and default. Use existing fields in the same class as a style
  reference.
- If main's lambda did string-to-bool coercion using
  `os.environ.get("X", "0") == "1"` or
  `os.getenv("X", "False").lower() in ("true", "1")`, you can rely on
  pydantic's native bool parsing — no validator needed.
- If main's lambda did anything more interesting (custom enum mapping,
  optional-with-fallback, multiple alias names), add a
  `@field_validator` or `validation_alias=AliasChoices(...)` to match.
- Mirror naming convention: env var `VLLM_FOO_BAR` becomes field
  `foo_bar` (the `VLLM_` prefix is stripped by `_SUB_CONFIG`'s
  `env_prefix="VLLM_"`). Env vars without the `VLLM_` prefix need an
  explicit `validation_alias` since they bypass `env_prefix`.
- `compile_factors()` polarity: if the branch represents compile factors
  as an `ignored_factors` *exclude*-set (the inverse of main's explicit
  include-list), then a main-side "add var to the include-list" ports to
  the *opposite* edit on the branch — "add var to the ignore set".
  Confirm the polarity against an existing neighbor before editing.
- Field placement: the existing `*Settings` class names on the branch
  do not always reflect topical purity (e.g. compilation- and
  distributed-flavored fields may already live in `QuantSettings`).
  Locate the insertion point by finding the NEAREST NEIGHBOR field with
  grep and then identifying the enclosing class — never trust a class
  label inferred from topic alone. Adding new fields adjacent to their
  semantic siblings preserves the branch's existing convention even
  when the host class is misleadingly named.

### 4. Verify

```bash
# No conflict markers.
grep -n "<<<<<<< \|>>>>>>> \|=======" vllm/envs.py

# File parses and the settings model can be instantiated.
.venv/bin/python -c "import vllm.envs; print(vllm.envs.envs)"

# Lint.
pre-commit run --files vllm/envs.py

# Spot-check that ported env vars are accessible.
.venv/bin/python -c "import vllm.envs as e; print(e.VLLM_USE_BREAKABLE_CUDAGRAPH)"
```

### 5. Commit

The merge resolution commit must be signed off per vLLM's DCO
requirement. Use `gcsm "<message>"` (alias for `git commit --signoff
--message`) so the `Signed-off-by: <name> <email>` trailer is added
automatically.

The merge resolution commit message must:

- State that the legacy `TYPE_CHECKING` block and `environment_variables`
  dict were dropped wholesale (already superseded by pydantic models on
  this branch).
- Enumerate every main-side PR that touched `envs.py` since the merge
  base, with a one-line description of what was ported for each.
- Explicitly note any main-side change that was intentionally **not**
  ported, with reasoning.

## Out of scope

- Other files in the merge are not affected by this playbook. Their
  conflicts (if any) are resolved with their own context.
- Running the full vLLM test matrix is the human submitter's
  responsibility per `AGENTS.md`.

## Risks and mitigations

- **Risk:** A main-side commit silently drops a new env var because the
  port list missed it. **Mitigation:** Step 1's enumeration command is
  authoritative. Run it every time and copy its output verbatim into the
  commit message — that creates an audit trail.
- **Risk:** A ported `Field` has the wrong default or type because the
  legacy lambda's parsing semantics were non-obvious. **Mitigation:**
  When in doubt, read the lambda carefully. The most common parsing
  patterns and their pydantic equivalents are listed in step 3.
- **Risk:** A new env var doesn't have an obvious home among the
  existing `*Settings` classes. **Mitigation:** Group by topic; if no
  class fits, create a new one and add it to the top-level `Settings`
  class. Don't shoehorn unrelated flags into a single class.

---

## Appendix A: 2026-05-14 snapshot (first execution)

This appendix records the conflict found on the first execution of this
playbook against branch `vrdn-23/refactor-envs-to-use-pydantic-settings`.
Future executions should re-run step 1 from scratch — these numbers will
be stale.

- **Merge base at time of capture:** `256dbcaab`.
- **Conflict regions in working tree:**
  - Lines 13–277: legacy `TYPE_CHECKING` block (main) vs. new pydantic
    imports (branch).
  - Lines 2625–4021: legacy `environment_variables` dict (main) vs.
    nothing on the branch side.
- **Main-side commits touching `vllm/envs.py` since merge base
  (initial run, 2026-05-14):**
  - `ae4f59f0e` (#39337) — `VLLM_USE_V2_MODEL_RUNNER` widened from
    `bool` (default `False`) to `bool | None` (default `None`), with
    `maybe_convert_bool` parsing. Tri-state semantics: unset means "use
    config default".
- **Additional commits found on re-run (2026-05-18):**
  - `8a56da384` (#42304) — adds `VLLM_USE_BREAKABLE_CUDAGRAPH: bool =
    False` (compilation flag).
  - `36e74c9ea` (#42689) — adds four KV-connector env vars:
    `VLLM_MOONCAKE_STORE_TIER_LOG: bool = False`,
    `VLLM_MOONCAKE_DISK_STAGING_USABLE_RATIO: float = 0.9`,
    `MOONCAKE_PREFERRED_SEGMENT: str | None = None`,
    `MOONCAKE_REQUESTER_LOCAL_HOSTNAME: str | None = None`. Note the
    last two have no `VLLM_` prefix and need explicit
    `validation_alias`.

## Appendix B: 2026-06-16 execution

Merge of `origin/main` (`520828789`) into the branch, base `ba94a3b99`.
13 main-side commits touched `vllm/envs.py`. Resolved take-ours + ported:

- **Additions (11 new vars):** `VLLM_MAX_AUDIO_DECODE_DURATION_S`,
  `VLLM_MAX_AUDIO_PREPROCESS_WORKERS` (`MediaSettings`);
  `VLLM_REGEX_COMPILATION_TIMEOUT_S`,
  `VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS` (`ServerSettings`);
  `VLLM_FASTSAFETENSORS_QUEUE_SIZE`, `VLLM_TRITON_FORCE_FIRST_CONFIG`,
  `VLLM_DEEPEP_V2_ALLOW_HYBRID_MODE`, `VLLM_DEEPEP_V2_PREFER_OVERLAP`,
  `VLLM_DEEPEP_V2_ALLOW_MULTIPLE_REDUCTION` (`QuantSettings` —
  this is where their nearest-neighbor anchors `use_triton_awq` and
  `deepep_low_latency_use_mnnvl` actually live on the branch, despite
  being topically compilation- and distributed-flavored; an earlier
  draft of this run mistakenly named `CompilationSettings` /
  `DistributedSettings` based on topic, which the field-placement bullet
  in step 3 now warns against);
  `VLLM_MXFP8_EMULATION_DEQUANT_AT_LOAD` (`RocmSettings`);
  `VLLM_WSL2_ENABLE_PIN_MEMORY` (`ConnectorSettings`). The 3
  timeout/limit vars were added to the `ignored_factors` exclude-set
  in `compile_factors()`.
- **Modification:** `VLLM_ENFORCE_STRICT_TOOL_CALLING` default
  `False` -> `True` (#45003), with description tightened.
- **Deletions (#44992):** removed 11 deprecated vars
  (`VLLM_MXFP4_USE_MARLIN`, `VLLM_USE_FLASHINFER_MOE_FP16/FP8/FP4`,
  `VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8(_CUTLASS)`,
  `VLLM_USE_FLASHINFER_MOE_MXFP4_BF16`, `VLLM_FLASHINFER_MOE_BACKEND`,
  `VLLM_USE_NVFP4_CT_EMULATIONS`, `VLLM_NVFP4_GEMM_BACKEND`,
  `VLLM_USE_FBGEMM`) plus the `_warn_deprecated_moe_backend_envs`
  / `_warn_deprecated_backend_envs` `model_validator`s, the
  `_parse_mxfp4` `field_validator`, the now-dead `_warn_deprecated_env`
  helper, and `import warnings`. Kept `use_flashinfer_moe_int4`
  (not deprecated, sits among the deleted cluster),
  `_parse_triton_attn_use_td`, `_env_set`. Net `vllm/envs.py` delta
  was −81 lines.
- **Not ported:** `VLLM_RPC_TIMEOUT` dead-env removal (#45777) —
  already absent on the branch.
- **`tests/test_envs.py`:** ported only the expanded
  `test_precompiled_install_flags_are_orthogonal` (main did not add
  test classes since the merge base; the branch had already removed the
  helper-function test classes for the now-removed
  `env_with_choices`-style helpers).

## Appendix C: 2026-07-06 execution

Merge of `origin/main` (`39a1d32b59`) into the branch, base `a46abb7ae6`.
Only `vllm/envs.py` conflicted this run (`tests/test_envs.py` was clean).
10 main-side commits touched `vllm/envs.py`. Resolved take-ours + ported:

- **Additions (5 new vars):**
  `VLLM_ROCM_USE_AITER_CUSTOM_AR` (`bool = True`, next to `rocm_use_aiter`,
  #46065); `VLLM_MAX_IMAGE_PIXELS` (`int = 178_956_970`, next to
  `max_audio_preprocess_workers`, #47010 — also added to the
  `ignored_factors` exclude-set); `VLLM_GPU_SYNC_CHECK`
  (`Literal["warn","error"] | None = None`, next to `triton_attn_use_td`,
  #44800 — main used `env_with_choices`; on the branch a `Literal` `Field`
  gives the same reject-on-invalid behavior natively, confirmed by test);
  `VLLM_MOONCAKE_LOAD_RECV_THREADS` (`int = 1`, next to
  `mooncake_store_tier_log`, #45971); `VLLM_MOE_SKIP_PADDING`
  (`bool = False`, next to `use_fused_moe_grouped_topk`, #46428). All parse
  natively — no `field_validator` needed.
- **Modification:** `VLLM_ROCM_QUICK_REDUCE_QUANTIZATION` gained `INT3` in
  its `Literal[...]` choices + description (#45666).
- **Deletions:** none.
- **Not ported (deliberate no-ops):**
  - `VLLM_ENFORCE_STRICT_TOOL_CALLING` (#45892) — main only reflowed the
    lambda; default was already `True` and the branch `Field` is already
    `default=True`. Cosmetic.
  - `VLLM_PORT` doc-URL string (#35530) — the branch already rewrote this
    to its own valid `configuration/env_vars` URL; porting main's
    `latest/...` over the branch's `stable/...` is churn.
  - `VLLM_USE_PACKED_HMA_KV_CACHE` — **added** (#46205) then **removed**
    (#46252) within this merge window; net-neutral, nothing to port. A
    naive marker-by-marker resolution would have re-introduced it.
- **`docs/configuration/env_vars.md`:** auto-generated at mkdocs build time
  by `docs/mkdocs/plugins/gen_env_vars.py` from the pydantic fields — the
  5 new `Field(description=...)` flow through automatically, no manual edit.
- **`tests/test_envs.py`:** not conflicted this run; untouched.

Net `vllm/envs.py` delta: +57 −9.
