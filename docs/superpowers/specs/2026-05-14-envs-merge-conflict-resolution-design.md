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

Then grep the merged tree for callers of every var on the list:

```bash
grep -rn "VLLM_THE_NEW_VAR" vllm/ tests/ | grep -v "^vllm/envs.py"
```

A var whose readers are **already merged** is a mandatory port — skipping
it is an `AttributeError` at runtime, not a missing feature. A var with no
readers yet is still ported, but a mistake there is inert. Do the same for
deletions: a main-side removal whose callers moved to a helper (rather than
to another env var) is safe to drop.

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
- `env_with_choices(...)` ports to a `Literal[...]` `Field` — the
  reject-on-invalid behavior is native. But check the call for
  `case_sensitive=False`: `Literal` is case-*sensitive*, so that flag needs
  a matching `@field_validator(..., mode="before")` returning
  `v.lower() if isinstance(v, str) else v` (see `_lower_mm_hasher`,
  `_lower_float32`). Omitting it silently narrows what main accepted.
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

#### 4a. Env var set parity (the mechanical safety net)

Reading 1600 conflicted lines is where a var gets silently dropped. Don't.
Diff the two name sets instead — main's `environment_variables` keys against
the branch's `_VAR_TO_PATH`. Save this as a scratch file (it needs no deps
and never imports main's `envs.py`, only parses it):

```python
import ast, sys
import vllm.envs as e

tree = ast.parse(open(sys.argv[1]).read())  # path to main's envs.py
runtime = set()
for node in ast.walk(tree):
    tgt = node.target if isinstance(node, ast.AnnAssign) else (
        node.targets[0] if isinstance(node, ast.Assign) and
        len(node.targets) == 1 else None)
    if isinstance(tgt, ast.Name) and tgt.id == "environment_variables":
        runtime = {k.value for k in node.value.keys
                   if isinstance(k, ast.Constant)}

ours = set(e._VAR_TO_PATH)
print("IN MAIN, NOT ON BRANCH:", sorted(runtime - ours))
print("ON BRANCH, NOT IN MAIN:", sorted(ours - runtime))
```

Extract main's version with `git show :3:vllm/envs.py > /tmp/theirs.py`
(stage 3 = theirs) and run `.venv/bin/python parity.py /tmp/theirs.py`.

Both lists must be empty **except** for known intentional divergences — carry
these forward and re-confirm each against the merge base rather than
assuming: if a name appears in both `base` and `theirs` unchanged, this merge
window didn't touch it and the divergence is pre-existing.

- `VLLM_TRITON_ATTN_USE_TD` — main keeps it as a deprecation shim for
  `VLLM_TRITON_USE_TD`; the branch dropped the shim (2026-07-23, user
  decision).

Note main's `TYPE_CHECKING` block is *not* authoritative — it drifts from the
runtime dict (see Appendix E's scale-constant case). Trust
`environment_variables`; that is what `__getattr__` actually reads.

#### 4b. The rest

```bash
# No conflict markers.
grep -n "<<<<<<< \|>>>>>>> \|=======" vllm/envs.py

# File parses and the settings model can be instantiated. Note `envs.envs` is
# not an attribute — `__getattr__` only serves names in `_VAR_TO_PATH`.
.venv/bin/python -c "import vllm.envs as e; print(type(e._get_settings()).__name__, len(e._VAR_TO_PATH))"

# Lint.
.venv/bin/pre-commit run --files vllm/envs.py tests/test_envs.py

# Tests. test_envs_pydantic.py is branch-only and pins branch-specific
# behavior (dir() exposure, the shim, alias fallbacks) — a ported deletion
# often needs a matching edit there, so always run it too.
.venv/bin/python -m pytest tests/test_envs.py tests/test_envs_pydantic.py \
    tests/docs/test_env_vars_gen.py -q

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

## Appendix D: 2026-07-23 execution

Merge of `origin/main` (`46f01a50ac`) into the branch, base `34e6dfced8`.
`vllm/envs.py` and `docs/configuration/env_vars.md` conflicted;
`tests/test_envs.py` auto-merged. 11 main-side commits touched
`vllm/envs.py`. Resolved take-ours + ported:

- **Additions (12 new vars):**
  `VLLM_MAX_COMPLETION_PROMPTS` (`int = 1024`, `ServerSettings`, next to
  `max_n_sequences`, #47845); `VLLM_ENABLE_STARTUP_PLAN` (`bool = False`,
  `PathSettings`, next to `tuned_config_folder`, #47388);
  `VLLM_BUILD_COMMIT` (`str = "unknown"`), `VLLM_BUILD_PIPELINE`
  (`str = "local"`), `VLLM_BUILD_URL` (`str = ""`), `VLLM_IMAGE_TAG`
  (`str = ""`) (`BuildSettings`, next to `docker_build_context`, #45313);
  `VLLM_FLASHINFER_AUTOTUNE_SKIP_OPS`
  (`Annotated[list[str] | None, NoDecode] = None`, `FlashInferSettings`,
  next to `flashinfer_autotune_cache_dir`, #48268 — needs a
  `@field_validator(mode="before")` mirroring `_parse_plugins`: `None`
  stays `None`, a comma string splits and strips, drops empties);
  `VLLM_EC_SIDE_CHANNEL_HOST` (`str = "localhost"`),
  `VLLM_EC_SIDE_CHANNEL_PORT` (`int = 5601`) (`ConnectorSettings`, next to
  `nixl_side_channel_port`, #42433); `VLLM_P2P_SIDE_CHANNEL_HOST`
  (`str = "localhost"`), `VLLM_P2P_SIDE_CHANNEL_PORT` (`int = 5710`)
  (`ConnectorSettings`, #47636 — **mandatory:** auto-merged
  `test_p2p_side_channel_defaults_and_override` asserts both);
  `VLLM_DCP_Q_REPLICATE` (`bool = False`, `QuantSettings`, next to
  `use_deep_gemm_tma_aligned_scales`, #45964). All but the skip-ops list
  parse natively.
- **Modification:** `VLLM_MOE_SKIP_PADDING` default `False -> True`
  (#48979); trimmed the trailing "off by default because not all kernels
  support it yet" clause from the description (main removed it).
- **Rename (mandatory for correctness):** `VLLM_TRITON_ATTN_USE_TD ->
  VLLM_TRITON_USE_TD` (#45781) — field `triton_attn_use_td ->
  triton_use_td`, validator `_parse_triton_attn_use_td ->
  _parse_triton_use_td` (rebound to `"triton_use_td"`), description
  "Intel Xe2/Xe3" -> "Intel XPU". No deprecation shim (user decision);
  already-merged callers `triton_attn.py:574` and
  `triton_unified_attention.py:1029` read `envs.VLLM_TRITON_USE_TD`, so
  skipping the rename would `AttributeError`.
- **Deletions (#44749):** removed `rocm_use_aiter_paged_attn`,
  `tpu_bucket_padding_gap`, `tpu_most_model_len`, `ci_use_s3` (+ its
  `VLLM_CI_USE_S3` entry in the `compile_factors()` ignore-set), and
  `flashinfer_allreduce_fusion_thresholds_mb` (+ its bound
  `_parse_json_thresholds` `field_validator`). Kept `import json`
  (3 other uses) and `NoDecode` (used by other fields). No repo callers
  of the deleted vars remained (grep-verified).
- **`compile_factors()` ignore-set** (same exclude-set polarity as main
  this run — no inversion): **added** `VLLM_ENABLE_STARTUP_PLAN` (#47388),
  `VLLM_FLASHINFER_AUTOTUNE_SKIP_OPS` (#48268), `VLLM_XLA_CACHE_PATH` +
  `VLLM_CONFIG_ROOT` (#47573, location-derived paths); **removed**
  `VLLM_CI_USE_S3` (#44749).
- **`docs/configuration/env_vars.md`:** main injected the
  `VLLM_PORT`/Kubernetes warning into the static file; the branch moved
  that warning into the generator (`gen_env_vars.py:37-48`). Took ours;
  the new fields flow through the generator automatically.
- **Not ported:** nothing dropped silently — all 11 commits' `envs.py`
  deltas are covered above.

Net `vllm/envs.py` delta: +105 −47 (post-ruff-format; the reformat
reflowed touched `Field(...)` blocks). All 11 `test_envs.py` tests pass;
`pre-commit run --files vllm/envs.py docs/configuration/env_vars.md` clean.

## Appendix E: 2026-08-07 execution

Merge of `origin/main` (`bc37fc970e`) into the branch (`3a08f2fb59`), base
`89f6aa3a9e`. `vllm/envs.py` and `tests/test_envs.py` conflicted. 10
main-side commits touched `vllm/envs.py`. Resolved take-ours + ported:

- **Additions (5 new vars, all mandatory — every one has an already-merged
  caller, confirmed by the step-1 grep):**
  `VLLM_ROCM_USE_AITER_MOE_SITUV2_A8W4` (`bool = False`, `RocmSettings`, next
  to `rocm_use_aiter_moe`, #50582 — caller `_aiter_ops.py:1617`);
  `VLLM_USE_RUST_BENCH` (`bool = False`, `ServerSettings`, next to
  `use_rust_frontend`, #50081 — caller `cli/benchmark/main.py:24`);
  `VLLM_KIMI_K3_SHARD_SP_SHARED_EXPERT` (`bool = False`, `QuantSettings`, next
  to `moe_skip_padding`, #50656 — caller `kimi_k3/nvidia/model.py:142`);
  `VLLM_RAISE_ON_LOGIT_NANS` (`bool = False`, next to
  `compute_nans_in_logits`, #50323 — callers `gpu_model_runner.py:5729`,
  `gpu/async_utils.py:85`); `VLLM_ENABLE_COHERE_API` (`bool = False`,
  `ServerSettings`, next to `enable_responses_api_store`, #47189 — caller
  `cohere/api_router.py`). All parse natively; no `field_validator` needed.
- **Modifications (2):**
    - `VLLM_COMPUTE_NANS_IN_LOGITS` — main ORed the new
    `VLLM_RAISE_ON_LOGIT_NANS` into its lambda. Cross-field coupling, so on
    the branch it ports as a `@model_validator(mode="after")` on
    `QuantSettings` (`_raise_on_nans_implies_compute`) rather than a
    `field_validator`, which cannot see a sibling field.
    - `_resolve_rust_frontend_path` -> `_resolve_rust_cli_path` (#50081): the
    `model_validator` now fires on `use_rust_frontend or use_rust_bench`,
    with main's reworded warning. `rust_frontend_path`'s description updated
    to name both flags.
- **Deletions (4):** `VLLM_CPU_SGL_KERNEL` (#50801 — safe: callers moved to
  the `check_cpu_sgl_kernel()` helper in `layers/utils.py`, no env read
  remains) and `Q_SCALE_CONSTANT` / `K_SCALE_CONSTANT` / `V_SCALE_CONSTANT`
  (#49389, all three carried explicit `alias=`).
    - **Main is inconsistent here:** #49389 deleted the three scale constants
    from `environment_variables` but left their `TYPE_CHECKING` annotations,
    so on main today `envs.Q_SCALE_CONSTANT` raises `AttributeError` while
    mypy still believes it exists. Deleted on the branch per the PR's intent
    ("Remove deprecated calculate_kv_scales runtime KV scale calculation") —
    user decision. This is the case that motivated the "TYPE_CHECKING block
    is not authoritative" note in step 4a.
- **Prose-only (1):** `video_loader_backend` description dropped the
  `"identity"` backend mention. Cosmetic on main; on the branch the
  description feeds `gen_env_vars.py`, so it is a real docs change.
- **`compile_factors()`:** no manual edit needed. The ignore-set is now
  derived from per-field `compile_factor` markers
  (`_NON_COMPILE_FACTORS`), so added/removed fields flow through
  automatically. Earlier appendices' manual ignore-set steps are obsolete.
- **`tests/test_envs.py`:** take-ours (206 lines), then ported main's
  `VLLMValidationError` import and its two `pytest.raises` swaps in
  `TestVllmMaxNSequences` (mandatory — `sampling_params.py` now raises it),
  plus `test_rust_bench_auto_path_missing_fails_fast` **adapted**: main calls
  `environment_variables["VLLM_RUST_FRONTEND_PATH"]()`, but on the branch the
  shim only reads an already-resolved attribute and can never raise, so the
  test constructs `envs.ServerSettings()` directly to trigger the validator.
- **`tests/test_envs_pydantic.py`:** dropped `"Q_SCALE_CONSTANT"` from
  `test_dir_exposes_known_vars`'s expected list (it was that test's example
  of an alias-exposed var; `MAX_JOBS` / `CUDA_HOME` already cover the case).
  Not conflicted — found only by running the suite.
- **Not ported:** nothing dropped silently; all 10 commits accounted for.
- **Ignored:** `tests/data/envs_snapshot.json` and
  `tests/data/compile_factors_baseline.json` are untracked scratch files
  with no test consumers.

Parity check (step 4a) after resolution: 281 branch fields vs 282 main
runtime entries, sole difference `VLLM_TRITON_ATTN_USE_TD` — verified
byte-identical in `base` and `theirs`, i.e. untouched by this merge window
and therefore the known pre-existing shim divergence, not a dropped port.

Net `vllm/envs.py` delta: +72 −44 (post-ruff-format). 54 tests pass across
`test_envs.py`, `test_envs_pydantic.py`, `test_env_vars_gen.py`;
`pre-commit run --files` clean on all three changed files.

## Appendix F: 2026-08-07 execution (second merge, same day)

Merge of `origin/main` (`4eccf906ca`) into the branch (`96fc10994c`), base
`bc37fc970e`. Only `vllm/envs.py` conflicted; `tests/test_envs.py` and
`tests/test_envs_pydantic.py` were untouched by main this window. **One**
main-side commit touched `vllm/envs.py` (+15 lines). Resolved take-ours +
ported:

- **Addition (1 new var, mandatory):** `VLLM_ROCM_AITER_MLA_ASM_PADDING`
  (`Literal["auto", "gluon", "asm"] = "auto"`, `RocmSettings`, next to
  `rocm_use_aiter_mla`, #50578). Main used
  `env_with_choices(..., case_sensitive=False)`; the `Literal` `Field` covers
  reject-on-invalid natively, but the case-insensitivity needs an explicit
  `_lower_aiter_mla_asm_padding` `@field_validator(mode="before")` — this is
  `RocmSettings`' first validator. Callers already merged:
  `mla/rocm_aiter_mla.py:114` plus four env assertions in
  `tests/kernels/attention/test_rocm_aiter_mla_head_padding.py`.
- **Modifications / deletions / prose:** none.
- **Not ported:** nothing; the single commit is fully accounted for.

Parity check (step 4a): 282 branch fields vs 283 main runtime entries, sole
difference `VLLM_TRITON_ATTN_USE_TD` (the known shim divergence).

Net `vllm/envs.py` delta: +21 −0. 54 tests pass across `test_envs.py`,
`test_envs_pydantic.py`, `test_env_vars_gen.py`; the 4
`test_rocm_aiter_mla_head_padding.py -k asm_padding_env` tests pass on
non-ROCm too (`_on_gfx950()` is just `False` there);
`pre-commit run --files vllm/envs.py` clean.

**Takeaway:** the playbook's cost is now dominated by step 1's enumeration,
not the port. A one-commit window took ~5 minutes end to end. Merging from
main *more often* keeps each conflict this size.

## Appendix G: 2026-08-13 execution

Merge of `origin/main` (`b216db3ed0`) into the branch (`db92061088`), base
`e644c8cd8c`. Only `vllm/envs.py` conflicted; main touched no test file this
window. 6 main-side commits touched `vllm/envs.py` (+55 −0). Resolved
take-ours + ported:

- **Additions (10 new vars, all mandatory — every one has an already-merged
  caller, confirmed by the step-1 grep):**
  `VLLM_MAX_STOP_STRINGS` (`int = 4`), `VLLM_MAX_NUM_BAD_WORDS`
  (`int = 128`), `VLLM_MAX_BAD_WORDS_TOTAL_TOKENS` (`int = 1024`)
  (`ServerSettings`, next to `max_completion_prompts`, #51447 — callers
  `openai/engine/protocol.py:32`, `anthropic/protocol.py:130`,
  `sampling_params.py:690`, `gpu/sample/bad_words.py`);
  `VLLM_MAX_AUDIO_DECODE_BYTES` (`int = 268_435_456`, `MediaSettings`, next to
  `max_audio_decode_duration_s`, #49948 — callers `multimodal/media/audio.py`,
  `speech_to_text/base/serving.py:129`);
  `VLLM_USE_DIRECT_DCP_A2A` / `_Q_GATHER` / `_KV_GATHER`
  (`bool | None = None`, `QuantSettings`, next to `dcp_q_replicate`, #50484 —
  caller `v1/attention/ops/dcp_utils.py`);
  `VLLM_KIMI_K3_GEMM_RS` (`bool = False`, `QuantSettings`, next to
  `kimi_k3_shard_sp_shared_expert`, #52079 — caller
  `kimi_k3/nvidia/model.py:153`);
  `VLLM_USE_HW_AGNOSTIC` (`bool = False`, `UsageSettings`, next to
  `disabled_kernels`, #49458 — caller
  `models/transformers/layers.py:26,48`);
  `VLLM_ADAPTIVE_VERIFICATION_PROFILE_CONTEXT_LEN` (`int = 8192`,
  `QuantSettings`, #47808 — caller
  `v1/worker/gpu/spec_decode/adaptive_verification.py:192`).
- **One shared tri-state validator:** the three `VLLM_USE_DIRECT_DCP_*` vars
  use `maybe_convert_bool` on main (`None` passthrough, else `bool(int(v))`).
  Ported as a single `_parse_direct_dcp` `@field_validator(..., mode="before")`
  over all three, mirroring the existing `_parse_humming_f16`. **Not optional:**
  bare `bool | None` would both *widen* (pydantic accepts `"true"`/`"yes"`,
  which `int()` rejects on main) and *narrow* (pydantic rejects `"2"`, which
  main coerces to `True`). Verified `"1" -> True`, `"0" -> False`,
  unset `-> None`.
- **Modifications / deletions / renames:** none.
- **`compile_factors()`:** `max_audio_decode_bytes` carries
  `json_schema_extra={"compile_factor": False}`, mirroring main's sole
  ignore-set addition (#49948). The other nine carry no marker and are
  therefore compile factors, matching main. Verified both polarities at
  runtime.
- **Field placement — the one genuine judgment call.**
  `VLLM_ADAPTIVE_VERIFICATION_PROFILE_CONTEXT_LEN` is the first case where
  the nearest-neighbor rule *did not decide*: main's left neighbor
  (`VLLM_SPARSE_INDEXER_MAX_LOGITS_MB`) lives in `TpuXpuSettings` on the
  branch, its right neighbor (`VLLM_USE_RAY_COMPILED_DAG_CHANNEL_TYPE`) in
  `DistributedSettings`. Neither fits a GPU spec-decode profiling knob, and
  the left anchor is itself a misfiled CUDA var. Placed in `QuantSettings`
  next to `compute_nans_in_logits` / `raise_on_logit_nans` — the branch's
  de-facto model-execution grab-bag, per Appendix E's precedent (user
  decision). **Rule refinement:** when main's two neighbors land in
  *different* branch classes, nearest-neighbor is silent; fall back to the
  topical grab-bag that already hosts the var's siblings rather than picking
  an anchor arbitrarily. Placement is functionally inert (`env_prefix` is
  `VLLM_` on every sub-class) but selects the docs section emitted by
  `gen_env_vars.py`.
- **Not ported:** nothing dropped silently; all 6 commits accounted for.

Parity check (step 4a): 292 branch fields vs 293 main runtime entries, sole
difference `VLLM_TRITON_ATTN_USE_TD` — the known shim divergence, re-confirmed
byte-identical in `base` and `theirs` (only line numbers shifted), i.e.
untouched by this merge window.

Net `vllm/envs.py` delta: +90 −0. 54 tests pass across `test_envs.py`,
`test_envs_pydantic.py`, `test_env_vars_gen.py`; `pre-commit run --files
vllm/envs.py` clean (ruff check/format, mypy, config-docstring validation).

**Consumer-suite caveat (macOS, no GPU).** The real acceptance test is the
already-merged callers, and three of their suites cannot fully run here.
Each failure was traced to an environment gap, not the port:

- `tests/test_request_input_bounds.py` — 22 passed (covers all three #51447
  vars).
- `tests/models/transformers/test_layer_registry.py` — 4 passed, 1 failed.
  The 2 initially-failing `caplog` assertions pass under
  `VLLM_CONFIGURE_LOGGING=0`: vLLM's `vllm` logger sets `propagate: False`
  (`vllm/logger.py:67`), so records never reach caplog's root handler. The
  env-var assertion itself (`resolved is fake_hw_layernorm.RMSNorm`) passed
  throughout. The remaining `test_hw_agnostic_matches_vllm_end_to_end`
  fails on `AttributeError: '_OpNamespace' '_C' object has no attribute
  'init_cpu_memory_env'` — no compiled extension on macOS.
- `tests/multimodal/media/test_audio.py` — collection error, `soundfile`
  not installed.
- `tests/distributed/test_dcp_direct_a2a_lse_reduce.py` — collection error,
  `multiprocess` not installed; GPU/distributed regardless.

These four suites are the human submitter's pre-merge check on a CUDA box.

**Takeaway:** first window where a new var had *no* defensible nearest
neighbor. The rule now has a documented tie-breaker. Also the first window
that was pure addition — no deletions, modifications, or renames — which is
what a 4-day merge cadence buys.
