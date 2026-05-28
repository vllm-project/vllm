# Code Notes: Runtime, Scheduling, and Structured Output

## 1) Thinking Budget Is a Cross-Layer Feature

This feature is intentionally distributed across API params, scheduler state, and model runner token handling:

- `vllm/sampling_params.py`: request inputs (`thinking_token_budget`, `continue_thinking`).
- `vllm/v1/core/sched/output.py`: scheduler -> worker contract (`requests_with_remaining_budget`, `end_thinking_token_id`).
- `vllm/v1/core/sched/scheduler.py`: per-step budget accounting and token-state tracking.
- `vllm/v1/worker/gpu_model_runner.py`: token truncation/forced end token and logprob realignment.
- `vllm/cohere/utils/__init__.py`: thinking token ID lookup + helper logic.

Takeaway:

- removing any one layer breaks the invariant that enforced output tokens still match returned logprobs.

## 2) Scheduler-Side Lifecycle Details

Scheduler behavior adds two request maps:

- `requests_to_start_thinking_idx`: request -> index where thinking began.
- `requests_with_remaining_budget`: request -> current remaining budget.

Lifecycle:

1. request add: initializes tracking when budget >= 0.
2. scheduling step: recomputes remaining budget from emitted tokens.
3. output processing: `handle_thinking_tokens` updates start/end state.
4. free/cancel: both maps are cleaned.

Important invariant:

- maps must be pruned on every request termination path (normal finish, abort, cancellation) to avoid stale state mutating future batched requests.

## 3) Worker-Side Forced End Thinking

`GPUModelRunner._force_end_thinking` performs per-request token list surgery:

- if generated tokens exceed remaining budget, truncates overflow,
- if remaining budget reaches zero, appends end-thinking token,
- then adjusts logprobs arrays to keep token/logprob alignment.

Subtle but important:

- async/sync batching can produce shorter token arrays than request index map; code guards for index mismatch before mutation.

Without this guard, mixed async paths can throw index errors or misattribute token edits.

## 4) Structured Output Safety Patch

In scheduler output handling:

- grammar FSM failure to advance now marks request as aborted and frees request state.

This avoids hanging/undefined structured-output requests when grammar cannot consume produced tokens.

Intent:

- fail fast and explicit instead of partial undefined execution.

## 5) XGrammar and Structural Tag Compatibility

`vllm/v1/structured_output/backend_xgrammar.py` changes:

- recursion depth now controlled by env (`VLLM_XGRAMMAR_RECURSION_DEPTH`),
- structural tags include `schema_type` propagation.

Why this matters:

- large outputs from Cohere models can hit recursion/stack constraints in grammar transitions,
- `schema_type` allows mixed tag modes (jsonschema + ebnf/tool grammar) without losing parser intent.

## 6) SHM Cache Lifecycle Fixes

Multimodal SHM changes span three files:

- `shm_object_storage.py`: explicit `close()` and destructor cleanup.
- `multimodal/cache.py`: cache close hook calls SHM close.
- `envs.py`: default SHM buffer name gets UUID to avoid process collisions.

Operational outcome:

- fewer stale `/dev/shm/VLLM_*` collisions across repeated CI runs and multiprocess startup.

## 7) Change Hotspots and Verification

High-conflict files:

- `vllm/v1/core/sched/scheduler.py`
- `vllm/v1/worker/gpu_model_runner.py`
- `vllm/cohere/utils/__init__.py`
- `vllm/v1/structured_output/backend_xgrammar.py`
- `vllm/distributed/device_communicators/shm_object_storage.py`

Validation checklist:

1. Run thinking-budget tests with logprobs enabled.
2. Verify forced end-thinking token appears exactly at budget boundary.
3. Confirm no logprob shape mismatch/assertion under async scheduling.
4. Force a grammar-advance failure and verify request aborts cleanly.
5. Run repeated multimodal startup/shutdown and verify no SHM name collision.

## 8) Cohere Auto-Config (Hardware Profile Application)

Hardware profiles are applied at engine boot via an opt-in hook inside
`EngineArgs.__post_init__`:

- `vllm/cohere/auto_config.py`: `apply_cohere_auto_config(engine_args)` --
  detects Cohere model architecture from the HF config, resolves matching
  profiles from `vllm/cohere/hardware_profiles.yaml` via CEL `when:`
  clauses bound to `{server.type, gpu.name}`, then mutates the live
  `EngineArgs` instance in place.
- `vllm/cohere/hardware_profiles.yaml`: bundled into the wheel via
  `setup.py` `package_data`. The `vllm-default` profile is always applied
  as a baseline; per-GPU profiles overlay it (later wins on conflicting
  keys).
- `vllm/engine/arg_utils.py`: at the end of `EngineArgs.__post_init__`
  the gate reads `os.environ.get("VLLM_ENABLE_COHERE_AUTO_CONFIG", "0").strip().lower() in ("true", "1")`.
  When truthy it does a lazy import and calls `apply_cohere_auto_config(self)`.
  Default is **off** -- non-Cohere launches do not import the module.

Subtle but important:

- The gate uses raw `os.environ.get` rather than
  `envs.VLLM_ENABLE_COHERE_AUTO_CONFIG` on purpose. `apply_cohere_auto_config`
  sets profile env vars (e.g. `VLLM_USE_V1`, `VLLM_ROCM_USE_AITER`) right
  after the gate; reading any `envs.*` attribute first would prime the
  `vllm.envs` cache with pre-profile values and shadow the writes.
- "User-set" detection compares each `EngineArgs` field's live value to
  its dataclass default (`f.default` / `f.default_factory()`). Fields
  matching the default are filled from the profile; fields with any
  user-supplied value are preserved unchanged. This means
  `LLM(model=..., max_model_len=8192)` keeps `max_model_len=8192` even
  when the profile sets a different value.
- Profile `env:` entries only land when the env var is currently unset;
  pre-existing `os.environ` values win and are logged at INFO so user
  overrides are visible.
- Type coercion delegates to upstream's `get_kwargs(EngineArgs)` for the
  field's argparse `type` fn, but `EngineArgs` may carry string-literal
  forward-ref annotations (e.g. `quantization_config:
  "dict[str, Any] | OnlineQuantizationConfigArgs | None"`) that upstream's
  `is_not_builtin` trips on. `_engine_arg_kwargs` calls
  `typing.get_type_hints(EngineArgs, include_extras=True)` to resolve them
  and rewrites any string `Field.type` to the resolved type before
  delegating to `get_kwargs`; originals are restored in `finally` so
  `EngineArgs.__dataclass_fields__` is never permanently mutated. If
  resolution raises (e.g. a forward ref names a `TYPE_CHECKING`-only
  symbol not bound at runtime), string-annotated fields fall back to
  `typing.Any | None` -- which keeps Optional handling so a YAML `null`
  stays `None` rather than becoming the literal string `"None"` -- while
  unrelated non-string fields are untouched and continue to coerce
  correctly.
- Failures inside `apply_cohere_auto_config` are caught and logged but
  never raised -- a malformed YAML or buggy CEL clause cannot break
  engine startup.

How callers opt in:

- CI shells (`run_tests.sh`, `run-bee-eval.sh`,
  `run-performance-benchmarks.sh`) `export VLLM_ENABLE_COHERE_AUTO_CONFIG=1`
  so every spawned `vllm serve` / `vllm bench` / pytest process picks
  profiles up automatically.
- Python entry points that build `LLM(...)` / `AsyncEngineArgs(...)`
  directly call
  `os.environ.setdefault("VLLM_ENABLE_COHERE_AUTO_CONFIG", "1")` in their
  `main()` / `__main__` / pytest entry function so standalone invocations
  self-configure without depending on the shell wrapper.

Validation: see
[`tests/cohere/cpu/test_auto_config.py`](../../../tests/cohere/cpu/test_auto_config.py)
([feature doc](../tests/features/auto_config.md)) for the full CPU unit
suite covering arch detection, CEL evaluation, type coercion, profile
resolution, and `__post_init__` opt-in semantics.
