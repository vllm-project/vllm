# Dual-precision residency and binding

Component C2 of the rollout precision scheduler. Package:
`vllm/model_executor/dual_precision/` (`policy_layers`, `loader`, `binding`,
`validation`, `__init__`), plus one attach call in
`vllm/v1/worker/gpu_model_runner.py`, the `BatchDescriptor.base_precision`
field in `vllm/forward_context.py`, and six knobs in `vllm/envs.py`.
The verl half lives in `verl/workers/config/rollout.py`,
`verl/workers/rollout/vllm_rollout/{vllm_async_server,utils}.py` and
`verl/workers/engine_workers.py`.

## Purpose

Keep two copies of the base model resident on one GPU, a BF16 copy (the one
LoRA trains against) and a low-precision shadow (GPTQ-packed INT4 or ModelOpt
NVFP4), and let every LoRA wrapper pick
one of them per forward without changing module topology. The scheduler
(C3/C4) decides *when* to use INT4 (long-tail decode with few live requests);
this component only makes both bases available and switchable.

## Mechanism

1. **Load** (`loader.attach_dual_precision`, called once from
   `GPUModelRunner.load_model` right after `load_lora_model`, inside the
   worker's CuMem `weights` pool). The INT4 checkpoint named by
   `VLLM_DUAL_PRECISION_INT4_MODEL` is loaded through the normal model loader
   with a cloned `VllmConfig` (`make_int4_vllm_config`: same everything except
   `model`/`hf_config_path`, `model_weights=""`, `quantization=None` so the
   shadow's own quant config is auto-detected, a fresh
   `CompilationConfig` so the shadow's attention layers register in their own
   static forward context, and its own `LoadConfig` with `load_format=auto`
   whatever the engine's is: verl's rollout default `load_format: dummy`
   used to reach the shadow through the clone and the store was
   `DummyModelLoader` noise, i.e. every INT4-phase token was garbage
   (integration defect 1); a dummy engine is announced at WARNING and
   `load_int4_shadow_model` refuses a dummy shadow config outright). The
   format is validated first
   (`validate_shadow_quantization`): Intel AutoRound `auto_round:auto_gptq`,
   plain GPTQ, compressed-tensors `pack-quantized` and ModelOpt NVFP4 are
   accepted; AWQ in any form raises `ValueError`, and ModelOpt FP8 is refused
   as not being a W4 shadow.
2. **Match** (`policy_layers`). Shadow `LinearBase` modules carrying a packed
   weight (`qweight`, `weight_packed`, ...) or an NVFP4 scale
   (`weight_global_scale`, `weight_scale_2` -- NVFP4 packs into the plain
   `weight` name, which BF16 has too) are matched by module name onto the
   BF16 model's LoRA wrappers (the wrapper sits at the linear's original
   name). `VLLM_DUAL_PRECISION_BF16_LAYERS` (default `first:3,last:3`) keeps
   whole transformer blocks BF16; `VLLM_DUAL_PRECISION_INT4_MODULES`
   (`all` | `mlp_only`) restricts to gate/up/down projections. Counts in the
   log line follow the archived semantics over every BF16 `LinearBase`:
   *attached* / *kept BF16 by policy* / *left BF16 (no shadow)*; a fourth
   count, *unwrapped* (quantized and eligible but not LoRA-wrapped, so not
   switchable), is logged as a warning when non-zero.
3. **Store**. Only the attached INT4 linears survive, owned by an
   `Int4ShadowLayerStore` registered on the model as
   `SHADOW_MODULE_NAME = "_vllm_dual_precision_int4_model"` *after* LoRA
   wrapping (the LoRA manager never sees the shadow). The temporary INT4 model
   is dropped (`gc.collect`, `empty_cache`). Because the store is a submodule
   allocated in the `weights` pool, sleep level 1 offloads and restores it
   with the BF16 weights; level 2 would discard it with no way to re-sync
   (hence verl's level-1 rule).
4. **Bind** (`binding`). Each wrapper gets a `DualPrecisionBinding`
   (`bf16`, `int4_or_fallback`, `layer_index`, mutable `active`) stored in
   its `__dict__`, registered in `compilation_config.static_forward_context`
   under `base_layer.prefix + ".dual_precision_base_linear"`, and a closure
   installed through C1's `set_base_forward_override`. The closure calls the
   opaque custom op `torch.ops.vllm.dual_precision_base_linear(layer_name,
   output_size, x, bias)`, whose body looks the binding up in the forward
   context and runs `active.quant_method.apply(active, x, bias)`. Outside a
   forward context (multimodal tower modules) the closure uses the BF16 base
   directly, as the plain sync path would. `bind_dual_precision(model,
   precision)` flips `active` for every binding; it never touches
   `_modules`, `_parameters` or `_buffers` (GEMMA4 audit invariant 1), is
   idempotent per (precision, analysis mask), and logs once per precision.
   Log-line contract: `Dual precision QLoRA base path bound: precision=%s,
   lora_base_layers=%d, rebound_layers=%d, int4_shadow_active=%d,
   analysis_bf16_layers=%s.` is emitted at **WARNING** (one line per
   precision / mask), because verl launches vLLM with
   `VLLM_LOGGING_LEVEL=WARN` and its `validate_rollout_run.py
   --expected-lora-layers` parses `lora_base_layers=` and `precision=int4
   ... int4_shadow_active=` from it (integration defect 3).
   The compiled graph sees one stable op, so one Dynamo graph serves both
   precisions and each precision captures its own CUDA graph
   (`BatchDescriptor.base_precision` is part of the graph key; C3/C4 wire the
   bind calls before capture/replay/eager forwards).
5. **Validation** (`validation`). One check is always on: the *sanity
   probe* runs one random input through the first attached layer, INT4 and
   BF16 base, and raises `RuntimeError` (naming the layer, the cosine and the
   shadow load format) unless the cosine exceeds `SANITY_MIN_COSINE = 0.5`
   (a real GPTQ shadow scores >= 0.9, random weights ~0), so a random shadow
   can never start serving. It runs at attach before any override is
   installed; when the engine loaded dummy base weights (verl syncs the
   trainer's weights later) the BF16 twin is noise, so the probe is deferred
   to the first INT4 bind after the first weight-load lifecycle event
   (`sanity_probe_pending` on the state); a deferred probe logs its success
   at WARNING so verl's default WARN log carries the positive evidence (the
   attach-time success stays INFO). Two further checks are off by default:
   `VLLM_DUAL_PRECISION_VALIDATE_SHADOW=1` compares every attached INT4
   linear with its BF16 twin on a random input and logs the ten worst
   cosines, deferred to the same point on a dummy engine
   (`shadow_validation_pending`) so it never scores the shadow against
   random base weights. `VLLM_DUAL_PRECISION_VALIDATE_LIFECYCLE=1` records fixed-input
   probes for the first six attached linears at load, re-runs them at the
   first INT4 bind (the baseline, which under a server is the CUDA-graph
   capture during init) and again at the first INT4 bind after every
   lifecycle event, logging `exact=True/False` per probe at WARNING (ERROR
   when a probe is no longer exact). Lifecycle events are marked through
   `mark_lifecycle_event(model, kind)`: the worker's `sleep` / `wake_up`
   (`sleep`, `wake_up`), the runner's `reload_weights`, the worker's
   `update_weights`, and every `model.load_weights(...)` call (the attach
   wraps the method on the instance, a `__dict__` entry, because verl's
   colocated worker extension streams the trainer's base weights with
   `model.load_weights` straight on the model). An event arms exactly one
   re-validation at the next INT4 bind, even an INT4 -> INT4 rebind across a
   wake-up; without an event no re-validation happens (integration defect
   4: the old first-bind-only probe ran before any sleep/wake or weight sync
   and proved nothing about the shadow at rollout time).

## Knobs (all registered in `vllm/envs.py`; defaults keep vanilla behavior)

| Env var | Default | Meaning |
|---|---|---|
| `VLLM_DUAL_PRECISION_ROLLOUT` | `0` | enable residency (`dual_precision_rollout_enabled()`) |
| `VLLM_DUAL_PRECISION_INT4_MODEL` | `""` | shadow checkpoint (GPTQ INT4 or ModelOpt NVFP4); required when enabled |
| `VLLM_DUAL_PRECISION_BF16_LAYERS` | `first:3,last:3` | blocks kept BF16 (`none`, `first:N`, `last:N`, `i`, `a-b`); every final run used `none` |
| `VLLM_DUAL_PRECISION_INT4_MODULES` | `all` | `all` or `mlp_only` (Gemma4 E2B/E4B QAT runs) |
| `VLLM_DUAL_PRECISION_VALIDATE_SHADOW` | `0` | numerical check at load |
| `VLLM_DUAL_PRECISION_VALIDATE_LIFECYCLE` | `0` | probes at load, re-check at first INT4 bind and after every sleep/wake-up/weight-load event |

verl: `actor_rollout_ref.rollout.model_path` (null: reuse the actor path)
and the C8 block `actor_rollout_ref.rollout.precision_scheduler.{enable,
int4_model, bf16_layers, int4_modules, validate_shadow, validate_lifecycle,
sleep_level}`, which verl translates into the env vars above for the server
actor (`verl.workers.config.precision_scheduler`). `sleep_level` null keeps
the engine default; `resolve_sleep_level` in that module applies it at both
`engine.sleep` sites and keeps level 1 when dual precision is enabled.

## Contracts with neighbours

* **C1 (`base_linear.py`)**: `BaseLinearLayerWithLoRA.base_forward_override`
  / `set_base_forward_override(fn)`; when set, `apply()` computes the base
  output through `fn(x, bias)` and applies LoRA synchronously afterwards
  (the dual-stream op is not used, decision 1). All dual-precision state
  lives in this package, keyed by module.
* **C3/C4**: consume `BASE_PRECISION_BF16/INT4`, `bind_dual_precision(model,
  precision, no_compile_layers)` (call before every capture, replay and eager
  forward with the batch's `base_precision`), `get_active_precision(model)`
  and `BatchDescriptor.base_precision`.
* **verl**: `SHADOW_MODULE_NAME` is imported by
  `_hide_dual_precision_shadow_model` (string fallback), which pops the store
  around `process_weights_after_loading` during weight sync: the Marlin
  repack is not idempotent and, on this vLLM base, asserts on a second visit.
  `aggressive_empty_cache` runs before `rollout.resume(tags=["weights"])`
  so the colocated trainer's allocator cache does not collide with the
  remapped BF16 + shadow weights.

## Dropped from the experimental tree, and why

* **AWQ block-state pairing** (paired norm parameters swapped in
  `_parameters`/`_buffers` per bind) and AWQ shadows altogether. GPTQ and
  NVFP4 only:
  on the Qwen3.5-9B AutoRound shadow 152 of 176 shared non-linear block
  tensors are bit-identical to BF16 and the remaining 24 (128-dim
  gated-deltanet norm vectors) differ at bf16 rounding level (max abs 0.004),
  so the pairing was a near no-op on the headline path; the INT4 path now
  uses the BF16 model's norms. Removing it also removes the only bind-time
  mutation of module dicts.
* **`_modules['base_layer']` swap** (commit 34e66a3): root cause of the
  `KeyError('weight')` under `@support_torch_compile` on Gemma4; replaced by
  the opaque op. `test_committed_modules_swap_violates_the_invariant` keeps
  it out.
* **Module-global caches** (`_BIND_LOGGED`, `_LIFECYCLE_VALIDATED`,
  wrapper discovery cache) and 15 `object.__setattr__` side attributes:
  replaced by one `DualPrecisionState` per model and one
  `DualPrecisionBinding` per wrapper.
* **Two `if self.lora_config` blocks in the runner**: one
  `attach_dual_precision()` call after LoRA load; discovery happens there,
  so `bind_dual_precision` no longer needs `no_compile_layers` (accepted for
  call-site symmetry, ignored).
* **`ROLLOUT_QLORA` gating of the feature flag**: `dual_precision_rollout_
  enabled()` reads only `VLLM_DUAL_PRECISION_ROLLOUT`; the LoRA fast path is
  C1's concern and the override contract works with or without it.
* **Analysis BF16 mask**: kept as `set_analysis_bf16_layers` (eager-only
  diagnostic used through `llm.apply_model`), now a field of the per-model
  state rather than extra attributes; the layer-sensitivity scripts that used
  it are not ported.
* **Marlin K-padding** for Nemotron stays with C9; the Nemotron golden skips
  until it merges.

## Measured numbers (2026-09-11, GPU 3, this branch)

* Qwen3.5-9B BF16 (`/data/huggingface/hub/models--Qwen--Qwen3.5-9B/...c2022362`)
  + Intel AutoRound INT4 (`models--Intel--Qwen3.5-9B-int4-AutoRound/...29688b89`),
  `BF16_LAYERS=none`: `Loaded 286 GPTQ shadow linear layers; attached 152 ...
  kept 0 ... left 134` (the wording of that line has since dropped "GPTQ",
  which no longer holds for every shadow; the counts are unchanged) (identical to the 697 archived runs under
  `/data/huanchen/verl/.codex-report/**`); shadow store 3.32 GiB; worst
  per-layer cosine 0.9856 (`layers.30.linear_attn.in_proj_ba`, rel-RMSE
  0.169). With the default `first:3,last:3`: 123 attached / 29 by policy
  (archived line matches). Lifecycle probes after `sleep(1)`, `wake_up` and a
  weight-sync repack: six probes `exact=True, max_abs=0`.
* Nemotron-Nano-9B-v2 + RedHatAI w4a16: 139 linears, 112 attached / 27
  fallback (the 27 mamba `conv1d` projections), from the module-name fixture;
  archived greedy audit in
  `.codex-report/precision-scheduling-validation/stage0/weight_audit/`
  (dual-resident W4 bit-identical across two runs; prefix agreement with
  standalone W4 of 671 and 215 tokens).
* Gemma4 E2B QAT, `mlp_only`: 213 linears, 70 attached / 141 by policy / 2
  fallback (the vision/audio embedding projections, `LinearBase` on this
  base; the archived log reported 0).
* Fixtures: `tests/model_executor/dual_precision/fixtures/*_modules.json`,
  derived from the checkpoints' safetensors headers by
  `derive_module_names.py`; the Qwen3.5-9B and Gemma4 lists were verified
  equal to a meta-device `initialize_model` of the real vLLM models.

## Known gaps

* `ReplicatedLinearWithLoRA.apply` used to call `self.base_layer(x)` directly
  and bypass the override; resolved by C1 (commit f121b54413 on the clean
  branch routes every LoRA linear `apply()` through the override when one is
  installed). `test_real_lora_wrapper_routes_apply_through_override` covers
  the base-class path.
* Bind call sites in `execute_model`, `_dummy_run` and CUDA-graph capture
  are C3/C4's; until they land the engine attaches the shadow but always
  serves BF16.

## Re-prefill after the switch (component C7; default-off ablation)

`VLLM_DUAL_PRECISION_REPREFILL=1` makes the scheduler preempt every surviving
request at the step on which the switcher (C4, `docs/design/precision_switch.md`)
reports the BF16 to INT4 switch, so the survivors' KV is recomputed under the
INT4 base instead of continuing from KV produced by BF16. Default `0`; with
every flag off the scheduler is vanilla. Code: `Scheduler.__init__`
(`precision_reprefill_enabled`, `_validate_precision_reprefill_config`),
`Scheduler._maybe_trigger_precision_reprefill`, `Request.precision_reprefill_done`
/ `precision_reprefill_output_offset`; tests `tests/v1/core/test_precision_reprefill.py`
(CPU) and `tests/v1/core/test_precision_reprefill_gpu.py` (gpu-smoke).

Mechanism. `schedule()` ticks the switcher, then, when re-prefill is on and
`switcher.last_switch` belongs to a rollout that has not been re-prefilled
yet, runs the trigger before the running loop: every running request that
has computed tokens and has not been re-prefilled is preempted through the
vanilla `_preempt_request` in reverse order (FCFS is preserved because
preemption prepends to the waiting queue), `precision_reprefill_done` is
set, the response length at the boundary is recorded in
`precision_reprefill_output_offset`, in-flight async output placeholders are
discarded exactly as `reset_prefix_cache(reset_running_requests=True)` does
(`async_tokens_to_discard = num_output_placeholders; num_output_placeholders = 0`),
waiting requests are marked done, and `prev_step_scheduled_req_ids` is cleared
so the model runner rebuilds the survivors as resumed requests. Because the
preempted list is populated before the running loop, the switching step is
an *idle engine step* (`total_num_scheduled_tokens == 0`) and the survivors
resume as prefill under the INT4 precision on the next step. The archived
evidence was produced with these idle-step semantics, which is why the
trigger was kept as a copy instead of being folded into a shared
preempt-all helper. Two log lines keep their archived format:
`Dual precision re-prefill request <id>: generated_tokens=%d, total_reprefill_tokens=%d`
per survivor and `Dual precision re-prefill triggered: rollout_index=%d,
committed_frontier=%d, applied_response_tokens=%d, unfinished_requests=%d,
preempted_requests=%d, total_reprefill_tokens=%d` (the experimental line
carried `threshold=` instead of the rollout/frontier fields).

Once per rollout, re-armed on drain. The idempotence key is the switch
event's `rollout_index`: the trigger fires at most once per switch, and the
switcher produces at most one switch per rollout. A drained scheduler ends
the rollout in the switcher (cohort-free specs) and the next batch arms a
new rollout with a new index, so a long-lived rollout engine re-prefills
once per batch; back-to-back cohort rollouts (a new cohort admitted before
the scheduler ever looked empty) get a new index too. A request that was
already re-prefilled in an earlier rollout and is still running is skipped
(the per-request latch), as in the experimental code. Any policy kind that
switches triggers it (`fixed_threshold`, `fixed_frontier`, EMA tables);
`uniform_w4` never reports a switch and never re-prefills, and the flag is
ignored with a warning when no policy is configured. Construction fails with
prefix caching (block hashes do not encode precision), KV connectors or EC
connectors.

Knobs.

| env (vllm/envs.py) | default | meaning |
|---|---|---|
| `VLLM_DUAL_PRECISION_REPREFILL` | `0` | preempt every survivor at the rollout's switch so its KV is recomputed under INT4; honoured only when `VLLM_DUAL_PRECISION_POLICY` produces a switch |

Dropped from the experimental tree (decision 5): `VLLM_REPREFILL_ONLY_ROLLOUT`
(a second gate that ran the trigger without INT4; one archived one-step run),
the separate threshold latch inside the trigger (the switch event is the
single source of the precision signal now), the `num_visible_output_tokens`
fold branch and the `check_stop` change that used it (dead in every code
path: output tokens survive preemption, so `num_output_tokens` keeps counting
from the pre-switch length and `max_tokens` is unaffected; the branch was
also non-monotone once the post-fold length passed the offset). The
`reprefill_done` / `reprefill_output_offset` fields are kept under the
`precision_reprefill_*` names; the offset is informational.

Evidence. The NLL study behind the default-off decision lives in verl
(`examples/precision_scheduler/analysis/reprefill_nll_study/README.md`):
on 16 Qwen3.5-9B long-tail traces with fake INT4, continuing from the BF16
state is *closer* to BF16 than re-prefilling under INT4 at every window
offset (dNLL reuse-reprefill -0.064 / -0.030 / -0.032 / -0.045 / -0.042
nats at offsets 0 / 512 / 1024 / 2048 / 4096, CI95 entirely negative;
KL-to-BF16 0.040 vs 0.097 at offset 0). The archived threshold studies
(`temporal_guard_120`, 72 completed runs; the `tail-k*-reprefill` natural
policies; the 27B `mixed_precision_reprefill` rollouts) ran with re-prefill
ON under the experimental scheduler; the later no-reprefill studies and every
headline dynamic-policy run ran with it OFF. GPU smoke (2026-09-11, GPU 2,
Qwen3.5-4B, `fixed_frontier:32`, two batches of eight greedy prompts, 128
tokens, eager; residency not enabled, so the preempt/recompute path is what
is exercised): one trigger per batch, `preempted_requests=8` ==
`unfinished_requests=8`, `num_preemptions == 1`, idle trigger step, both
batches identical, pre-switch tokens bit-identical to an uninterrupted run
on the same engine and 7/8 sequences identical over all 128 tokens (token
agreement 0.94).

### NVFP4 shadow (2026-09-15, RTX PRO 6000 Blackwell sm_120, pro6000-adapt branch)

Same BF16 base, shadow `models--AxionML--Qwen3.5-9B-NVFP4/...97aef923`
(ModelOpt NVFP4, 4-bit float weights, group 16), `BF16_LAYERS=none`,
`enforce_eager`: **152 attached / 0 by policy / 134 fallback, identical to the
AutoRound INT4 shadow above**, shadow store 3.62 GiB (against 3.32 GiB for
AutoRound), attach-time sanity probe cosine 0.9927, worst per-layer cosine
0.9849 (`layers.6.linear_attn.in_proj_ba`, rel-RMSE 0.174) against AutoRound's
0.9856. vLLM selects `FlashInferCutlassNvFp4LinearKernel` for the shadow GEMM.
The engine generates normally with the shadow attached
(`tests/model_executor/dual_precision/test_nvfp4_shadow_gpu.py`, 2 passed).

That the counts match exactly is the point: the residency and binding path is
not format-aware. A binding holds two `LinearBase` objects and the forward runs
through whichever is active, using that layer's own `quant_method`, so the only
format-specific code is the gate and the parameter predicate.
