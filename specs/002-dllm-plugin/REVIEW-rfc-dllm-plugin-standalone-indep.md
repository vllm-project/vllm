# Independent Review: RFC dLLM Plugin (spec-decode path reuse)

**Document reviewed:** [rfc-dllm-plugin-standalone.md](rfc-dllm-plugin-standalone.md)  
**Review type:** Independent (no reliance on prior reviews)  
**Scope:** Viability, weak points, risks, assumptions vs production deployments, impact on users and community.

---

## 1. Executive summary

The RFC proposes adding **block-based diffusion LLM (dLLM)** support to vLLM by **reusing the spec-decode data path** and making a **single engine change**: call the draft-token hook after every step when the model was executed, not only when `use_spec_decode` is true. All dLLM logic lives in a **plugin** (custom scheduler, worker, and model).

**Verdict:** The design is **technically viable** and the minimal core change is **sound and backward compatible**. Key **weak points** are: (1) strict, easy-to-get-wrong scheduler/worker/model pairing, (2) reliance on an interface the codebase explicitly marks as **not public**, (3) field overloading that hurts debugging and observability, and (4) underspecified first-step and async/structured-output behavior. **Production risk** is real if deployments use a dLLM model with the default scheduler/worker (state corruption, wrong outputs). **Community impact** is positive (ecosystem alignment, one engine for AR and dLLM) provided the project documents the implicit contract and supports operators with clear configuration and observability.

---

## 2. Viability assessment

### 2.1 Technical fit and codebase verification

**Data-shape alignment.** The RFC correctly argues that spec-decode and dLLM share the same abstract shape: one forward over a block → variable-length “finalized” tokens + fixed-size next block. Reusing `scheduled_spec_decode_tokens`, `sampled_token_ids`, `spec_token_ids`, and the draft-token hook is **coherent**.

**Engine change (sync path).** In `vllm/v1/engine/core.py`, `post_step()` (lines 410–418) currently runs the draft-token update only when:

```text
if not self.async_scheduling and self.use_spec_decode and model_executed:
```

Relaxing this to “whenever the model was executed” (e.g. drop `and self.use_spec_decode`) is **backward compatible**: the default worker’s `take_draft_token_ids()` returns `None` when `not self.num_spec_tokens or not self._draft_token_req_ids` (`gpu_model_runner.py` ~3932–3934), so for normal AR and non–spec-decode runs the hook is a no-op.

**Engine change (async path).** In `step_with_batch_queue()` (lines 514–526), the draft-token update is gated on `if self.use_spec_decode`. Changing this to run whenever `deferred_scheduler_output` exists and `take_draft_token_ids()` returns a non-None value allows the dLLM plugin to supply the next block without enabling `speculative_config`. For non-dLLM, non–spec-decode deferred cases, `take_draft_token_ids()` returns `None`, so behavior is unchanged.

**Default scheduler and commit-0.** In `vllm/v1/core/sched/scheduler.py`, `update_from_output` (lines 1320–1334) only adjusts `num_computed_tokens` (rollback for rejected spec tokens) when **both** `scheduled_spec_token_ids` and `generated_token_ids` are non-empty. So when the plugin reports **commit-0** (empty `sampled_token_ids`), the **default** scheduler does **not** roll back. Because `_update_after_schedule` has already advanced `num_computed_tokens` by `num_scheduled_tokens` (e.g. DRAFT_SIZE) before the step, using the default scheduler with a worker that sometimes returns empty `sampled_token_ids` while scheduling a block would **desync** KV cache and `num_computed_tokens`. The RFC’s reliance on a **plugin scheduler** to implement commit-0 rollback is therefore **necessary and correct**; the weak point is that the dLLM model **must never** run with the default scheduler.

**Conclusion:** The design is **viable**; the single engine change is minimal and safe for existing AR and spec-decode users. Correctness depends entirely on the dLLM stack (model + scheduler + worker) being used as a single unit.

### 2.2 Plugin encapsulation and configuration

- Custom scheduler and worker are selected via **config**: `SchedulerConfig.scheduler_cls` (str or type) and `ParallelConfig.worker_cls` (str, default `"auto"`). The RFC’s “user or launcher passes `--scheduler-cls` and `--worker-cls`” is **accurate**.
- General plugins register **models** via `ModelRegistry` / `vllm.general_plugins`; they do **not** register workers or schedulers. So the dLLM plugin **package** must ship scheduler and worker **classes**, and deployment must **explicitly** set both when serving a dLLM model. There is no “dLLM mode” flag that sets all three (model + scheduler + worker) today; the RFC correctly identifies that deployment/automation should treat the triple as one stack and recommends plugin-side validation.

---

## 3. Weak points

### 3.1 Strict scheduler/worker/model pairing and misconfiguration

- A dLLM model **must** run with the plugin’s scheduler and worker. Using the default scheduler (or default worker) with a dLLM model leaves commit-0 unhandled and can corrupt `num_computed_tokens` and KV cache.
- **Misconfiguration is easy:** A user (or a model registry) can select a dLLM model by name without setting `scheduler_cls` and `worker_cls`. The process then runs with the default scheduler and worker. The default worker may not even implement the dLLM forward (wrong interface), or it may run something that produces empty `sampled_token_ids` sometimes; the default scheduler will not roll back, leading to subtle or catastrophic state corruption.
- The RFC’s suggestion that the **plugin validate at startup** that the active scheduler and worker are its own classes (and raise a clear error otherwise) is **important** and should be a **required** part of the plugin contract, not optional.

### 3.2 Stability of the spec-decode path

- In `vllm/config/scheduler.py` (lines 174–176), the codebase logs a **warning** that the custom scheduler interface is “not public and compatibility may not be maintained.” The RFC **assumes** that `update_draft_token_ids`, `update_draft_token_ids_in_output`, and the spec-decode fields remain stable enough for a plugin to rely on.
- **Risk:** Future changes to spec-decode (e.g. new fields, different semantics for `sampled_token_ids` or `scheduled_spec_decode_tokens`) could break the dLLM plugin. Without an explicit **stability commitment** or a small, documented “dLLM extension contract,” production plugins are fragile across vLLM releases. The RFC’s “Limitations, risks, and open questions” section acknowledges this; it should be elevated to a **recommendation** that the project either document this contract and treat it as stable or reject the design.

### 3.3 Field overloading and observability

- The same fields (`spec_token_ids`, `scheduled_spec_decode_tokens`, `sampled_token_ids`) carry **different meaning** for dLLM (next block, input block, Committed tokens) vs spec-decode (draft tokens, verified + 1 token). Any logging, metrics, or profiling that assumes “speculative decoding” (e.g. acceptance rate, draft/accept counts) will be **wrong or misleading** for dLLM.
- **Production impact:** Operators and support need a **clear signal** that a run is dLLM (e.g. model type, config flag, or request attribute) so they do not mis-tune or mis-diagnose. The RFC mentions this under “Field overloading and observability” but does not specify how the signal should be exposed; that should be part of the MVP or a follow-up spec.

### 3.4 First decode step (prompt → first block)

- The RFC states that on the first decode step the input block “may be derived from prompt + padding with `<MASK>`” and that “the exact convention is plugin-defined.” It does **not** specify who builds the first block (scheduler, worker, or model), who sets initial `request.spec_token_ids` before the first schedule, or how padding/masking is agreed between components.
- **Risk:** Divergent plugin implementations and subtle bugs (e.g. wrong first-block layout, inconsistent mask placement) across dLLM architectures. This is a **spec gap** that should be closed in the RFC or a linked spec before or shortly after MVP.

### 3.5 Async scheduling and structured output

- The batch-queue path uses draft token ids for **grammar bitmask** when `use_spec_decode` is true. The RFC proposes running the draft update whenever deferred output exists and draft token ids are available. For dLLM, the “next block” is not necessarily grammar-validated in the same way; the RFC leaves the interaction of dLLM with **async scheduling** and **structured output** to “plugin implementation or a follow-up spec.”
- **Risk:** Unclear or incorrect behavior when dLLM is used with async scheduling or structured output (e.g. grammar applied to the plugin’s next block, or deferred-step logic assuming AR semantics). This should be clarified so the plugin contract is complete and the default path does not assume AR + grammar semantics for those tokens.

### 3.6 Scheduler interface: update_draft_token_ids and grammar

- The abstract interface (`vllm/v1/core/sched/interface.py`) documents `update_draft_token_ids` and `update_draft_token_ids_in_output` as applying “structured output grammar validation if needed.” For dLLM, the “next block” is produced by the model and may not be subject to the same grammar filtering. The RFC does not state whether the core will apply grammar to the plugin’s next block when a custom scheduler is used. If the default scheduler implementation applies grammar in these methods, the plugin scheduler may need to **override** and skip grammar for the dLLM next block, or the contract must explicitly state that when a custom scheduler is used, grammar application is the scheduler’s responsibility. This should be made explicit to avoid silent wrong behavior.

---

## 4. Risks for production and users

### 4.1 Correctness and deployment

| Risk | Effect | Mitigation |
|------|--------|------------|
| **Wrong pairing** (dLLM model + default scheduler/worker) | Desync of `num_computed_tokens` and KV cache; wrong outputs or crashes. | Treat (model + scheduler_cls + worker_cls) as one “dLLM stack”; plugin validation at startup; single preset or launcher that sets all three. |
| **First-step inconsistency** | Different plugins (or versions) build the first block differently; hard-to-reproduce bugs. | Specify first-step contract (who sets `request.spec_token_ids`, who builds first block, padding convention) in RFC or linked spec. |
| **Async/structured output** | Unclear or incorrect behavior when dLLM is used with grammar or deferred scheduling. | Clarify in RFC or follow-up: does core apply grammar to plugin’s next block? How does deferred path treat dLLM? |

### 4.2 Observability and support

- Metrics and logs that key off “spec decode” (e.g. acceptance rate, draft length) are **meaningless or wrong** for dLLM. Support and ops need a **clear way** to identify dLLM runs (e.g. config flag, model type, or request attribute) so they do not misinterpret or mis-tune. The RFC should recommend that the MVP (or a quick follow-up) define this signal.

### 4.3 Upstream evolution and maintenance

- The plugin relies on the **same** fields and hooks as spec-decode. If the project evolves spec-decode (e.g. new fields, different semantics) without treating the dLLM plugin as a stakeholder, the plugin can **break or change behavior** silently. The RFC’s “mild abuse” is an **implicit contract** that should be **documented in code** (e.g. in the engine and scheduler) so maintainers consider the plugin when changing that path. Optionally, the project could declare the hook and field set used by the dLLM plugin as a **stable extension point** for custom scheduler/worker plugins.

### 4.4 Performance

- Running the draft-token hook after every step when the model was executed has **negligible** cost for the default worker (it returns `None`). For the plugin worker, the hook is required. So **risk to existing AR and spec-decode users is low**.

---

## 5. Assumptions and applicability to production

| Assumption | Production applicability | Risk if wrong |
|------------|--------------------------|----------------|
| **Scheduler interface stable enough for a plugin** | The codebase **explicitly warns** that the custom scheduler interface is not public. Production plugins need a stable target across minor/patch releases. | Without a documented contract or stability commitment, production dLLM plugins can break on any spec-decode or scheduler change. |
| **User or config passes `--scheduler-cls` and `--worker-cls` for dLLM** | In production, config is often automation-driven (Helm, Terraform, internal launchers). If only the model is selected (e.g. from a registry) and scheduler/worker are omitted, the default stack runs → state corruption. | Wrong outputs, crashes, or hard-to-debug behavior. Mitigation: single “dLLM stack” preset and plugin-side validation that fails fast with an actionable error. |
| **One scheduler/worker pair per process** | Typical production runs one model (or one stack) per process. Mixing AR and dLLM in the same process would require the default scheduler to understand commit-0 or multiple modes; it does not. | Constrains multi-model or mixed AR+dLLM in one process; acceptable for many deployments. |
| **Default worker returns None when not spec-decode** | **Verified:** `take_draft_token_ids()` returns `None` when `not self.num_spec_tokens or not self._draft_token_req_ids`. So the new hook is a no-op for normal AR and non–spec-decode runs. | No production risk for existing users. |
| **First-step semantics (who builds first block, initial spec_token_ids)** | RFC defers to the plugin. In production, divergent implementations can cause subtle bugs or incompatibility across dLLM plugins. | Recommend specifying first-step contract in the RFC or a linked spec. |
| **Async scheduling and structured output** | RFC leaves interaction to follow-up. Batch-queue path uses draft token ids for grammar when `use_spec_decode` is true. For dLLM, the next block may not have the same grammar semantics. | Unclear behavior when dLLM is used with async scheduling or structured output; possible incorrect filtering or deferred-step bugs. |

---

## 6. Effect on actual users and the overall community

### 6.1 End users (API consumers)

- **Benefit:** Access to dLLM models (e.g. LLaDA2.x, WeDLM) with reported throughput/latency gains, using the same vLLM deployment they already use. No change for users who do not use dLLM models.
- **Risk:** If a deployment mistakenly runs a dLLM model with the default scheduler/worker, outputs can be wrong or the process can crash; from the API consumer’s perspective this can look like “vLLM is broken” or “this model is unstable.” **Clear errors and documentation** are essential.

### 6.2 Operators and platform teams

- **Burden:** Must treat the dLLM stack as a **single unit**: model + `--scheduler-cls` + `--worker-cls`. Automation (Helm, Terraform, launchers) must set all three when serving a dLLM model. Forgetting one leads to incorrect execution.
- **Observability:** Existing spec-decode metrics (e.g. acceptance rate, draft length) do **not** apply to dLLM. Operators need a **clear way** to know a run is dLLM so they do not mis-tune or mis-diagnose. The RFC should recommend implementing this signal (e.g. model type or config) for production support.

### 6.3 Plugin authors and model providers

- **Opportunity:** New dLLM architectures can be shipped as plugins without changing vLLM core, aligned with the existing plugin story.
- **Fragility:** Plugin authors depend on an interface the codebase currently warns is “not public.” Without an explicit stability commitment (or a documented “dLLM contract”), plugins may break on vLLM upgrades. The RFC’s recommendation to document the contract in the RFC and in code is important.

### 6.4 vLLM maintainers and community

- **Maintenance cost:** The engine change is small, but the **implicit contract** (same fields, different semantics when custom scheduler/worker are used) must be **documented** where the hook and fields live, so future spec-decode or scheduler changes do not silently break the dLLM plugin.
- **Ecosystem alignment:** SGlang, Ollama, and LMDeploy already support or ship dLLM-style inference. Adding dLLM via plugin keeps vLLM relevant for users who want one engine for both AR and dLLM and reduces incentive to switch stacks solely for dLLM.

### 6.5 Summary table

| Stakeholder        | Net effect |
|--------------------|------------|
| End users          | Positive if dLLM is desired; no impact if not. Risk of wrong outputs if deployment is misconfigured. |
| Operators          | Extra configuration discipline; need dLLM-aware observability. |
| Plugin authors     | New capability with dependency on an currently “unstable” interface. |
| vLLM maintainers   | Low code cost; documentation and stability commitment for the reused path. |
| Community/ecosystem| Keeps vLLM competitive on dLLM; aligns with plugin-based extensibility. |

---

## 7. Recommendations

1. **Document the implicit contract** in the engine and scheduler: when a custom scheduler and worker are used, the same fields may carry dLLM semantics; list those fields and the single engine change so maintainers consider the plugin when evolving that path. Optionally, declare this as a **stable extension point** for custom scheduler/worker plugins.
2. **Specify first-step semantics** in the RFC or a linked spec: who builds the first input block, how `request.spec_token_ids` is initialized before the first schedule, and how padding/masking works (prompt + `<MASK>`).
3. **Clarify async and structured output:** Whether the core applies grammar (or other structured-output logic) to the “next block” from the plugin, and how the deferred path treats dLLM. Document that the plugin scheduler may need to override `update_draft_token_ids` / `update_draft_token_ids_in_output` to skip grammar for the dLLM next block if the default applies it.
4. **Improve deployability:** Prefer a **single entry point** (e.g. “use dLLM” flag or preset) that sets model + scheduler_cls + worker_cls; require **plugin-side validation** that the active scheduler and worker are the plugin’s when the plugin’s model is loaded, with a clear, actionable error on mismatch.
5. **Observability:** Define a **clear signal** (e.g. request/engine attribute or config) to mark dLLM runs so logs and metrics can treat them separately from AR and spec-decode. Include this in the MVP or a quick follow-up.
6. **Stability:** If the project accepts this design, consider documenting the spec-decode hook and field set used by the dLLM plugin as a **stable contract** for custom scheduler/worker plugins (or a dedicated “dLLM extension point”) so plugin authors and production deployments have a clear compatibility target.

---

## 8. Conclusion

- **Viability:** The design is **viable**; the minimal engine change is correct and backward compatible. Correctness depends on the dLLM stack (model + scheduler + worker) being used as a single unit.
- **Weak points:** Strict scheduler/worker/model pairing (and easy misconfiguration), unstable scheduler interface, field overloading (debugging/observability), underspecified first step and async/structured-output behavior, and potential grammar application to the plugin’s next block.
- **Risks:** Production config mistakes (wrong scheduler/worker) can cause incorrect execution or state corruption; spec-decode evolution can break the plugin; observability and support are harder without an explicit dLLM signal.

The RFC is a **reasonable** way to bring dLLM into the vLLM ecosystem with minimal core change and good encapsulation, **provided**:

1. The plugin is **always** used as a coherent stack (model + scheduler + worker) and the plugin **validates** this at startup.
2. First-step and async/grammar behavior are **specified** (in the RFC or a linked spec).
3. The project **documents** (and optionally commits to) the stability contract for the spec-decode path used by the plugin.
4. Operators have a **clear** way to configure the dLLM stack and to distinguish dLLM runs in observability.

---

## 9. References (codebase)

- Engine: `vllm/v1/engine/core.py` — `post_step()` (lines 410–418), `step_with_batch_queue()` (lines 514–526), `use_spec_decode`.
- Scheduler: `vllm/v1/core/sched/scheduler.py` — `update_from_output` (spec-decode rollback at 1320–1334), `_update_after_schedule` (num_computed_tokens advance), `scheduled_spec_decode_tokens` / `spec_token_ids` usage.
- Scheduler interface: `vllm/v1/core/sched/interface.py` — `update_draft_token_ids`, `update_draft_token_ids_in_output` (grammar mentioned in docstrings).
- Worker: `vllm/v1/worker/gpu_model_runner.py` — `take_draft_token_ids()` (3932–3936), `num_spec_tokens`, `_draft_token_req_ids`.
- Config: `vllm/config/scheduler.py` — `get_scheduler_cls()` and custom-scheduler warning (174–176); `vllm/config/parallel.py` — `worker_cls`.
