# Review: RFC dLLM Plugin (spec-decode path reuse)

**Document reviewed:** [rfc-dllm-plugin-standalone.md](rfc-dllm-plugin-standalone.md)  
**Reviewer:** (agent review)  
**Scope:** Viability, weak points, risks, assumptions vs production deployments, and impact on actual users and the overall community.

---

## 1. Executive summary

The RFC proposes adding **block-based diffusion LLM (dLLM)** support in vLLM by **reusing the existing spec-decode data path** and making a **single engine change**: call the draft-token hook after every step when the model was executed, not only when `use_spec_decode` is true. All dLLM logic is encapsulated in a **plugin** (custom scheduler + worker + model). The design is **technically viable** and the minimal core change is sound and backward compatible. Weak points center on **strict scheduler/worker pairing**, **stability of the spec-decode interface**, **field overloading** (debugging/observability), and **underspecified first-step and async behavior**. Production risk is manageable if deployment always uses the plugin as a coherent stack and the project documents the hidden contract for the spec-decode path used by the plugin. **Impact on users and community** is positive (dLLM available without forking, ecosystem alignment with SGlang/Ollama/LMDeploy) but requires clear docs and validation to avoid misconfiguration and support burden.

---

## 2. Viability assessment

### 2.1 Technical fit

- **Data-shape alignment:** Spec-decode uses one forward over a draft block → verified prefix + 1 new token + next draft block. dLLM uses one forward over an input block → 0..DRAFT_SIZE committed tokens + next input block. The shapes align: block in, variable-length “finalized” tokens out, fixed-size next block. Reusing `scheduled_spec_decode_tokens`, `sampled_token_ids`, `spec_token_ids`, and the draft-token hook is coherent.
- **Codebase verification:** The exact engine locations cited in the RFC exist and behave as described:
  - **Sync path:** `vllm/v1/engine/core.py` `post_step()` (lines 410–418): condition `if not self.async_scheduling and self.use_spec_decode and model_executed`; relaxing to “whenever model was executed” is backward compatible because the default worker returns `None` from `take_draft_token_ids()` when not doing spec-decode.
  - **Async path:** `step_with_batch_queue()` (lines 514–526): draft-token update is gated on `self.use_spec_decode`; changing to “whenever deferred output exists and draft token ids are available” allows the plugin to supply the next block without requiring `speculative_config`.
- **Default worker behavior:** `take_draft_token_ids()` returns `None` when `not self.num_spec_tokens or not self._draft_token_req_ids` (gpu_model_runner.py). So for normal AR runs the new hook is a no-op.

**Conclusion:** The design is **viable**; the single engine change is minimal and safe for non-dLLM users.

### 2.2 Plugin encapsulation

- Custom scheduler and worker are selected via `--scheduler-cls` and `--worker-cls` (qualified string names; passing a class is deprecated per worker_base.py). Platform plugins (e.g. CUDA/ROCm) set `worker_cls` in `check_and_update_config`; the same mechanism could set both scheduler and worker for a dLLM “platform,” or users pass both explicitly. The RFC’s “plugin supplies scheduler + worker” approach is **achievable**.

---

## 3. Weak points

### 3.1 Default scheduler and commit-0

- In the default scheduler’s `update_from_output`, the rollback of `num_computed_tokens` (for rejected spec tokens) runs only when **both** `scheduled_spec_token_ids` and `generated_token_ids` are non-empty (scheduler.py ~1320–1334). So when the plugin reports **commit-0** (empty `sampled_token_ids`), the default scheduler does **not** roll back.
- `_update_after_schedule` has already advanced `num_computed_tokens` by `num_scheduled_tokens` (e.g. DRAFT_SIZE) before the step. So if the **default** scheduler were ever used with a worker that sometimes returns empty `sampled_token_ids` while still scheduling a block, state would be wrong (KV and `num_computed_tokens` out of sync).
- The RFC correctly relies on the **plugin scheduler** to implement commit-0 (roll back by full scheduled length when committed count is 0). The weak point is **strict coupling**: dLLM must **always** run with the plugin scheduler (and plugin worker). Any deployment that uses the dLLM model with the default scheduler is unsafe.

### 3.2 Scheduler interface stability

- `SchedulerConfig.get_scheduler_cls()` (config/scheduler.py) already logs a warning that the custom scheduler interface is **not public** and compatibility may not be maintained. The RFC assumes `update_draft_token_ids` / `update_draft_token_ids_in_output` and the spec-decode-related fields remain stable. That is a **long-term maintenance risk**: future changes to spec-decode (e.g. new fields, different semantics) could break the dLLM plugin without the core explicitly “owning” the dLLM contract.

### 3.3 Field overloading and observability

- Reusing `spec_token_ids`, `scheduled_spec_decode_tokens`, and `sampled_token_ids` for dLLM avoids new core types but overloads meaning. Debugging, profiling, and metrics that assume “speculative decoding” (e.g. draft/accept counts, acceptance rate) will be wrong or misleading for dLLM. Production users need a clear way to distinguish dLLM from spec-decode (e.g. model type or a plugin flag) in logs and metrics.

### 3.4 First step (prompt → first block)

- The RFC states that on the first decode step the input block “may be derived from prompt + padding with `<MASK>`” but does not specify **who** builds it (scheduler, worker, or model) or how the scheduler learns the first block (e.g. who sets initial `request.spec_token_ids` before the first schedule). That is a **spec gap** that can cause divergent plugin implementations and subtle bugs (e.g. wrong first-block layout or mask placement).

### 3.5 Async scheduling and structured output

- The batch-queue path currently uses draft token ids for **grammar bitmask** when `use_spec_decode` is true. The RFC proposes running the draft update whenever deferred output exists and draft token ids are available. For dLLM, the “next block” is not necessarily grammar-validated in the same way. The interaction between deferred scheduling, structured output, and dLLM (e.g. whether the core ever applies grammar to the plugin’s next block) should be spelled out so the plugin contract is clear and the default path doesn’t assume AR + grammar semantics for those tokens.

### 3.6 Worker/scheduler selection and UX

- The RFC assumes “user or launcher passes `--scheduler-cls` and `--worker-cls`.” If only the dLLM **model** is loaded (e.g. via a general plugin) and the user forgets to switch scheduler and worker, the **default** scheduler and worker run. The default worker would then treat the run as normal AR (or fail if the model doesn’t match the standard interface), and the default scheduler would not handle commit-0. So **configuration errors are easy** and lead to incorrect state or hard-to-debug behavior. Documentation, examples, and ideally **validation** (e.g. plugin checks that the configured scheduler/worker are its own) are important.

---

## 4. Risks for production and users

### 4.1 Correctness and deployment

- **Wrong pairing:** Using a dLLM model with the default scheduler/worker can desync `num_computed_tokens` and KV cache (especially on commit-0), producing wrong outputs or crashes. Production configs must **always** pair the dLLM model with the plugin scheduler and worker. Any automation (Helm, Terraform, etc.) must encode this triple (model + scheduler_cls + worker_cls).

### 4.2 Observability and support

- Logs and metrics that key off “spec decode” (e.g. acceptance rate, draft length) will be meaningless or incorrect for dLLM. Support and ops need a clear signal that a run is dLLM so they don’t misinterpret or mis-tune.

### 4.3 Performance

- Relaxing the engine condition so the draft-token hook runs after every step has **negligible** impact for the default worker (it returns `None`). For the plugin worker, the hook is required. So risk to existing AR/spec-decode users is small.

### 4.4 Upstream evolution

- The **community/maintainer** risk is that the spec-decode path is evolved (e.g. for new speculative schemes or structured output) without treating the dLLM plugin as a stakeholder. Because the plugin relies on the same fields and hooks, such changes can break the plugin or change its semantics. The RFC’s “mild abuse” is a **hidden contract** that needs to be documented in the scheduler/spec-decode area so future changes consider dLLM.

---

## 5. Assumptions vs production reality

| Assumption | Applicability / risk |
|------------|----------------------|
| “One scheduler/worker pair per process” | **Critical.** Mixing AR and dLLM in the same process with different semantics would require the default scheduler to handle commit-0 or multiple modes; it doesn’t. So one process = one “mode” (AR or dLLM). Acceptable for many deployments; constrains multi-model or mixed AR+dLLM in one process. |
| Scheduler interface stable enough for a plugin | **At risk.** The codebase already warns that the custom scheduler interface is not public. Either the project commits to stability for this hook/field set (and documents it), or the plugin will be fragile across releases. |
| User/config passes `--scheduler-cls` and `--worker-cls` for dLLM | **Easy to get wrong.** Production and docs should make the triple (model + scheduler + worker) a single “dLLM stack” (e.g. one flag or preset that sets all three). |
| Default worker returns `None` when not spec-decode | **Holds.** So the engine change is safe for non-dLLM, non–spec-decode runs. |

---

## 6. Effect on actual users and the overall community

### 6.1 End users (API consumers)

- **Benefit:** Access to dLLM models (e.g. LLaDA2.x, WeDLM) via the same vLLM deployment they already use, with reported throughput/latency gains. No change for users who do not use dLLM models.
- **Risk:** If a deployment mistakenly runs a dLLM model with the default scheduler/worker, outputs can be wrong or the process can crash; from the API consumer’s perspective this looks like “vLLM is broken” or “this model is unstable.” Clear errors and docs reduce this.

### 6.2 Operators and platform teams

- **Burden:** Must treat the dLLM stack as a single unit: model + `--scheduler-cls` + `--worker-cls`. Any automation (Helm, Terraform, internal launchers) must set all three when serving a dLLM model. Forgetting one yields incorrect execution.
- **Observability:** Existing spec-decode metrics (e.g. acceptance rate, draft length) do not apply to dLLM; operators need a way to know a run is dLLM so they don’t mis-tune or mis-diagnose. The RFC’s “Limitations” note this; implementing a clear signal (e.g. model type or config) is important for production support.

### 6.3 Plugin authors and model providers

- **Opportunity:** New dLLM architectures can be shipped as plugins without changing vLLM core. Aligns with the existing plugin story (e.g. bart-plugin, general_plugins).
- **Fragility:** Reliance on spec-decode fields and the draft-token hook means plugin authors depend on an interface the codebase currently warns is “not public” (`config/scheduler.py` lines 174–176). Without an explicit stability commitment (or a small, documented “dLLM contract”), plugins may break on vLLM upgrades.

### 6.4 vLLM maintainers and community

- **Maintenance cost:** The engine change is small, but the **hidden contract** (same fields, different semantics when custom scheduler/worker are used) must be documented where the hook and fields live, so future spec-decode or scheduler changes don’t silently break the dLLM plugin.
- **Ecosystem alignment:** The RFC correctly notes that SGlang, Ollama, and LMDeploy already support or ship dLLM-style inference. Adding dLLM via plugin keeps vLLM relevant for users who want one engine for both AR and dLLM and reduces incentive to switch stacks solely for dLLM.

### 6.5 Summary

| Stakeholder        | Net effect |
|--------------------|------------|
| End users          | Positive if dLLM is desired; no impact if not. Risk of wrong outputs if deployment is misconfigured. |
| Operators          | Extra configuration discipline; need dLLM-aware observability. |
| Plugin authors     | New capability with dependency on an currently “unstable” interface. |
| vLLM maintainers   | Low code cost; documentation and stability commitment for the reused path. |
| Community/ecosystem| Keeps vLLM competitive on dLLM; aligns with plugin-based extensibility. |

---

## 7. Assumptions vs production deployments (detailed)

| Assumption | Production applicability | Risk if wrong |
|------------|--------------------------|---------------|
| **One scheduler/worker pair per process** | Typical production runs one model (or one “stack”) per process. Mixed AR + dLLM in the same process would require the default scheduler to understand commit-0 or multiple modes; it does not. | Constrains multi-model or mixed AR+dLLM in one process; acceptable for many deployments. |
| **Scheduler interface stable enough for a plugin** | The codebase explicitly warns that the custom scheduler interface is “not public and compatibility may not be maintained” (`vllm/config/scheduler.py` ~175–176). Production plugins need a stable target across vLLM minor/patch releases. | Without a documented contract or stability commitment, production plugins can break on any spec-decode or scheduler change. |
| **User or config passes `--scheduler-cls` and `--worker-cls` for dLLM** | In production, config is usually managed by automation. If the dLLM model is selected by name only (e.g. from a model registry) and scheduler/worker are not set, the default stack runs and state corrupts (e.g. commit-0 not rolled back). | Wrong outputs, crashes, or hard-to-debug behavior. Mitigation: single “dLLM mode” preset or plugin validation that fails fast with an actionable error. |
| **Default worker returns `None` when not spec-decode** | Verified: `take_draft_token_ids()` returns `None` when `not self.num_spec_tokens or not self._draft_token_req_ids` (gpu_model_runner.py ~3932–3936). So the new hook is a no-op for normal AR and non–spec-decode runs. | No production risk for existing AR/spec-decode users. |
| **First-step semantics (who builds first block, initial `spec_token_ids`)** | RFC defers this to the plugin. In production, different plugin implementations could diverge (e.g. who pads with `<MASK>`, who sets `request.spec_token_ids` before first schedule), leading to subtle bugs or incompatibility across dLLM plugins. | Recommend specifying first-step contract in the RFC or a linked spec. |
| **Async scheduling and structured output** | Batch-queue path uses draft token ids for grammar bitmask when `use_spec_decode` is true. For dLLM, the “next block” may not have the same grammar semantics. RFC leaves interaction to follow-up. | Unclear behavior when dLLM is used with async scheduling or structured output; could cause incorrect filtering or deferred-step bugs. |

---

## 8. Recommendations

1. **Document the hidden contract** in the engine/scheduler: when a custom scheduler and worker are used, the same fields may carry dLLM semantics; list those fields and the single engine change so maintainers don’t break the plugin inadvertently.
2. **Specify first-step semantics** in the RFC or a linked spec: who builds the first input block, how `request.spec_token_ids` is initialized for the first schedule, and how padding/masking works (prompt + `<MASK>`).
3. **Clarify async + structured output:** Whether the core ever applies grammar (or other structured-output logic) to the “next block” from the plugin, and how that interacts with dLLM.
4. **Improve deployability:** Prefer a single entry point (e.g. “use dLLM” flag or preset) that sets model + scheduler_cls + worker_cls; add plugin-side validation that the active scheduler/worker are the plugin’s when the plugin’s model is loaded.
5. **Observability:** Define a clear way (e.g. request/engine attribute or config) to mark dLLM runs so logs and metrics can treat them separately from AR and spec-decode.
6. **Stability:** If the project accepts this design, consider documenting the spec-decode hook and field set used by the dLLM plugin as a **stable contract** for custom scheduler/worker plugins (or a dedicated "dLLM extension point") so plugin authors and production deployments have a clear compatibility target.

---

## 9. Conclusion

- **Viability:** The design is **viable**; the minimal engine change is correct and backward compatible.
- **Weak points:** Commit-0 and strict scheduler/worker pairing, unstable scheduler interface, field overloading (debugging/observability), and underspecified first step and async/grammar behavior.
- **Risks:** Production config mistakes (wrong scheduler/worker) can cause incorrect execution; spec-decode evolution can break the plugin; observability and support are harder without an explicit dLLM signal.

The RFC is a reasonable way to bring dLLM into the vLLM ecosystem with minimal core change and good encapsulation, **provided** the plugin is always used as a coherent stack (model + scheduler + worker), the first-step and async/grammar behavior are specified, and the project explicitly accepts and documents the stability contract for the spec-decode path used by the plugin.

---

## 10. References (codebase)

- Engine: `vllm/v1/engine/core.py` — `post_step()` (~410), `step_with_batch_queue()` (~514–526), `use_spec_decode` (~154).
- Scheduler: `vllm/v1/core/sched/scheduler.py` — `update_from_output` (spec-decode rollback ~1320–1334), `_update_after_schedule` (~944–966), `scheduled_spec_decode_tokens` / `spec_token_ids` usage.
- Worker: `vllm/v1/worker/gpu_model_runner.py` — `take_draft_token_ids()` (~3932), `use_spec_decode` from `scheduled_spec_decode_tokens` (~1680).
- Config: `vllm/config/scheduler.py` — `get_scheduler_cls()` and custom-scheduler warning; `vllm/v1/worker/worker_base.py` — worker_cls as qualified string.
