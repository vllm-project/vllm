# Research: dLLM Plugin

**Branch**: `002-dllm-plugin` | **Date**: 2025-03-01

Decisions and rationale for the dLLM plugin implementation, including patterns taken from the [vLLM BART plugin](https://github.com/vllm-project/bart-plugin).

---

## 1. Plugin registration pattern (bart-plugin)

**Decision**: Use the same entry-point and registration pattern as the BART plugin.

**Rationale**:
- BART plugin uses `vllm.general_plugins` with a single callable (e.g. `bart=vllm_bart_plugin:register_bart_model`). The callable is invoked when vLLM loads plugins (`load_general_plugins()`), which runs before model resolution.
- Registration is done via `ModelRegistry.register_model(arch_name, model_class_qualname)`. The arch name (e.g. `BartForConditionalGeneration`) is what appears in HuggingFace config or is resolved when loading; the qualname is the plugin’s model class (e.g. `vllm_bart_plugin.bart:BartForConditionalGeneration`).
- BART registers multiple architectures from one plugin (BART and Florence2), so one plugin → multiple `ModelRegistry.register_model` calls is already a supported pattern.
- Re-entrant: the registration function can be called multiple times (e.g. in multiple processes); BART uses `get_supported_archs()` to avoid double-registering.

**Alternatives considered**:
- New entry point group for “dLLM-only” plugins: rejected to keep a single extension mechanism and avoid fragmenting the plugin API.
- Register only one model per plugin: rejected; spec FR-003 and BART precedent both support multiple architectures per plugin.

**References**: [bart-plugin](https://github.com/vllm-project/bart-plugin) `setup.py`, `pyproject.toml`, `vllm_bart_plugin/__init__.py`; vLLM [Plugin System](https://docs.vllm.ai/en/latest/design/plugin_system/).

---

## 2. Where the dLLM execution path lives (core vs plugin)

**Decision**: The dLLM step contract (scheduler, worker, DllmStepOutput, next_dllm_input_token_ids, scheduled_dllm_input_tokens) lives in **vLLM core** (v1). The plugin only supplies **model implementations** (model class per architecture).

**Rationale**:
- The plugin system today allows registering **models** via `ModelRegistry`; it does not allow registering new scheduler or worker behavior. So the engine must already “know” how to run a step when the model returns committed tokens and next-step input.
- Putting the dLLM step loop in core (once) lets any plugin-registered dLLM model work without each plugin reimplementing scheduler/worker logic. This matches “add new dLLM architectures by publishing a plugin” (SC-003).
- BART does not change the engine’s step semantics; it only adds a model class. Similarly, dLLM plugins add model classes; the engine’s handling of dLLM step output is a one-time addition in core.

**Alternatives considered**:
- Full “dLLM engine” in a plugin (e.g. platform plugin): would require a new plugin type or platform that replaces scheduler/worker behavior. Not supported by current plugin docs and would complicate multi-worker and preemption.
- Hybrid “core calls into plugin for step logic”: possible but duplicates contract and validation in every plugin; core-owned step contract is simpler and safer.

---

## 3. Plugin package layout and naming

**Decision**: Plugin package name `vllm-dllm-plugin` (PyPI) with top-level package `vllm_dllm_plugin`. Entry point name `dllm` (e.g. `dllm=vllm_dllm_plugin:register_dllm_model`). Optional verify script `verify_plugin.py` and `tests/` for registration and smoke inference.

**Rationale**:
- Aligns with `vllm-bart-plugin` / `vllm_bart_plugin` naming. Users can `pip install vllm-dllm-plugin` and use `VLLM_PLUGINS=dllm` to load only this plugin.
- Single registration function that registers all dLLM architectures provided by the plugin (e.g. LLaDA2, LLaDA2.1, stub); BART does the same (BART + Florence2).

**Alternatives considered**:
- One package per architecture (e.g. `vllm-llada-plugin`): would work but increases package sprawl; one plugin with multiple archs is simpler (FR-003, BART precedent).

---

## 4. Model class contract for plugin dLLM models

**Decision**: A plugin-registered dLLM model class must (1) be loadable and runnable by vLLM’s existing model loader and worker; (2) declare that it is a dLLM (e.g. config or base class) so the engine uses the dLLM path; (3) produce per-step output that the worker can convert into `DllmStepOutput` (committed_token_ids, next_step_input_token_ids per request). The exact interface (forward signature, where commit/next-input are computed) is defined in the contract and implemented by the model runner when it detects a dLLM model.

**Rationale**:
- vLLM core owns the step contract (DllmStepOutput, validation, scheduler application). The model class owns the **semantics** (attention mask, block size, commit rule). The worker/model runner bridges the two: it calls the model, then runs model-specific logic to fill DllmStepOutput. That logic can live in the model class (e.g. a method) or in a small adapter in core that knows how to call the plugin model.
- BART model class implements vLLM’s model interfaces (embedding, layers, etc.); it does not implement scheduler logic. Similarly, dLLM model class implements the model forward and exposes block size and step output semantics; the runner fills DllmStepOutput and validates lengths.

**Alternatives considered**:
- Plugin returns DllmStepOutput directly from the model: would require the plugin to depend on vLLM’s output types and worker contract; tighter coupling. Prefer core owning the output type and the runner building it from model logits/state.

---

## 5. Prefix caching and bidirectional prefill

**Decision**: Same as 001-dllm-integration and spec FR-005: prefix cache valid only up to committed length; when the model uses a fully bidirectional prefill mask, prefix caching is disabled (core or model config must declare this so the engine does not reuse prefix for that model).

**Rationale**: Spec and pitch document already define this; plugin model can declare “bidirectional_prefill” in config so core disables prefix caching.

---

## 6. Error when model not provided by any plugin

**Decision**: When the user requests a model name that resolves to an architecture that no loaded plugin has registered, vLLM’s existing “unknown model” or “model not found” path applies. Plugin docs and README should state that installing the dLLM plugin is required for dLLM model names (e.g. LLaDA2). No new error code required; clear message is sufficient (FR-006).

**Rationale**: Model resolution already fails when the architecture is not in ModelRegistry; the only requirement is that the message is clear. Plugin authors can document “install vllm-dllm-plugin” in their model card or README.

---

## 7. Version compatibility (plugin vs vLLM)

**Decision**: Plugin declares a minimum vLLM version in `install_requires` (e.g. `vllm>=X.Y.Z`). Compatibility is the plugin author’s responsibility; optional runtime check (e.g. at registration time) can log a warning if vLLM version is below a tested version. No core change required.

**Rationale**: BART plugin uses `vllm>=0.14.0` (or similar); this is the standard approach. Spec assumption: “Compatibility between a plugin and a vLLM version is the responsibility of the plugin author.”
