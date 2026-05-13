# Data Model: dLLM Plugin

**Branch**: `002-dllm-plugin` | **Date**: 2025-03-01

Entities and state for the dLLM plugin feature. Core execution state (Request, DllmStepOutput, SchedulerOutput, etc.) matches the design in [001-dllm-integration/data-model.md](../001-dllm-integration/data-model.md); this document adds **plugin-side** entities and references.

---

## 1. Plugin package (installable)

**Location**: Separate repository or package tree (e.g. `vllm-dllm-plugin`).

| Concept | Description |
|--------|--------------|
| Package name | e.g. `vllm-dllm-plugin` (PyPI); top-level Python package `vllm_dllm_plugin`. |
| Entry point | `vllm.general_plugins` → `dllm = vllm_dllm_plugin:register_dllm_model`. |
| Dependencies | `vllm>=X.Y.Z`, `torch`, `transformers` (or as needed for model impl). |
| Re-entrancy | Registration function must be safe to call multiple times (e.g. check `get_supported_archs()` before registering). |

**Validation**: Package installs without error; `entry_points` visible via `importlib.metadata.entry_points(group="vllm.general_plugins")` when installed.

---

## 2. Registration function

**Location**: `vllm_dllm_plugin/__init__.py` (or equivalent).

| Responsibility | Description |
|----------------|-------------|
| Called by | vLLM’s `load_general_plugins()` (and thus before model resolution). |
| Must | Call `ModelRegistry.register_model(arch_name, model_class_qualname)` for each dLLM architecture the plugin provides. |
| May | Use `ModelRegistry.get_supported_archs()` to avoid double registration. Log success/failure. |
| Must not | Assume process singleton; may be invoked in multiple processes. |

**Example** (pattern from [bart-plugin](https://github.com/vllm-project/bart-plugin)):

- `register_dllm_model()` → `ModelRegistry.register_model("LLaDA2", "vllm_dllm_plugin.llada:LLaDA2")` and optionally `register_model("LLaDA2_1", "...")`.

---

## 3. Model class (plugin-provided)

**Location**: Plugin package (e.g. `vllm_dllm_plugin/llada.py`).

| Responsibility | Description |
|----------------|-------------|
| Loadable by | vLLM model loader (same as any ModelRegistry-registered model). |
| Architecture name | Must match the name passed to `register_model`; typically matches HuggingFace config `architectures` or a name the loader resolves to. |
| dLLM semantics | Implements block-step semantics: LOOKAHEAD_SIZE, attention mask, and logic to produce committed tokens and next-block input from logits. The **worker** (in core) calls the model and builds `DllmStepOutput` from model output; the model does not return DllmStepOutput directly. |
| Config | May expose `lookahead_size`, `bidirectional_prefill` (or equivalent) so core can disable prefix caching when needed. |

**Validation**: When the engine loads this model, it detects dLLM (e.g. from config or base class) and uses the dLLM step path; worker validates committed/next_step lengths before return.

---

## 4. Core state (reference)

The following entities live in **vLLM core** (v1) and are unchanged from the 001 design. The plugin does not define these; the engine uses them when a plugin-registered dLLM model is loaded.

| Entity | Source | Description |
|--------|--------|-------------|
| Request.next_dllm_input_token_ids | vllm/v1/request.py | Length LOOKAHEAD_SIZE when set; next block input for the request. |
| DllmStepOutput | vllm/v1/outputs.py | req_ids, committed_token_ids, next_step_input_token_ids. |
| ModelRunnerOutput.dllm_step_output | vllm/v1/outputs.py | Optional; when set, scheduler uses it for append/commit. |
| SchedulerOutput.scheduled_dllm_input_tokens | vllm/v1/core/sched/output.py | req_id → list of length LOOKAHEAD_SIZE. |

Full definitions and state transitions: [001-dllm-integration/data-model.md](../001-dllm-integration/data-model.md).

---

## 5. State transitions (plugin load and run)

1. **Process start**: vLLM calls `load_general_plugins()` → plugin’s `register_dllm_model()` runs → `ModelRegistry` gains one or more dLLM arch names.
2. **User starts vLLM with model X**: Model loader resolves X to an architecture; if that architecture was registered by the plugin, the loader loads the plugin’s model class. Engine detects dLLM and uses dLLM step path.
3. **Per-step**: Same as 001: scheduler sends `scheduled_dllm_input_tokens`, worker returns `dllm_step_output`, scheduler updates Request and KV (see 001 data-model state transitions).
