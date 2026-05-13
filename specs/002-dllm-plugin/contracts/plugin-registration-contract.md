# Contract: dLLM Plugin Registration

**Branch**: `002-dllm-plugin` | **Date**: 2025-03-01

This contract defines what a dLLM plugin package must do to register with vLLM and how vLLM discovers and uses it. It is the **external** contract for plugin authors (in addition to the engine–worker dLLM step contract, which is internal to vLLM).

---

## 1. Entry point

| Requirement | Detail |
|-------------|--------|
| Group | `vllm.general_plugins` (same as other model plugins). |
| Name | Plugin author chooses (e.g. `dllm`). Used with `VLLM_PLUGINS` to filter. |
| Value | Fully qualified callable, e.g. `vllm_dllm_plugin:register_dllm_model`. Must be a no-argument function (or callable that vLLM can invoke with no args). |

**Example** (pyproject.toml):

```toml
[project.entry-points."vllm.general_plugins"]
dllm = "vllm_dllm_plugin:register_dllm_model"
```

---

## 2. Registration function behavior

| Requirement | Detail |
|-------------|--------|
| Idempotent / re-entrant | May be called multiple times (e.g. in different processes). Must not assume single process. |
| Action | For each dLLM architecture the plugin provides, call `ModelRegistry.register_model(arch_name, model_class_qualname)`. |
| arch_name | String that the model loader will resolve (e.g. matches HuggingFace config `architectures` or a known alias). |
| model_class_qualname | String of the form `"package.module:ClassName"` so vLLM can import and instantiate the model class. |
| Optional | Check `arch_name not in ModelRegistry.get_supported_archs()` before registering to avoid duplicate registration in the same process. |
| Errors | On failure (e.g. import error), the function may log and re-raise so vLLM or the user sees the error. |

---

## 3. Model class contract (summary)

| Requirement | Detail |
|-------------|--------|
| Implement | vLLM’s model interfaces so the model loader can load weights and the worker can run forward. |
| dLLM behavior | Model is used in a way that produces, per step, committed tokens (0..LOOKAHEAD_SIZE) and next-block input (length LOOKAHEAD_SIZE). The **worker** in vLLM core builds `DllmStepOutput` from the model’s forward output; the model does not return DllmStepOutput. |
| Config | Model (or its config) should expose block size and, if applicable, whether prefill is bidirectional (so core can disable prefix caching). |

The internal engine–worker contract for dLLM steps is defined in [001-dllm-integration/contracts/dllm-step-contract.md](../../001-dllm-integration/contracts/dllm-step-contract.md). Plugin authors do not implement that contract directly; they implement the model, and vLLM’s worker implements the step output and validation.

---

## 4. Discovery and loading

| Step | Who | What |
|------|-----|------|
| 1 | vLLM | Calls `load_plugins_by_group("vllm.general_plugins")` and invokes each returned callable (e.g. at engine init or before model resolution). |
| 2 | Plugin | `register_dllm_model()` runs and registers one or more arch names with ModelRegistry. |
| 3 | User | Starts vLLM with `model=<name>` (e.g. a HuggingFace model ID or path whose config has an architecture registered by the plugin). |
| 4 | vLLM | Resolves architecture, loads the plugin’s model class, detects dLLM, and uses the dLLM step path for decode. |

---

## 5. Version and compatibility

| Requirement | Detail |
|-------------|--------|
| Plugin declares | Minimum vLLM version in `install_requires` (e.g. `vllm>=0.6.0`). |
| Compatibility | Plugin author’s responsibility to test against target vLLM version(s). Optional: plugin can log a warning at registration if vLLM version is below a tested version. |
