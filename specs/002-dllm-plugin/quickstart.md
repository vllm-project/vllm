# Quickstart: dLLM Plugin

**Branch**: `002-dllm-plugin` | **Date**: 2025-03-01

How to run and test the dLLM plugin once implemented. Assumes (1) vLLM has the dLLM execution path (from 001 design), and (2) a plugin package (e.g. `vllm-dllm-plugin`) exists.

---

## 1. Install vLLM and the plugin

```bash
# Install vLLM (with dLLM support)
pip install vllm

# Install the dLLM plugin (from source or PyPI)
pip install vllm-dllm-plugin
# Or from source:
# cd vllm-dllm-plugin && pip install -e .
```

---

## 2. Verify plugin loading

```bash
# Optional: load only the dLLM plugin
export VLLM_PLUGINS=dllm

# Check that the plugin’s model architecture(s) are registered
python -c "
from vllm.plugins import load_general_plugins
load_general_plugins()
from vllm.model_executor.models import ModelRegistry
archs = ModelRegistry.get_supported_archs()
# Expect at least one dLLM arch, e.g. LLaDA2 or a stub name
print('Registered archs (sample):', list(archs)[:20])
"
```

Or run a small verify script (if the plugin ships one, similar to [bart-plugin’s verify_plugin.py](https://github.com/vllm-project/bart-plugin/blob/master/verify_plugin.py)):

```bash
python verify_plugin.py
```

---

## 3. Run inference with a dLLM model

Use a model name that resolves to an architecture registered by the plugin (e.g. a HuggingFace model ID whose config lists that architecture, or a local path with the same).

**Offline (Python)**:

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="<model_name_or_path>",  # e.g. LLaDA2 or stub
    max_model_len=2048,
    gpu_memory_utilization=0.9,
)
params = SamplingParams(temperature=0.0, max_tokens=256)
outputs = llm.generate(["Hello, world"], params)
for o in outputs:
    print(o.outputs[0].text)
```

**Server**:

```bash
python -m vllm.entrypoints.openai.api_server \
  --model <model_name_or_path> \
  --served-model-name dllm-model
```

Then send completion requests to the served model. Ensure output length equals prompt length + sum of committed tokens per step (block-step semantics).

---

## 4. Run tests

**vLLM repo** (core dLLM path and plugin discovery):
- Existing plugin tests: `tests/plugins_tests/`
- Add or extend tests that load the dLLM plugin (or a stub plugin) and run one request with a dLLM model, asserting step semantics and output length.

**Plugin repo** (if the plugin has its own tests):
```bash
cd vllm-dllm-plugin
pip install -e ".[dev]"
pytest tests/
```

---

## 5. Environment variables (reference)

| Variable | Effect |
|----------|--------|
| `VLLM_PLUGINS=all` | Load all general plugins (default). |
| `VLLM_PLUGINS=dllm` | Load only the dLLM plugin. |
| `VLLM_PLUGINS=none` | Disable general plugins. |

See [vLLM Plugin System](https://docs.vllm.ai/en/latest/design/plugin_system/) and [bart-plugin README](https://github.com/vllm-project/bart-plugin) for more.
