# Eagle-1 Speculators Examples

This directory contains example scripts for serving Eagle-1 models in speculators format using vLLM.

## Important Note on VLLM_DISABLE_COMPILE_CACHE

The example scripts use the `VLLM_DISABLE_COMPILE_CACHE=1` environment variable. This is necessary because:

1. **CUDA Graph Caching**: vLLM caches CUDA graphs by default and reuses them for subsequent model runs to improve performance.

2. **Model Definition Conflicts**: When two models with the same class definition but different architectures (e.g., Eagle vs Eagle with layernorms) are run consecutively, the cached CUDA graph from the first model is incorrectly reused for the second model.

3. **Torch.compile Errors**: This cache reuse causes torch.compile errors because the cached graph expects a different number of parameters than what the second model actually has.

4. **Clean Model Loading**: Disabling the compile cache ensures each model loads with a fresh CUDA graph compilation, avoiding conflicts between different Eagle variants.

## Example Commands

### Recommended Approach (with compile cache disabled):

```bash
VLLM_USE_V1=1 VLLM_DISABLE_COMPILE_CACHE=1 vllm serve nm-testing/eagle-llama3.1-8b-instruct-converted-0717 --port 5000
```

### Alternative Approach (not recommended):
You can also use `--enforce-eager` to completely disable torch.compile:

```bash
VLLM_USE_V1=1 vllm serve nm-testing/eagle-llama3.1-8b-instruct-converted-0717 --enforce-eager --port 5000
```

**Warning**: Using `--enforce-eager` disables torch.compile entirely, which may lead to degraded performance. The `VLLM_DISABLE_COMPILE_CACHE=1` approach is preferred as it still allows torch.compile optimizations while avoiding cache-related issues.

## Available Scripts

- `serve_llama.sh`: Serves the standard Eagle-1 Llama model
- `serve_llama_f.sh`: Serves the Eagle-1 Llama model with float16 weights (FC variant)

Both scripts automatically redirect output to `output.txt` for debugging purposes.
