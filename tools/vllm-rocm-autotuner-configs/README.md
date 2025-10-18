# vLLM ROCm Auto-Tuner Configurations

Optimized environment variable settings for vLLM on ROCm-enabled GPUs.

## Installation

```bash
# In the repo root directory
pip install -e .                   # Base install
pip install -e ".[gpu-detect]"     # With GPU detection
pip install -e ".[dev]"            # With dev tools
```

## Quick Start

```bash
# Auto-detect and apply optimal settings
vllm serve /data/models/your-model/

# Use specific recipe
vllm serve /data/models/your-model/ --rocm-tuner-recipe optimal

# With HuggingFace model ID (no duplicate model specification needed)
vllm serve --model amd/Llama-3.3-70B-Instruct-FP8-KV --rocm-tuner-is-hf

# Explicit config file
vllm serve --model /data/models/your-model/ --rocm-tuner-config /path/to/rocm_config_gfx950.json

# Disable tuner
vllm serve --model your-model --rocm-tuner-disable
```

## Supported GPUs

| Architecture | GPU Series     | Status |
|--------------|----------------|--------|
| gfx942       | MI300X, MI300A | ✅ Supported |
| gfx950       | MI350X, MI355X | ✅ Supported |

## How It Works

Models are matched by **signature** - a unique identifier from their config.json:

```log
GptOssForCausalLM_36L_2880H_64A
└─ Architecture   │   │      └─ Attention heads
                  │   └─ Hidden size  
                  └─ Layers
```

If no exact match is found, the tuner gracefully exits without affecting vLLM operation.

## Platform Safety

- **NVIDIA GPUs**: Tuner disabled automatically, no changes made
- **No AMD GPU detected**: Tuner exits gracefully with clear message
- **Unsupported AMD GPU**: Tuner exits gracefully, lists supported architectures
- **Missing config**: Tuner exits gracefully, vLLM uses defaults
- **Model not in config**: Tuner exits gracefully, vLLM uses defaults

## Adding New Models

### For Local Models

```bash
# Generate signature
python tools/generate_signature.py /data/models/your-model/

# Output shows:
# Signature: GptOssForCausalLM_36L_2880H_64A
# Add to config: ...
```

Manually add to `configs/rocm_config_gfx950.json`:

```json
{
  "model_configs": {
    "openai/gpt-oss-120b": {
      "signature": "GptOssForCausalLM_36L_2880H_64A",
      "recipes": [
        {
          "name": "optimal",
          "rank": 1,
          "description": "Optimized for throughput",
          "env_vars": {
            "VLLM_ROCM_USE_AITER": "1"
          },
          "cli_args": {
            "block-size": 64,
            "gpu-memory-utilization": 0.92
          }
        }
      ]
    }
  }
}
```

### For HuggingFace Models

```bash
# Auto-fetch configs and add signatures
pip install huggingface-hub  # Required

# Dry run first
python tools/add_signatures.py --dry-run

# Add signatures to all configs
python tools/add_signatures.py

# Add to specific config
python tools/add_signatures.py --config configs/rocm_config_gfx950.json
```

This automatically fetches config.json from HuggingFace and generates signatures for all models in your config files that have HuggingFace IDs (format: `org/model-name`).

## vLLM Integration

The tuner integrates automatically when vLLM starts:

1. Detects your GPU architecture
2. Loads the appropriate config file from package
3. Reads your model's config.json
4. Matches by signature or model ID
5. Applies environment variables
6. Provides CLI argument defaults (user args take precedence)

## CLI Arguments

All CLI arguments can be found under `vllm/vllm/entrypoints/cli/autotune.py`, and are:

```bash
--rocm-tuner-config PATH        # Explicit config file path (overrides auto-detection)
--rocm-tuner-recipe NAME        # Recipe name to use (default: rank 1 "optimal")
--rocm-tuner-is-hf              # Model is HuggingFace ID (uses --model arg for matching)
--rocm-tuner-use-defaults       # Apply default_config settings from config file
--rocm-tuner-model-id           # Explicit model ID for config matching (overrides signature detection)
--rocm-tuner-disable            # Disable tuner completely (skip all optimizations)
```

### Using `--rocm-tuner-is-hf`

When your model argument is already a HuggingFace model ID, use this flag to avoid duplication:

```bash
# Before (redundant):
vllm serve --model amd/Llama-3.3-70B-Instruct-FP8-KV --rocm-tuner-model-id amd/Llama-3.3-70B-Instruct-FP8-KV

# After (clean):
vllm serve --model amd/Llama-3.3-70B-Instruct-FP8-KV --rocm-tuner-is-hf
```

## Config File Discovery

Configs are searched in this order:

1. **Package configs** (recommended) - Bundled with installation
2. **Explicit path** via `--rocm-tuner-config`

Previous versions searched additional locations (`./`, `~/.config/`, `/etc/`), but these have been removed for simplicity. Use `--rocm-tuner-config` for custom configs.

## Troubleshooting

```bash
# Check GPU architecture
vllm-rocm-detect

# Check GPU architecture (JSON output)
vllm-rocm-detect --json

# Check available configs
python -c "import vllm_rocm_autotuner_configs; \
print(vllm_rocm_autotuner_configs.list_available_configs())"

# Generate signature for your model
python tools/generate_signature.py /path/to/model

# Verify signature in config
grep -A2 "signature" src/vllm_rocm_autotuner_configs/configs/rocm_config_gfx950.json
```

### Development tools

In `tools/` (not installed with package):

- **`generate_signature.py`** - Generate signature for a model

  ```bash
  python tools/generate_signature.py /path/to/model
  ```

- **`add_signatures.py`** - Batch add signatures for HF models

  ```bash
  python tools/add_signatures.py --dry-run  # Preview
  python tools/add_signatures.py            # Apply
  ```
