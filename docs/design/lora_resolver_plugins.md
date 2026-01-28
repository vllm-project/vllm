# LoRA Resolver Plugins

This directory contains vLLM's LoRA resolver plugins built on the `LoRAResolver` framework.
They automatically discover and load LoRA adapters from a specified local storage path, eliminating the need for manual configuration or server restarts.

## Overview

LoRA Resolver Plugins provide a flexible way to dynamically load LoRA adapters at runtime. When vLLM
receives a request for a LoRA adapter that hasn't been loaded yet, the resolver plugins will attempt
to locate and load the adapter from their configured storage locations. This enables:

- **Dynamic LoRA Loading**: Load adapters on-demand without server restarts
- **Multiple Storage Backends**: Support for filesystem, S3, and custom backends. The built-in `lora_filesystem_resolver` requires a local storage path, while the built-in `hf_hub_resolver` will pull LoRA adapters from Huggingface Hub and proceed in an identical manner. In general, custom resolvers can be implemented to fetch from any source.
- **Automatic Discovery**: Seamless integration with existing LoRA workflows
- **Scalable Deployment**: Centralized adapter management across multiple vLLM instances

## Prerequisites

Before using LoRA Resolver Plugins, ensure the following environment variables are configured:

### Required Environment Variables

1. **`VLLM_ALLOW_RUNTIME_LORA_UPDATING`**: Must be set to `true` or `1` to enable dynamic LoRA loading
   ```bash
   export VLLM_ALLOW_RUNTIME_LORA_UPDATING=true
   ```

2. **`VLLM_PLUGINS`**: Must include the desired resolver plugins (comma-separated list)
   ```bash
   export VLLM_PLUGINS=lora_filesystem_resolver
   ```

3. **`VLLM_LORA_RESOLVER_CACHE_DIR`**: Must be set to a valid directory path for filesystem resolver
   ```bash
   export VLLM_LORA_RESOLVER_CACHE_DIR=/path/to/lora/adapters
   ```

### Optional Environment Variables

- **`VLLM_PLUGINS`**: If not set, all available plugins will be loaded. If set to empty string, no plugins will be loaded.

## Available Resolvers

### lora_filesystem_resolver

The filesystem resolver is installed with vLLM by default and enables loading LoRA adapters from a local directory structure.

#### Setup Steps

1. **Create the LoRA adapter storage directory**:
   ```bash
   mkdir -p /path/to/lora/adapters
   ```

2. **Set environment variables**:
   ```bash
   export VLLM_ALLOW_RUNTIME_LORA_UPDATING=true
   export VLLM_PLUGINS=lora_filesystem_resolver
   export VLLM_LORA_RESOLVER_CACHE_DIR=/path/to/lora/adapters
   ```

3. **Start vLLM server**:
   Your base model can be `meta-llama/Llama-2-7b-hf`. Please make sure you set up the Hugging Face token in your env var `export HF_TOKEN=xxx235`.
   ```bash
   python -m vllm.entrypoints.openai.api_server \
       --model your-base-model \
       --enable-lora
   ```

#### Directory Structure Requirements

The filesystem resolver expects LoRA adapters to be organized in the following structure:

```text
/path/to/lora/adapters/
├── adapter1/
│   ├── adapter_config.json
│   ├── adapter_model.bin
│   └── tokenizer files (if applicable)
├── adapter2/
│   ├── adapter_config.json
│   ├── adapter_model.bin
│   └── tokenizer files (if applicable)
└── ...
```

Each adapter directory must contain:

- **`adapter_config.json`**: Required configuration file with the following structure:
  ```json
  {
    "peft_type": "LORA",
    "base_model_name_or_path": "your-base-model-name",
    "r": 16,
    "lora_alpha": 32,
    "target_modules": ["q_proj", "v_proj"],
    "bias": "none",
    "modules_to_save": null,
    "use_rslora": false,
    "use_dora": false
  }
  ```

- **`adapter_model.bin`**: The LoRA adapter weights file

#### Usage Example

1. **Prepare your LoRA adapter**:
   ```bash
   # Assuming you have a LoRA adapter in /tmp/my_lora_adapter
   cp -r /tmp/my_lora_adapter /path/to/lora/adapters/my_sql_adapter
   ```

2. **Verify the directory structure**:
   ```bash
   ls -la /path/to/lora/adapters/my_sql_adapter/
   # Should show: adapter_config.json, adapter_model.bin, etc.
   ```

3. **Make a request using the adapter**:
   ```bash
   curl http://localhost:8000/v1/completions \
       -H "Content-Type: application/json" \
       -d '{
           "model": "my_sql_adapter",
           "prompt": "Generate a SQL query for:",
           "max_tokens": 50,
           "temperature": 0.1
       }'
   ```

#### How It Works

1. When vLLM receives a request for a LoRA adapter named `my_sql_adapter`
2. The filesystem resolver checks if `/path/to/lora/adapters/my_sql_adapter/` exists
3. If found, it validates the `adapter_config.json` file
4. If the configuration matches the base model and is valid, the adapter is loaded
5. The request is processed normally with the newly loaded adapter
6. The adapter remains available for future requests

## Advanced Configuration

### Multiple Resolvers

You can configure multiple resolver plugins to load adapters from different sources:

'lora_s3_resolver' is an example of a custom resolver you would need to implement

```bash
export VLLM_PLUGINS=lora_filesystem_resolver,lora_s3_resolver
```

All listed resolvers are enabled; at request time, vLLM tries them in order until one succeeds.

### Custom Resolver Implementation

To implement your own resolver plugin:

1. **Create a new resolver class**:
   ```python
   from vllm.lora.resolver import LoRAResolver, LoRAResolverRegistry
   from vllm.lora.request import LoRARequest
   
   class CustomResolver(LoRAResolver):
       async def resolve_lora(self, base_model_name: str, lora_name: str) -> Optional[LoRARequest]:
           # Your custom resolution logic here
           pass
   ```

2. **Register the resolver**:
   ```python
   def register_custom_resolver():
       resolver = CustomResolver()
       LoRAResolverRegistry.register_resolver("Custom Resolver", resolver)
   ```

## Troubleshooting

### Common Issues

1. **"VLLM_LORA_RESOLVER_CACHE_DIR must be set to a valid directory"**
   - Ensure the directory exists and is accessible
   - Check file permissions on the directory

2. **"LoRA adapter not found"**
   - Verify the adapter directory name matches the requested model name
   - Check that `adapter_config.json` exists and is valid JSON
   - Ensure `adapter_model.bin` exists in the directory

3. **"Invalid adapter configuration"**
   - Verify `peft_type` is set to "LORA"
   - Check that `base_model_name_or_path` matches your base model
   - Ensure `target_modules` is properly configured

4. **"LoRA rank exceeds maximum"**
   - Check that `r` value in `adapter_config.json` doesn't exceed `max_lora_rank` setting

### Debugging Tips

1. **Enable debug logging**:
   ```bash
   export VLLM_LOGGING_LEVEL=DEBUG
   ```

2. **Verify environment variables**:
   ```bash
   echo $VLLM_ALLOW_RUNTIME_LORA_UPDATING
   echo $VLLM_PLUGINS
   echo $VLLM_LORA_RESOLVER_CACHE_DIR
   ```

3. **Test adapter configuration**:
   ```bash
   python -c "
   import json
   with open('/path/to/lora/adapters/my_adapter/adapter_config.json') as f:
       config = json.load(f)
   print('Config valid:', config)
   "
   ```
