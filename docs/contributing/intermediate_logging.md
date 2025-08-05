# Intermediate Tensor Logging

This document provides guidance on using the intermediate tensor logging feature in vLLM, which allows you to capture and save intermediate tensors during model execution.

## Overview

The intermediate tensor logging feature enables you to:

- Log input and output tensors from a configured set of filters
- Filter modules by name using regex patterns
- Filter module fwd call index (e.g. dump 2nd call of forward pass on same module)
- Filter tensors by device
- Filter whole model fwd step id

## Usage

### Enabling via parameters or config file

**Offline Inference example**

Dump all modules, all devices for step 0 (default behavior)

```bash
python3 ./examples/offline_inference/llm_engine_example.py --model "meta-llama/Llama-3.1-8B-Instruct"  --enforce-eager  --intermediate-log-config '{"enabled": true}'
```

Dump first layers module, all devices for step 0

```bash
python3 ./examples/offline_inference/llm_engine_example.py --model "meta-llama/Llama-3.1-8B-Instruct"  --enforce-eager  --intermediate-log-config '{"enabled": true, "module_call_match": "layers\\.0\\."}'
```

Dump customized layers, devices, steps through a config file

The configuration file should be a JSON file with the following structure:

```json
{
  "output_dir": "/tmp/vllm_intermediates",
  "module_call_match": ["layers\\.0\\.(?!.*rotary_emb).*", "rotary_emb:0", "embed_tokens", "model\\.norm"],
  "log_step_ids": [0, 1],
  "device_names": ["cuda:0"]
}
```

```bash
python3 ./examples/offline_inference/llm_engine_example.py --model "meta-llama/Llama-3.1-8B-Instruct"  --enforce-eager  --intermediate-log-config-path $HOME/intermediate_logging_config.json
```

#### Configuration Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `output_dir` | string | Directory where to save the intermediate tensors | `/tmp/vllm_intermediates` |
| `module_call_match` | array | Regex patterns to filter module names, if limti to ith call only, add `:i` | `null` (log all modules) |
| `log_step_ids` | array | List of step IDs to log | `[0]` |
| `max_tensor_size` | integer | Maximum number of elements in tensors to log | `null` (no limit) |
| `device_names` | array | List of device names to log | `[]` (log all devices) |

### Output Directory Structure

When you enable intermediate logging, the system creates a timestamped directory under your specified `output_dir`. This helps organize multiple logging sessions:

```
/tmp/vllm_intermediates/010fed05-4a36-4c19-ab44-7cd67e3f63ce/
└── step_0
    ├── model.embed_tokens
    │   ├── inputs_0_cuda_0.pt
    │   ├── inputs.json
    │   ├── outputs_cuda_0.pt
    │   └── outputs.json
    ├── model.layers.0.input_layernorm
    │   ├── inputs_0_cuda_0.pt
    │   ├── inputs.json
    │   ├── outputs_cuda_0.pt
    │   └── outputs.json
    └── step_1/
        └── ...
```

Each tensor is saved in a `.pt` file containing the full PyTorch tensors (can be loaded with `torch.load()`)
