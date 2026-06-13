# Debug Tensor Dump

The tensor dump feature captures intermediate activations from every leaf
module of the model during inference.  Each forward pass produces a `.pt`
file that you can load with `torch.load()` for offline analysis and
debugging.

## Quick Start

Set the `VLLM_DEBUG_TENSOR_DUMP_OUTPUT_FOLDER` environment variable to
an output directory:

```bash
VLLM_DEBUG_TENSOR_DUMP_OUTPUT_FOLDER=./dump \
    python -m vllm.entrypoints.openai.api_server --model <model>
```

After sending requests, the dump directory will contain per-worker
sub-directories with `.pt` files:

```text
dump/
  TP0_PP0_Rank0_pid12345/
    Pass00000.pt
    Pass00001.pt
    ...
```

Each file is a dictionary mapping fully-qualified module names to their
output tensors (on CPU).

## Environment Variables

| Variable | Type | Default | Description |
| --- | --- | --- | --- |
| `VLLM_DEBUG_TENSOR_DUMP_OUTPUT_FOLDER` | `str` | (unset) | Output directory. Feature is disabled when unset. |
| `VLLM_DEBUG_TENSOR_DUMP_LAYERS` | `str` | (unset) | Comma-separated layer indices to dump (e.g. `"0,1,31"`). All layers when unset. |
| `VLLM_DEBUG_TENSOR_DUMP_SKIP_PASSES` | `int` | `0` | Number of initial forward passes to skip (useful for skipping warmup passes). |
| `VLLM_DEBUG_TENSOR_DUMP_TOP_LEVEL_MODULE_NAME` | `str` | `"model"` | Name of the top-level sub-module inside the model to hook. |
| `VLLM_DEBUG_TENSOR_DUMP_LAYERS_MODULE_NAME` | `str` | `"layers"` | Name of the layers container inside the top-level module. |

## Notes

- When tensor dump is enabled, `torch.compile` and CUDA graph capture
  are automatically disabled.  This has a significant performance impact
  and should only be used for debugging.
- **Not safe for multi-model serving.** Enabling tensor dump removes the
  custom `__call__` from `@support_torch_compile` classes at the Python
  class level.  This is a global, irreversible change within the process
  and will affect all models loaded in the same worker.  Only use this
  feature in single-model debugging scenarios.
- Tensors are moved to CPU immediately inside the forward hook, which
  adds overhead per forward pass.
- The dump directory is organized by TP/PP rank and process ID, so
  multi-GPU setups produce separate files per worker.

## Loading Dumps

```python
import torch

data = torch.load("dump/TP0_PP0_Rank0_pid12345/Pass00000.pt",
                   weights_only=True)
for name, tensor in sorted(data.items()):
    print(f"{name}: shape={tensor.shape}, dtype={tensor.dtype}")
```
