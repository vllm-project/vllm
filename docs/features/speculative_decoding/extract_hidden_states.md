# Hidden State Extraction

The Hidden State Extraction feature allows vLLM to save intermediate layer activations from a target model during inference. This is useful for training [EAGLE](eagle.md)-style draft models, knowledge distillation, or offline analysis of model internals.

!!! note
    It is possible to save the last-layer's output hidden states by passing `num_hidden_layers` as a layer id. Note that these are _not_ normalized using the output norm.

## Offline Example

```python
import tempfile

from vllm import LLM, SamplingParams
from vllm.config.kv_transfer import KVTransferConfig
from vllm.distributed.kv_transfer.kv_connector.v1 import (
    example_hidden_states_connector,
)

with tempfile.TemporaryDirectory() as tmpdir:
    llm = LLM(
        model="Qwen/Qwen3-8B",
        speculative_config={
            "method": "extract_hidden_states",
            "num_speculative_tokens": 1,
            "draft_model_config": {
                "hf_config": {
                    "eagle_aux_hidden_state_layer_ids": [1, 2, 3, 4],
                },
            },
        },
        kv_transfer_config=KVTransferConfig(
            kv_connector="ExampleHiddenStatesConnector",
            kv_role="kv_producer",
            kv_connector_extra_config={
                "shared_storage_path": tmpdir,
            },
        ),
    )

    outputs = llm.generate(
        ["The future of AI is"],
        SamplingParams(max_tokens=1),
    )

    for output in outputs:
        path = output.kv_transfer_params["hidden_states_path"]
        obj = example_hidden_states_connector.load_hidden_states(path)
        print(f"token_ids: {obj['token_ids'].shape}")
        print(f"hidden_states: {obj['hidden_states'].shape}")
```

A complete example is available at [`examples/features/speculative_decoding/extract_hidden_states_offline.py`](../../../examples/features/speculative_decoding/extract_hidden_states_offline.py).

## Online Example

For improved performance, it is recommended to use a RAM-mounted file system such as `/dev/shm/` for online usage in which the client cleans up the files soon after they are generated.

```bash
vllm serve Qwen/Qwen3-8B \
    --speculative_config '{"method": "extract_hidden_states", "num_speculative_tokens": 1, "draft_model_config": {"hf_config": {"eagle_aux_hidden_state_layer_ids": [1, 2, 3, 4]}}}' \
    --kv_transfer_config '{"kv_connector": "ExampleHiddenStatesConnector", "kv_role": "kv_producer", "kv_connector_extra_config": {"shared_storage_path": "/dev/shm/hidden_states"}}'
```

## Per-Request Options

Both offline and online modes support per-request options via `kv_transfer_params`:

| Parameter | Default | Description |
| --- | --- | --- |
| `hidden_states_path` | Auto-generated | Custom file path for saving hidden states. If not set, files are saved to `<shared_storage_path>/<request_id>.safetensors`. Requires `allow_custom_save_path` to be enabled in the server config. |
| `include_output_tokens` | `False` | When `True`, save hidden states for both prompt and generated output tokens. When `False`, only prompt token hidden states are saved. |

### Offline usage

Pass per-request options via `extra_args` on `SamplingParams`:

```python
SamplingParams(
    max_tokens=32,
    extra_args={
        "kv_transfer_params": {
            "hidden_states_path": "/tmp/my_output.safetensors",
            "include_output_tokens": True,
        }
    },
)
```

### Online usage

Pass `kv_transfer_params` as a top-level field in the API request:

```json
{
    "model": "Qwen/Qwen3-8B",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 32,
    "kv_transfer_params": {
        "hidden_states_path": "/tmp/my_output.safetensors",
        "include_output_tokens": true
    }
}
```

## Configuration

The `kv_connector_extra_config` dict accepts these server-level options:

| Parameter | Default | Description |
| --- | --- | --- |
| `shared_storage_path` | `/tmp` | Directory where hidden state files are saved (used when `hidden_states_path` is not set per-request) |
| `allow_custom_save_path` | `False` | Allow API clients to specify custom file paths via `hidden_states_path`. When disabled, client-provided paths are ignored with a warning. Enable only with trusted clients — custom paths can write to arbitrary locations on the server. |
| `num_writer_threads` | `8` | Thread pool size for async disk writes |
| `use_synchronization_lock` | `True` | Use file locks so concurrent readers block until writes complete. Can be disabled for batch generation where synchronization is not needed. |
| `hs_store_device` | `device` | Where hidden states are held while a request is running. `device` (default) stores them in the device KV cache pool and gathers them at request finish. `cpu` streams them every step into per-request CPU pages on tensor-parallel rank 0, keeping the extract-hidden-states layer out of the device KV pool (useful when that pool is memory constrained, e.g. on hybrid/MLA models); the model runner then skips the layer's device KV cache spec, so no pool is allocated for it. Supported on GPU and on NPU via vllm-ascend. |
| `hs_gather_chunk_tokens` | `4096` | With `hs_store_device=cpu`, number of tokens copied per chunk when streaming hidden states device-to-host. `0` disables chunking. |

!!! note "CPU-pages prefix backfill (engine prefix cache aligned)"
    With `hs_store_device=cpu` and the engine prefix cache enabled, tensor-parallel rank 0 keeps a CPU block table with one slot per engine BlockPool block, addressed by the same block ids as the device KV. Host RAM cost is `num_gpu_blocks × block_size × num_extracted_layers × hidden_size × 2` bytes (bf16) — the same total as the engine KV cache it mirrors, and no extra knob: capacity and eviction follow the engine pool. A request whose prefix hits the engine prefix cache backfills the skipped rows from its own block ids; provenance is guaranteed by the engine's own matching, so multimodal and prompt-embeds requests are covered too. Every streamed step also mirrors its rows into the table, and a block id reused by a new request overwrites its slot, so stale rows evict exactly when the engine evicts them. Rows whose block was never scattered (the table was enabled mid-run, or a streaming step was dropped) stay NaN-filled in the saved file and are flagged by the writer's audit log — holes are auditable and never silently filled with stale data. No table is allocated when the engine prefix cache is off. Disable prefix caching if exact hidden states are required.

## Output Format

Each request produces a `.safetensors` file containing:

- **`hidden_states`** — shape `[num_tokens, num_extracted_layers, hidden_size]`
- **`token_ids`** — shape `[num_tokens]`

The file path is returned in `output.kv_transfer_params["hidden_states_path"]`. Use `load_hidden_states()` from the connector module to read the file with proper synchronization.

!!! note
    Chunked prefill is supported in both storage modes. With `hs_store_device=device`, hidden states are gathered from the finished request's KV cache blocks, which accumulate across chunks. With `hs_store_device=cpu`, each step's rows are appended to the request's CPU page at that step's `num_computed_tokens` offset, so a prompt split across multiple prefill steps is reassembled in order. An earlier version of this feature required disabling chunked prefill; that restriction no longer applies.
