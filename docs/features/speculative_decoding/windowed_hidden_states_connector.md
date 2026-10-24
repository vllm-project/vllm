# Window capture with ExampleHiddenStatesConnector (experimental)

This implementation filters cache writes and exports response windows through
the existing file connector. It does **not** provide window-only GPU block
allocation, real speculative-drafter coexistence, or early window publication.
End-to-end CUDA validation is required before deployment.

```python
from vllm import LLM, SamplingParams
from vllm.distributed.kv_transfer.kv_connector.v1.example_hidden_states_connector import (
    load_hidden_states, cleanup_hidden_states,
)

llm = LLM(
    model="Qwen/Qwen3-0.6B",
    enable_prefix_caching=False,
    async_scheduling=False,
    enable_chunked_prefill=True,
    speculative_config={
        "method": "extract_hidden_states",
        "num_speculative_tokens": 1,
        "draft_model_config": {"hf_config": {
            "eagle_aux_hidden_state_layer_ids": [1, 2, 3],
            "hidden_states_layout": "auxiliary_plus_final",
        }},
    },
    kv_transfer_config={
        "kv_connector": "ExampleHiddenStatesConnector",
        "kv_role": "kv_producer",
        "kv_connector_extra_config": {
            "shared_storage_path": "/dev/shm/features",
            "enable_window_capture": True,
            "max_inflight_export_bytes": 64 * 1024 * 1024,
        },
    },
)
output = llm.generate("Explain matrix multiplication", SamplingParams(
    max_tokens=192,
    extra_args={"kv_transfer_params": {
        "hidden_state_window": {"start": 128, "end": 160, "min_rows": 1},
    }},
))[0]
params = output.kv_transfer_params
if "hidden_states_path" in params:
    data = load_hidden_states(params["hidden_states_path"])
    # hidden_positions is authoritative, including partial windows.
    print(data["hidden_positions"], data["hidden_states"].shape)
    cleanup_hidden_states(params["hidden_states_path"])
else:
    print(params["hidden_state_capture_skip_reason"])
```

Use `VLLM_USE_V2_MODEL_RUNNER=0`. The extractor requires a CUDA decoder-only
model implementing the existing auxiliary-output interface, PP=CP=1,
prefix caching disabled and synchronous scheduling. Stop strings and resumable
input are rejected; EOS, stop-token IDs and max-token termination use accepted
output length. Real EAGLE3/DFlash/DSpark drafters cannot share the single
speculative-method slot with this extractor. TP follows the existing rank-zero
export contract and still requires deployment validation.

The request option is response-relative and half-open. Hidden row for response
token `r` has absolute position `prompt_len - 1 + r`. No final generated-token
replay is required, because its predictor row is already computed. A partial
window is exported if it meets `min_rows`; otherwise the request returns
`hidden_state_capture_skip_reason="insufficient_rows"` and no path.

Layer selection is engine-wide, not request-specific. IDs follow the target's
existing auxiliary-output convention; for Llama/Qwen they denote layer-input
residuals, with ID `num_hidden_layers` denoting the final **pre-norm** residual.
Window mode requires auxiliary IDs in ascending model order. `hidden_states_layout`
can be `auxiliary`, `final`, or `auxiliary_plus_final`. The latter two use the
target's final output tensor (normalized in Llama/Qwen), not the last auxiliary
residual. Final is appended after auxiliaries and has layer ID `-1` in the file.
The model's output contract must be verified for other architectures.

Window files contain:

| Key | Meaning |
| --- | --- |
| `hidden_states` | `[rows, layers, hidden_size]`, original activation dtype |
| `hidden_positions` | Absolute predictor positions |
| `response_positions` | Corresponding accepted response offsets |
| `token_ids` | Input tokens at `hidden_positions`, preserving file semantics |
| `target_token_ids` | Accepted response tokens predicted by those rows |
| `layer_ids` | Aux IDs in order, followed by `-1` for final if included |
| `final_is_normalized` | Final layout uses the target final output contract |
| `hidden_window` | Requested response-relative `[start, end]` |

Safetensors header metadata carries request ID, layout, schema version, and
coordinate. Existing non-window files keep their two-tensor format.

Scheduling metadata updates offset bounds for each allocated physical block.
The masked store uses stable GPU range buffers, so replay reads current bounds
instead of baking request windows into a CUDA graph. Non-window rows are not
written into live hidden-state slots, but their blocks remain allocated by
`FullAttentionManager`. Prefix caching is disabled to prevent reuse of partially
populated hidden-state blocks across requests.

At request completion the connector selects accepted window slots on CPU, gathers
only those rows on GPU, and copies them to CPU. A CUDA event fences block release;
`finished_sending` is emitted only when it completes. The writer waits for that
event, writes a temporary file, atomically replaces the final file, and releases
the companion flock. Consumers must use the lock-aware loader. In window mode,
aborted, errored and insufficient-row requests close/remove unused locks and are
not exported; non-window requests keep the legacy export path. Write failures
write an `.error` file before releasing the lock, and the load raises rather than
exposing partial tensors. Shutdown drains writers and closes remaining
descriptors.

Requests with a malformed or unsupported `hidden_state_window` (for example when
`enable_window_capture` is off, or `end` exceeds `max_window_rows`) are not
captured and return `hidden_state_capture_skip_reason="invalid_window"`.
`max_window_rows` defaults to `max_model_len` and caps the window length.

`max_inflight_export_bytes` bounds live **window-export** pinned allocations per
exporting worker, including writer backlog. A window larger than the budget uses
pageable final storage and a reusable pinned staging buffer, synchronizing each
chunk before reusing it. This deliberately adds completion-time host waits, and
when the budget is exhausted the engine loop waits for earlier writes to finish
(a failed earlier write does not fail the waiting request). Size the budget and
`max_window_rows` for your storage latency.
It does not cap PyTorch's cached pinned allocator reserve, other vLLM pinned
buffers, or legacy full-export requests mixed into the same engine.

For `R` exported rows, `L` layers, hidden width `H`, and element size `D`, D2H
payload is `R*L*H*D`. Window export retains at most the configured budget in
live pinned buffers, with GPU gathers bounded by the same byte budget. Large
windows still require `R*L*H*D` pageable final storage. The existing proposer
retains its batch-sized interleaving buffer; copying layers into it avoids the
additional full-batch `torch.stack` allocation. Target-model auxiliary activation
generation is unchanged. Counters are available from the worker connector's
`get_window_capture_metrics()`; copy-wait time is a host wait, not GPU duration.
