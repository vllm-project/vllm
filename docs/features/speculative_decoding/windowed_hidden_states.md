# Windowed hidden-state capture

The AuxOutput Connector can return hidden rows for a request-local response window.
It captures selected rows during generation and skips capture copies outside the
window. It does not allocate a full-response GPU hidden-state cache.

```python
import os
from dataclasses import asdict

os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "1"

from vllm import LLM, SamplingParams
from vllm.distributed.aux_output_connector.hidden_states import HiddenStateCapturePlan

llm = LLM(
    model="Qwen/Qwen3-1.7B",
    async_scheduling=False,
    aux_output_config={"enable_hidden_states": True},
)
prompt_ids = llm.get_tokenizer().encode("Explain how GPUs work.")
plan = HiddenStateCapturePlan.from_window(
    request_id="",  # Bound to the engine's request ID by the connector.
    prompt_len=len(prompt_ids),
    start=1000,
    end=2000,
    collection_id="training-step-1",
    min_rows=1000,
)
params = SamplingParams(
    max_tokens=2048,
    extra_args={"hidden_state_capture": asdict(plan)},
)
output = llm.generate({"prompt_token_ids": prompt_ids}, params)[0]
if output.hidden_state_capture is not None:
    result = output.hidden_state_capture
    print(result.hidden_positions, result.hidden_states.shape)
else:
    print(output.hidden_capture_skip_reason)
```

Windows use zero-based, half-open response coordinates. `[1000, 2000)` contains
1000 rows. Absolute hidden position is `prompt_len - 1 + response_index`; row `p`
predicts token `p + 1`. Results contain only rows associated with delivered tokens,
including when speculative proposals are rejected or the request stops early.
If generation ends before the window, the skip reason is `ended_before_window`.
If fewer than `min_rows` are available, the reason is `insufficient_rows`.

`hidden_states` is a CPU tensor. `hidden_positions`, layout, auxiliary layer IDs,
request ID, collection ID, copy bytes and copy submission time accompany the result.
The connector settles and releases its capture state when the window completes;
the frontend retains the result until delivery. Cumulative/final-only output emits
the result at completion, while delta output can deliver it when the window settles.

Final-layer capture reuses normal model outputs. Auxiliary layouts reuse the
layers already enabled for the configured speculative method: `aux_final` appends
normalized final hidden states, and `dflash_aux` contains only auxiliary outputs.
Requested layer IDs must match the runner's selected layers in order. The connector
reports `aux_layers_unavailable` for an incompatible layout. Capturing does not
disable auxiliary outputs needed by the speculative decoder itself.

`logprob_topk` optionally includes window-aligned top-k raw logprobs and token IDs.
These are computed before sampling penalties, temperature or grammar masks.
After preemption, recomputed positions already delivered to the caller are excluded
before hidden rows and logits are selected. Rejected speculative positions remain
eligible for capture when they are computed again.

Supported execution uses CUDA Model Runner V2, synchronous scheduling, a causal
decoder, one completion per request, and TP with PP=CP=1. Stop strings, streaming
input, adaptive verification and batch-sharded sampling are unsupported. Hidden
capture works for dense models without enabling routed experts or prefix caching.
Routed-experts output keeps its own existing compatibility requirements and block
storage; hidden capture does not use that storage.

For a short correctness and performance comparison, run
`benchmarks/benchmark_windowed_hidden_states.py --model /path/to/model --result result.json`.
It compares full/window hidden values, generated tokens and raw logprobs, then
measures capture off, full, window and an unreachable window. Use `--native-baseline`
for an additional run with the connector disabled, and `--draft-model` to exercise
EAGLE3. Keep the GPU exclusive during timing and compare identical configurations.
For strict cross-run equality, set `VLLM_BATCH_INVARIANT=1` and record that setting
with the results, since it can select different execution kernels.
Rerun these measurements after changes to the connector or output delivery path;
measurements from another implementation do not validate the current data path.
