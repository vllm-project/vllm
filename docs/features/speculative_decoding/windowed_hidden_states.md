# Windowed hidden-state capture

The AuxOutput Connector can return hidden rows for a request-local response window.
It captures selected rows during generation and skips capture copies outside the
window. It does not allocate a full-response GPU hidden-state cache.

```python
import os

os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "1"

from vllm import LLM, SamplingParams

llm = LLM(
    model="Qwen/Qwen3-1.7B",
    async_scheduling=False,
    max_num_seqs=8,
    aux_output_config={"enable_hidden_states": True},
)
prompt_ids = llm.get_tokenizer().encode("Explain how GPUs work.")
params = SamplingParams(
    max_tokens=2048,
    extra_args={"hidden_state_capture": {
        "start": 1000,
        "end": 2000,
        "collection_id": "training-step-1",
        "min_rows": 1000,
    }},
)
output = llm.generate({"prompt_token_ids": prompt_ids}, params)[0]
if output.hidden_state_capture is not None:
    result = output.hidden_state_capture
    print(result.hidden_positions, result.hidden_states.shape)
else:
    print(output.hidden_capture_skip_reason)
```

Windows use zero-based, half-open response coordinates. `[1000, 2000)` contains
1000 rows. The engine binds these offsets after tokenization, so text prompts do
not require the caller to predict the token count. Legacy `asdict(plan)` inputs
retain response offsets when their client prompt length differs from the engine.
Absolute hidden position is `prompt_len - 1 + response_index`; row `p`
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
The requested top-k must not exceed `max_logprobs` (unless it is -1) or the model
vocabulary. Window length must not exceed `max_model_len`. Admission bounds
retained payload bytes multiplied by `max_num_seqs` using
`aux_output_config.max_hidden_capture_bytes` (256 MiB by default). This budget
includes hidden values and top-k values/IDs; model logits and allocator caches
are separate. Increase it explicitly for wider layouts or longer windows.
Unavailable logit rows produce a request-local skip reason rather than a CUDA
device assertion.
After preemption, recomputed positions already delivered to the caller are excluded
before hidden rows and logits are selected. Rejected speculative positions remain
eligible for capture when they are computed again.

Supported execution uses CUDA Model Runner V2, synchronous scheduling, a causal
decoder, one completion per request, and TP with PP=CP=1. Stop strings, streaming
input, adaptive verification and batch-sharded sampling are unsupported. Hidden
capture works for dense models without enabling routed experts or prefix caching.
The validated model configurations are final hidden output on TP1, DSpark
`aux_final` on TP1/TP2, and EAGLE3 `aux_final` on TP1.
`dflash_aux` and simultaneous routed-experts plus hidden output are
experimental and are not part of the validated model matrix.
Routed-experts output keeps its own existing compatibility requirements and block
storage; hidden capture does not use that storage.

This API returns Python tensors through `LLM`/`AsyncLLM`. The OpenAI-compatible
HTTP schema does not expose the nested capture request or its result. Existing
`extract_hidden_states` instead exports speculative training features through
its own connector; response-window capture shares auxiliary model outputs but
does not replace that export workflow.

For correctness validation, compare full/window hidden values, generated tokens
and raw logprobs. For performance validation, measure capture off, full, window
and an unreachable window. Keep the GPU exclusive during timing and compare
identical configurations.
For strict cross-run equality, set `VLLM_BATCH_INVARIANT=1` and record that setting
with the results, since it can select different execution kernels.
