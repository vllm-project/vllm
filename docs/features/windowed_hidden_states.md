# Windowed hidden-state capture

The AuxOutput Connector can return hidden rows for a request-local position range.
It captures selected rows while the request runs and skips capture copies outside
the range. It does not allocate a full-response GPU hidden-state cache.

Two workloads use the same primitive:

- **Drafter training data**: capture a window of generated tokens (for example,
  response tokens 1000 to 2000) during rollout, optionally with the auxiliary
  layers used by EAGLE3, DSpark or DFlash.
- **Full-vocabulary on-policy distillation**: a teacher server scores a student
  trajectory in one prefill and returns the final hidden rows of the trajectory.
  The trainer rebuilds the teacher's full-vocabulary logits with the teacher's
  output head instead of transferring dense logits.

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
aux = output.aux_outputs
if aux is not None and aux.hidden_state_capture is not None:
    result = aux.hidden_state_capture
    print(result.hidden_positions, result.hidden_states.shape)
elif aux is not None:
    print(aux.hidden_capture_skip_reason)
```

`RequestOutput.aux_outputs` is the single container for settled AuxOutput
Connector results; it is `None` when nothing was settled for that output.

Windows use zero-based, half-open response coordinates. `[1000, 2000)` contains
1000 rows. The engine binds these offsets after tokenization, so text prompts do
not require the caller to predict the token count. Legacy `asdict(plan)` inputs
retain response offsets when their client prompt length differs from the engine.
Absolute hidden position is `prompt_len - 1 + response_index`; row `p`
predicts token `p + 1`. Results contain only rows associated with delivered tokens,
including when speculative proposals are rejected or the request stops early.
If generation ends before the window, the skip reason is `ended_before_window`.
If fewer than `min_rows` are available, the reason is `insufficient_rows`.
Prompt segments must contain every requested row, regardless of `min_rows`;
missing prompt rows return `prompt_rows_unavailable`. Inconsistent tensor fields
across chunks return `tensor_fields_mismatch`. Both failures are request-local.
Auxiliary-layer capture requires a speculator that produces auxiliary hidden
states; requests without one are rejected during input validation.

`result.tensors` holds CPU tensors keyed by name: `hidden_states`, plus
`logprobs` and `logprob_token_ids` when `logprob_topk` is set (also available as
attributes). `hidden_positions`, layout, auxiliary layer IDs, `segments`,
collection ID, copy bytes and `copy_ms` accompany the result.
`copy_ms` measures CPU allocation and asynchronous-copy submission time,
not completed GPU-to-CPU transfer time.
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
are separate. Increase it explicitly for wider layouts or longer windows. For example, final
BF16 hidden of width 4096 with max_num_seqs=256 permits at most 128 rows
under the default budget, even if only one capture request is active. This is
a static admission bound, not dynamic accounting of in-flight allocations.
Unavailable logit rows produce a request-local skip reason rather than a CUDA
device assertion.
After preemption, recomputed positions already delivered to the caller are excluded
before hidden rows and logits are selected. Rejected speculative positions remain
eligible for capture when they are computed again.

## Prompt rows for teacher scoring

Set `"coordinate": "absolute"` to address rows from the start of the prompt.
Rows before `prompt_len - 1` form the `prompt` segment; later rows form the
`output` segment, and a range crossing the boundary returns both (see
`result.segments`). To score a student trajectory `x + y` with a teacher:

```python
scoring = SamplingParams(
    max_tokens=1,
    extra_args={"hidden_state_capture": {
        "coordinate": "absolute",
        "start": len(x) - 1,
        "end": len(x) + len(y) - 1,
        "min_rows": len(y),
        "collection_id": "opd-step-1",
    }},
)
output = llm.generate({"prompt_token_ids": x + y}, scoring)[0]
teacher_hidden = output.aux_outputs.hidden_state_capture.hidden_states
```

Row `p` predicts token `p + 1`, so these rows give the teacher distribution for
every token of `y`. Prompt rows settle as soon as they are computed; rows from
intermediate chunked-prefill steps are held in the connector and delivered with
the step that samples. Prefix-cache hits are limited to the rows before `start`
(and external KV-connector hits are skipped), because cached rows are not
recomputed. `logprob_topk` is rejected for prompt rows because prefill only
computes logits for the last row; use `prompt_logprob_token_ids` for prompt
top-k scores.

Prompt capture rejects `--kv-sharing-fast-prefill`, which can skip computation
needed for complete prompt hidden rows. By default, capture alone may reuse
prefix-cache blocks strictly before `start`; set
`SamplingParams(skip_reading_prefix_cache=True)` to force recomputation.
This differs from prompt-logprob requests, which default to skipping cache reads.
Cached and recomputed execution need not be bitwise identical without
`VLLM_BATCH_INVARIANT=1`.

For layouts that include the final layer, `result.logits_contract` describes how
the sampler turns the final hidden rows into logits:
`vocab_size` (logit columns kept), `logit_scale`, `soft_cap`, and
`tie_word_embeddings` (whether the output head shares the input embedding).
The logits are `soft_cap(h @ W_head.T)[:, :vocab_size] * logit_scale`, with
soft-capping skipped when `soft_cap` is `None`.
The contract is emitted only for the exact, unquantized `Qwen3ForCausalLM` class
with its standard `compute_logits`, a standard `LogitsProcessor`, the default
`ModelState.compute_sample_logits`, and LoRA disabled. Subclasses and custom
projection paths do not receive a contract. It also identifies
`lm_head_weight` (`lm_head.weight`, or `model.embed_tokens.weight` when tied).
`served_model_name` is the public service alias; it is null when the configured
name is a local path. `revision` is the configured revision and may be null.
These fields are not content hashes; the caller must ensure the reconstruction head uses
the same checkpoint contents, including any weight updates.
Other model implementations can still return hidden tensors, but their contract
is `None`; teacher payload scoring requires a supported reconstruction contract.
Reconstruction is numerically equivalent but not guaranteed bitwise identical,
because the trainer may accumulate in a different precision.

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

The Python API returns tensors through `LLM`/`AsyncLLM`. For HTTP, enable the
same runner and connector configuration:

```bash
VLLM_USE_V2_MODEL_RUNNER=1 vllm serve Qwen/Qwen3-1.7B \
  --no-async-scheduling --max-num-seqs 1 \
  --aux-output-config '{"enable_hidden_states": true}'
```

Send the capture window to `POST /v1/completions` (or through the OpenAI client's
`extra_body`):

```json
{
  "model": "Qwen/Qwen3-1.7B",
  "prompt": "Explain GPUs briefly.",
  "max_tokens": 32,
  "hidden_state_capture": {
    "start": 0, "end": 4, "min_rows": 4,
    "collection_id": "training-step-1", "logprob_topk": 3
  }
}
```

Add `"coordinate": "absolute"` to capture prompt rows for teacher scoring.
Each choice contains `hidden_state_capture` or `hidden_capture_skip_reason`.
The capture contains window/position metadata, `segments`, layout, layer IDs,
request ID, collection ID, `logits_contract`, and a `tensors` map with
`hidden_states` and optional `logprobs` and `logprob_token_ids`. Each tensor
contains `data` (Base64-encoded contiguous native-endian bytes), `dtype` (PyTorch
name without `torch.`), and `shape`. Decode without losing BF16 precision:

```python
import base64
import torch

def decode_tensor(payload):
    return torch.frombuffer(
        bytearray(base64.b64decode(payload["data"])),
        dtype=getattr(torch, payload["dtype"]),
    ).reshape(payload["shape"])
```

HTTP capture supports non-streaming completions with `n=1`, including batched
prompts; each choice has its own capture. Streaming, beam search and chat
completions are not supported. The Python API's layout, memory budget and
runner restrictions also apply. Base64 increases the tensor payload size by
approximately one third.

Existing
`extract_hidden_states` instead exports speculative training features through
its own connector; response-window capture shares auxiliary model outputs but
does not replace that export workflow.

For correctness validation, compare full/window hidden values, generated tokens
and raw logprobs. For performance validation, measure capture off, full, window
and an unreachable window. Keep the GPU exclusive during timing and compare
identical configurations. Also measure capture disabled with asynchronous
scheduling: this feature requires synchronous scheduling for the entire engine,
including requests outside the capture window or without a capture plan. Zero
hidden-copy bytes outside the window do not imply the same throughput as an
asynchronously scheduled deployment.
For strict cross-run equality, set `VLLM_BATCH_INVARIANT=1` and record that setting
with the results, since it can select different execution kernels.
