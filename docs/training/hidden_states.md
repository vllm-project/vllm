# Hidden-state outputs

Return the exact inputs to `lm_head` for selected generated positions or explicit
prompt scoring. Hidden
states travel inline with the final completion as one tensor, preserving their
original dtype. A trainer can apply the matching model head to obtain the full
vocabulary distribution. Use the engine's sampled-token logprobs for behavior
probabilities; a different head GEMM can produce different rounding.

## Enable and request

This initial implementation supports unquantized Qwen3 on CUDA Model Runner V2,
without LoRA, pipeline/context parallelism, batch-sharded sampling or adaptive
verification. Unsupported model logits paths fail during startup. Async
scheduling is not supported yet; this restriction applies to the whole engine.
The output APIs are implemented by the Python frontend; the Rust frontend does
not expose these request or response fields.

```bash
vllm serve Qwen/Qwen3-0.6B \
    --enable-return-hidden-states --no-async-scheduling
```

```python
from vllm import LLM, SamplingParams

llm = LLM(model="Qwen/Qwen3-0.6B", enable_return_hidden_states=True,
          async_scheduling=False)
tokens = [100, 200, 300, 400]
start = len(tokens) - 1
outputs = llm.generate(
    {"prompt_token_ids": tokens},
    SamplingParams(max_tokens=32, aux_outputs=("hidden_states",),
                   hidden_states_start=start + 8, hidden_states_end=start + 16),
)
completion = outputs[0].outputs[0]
hidden = completion.hidden_states  # CPU torch.Tensor [rows, hidden_size]
position = completion.hidden_states_start
metadata = llm.collective_rpc("get_hidden_states_metadata")[0]
```

Position `p` processes token `p` and predicts token `p + 1`. With engine-side
prompt length `P` and retained output length `O`, generated positions cover
`[P - 1, P + O - 1)`. Start/end are inclusive/exclusive absolute positions after
multimodal expansion. The returned range is the intersection with that generated
range. Defaults (`start=0`, `end=None`) return all generated rows. Rows outside
the requested window are not copied to the host. Generation ending before the
window returns `[0, hidden_size]`; no request returns `None` for requested rows.
Internal missing rows fail the request. An aborted request is not a training
sample; aborting before execution returns an empty tensor.

Each completion in `n > 1` owns its own tensor. Streaming returns the tensor only
in the terminal completion; intermediate deltas contain no hidden states. Stop
strings and speculative acceptance trim the rows to the retained token range.
Preemption recomputes history without returning previously sampled rows again.

Requests cannot use penalties or `logit_bias`. Truncation and token masks require
an explicit positive `top_k <= 256`; untruncated requests may use `top_k=0`.
The single-completion raw payload limit is
`aux_output_config.max_hidden_capture_bytes` (default 256 MiB). Admission uses
the largest possible intersection of the window and generated positions.
This cap does not bound all concurrent completions or HTTP encoding buffers.

## HTTP

`/v1/completions` and `/v1/chat/completions` accept `aux_outputs`,
`hidden_states_start` and `hidden_states_end` as vLLM extension fields.
`/inference/v1/generate` accepts them inside `sampling_params`.
Enable `/inference` routes with `--enable-scale-out`, including the metadata
query below.

Terminal choices include `hidden_states` (base64 `.npy`), `hidden_states_dtype`
and `hidden_states_start`. These fields are omitted when not requested.
BF16 uses uint16 bit patterns, avoiding a dependency on `ml_dtypes`:

```python
import base64
import io
import numpy as np
import torch

array = np.load(io.BytesIO(base64.b64decode(choice["hidden_states"])),
                allow_pickle=False)
hidden = torch.from_numpy(array)
if choice["hidden_states_dtype"] == "bfloat16":
    hidden = hidden.view(torch.bfloat16)
```

## Model metadata and versions

Python `collective_rpc("get_hidden_states_metadata")` and
`GET /inference/v1/hidden_states_metadata` expose the effective vocabulary size,
head dtype, scale, soft cap, transform order, tied-head flag and checkpoint
revision. Metadata contains no checkpoint filesystem path. The consumer must use
the matching checkpoint, including the appropriate weights after an update;
`revision=None` does not identify local weights. Startup logs print the same
metadata once. No reconstruction contract is repeated per completion.

`CompletionOutput.weight_versions` spans use output-token indexes. Hidden row
`i` belongs to output index `hidden_states_start + i - (P - 1)`. HTTP hidden
responses include the existing version spans automatically. Weight updates must
use vLLM's pause/update/version/resume workflow.

## OPD teacher prompt scoring

Request `prompt_hidden_states` to score input positions for full-vocabulary
distillation. This explicit extension is separate from the sampling-following
`hidden_states` output. Both use `hidden_states_start/end` as absolute bounds:
prompt rows cover `[0, P - 1)`, generated rows cover `[P - 1, P + O - 1)`.
Request either output or both; each returns its own intersection with the bounds.

For a context `x` and target tokens `y`, prefill `x + y` once:

```python
teacher = llm.generate(
    {"prompt_token_ids": x + y},
    SamplingParams(
        temperature=0,
        max_tokens=1,
        aux_outputs=("prompt_hidden_states",),
        hidden_states_start=len(x) - 1,
        hidden_states_end=len(x) + len(y) - 1,
    ),
)[0].outputs[0]
rows = teacher.prompt_hidden_states  # [len(y), hidden_size], original dtype
start = teacher.prompt_hidden_states_start
versions = teacher.prompt_hidden_states_weight_versions
```

The extra generated token is not part of the prompt payload. Apply the matching
checkpoint's output head and the model-level metadata to reconstruct raw logits;
no top-k truncation is introduced. Token IDs remain available in the input `y`.
Use `prompt_logprob_token_ids` when only scores for chosen candidate IDs are needed.

Each prefill chunk delivers its selected rows through engine IPC to the frontend.
The final completion returns one complete tensor. Preemption replay preserves
the first received rows, missing rows fail the request, and cancellation releases
the request's accumulated chunks. No worker carry or scheduler payload buffer is
required. Stop strings trim generated rows without trimming scored prompt rows.

Prompt scoring defaults to `skip_reading_prefix_cache=True`. Set it explicitly
to `False` to reuse only the prefix before the first selected row; external KV
reads are disabled for this bounded request. `kv_sharing_fast_prefill` is rejected.
The per-completion byte limit includes both requested output ranges.

HTTP completions, chat and `/inference/v1/generate` accept the same auxiliary
output name and bounds. Terminal choices include `prompt_hidden_states`
(base64 `.npy`), `prompt_hidden_states_dtype`, `prompt_hidden_states_start`, and
`prompt_hidden_states_weight_versions`. Fields are omitted when not requested.
The prompt version spans index rows of the returned prompt tensor, independently
of the generated-token `weight_versions` spans. The same BF16 uint16 decoding
shown above applies.

## Scope

Auxiliary drafter layers, top-k tensor delivery, store mode and async scheduling
are separate follow-ups. Prompt scoring is an explicit extension; generated
`hidden_states` retains its sampling-following meaning. Earlier benchmarks of
the combined prompt/auxiliary producer must be rerun for this implementation.

Prompt-only capture returns the raw model rows independently of sampling
penalties, logit bias or truncation. The generated-row sampling restrictions
apply when `aux_outputs` includes `hidden_states`, including requests for both
outputs. A missing required row fails its request; synchronous Python consumers
receive `finish_reason="error"` and a structured `RequestOutput.error`. Other
requests in the batch continue.
