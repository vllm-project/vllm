# Indexer Replay

Enable `--enable-return-indexer-topk` to return ordinary sparse-attention
selections through the local AuxOutput SHM connector. This path stores raw
`int32` token IDs and does not use a Profile or a model-specific codec.

```bash
vllm serve deepseek-ai/DeepSeek-V3.2 --enable-return-indexer-topk
```

The adapters cover DeepSeek V3.2 and GLM-5.2 DSA, including their shared
reference MLA path. Dots3 NOTE's overridden forward also captures its raw
token selections; GLM4-MoE-Lite uses the reference path when configured for DSA.
Support is an implementation claim pending model validation.

DeepSeek V4/V4.1 compressed-cache coordinates, GLM-5.3 KPool, Qwen4Exp QSA,
Step5 Region selection and HY-V4 are deferred to the dependent Profile PR.
MiniMax-M3 has no adapter. A model/backend without ordinary capture sources
fails at startup instead of returning an empty replay.

## Output

Offline generation returns `CompletionOutput.indexer_topk` as an array with
shape `[num_forwarded_tokens, num_indexer_sources, index_topk]`, dtype `int32`.
Source planes are ordered by their actual layer IDs. IndexCache consumers
reuse the source plane selected by the model configuration; a shared consumer
does not create another plane. All sources must have the same TopK width.
Order, padding and invalid IDs are preserved exactly as captured.

Python Chat Completions, Completions and Token-in-Token-out responses include
`indexer_topk` on the terminal choice/chunk. The value is a base64-encoded NumPy
`.npy` array; the field is omitted when disabled or no rows were produced.
These are intentional vLLM extensions, absent from the
[OpenAI OpenAPI specification](https://raw.githubusercontent.com/openai/openai-openapi/master/openapi.yaml).

```python
import base64
import io

import numpy as np

indices = np.load(io.BytesIO(base64.b64decode(choice["indexer_topk"])))
```

`SamplingParams.indexer_topk_prompt_start` skips an already returned prompt
prefix independently of `routed_experts_prompt_start`. The final sampled token
has no selection row because it has not been forwarded yet.

## Storage and limits

Indexer and routed-expert return can be enabled independently or together.
They share the existing AuxOutput block hashes, asynchronous output event,
accepted-token cursors, prefix restoration and finish lifecycle. Model identity,
source layers and TopK width identify the SHM namespace. R3-only storage keeps
its existing format.

Indexer storage is `4 * num_indexer_sources * index_topk` bytes/token. Startup
checks SHM capacity against all KV blocks and reports host staging estimates.
Capture buffers and bounded snapshots are allocated before KV memory planning.

MRV2 generate runner and prefix caching are required. Existing AuxOutput limits
on PP, DCP/PCP, adaptive verification and remote PD still apply. Rust retains the
raw engine wire field; Rust HTTP delivery is not implemented. SHM keeps its
immutable first-writer behavior. CUDA Graph, real TP/DP, model quality and
performance require later validation; no tests or model evaluations were run
for this implementation at the user's request.
