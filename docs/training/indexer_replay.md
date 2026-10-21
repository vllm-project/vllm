# Indexer selection return

Indexer Replay returns the sparse selections made during a rollout, so a
trainer can reuse them. Selection semantics depend on the model: token IDs,
compressed-cache coordinates, Blocks, Pools or Regions. Always retain the
returned `indexer_profile` with the data.

This implementation is under review. CPU tests, model evaluations and hardware
end-to-end comparisons have not been run for this revision.

## Enable capture

```bash
VLLM_USE_V2_MODEL_RUNNER=1 vllm serve deepseek-ai/DeepSeek-V3.2 \
    --enable-return-indexer-topk
```

For offline generation:

```python
from vllm import LLM, SamplingParams

llm = LLM("deepseek-ai/DeepSeek-V3.2", enable_return_indexer_topk=True)
outputs = llm.generate("Hello", SamplingParams(max_tokens=8))
indices = outputs[0].outputs[0].indexer_topk
profile = outputs[0].outputs[0].indexer_profile
```

The full configuration also controls encoding and output format:

```bash
--aux-output-config '{"enable_return_indexer_topk":true,"indexer_encoding":"compact","indexer_output_format":"packed"}'
```

Routed-expert return can be enabled independently or at the same time.
Indexer-only mode does not bind the MoE router. Model Runner V2, generate
runner and prefix caching are required. The existing AuxOutput restrictions
on PP, DCP/PCP, adaptive verification and remote PD connectors remain in effect.

## Read the output

Offline, raw selections are an `int32` NumPy array. The first dimension is
logical token position and the second is **source indexer**, ordered by
`profile["sources"]`. `layer_id` identifies each producer and
`consumer_layer_ids` describes the attention layers reusing that selection.
Ordinary attention layers without an indexer are not added as sources.

The returned logical range is:

```text
[indexer_topk_prompt_start, prompt_length + accepted_output_length - 1)
```

The last sampled token has no forward selection. Set
`SamplingParams.indexer_topk_prompt_start` to omit already-returned prompt
positions in a later turn. It is independent of `routed_experts_prompt_start`
and must be between zero and prompt length, inclusive.

The Python Chat Completions, Completions and token-in/token-out APIs return
`indexer_topk` as a base64 `.npy` array in each terminal choice. Streaming
responses contain it only on the choice's final chunk. These fields are
vLLM extensions, absent from the
[OpenAI OpenAPI 2.3.0 specification](https://raw.githubusercontent.com/openai/openai-openapi/master/openapi.yaml)
checked on 2026-10-20. Disabled responses omit the new fields.

```python
import base64
import io
import numpy as np

indices = np.load(io.BytesIO(base64.b64decode(choice["indexer_topk"])))
```

Packed mode returns `indexer_payload` (bytes offline, base64 over HTTP) and
`indexer_profile`, without expanding the data at the API server:

```python
from vllm.distributed.aux_output_connector.profile import (
    AuxOutputProfile,
    decode_indexer,
)

profile = AuxOutputProfile.from_dict(choice["indexer_profile"])
payload = base64.b64decode(choice["indexer_payload"])
rows = np.frombuffer(payload, dtype=np.int32).reshape(-1, *profile.shape_per_token)
decoded = decode_indexer(rows, profile)
```

The profile's shape and dtype describe the **stored** rows. Each source's
codec explains their interpretation; packed Pool IDs must not be treated as
raw token IDs. Profile version mismatches fail explicitly.

## Model layouts

| Model/backend | Persistent source layout | Decoded selection |
| --- | --- | --- |
| DeepSeek V3.2 / GLM-5.2, NVIDIA/AMD | `K` int32 | Logical token IDs, with original order and `-1` padding |
| DeepSeek V4 / V4.1, NVIDIA/AMD | `K` int32 per actual source | Compressed-cache coordinates; source ratio and consumer mapping are explicit |
| GLM-5.3 KPool, NVIDIA/AMD | `K/P` packed Pool/validity words and `P-1` literal tail entries | Exact expanded history and causal tail, including short-context padding |
| GLM-5.3 without Pool compression | Raw int32 selection | Logical token IDs |
| Qwen4Exp NVIDIA | `K/R` Block IDs, query position, visible-Block count | Expanded token IDs, causal tail, padding and a final valid-count column |
| Qwen4Exp AMD | Raw `K+R-1` token IDs | Backend's actual selection; no trailing count column |
| Step5 NVIDIA | `G * (K+2)` int32 | Separate KV groups, selected historical Regions and current Region, padding, original selection width in the final column |
| HY-V4 NVIDIA / Dots3 / GLM4-MoE-Lite sparse MLA | Raw token IDs at actual producer layers | Explicit source mapping; ordinary/SWA layers excluded |

`K` is effective topk, `P` the Pool size, `R` the QSA compression ratio and
`G` the global KV-group count. `indexer_encoding="raw"` disables compact Pool
and QSA codecs. All layouts retain selection order. For Step5, reconstruct
the exact original variable-width selection for a token/source/group with
`row[:int(row[-1])]`; the trailing value is width metadata, not a Region ID.
`ratio` is the number of tokens represented by a Region, Block or cache row;
it does not imply those representations are interchangeable.

MiniMax-M3's per-head Block/indexer implementation has been audited but has no
adapter in this revision. Models/backends without a registered source fail at
worker initialization. CPU-only tests do not imply CPU inference support.
The Rust raw engine client retains the new IPC fields; Rust HTTP delivery is
not implemented.

## SHM lifecycle and memory

Scheduler metadata supplies the same request IDs, generations and incremental
KV-compatible hashes used by routed-expert return. Source hooks copy selections
before a shared index buffer is overwritten. Stable step snapshots are copied
on the existing output stream; after its completion event, the Worker trims
rejected speculative rows and publishes complete hash Blocks to local SHM.
Cached prefixes are read from SHM without re-running their forward pass.
Request tails, preemption rewind and cleanup use the existing AuxOutput state
machine. R3 and Indexer share a fixed record when both are enabled.

Per output worker, minimum persistent capacity is:

```text
KV_blocks * Scheduler_block_size * (R3_bytes_per_token + Indexer_bytes_per_token)
```

`max_bytes` smaller than this capacity fails at startup, with the current and
maximum supported KV-block counts. Increase the host budget, use compact
encoding, or reduce KV-cache capacity. The diagnostic also estimates host
staging, metadata, runtime reserve and local DP output workers. TP workers
participate in capture, but only the output rank creates the SHM Store.
GPU capture buffers and the bounded concurrent snapshot pool are allocated
inside model-load memory profiling, before KV capacity is planned.

For `K=2048`, GLM KPool `P=4` stores 2,060 bytes/source/token instead of 8,204;
QSA NVIDIA `R=4` stores 2,056 instead of 8,208. Multiply by actual source count
and KV token capacity, then add R3 and staging. These are format sizes, not
measured performance results. Host estimates do not atomically reserve memory
across independently launched deployments.

The SHM Store keeps the existing immutable first-writer behavior. A hash reused
after recomputation can retain an earlier valid selection for that logical
position. Evicted/missing objects produce an explicit error; they are not
silently replaced with zero rows. Cross-host SHM, Mooncake and a new PD protocol
are outside this implementation.
