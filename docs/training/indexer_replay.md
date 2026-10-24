# Indexer Replay

Enable `--enable-return-indexer-topk` to return ordinary sparse-attention
selections through the local AuxOutput SHM connector. This path stores raw
`int32` indices and does not use a Profile or a model-specific codec.

```bash
vllm serve deepseek-ai/DeepSeek-V3.2 --enable-return-indexer-topk
```

The adapters cover DeepSeek V3.2 / GLM-5.2 DSA and their shared reference MLA
path, plus DeepSeek V4's common GPU attention path. Dots3 NOTE also captures raw
token selections; GLM4-MoE-Lite uses the reference path when configured for DSA.
Support is an implementation claim pending model validation.

DeepSeek V4.1, GLM-5.3 KPool, Qwen4Exp QSA, Step5 Region selection and HY-V4
are deferred to the dependent Profile PR.
MiniMax-M3 has no adapter. A model/backend without ordinary capture sources
fails at startup instead of returning an empty replay.

## Output

Offline generation returns `CompletionOutput.indexer_topk` as an array with
shape `[num_forwarded_tokens, num_indexer_sources, index_topk]`, dtype `int32`.
Source planes are ordered by their actual layer IDs. IndexCache consumers
reuse the source plane selected by the model configuration; a shared consumer
does not create another plane. All sources must have the same TopK width.
Order, padding and invalid IDs are preserved exactly as captured.

DeepSeek V4 captures only actual C4A Indexer layers; HCA/SWA layers do not add
planes. Its indices address the compressed KV cache and are returned unchanged,
as consumed by Attention. They are not expanded to original token positions.
This fixed-width raw format requires no Profile or codec.

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

GPUWorker initializes and binds both capturers after leaving the weights memory
pool and before KV-cache memory profiling. The Connector owns their buffers and
bindings; SHM storage is initialized separately after KV capacity is resolved.
Capture buffers therefore participate in automatic KV budgeting and stay outside
the weights pool during weight reload or sleep. Explicit KV memory budgets remain
manual. Shutdown releases captured graphs before unbinding and freeing capture.

Indexer storage is `4 * num_indexer_sources * index_topk` bytes/token. Startup
adds the byte widths of all enabled artifacts, including routed experts. For
`N` KV blocks and `B` tokens/block, required SHM per output worker is
`N * B * combined_bytes_per_token`.

Before creating the Store, startup queries system available RAM and the remaining
container cgroup memory, using the smaller value. The Store uses anonymous shared
`mmap`, so `/dev/shm` mount free space is not its capacity limit; see the
[Linux tmpfs documentation](https://www.kernel.org/doc/html/latest/filesystems/tmpfs.html).
The check reserves runtime headroom, accounts for staging and estimated metadata,
and divides the remaining budget across local DP output workers. It then rounds
down to complete SHM objects and infers the maximum GPU KV block count. The same
check applies when only routed-expert return is enabled.

If the arenas cannot fit or a configured `aux_output_config.max_bytes` cannot
cover all KV blocks, startup fails with required/available SHM and a block limit.
Use the reported `--num-gpu-blocks-override N` (Python:
`num_gpu_blocks_override=N`) or a smaller positive count. Unset
`aux_output_config.max_bytes` so the arena follows the chosen KV block count, or
resize it within the reported host budget. A smaller `--max-model-len` may be
needed to fit the resulting KV capacity.

Alternatively, reduce `--kv-cache-memory-bytes` or `--gpu-memory-utilization`.
An explicit KV memory byte budget takes precedence over utilization; no linear
conversion from CPU RAM to a GPU utilization percentage is assumed. If staging
alone exhausts the budget, the error asks for more RAM/container memory or smaller
`--max-num-batched-tokens`/`--max-num-seqs` instead of suggesting zero blocks.
This is a startup estimate, not an atomic host-memory reservation; concurrent
allocations and later output accumulation can change the available memory.
Capture buffers and bounded snapshots are allocated before KV memory planning.

MRV2 generate runner and prefix caching are required. Existing AuxOutput limits
on PP, DCP/PCP, adaptive verification and remote PD still apply. Rust's token
frontend accepts `sampling_params.indexer_topk_prompt_start` on
`/inference/v1/generate`, independently of the routed-expert offset. Its LLM
facade accumulates raw rows across engine updates and stream intervals; the
terminal choice/chunk carries the same base64 NumPy array as Python. Disabled
or absent Indexer data omits the field. Rust Chat/Completions artifact delivery
is not implemented, matching its existing routed-expert support boundary.
SHM keeps its immutable first-writer behavior. CUDA Graph, real TP/DP, model
quality and performance require later validation. Rust compilation and route
regressions require a Unix Rust environment; they have not run on the local
Windows host.
