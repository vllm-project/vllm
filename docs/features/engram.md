# Engram: conditional memory via n-gram lookups

Engram is a *conditional memory* module attached to specific layers of a
transformer backbone. At each position, the preceding few tokens are hashed
into large static embedding tables, and the retrieved rows are fused into the
hidden state, gated by how well they match it. Where Mixture-of-Experts scales
capacity through conditional computation, Engram scales it through conditional
memory: parameters that are never computed over, only looked up, so capacity
grows at O(1) per-token cost. The mechanism and its design rationale are
described in [Conditional Memory via Scalable Lookup: A New Axis of Sparsity
for Large Language Models](https://arxiv.org/abs/2601.07372).

vLLM enables Engram automatically whenever the checkpoint declares Engram
layers, so basic serving needs no extra flags. This page explains what Engram
does and how to control its memory footprint and multi-GPU topology.

## Supported models

Engram layers are declared by the checkpoint, not chosen at serve time:

| Architecture | HF config field | Notes |
| --- | --- | --- |
| `DeepseekV41ForCausalLM` | `engram_layer_ids` | e.g. `deepseek-ai/DeepSeek-V4.1-Flash` |
| `Qwen4ExpForCausalLM` / `Qwen4ExpForConditionalGeneration` | `ple_layer_ids` | same mechanism, called PLE |

Passing `--engram-config` for a model without n-gram embedding layers fails at
startup, as does running on a non-CUDA-alike platform.

## How it works

Engram processes each position in two phases: *retrieval* and *fusion*.

**Retrieval** maps the local token context to static table rows:

1. **Tokenizer compression.** At startup, vLLM builds a many-to-one map from
   the tokenizer that collapses tokens which normalize alike (NFKC, accent
   stripping, lowercasing, whitespace collapsing), so `" The"`, `"the"` and
   `"THE"` hash identically. This concentrates semantic density into a much
   smaller id space; its size must match `engram_compressed_vocab_size` from
   the checkpoint, or startup fails — every hash multiplier derives from it.
2. **Multi-head hashing.** For each n-gram order `n` (up to
   `engram_max_ngram_size - 1`), several hash heads each apply an independent
   multiplicative-XOR hash over the compressed suffix n-gram. Every (n-gram
   order, head) pair indexes its own prime-sized bucket range in the layer's
   table — disjoint across pairs, so collisions cannot collide twice — and
   the retrieved rows are concatenated into one memory vector per position.

**Fusion** turns the retrieved rows into a context-aware contribution to the
hidden state. The rows are context-independent priors and may be noisy (hash
collisions, polysemy), so they are gated: a `wkv` projection produces one key
per hyper-connection branch plus a shared value, and each branch computes a
sigmoid gate over the normalized dot product between its hidden state —
which has already aggregated global context through preceding attention —
and its key. If the retrieved memory contradicts the current context, the
gate tends toward zero and suppresses it. The gated value is added
residually into the branch's stream (`hidden + gate * value`); positions that
take no part in an n-gram (e.g. image spans) are masked so they pass through
untouched.

Engram sits only on the layers listed in the checkpoint. Placement is a
deliberate hardware-algorithm trade-off: early enough that the backbone is
relieved from reconstructing static patterns in its first layers, deep enough
that vLLM can hide table lookups behind the compute of the preceding layers
(see below).

In vLLM, hashing, lookup, and gating run as fused Triton kernels over FP8
rows with ue8m0 block scales. The layer layout — table sizes, n-gram size,
head count — is fixed by the checkpoint and is not user-configurable. Only the
storage and sharding described below are configurable, via
`--engram-config`, which takes a JSON object and can also be set field by
field as `--engram-config.<field>`
([CLI reference](../cli/serve.md#-engram-config)).

## CPU offload

Engram tables dwarf typical embedding weights: [DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash)
carries two Engram layers of roughly 384M table rows each, with 256-byte FP8
entries plus their ue8m0 block scales — about 200 GB of table weights alone.
Because lookups are deterministic — the indices are known from the token ids
alone, before the forward pass — the tables need not sit in GPU memory at
all. By default vLLM stores them in **pinned host memory** and serves lookups
directly from the Triton kernel through unified virtual addressing (UVA),
prefetching on a side CUDA stream so transfers overlap with the compute of
preceding layers. The freed GPU memory goes to the KV cache.

`cpu_offload` defaults to `true` (on): it follows the
`VLLM_PLE_CPU_OFFLOAD` environment variable, which also defaults to `true`.
An explicit `--engram-config` value takes precedence over the environment
variable.

To keep the tables resident on GPU instead (e.g. when host memory is
scarce):

```bash
vllm serve deepseek-ai/DeepSeek-V4.1-Flash \
  --engram-config.cpu_offload false
```

or equivalently `VLLM_PLE_CPU_OFFLOAD=0`. CPU offload requires a GPU with UVA
support; vLLM fails fast when it is unavailable.

## Data-parallel topologies (DeepSeek V4.1)

By default (`embedding_across_dp: false`), each DP replica keeps its own
TP-sharded copy of the tables — no cross-replica communication, but one host
copy per replica.

With `embedding_across_dp: true`, the hash heads are sharded across all
TP × DP ranks into a single table copy. Each step gathers the hash ids of
every co-located DP replica, each rank looks up the heads it owns, and the
rows are exchanged back. This trades per-step DP collectives for a much
smaller table footprint:

```bash
vllm serve deepseek-ai/DeepSeek-V4.1-Flash \
  --tensor-parallel-size 2 --data-parallel-size 4 \
  --engram-config '{"embedding_across_dp": true}'
```

`embedding_across_dp` is not supported with elastic expert parallelism yet.
Qwen4Exp PLE tables are ETP-sharded instead and honor `cpu_offload` only.

## Sharing host tables across DP replicas

When the tables are CPU-offloaded and DP replicas are co-located on one node,
`dp_shared_memory` stores **one copy of each TP shard in `/dev/shm`**, shared
by all co-located replicas through `cudaHostRegister`. Each replica then
prefetches only its own tokens — no per-step Engram DP collectives — and host
memory drops from one table copy per replica to one per node.

`dp_shared_memory` defaults to enabled whenever the other settings allow it,
and falls back to per-replica (or DP-sharded) tables with a warning when:

- the DP replicas are not co-located on a single node,
- `/dev/shm` cannot hold the full tables — in containers, raise the limit
  with `--shm-size` or `--ipc=host`,
- expert parallelism is elastic, which is unsupported.

Requirements: `cpu_offload` enabled, `--data-parallel-size > 1`, and a shared
IPC namespace across the co-located replicas.

## Limitations

- CUDA-alike platforms only.
- DeepSeek V4.1 Engram does not support DBO or microbatching; disable
  `--enable-dbo` and set `--ubatch-size 0`.
- `dp_shared_memory` requires `cpu_offload`, `data_parallel_size > 1`, and is
  unsupported with elastic expert parallelism.
