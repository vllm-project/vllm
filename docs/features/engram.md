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
or packed MXFP4 rows with ue8m0 block-32 scales. The layer layout — table
sizes, n-gram size, head count — is fixed by the checkpoint and is not user-configurable. Only the
storage and sharding described below are configurable, via
`--engram-config`, which takes a JSON object and can also be set field by
field as `--engram-config.<field>`
([CLI reference](../cli/serve.md#-engram-config)).

DeepSeek V4.1 checkpoints select packed MXFP4 tables with
`quantization_config.engram_embed_quant: "mxfp4"`; absent or `"fp8"` keeps
FP8 tables. MXFP4 stores two E2M1 values per byte, with the even column in
the low nibble. If specified, `engram_embed_block_size` must be `32`,
`engram_embed_scale_dtype` must be `"ue8m0"`, and `engram_embed_layers` must
match `engram_layer_ids`. Mixed per-layer table formats are not supported.
These fields are read from checkpoint metadata.

## CPU offload

Engram tables dwarf typical embedding weights: [DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash)
carries two Engram layers of roughly 384M table rows each, with 256-byte FP8
entries plus their ue8m0 block scales — about 200 GB of table weights alone.
The MXFP4 tables in `nvidia/DeepSeek-V4.1-Flash-NVFP4` use 128-byte rows
plus 8 scale bytes, reducing the full tables to about 104 GB (97 GiB).
Packed tables work with CPU offload, shared host storage, and GPU peer reads.
Because lookups are deterministic — the indices are known from the token ids
alone, before the forward pass — the tables need not sit in GPU memory at
all. By default vLLM stores them in **pinned host memory** and serves lookups
directly from the Triton kernel through unified virtual addressing (UVA),
prefetching on a side CUDA stream so transfers overlap with the compute of
preceding layers. The freed GPU memory goes to the KV cache.

`cpu_offload` defaults to `true` (on)

To keep the tables resident on GPU instead (e.g. when host memory is
scarce):

```bash
vllm serve deepseek-ai/DeepSeek-V4.1-Flash \
  --engram-config.cpu_offload false
```

CPU offload requires a GPU with UVA support;
vLLM fails fast when it is unavailable.
The exception is Qwen4Exp on NVIDIA GPUs, for example under NVIDIA
Confidential Computing. It keeps the table in host memory and copies each
step's rows to the GPU before the forward, as
[disk offload](#disk-offload-qwen4exp) does. Each DP replica then keeps its
own table, even with `dp_shared_memory`.

## Disk offload (Qwen4Exp)

`disk_offload` keeps Qwen4Exp PLE tables in the checkpoint files: no table
is allocated in GPU or pinned host memory, and nothing is copied to an offload
directory. Each step reads that step's rows from the safetensors files on the
host. Use it where neither `cpu_offload` nor a resident table fits, for example
on unified-memory GPUs such as DGX Spark, where pinned host memory comes out of
the same pool as the weights and the KV cache:

```bash
vllm serve <qwen4exp-checkpoint> --engram-config '{"disk_offload": true}'
```

The Qwen4Exp implementation runs on CUDA and takes precedence over `cpu_offload`.

Rows are read with `preadv` and copied to fixed device buffers before the
forward, so CUDA graphs replay unchanged. They stay in the checkpoint format
(BF16, FP8 or NVFP4) and are decoded on the GPU, so an unquantized table must
match `--dtype`. Rows that were read recently are served from the OS page
cache, which is shared by co-located ranks, not counted in the worker's RSS,
and reclaimed by the kernel under memory pressure.

Costs and limits:

- The read needs the step's token ids, so it waits for the previous step on
  the GPU, and every TP rank reads all of its rows.
- A row that is not in the page cache is read from disk. For a fast first
  request, read the PLE shard files once after the server has started.
- The checkpoint files must not change while the server runs. Truncation stops
  the engine with an error naming the file; deleting or renaming is safe.
- Weight reload and elastic EP are unsupported. Loading fails when tensors are
  not file-backed safetensors views (`.bin`, runai, `fastsafetensors`,
  `instanttensor`, or `--safetensors-load-strategy eager`);
  `--load-format dummy` serves zeros.

## Disk offload (HY4 Flash)

On NVIDIA CUDA, HY4 Flash can read the Engram rows for each step directly
from a lazy-loaded safetensors checkpoint. This does not change the model's
GPU requirements:

```bash
vllm serve tencent/Hy4-flash-FP8 --language-model-only --enforce-eager \
  --tensor-parallel-size 4 --enable-expert-parallel --max-model-len 4096 \
  --safetensors-load-strategy lazy --engram-config '{"disk_offload": true}'
```

Each tensor-parallel rank reads only its own hash heads. Both BF16 rows and
FP8 rows with their scales stay in the checkpoint files; this replaces the
pinned host tables, not the model's other weights or KV cache.

Reads synchronize with the GPU and can be slower than pinned RAM, especially
on page-cache misses. Use a local SSD and keep the checkpoint files unchanged
while serving. Weight reload, shared or cross-DP Engram tables, and CUDA
graphs are not supported with HY4 Flash disk offload.

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
Qwen4Exp PLE tables use ETP sharding by default. On NVIDIA they can also use
the cross-engine Engram pool described below.

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

On ROCm, `VLLM_ROCM_ENGRAM_HIP_VMM=1` places the shared tables in HIP VMM host
memory instead, which is faster to fill than `/dev/shm`. It is off by default:
on affected amdgpu drivers (for example 6.16.6) the tables' host memory is not
freed after the replicas exit, until reboot
([ROCm/rocm-systems#13573](https://github.com/ROCm/rocm-systems/issues/13573)).
Any HIP VMM failure falls back to `/dev/shm` on every replica.

## Splitting tables across engines (DeepSeek V4.1)

Engines that can reach each other's GPUs, such as the prefill and decode
engines of a disaggregated deployment, can split each table's rows over all of
their GPUs instead of keeping a copy per engine. Each GPU holds
1/`peer_group_size` of every table and reads the other rows from their owner:

```bash
# Every engine of the group, e.g. 9 DEP2 prefill engines (18 GPUs):
--engram-config.cpu_offload false \
--engram-config.peer_group_size 18 \
--engram-config.peer_group_store "$HOME/engram-peers/$SLURM_JOB_ID/store"
```

`peer_group_size` is the sum of the engines' GPU counts. `peer_group_store` is
a file the engines meet through once at startup: give every engine the same
path, reachable by all of them (a shared filesystem across nodes) and fresh per
launch. `peer_group_transport` picks how rows move:

- `nvlink` (default): with `cpu_offload=true`, each shard lives in Extended
  GPU Memory (EGM) on its owner's local Grace NUMA node. Set
  `cpu_offload=false` to place shards in HBM. Both placements require CUDA
  FABRIC handles, an IMEX channel, and peer access over NVLink. EGM also
  requires host NUMA VMM support and enough free Grace memory.
- `rdma`: the owner's NIC gathers rows from its HBM, or from its host memory
  with `cpu_offload` enabled, into RDMA WRITEs (RoCE v2 or InfiniBand, through
  libibverbs) that the fetching GPU builds and hands to the owner itself (it
  rings the NIC's doorbell); no SM moves rows. Each GPU uses the active mlx5
  RDMA device closest to it (behind its PCIe switch, else on its NUMA node)
  and starts a proxy process that posts the WRs it is handed.
  Peers exchange memory keys at startup, so the network must be trusted.

`peer_group_fetch` picks where the fetch kernels run: `sync` (default) on the
compute stream, started before the decoder layers with each Engram layer
waiting for its own rows; `async` on a side stream. Full CUDA graphs capture
both; piecewise ones break at the start and at each Engram layer's wait.

For Qwen3.8-Flash-Next-FP8, use `peer_group_fetch=sync` with either transport.
For a cross-node group, set `peer_group_transport=rdma`. Set `cpu_offload=true`
for pinned host shards or false for HBM shards. Start every engine
with the same checkpoint and revision. Each worker loads the FP8 PLE rows
assigned to its peer rank and keeps a local copy of the table's scale. The PLE
layer reads from the pool on the compute stream. Set `disk_offload` and
`embedding_across_dp` to false (their defaults).

Use a full checkpoint with the pool. The `ipc_cache` and `sharded_state` loaders
and online weight updates are unsupported. Restart the peer group to change
weights.

Prefill and decode engines can share one group (each GPU holds less of every
table) or form one group each (each group holds a full copy, and neither slows
the other). With `rdma`, engines that step independently in one group, such as
prefill and decode engines or several aggregated engines, share the owners'
NICs: `peer_group_rdma_inflight` (default 8, 0 for no limit) bounds the
prefill-sized gathers each NIC keeps in flight, so one engine's prefill delays
another's decode fetches only slightly while prefill fetches keep their rate.
The DP ranks of one engine step together, so it does not matter within one
engine.

With `cpu_offload`, each GPU's NIC reads the host shards of its own node
itself, so the engines of a node must run as one user in one PID namespace
(they open each other's shards through `/proc`), and no traffic goes from one
NIC of a node to another. That is required on rail-optimized fabrics, whose
NICs of one node cannot reach each other. Point `peer_group_thp_dir` at a tmpfs
mounted with `huge=always` to put host shards on huge pages: NICs gather rows
at random and miss their address-translation cache on nearly every 4 KB page.
Mount it with `mount -t tmpfs -o huge=always,size=<size> none <dir>`, or
inside `unshare -Urm` without root.

Both require TP=1 and PP=1. Weights cannot be reloaded, and restarting one
engine means restarting the group.

## Limitations

- CUDA-alike platforms only.
- DeepSeek V4.1 Engram does not support DBO or microbatching; disable
  `--enable-dbo` and set `--ubatch-size 0`.
- `dp_shared_memory` requires `cpu_offload`, `data_parallel_size > 1`, and is
  unsupported with elastic expert parallelism.
- MXFP4 tables use BF16 row exchange when DP-sharded; the batched FP8
  exchange optimization is disabled for these tables.
