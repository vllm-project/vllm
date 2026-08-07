# Sequence-parallel communication in parallel linear layers

This document describes the sequence-parallel (SP) implementation and the
boundary of the parallel-linear refactor.

## Pre-refactor implementations

vLLM has two independent features called sequence parallelism:

- The compilation pass in
  `vllm/compilation/passes/fusion/sequence_parallelism.py` rewrites
  all-reduce and RMSNorm patterns to reduce-scatter, local normalization, and
  all-gather. It is model-independent and requires full-graph compilation.
- MoE sequence parallelism is selected by
  `ParallelConfig.use_sequence_parallel_moe`. Models shard their token
  dimension so expert work can use the tensor-parallel ranks as sequence
  ranks. Most of this path was implemented explicitly in model code.

Before this refactor, the model-side implementations fell into the following
groups.

| Implementation | Model families | Model-side communication |
| --- | --- | --- |
| Legacy MoE blocks | AXK1, GPT-OSS, GraniteMoE, InternS1 Pro, Llama 4, MiMo V2, Nemotron-H, OpenPangu, Qwen3-MoE | `sequence_parallel_chunk` before MoE and all-gather after MoE |
| Attention-to-MoE bridge | DeepSeek V2, Qwen3-Next, Qwen3.5 (through Qwen3-Next), DeepSeek V3.2 | all-gather before attention and reduce-scatter after its row-parallel output |
| Transformers backend | Generic Transformers MoE fuser | `sequence_parallel_chunk` and all-gather in the fuser |
| New model implementation | Kimi K3 | all-gather/reduce-scatter around attention and around sharded dense MLPs; additional gathers at model and MTP boundaries |
| New model implementation | DeepSeek V4 | all-gather/reduce-scatter around attention; additional gathers in model, dSPARK, and MTP boundaries |

The inventory excludes collectives that do not represent token-dimension SP,
including vocabulary/logits gathers, vision Q/K gathers, pipeline transfers,
and Inkling's hidden-dimension reduce-scatter/all-gather pair.

Boundary gathers are not parallel-linear operations. Final hidden states,
auxiliary hidden states, MTP inputs/outputs, and pipeline boundaries may still
need explicit model orchestration after the linear migration.

## Unified entry point

SP collectives live in `vllm.distributed.communication_op`:

- `sequence_parallel_all_gather` gathers dimension 0 without changing the
  gathered shape.
- `sequence_parallel_reduce_scatter` sums partial results and scatters token
  shards along dimension 0 without adding padding.
- Both operations first use a device communicator's custom SP collective when
  available, then fall back to the regular TP collective.

`vllm.models.common.ops.sequence_parallel` remains as a compatibility facade
for existing Kimi K3 and DeepSeek V4 call sites.

Parallel linear layers expose one `sequence_parallel` switch:

```text
local token shard
  -> ColumnParallelLinear.prepare_input() -> all-gather
  -> column-parallel computation
  -> row-parallel computation
  -> RowParallelLinear.reduce_output() -> reduce-scatter
  -> local token shard
```

The default is `False`, so existing layers retain all current TP behavior.
When enabled, a column-parallel layer gathers its input token shards before
the quantization method runs. A row-parallel layer reduce-scatters its partial
output instead of all-reducing it. `MergedColumnParallelLinear` and
`QKVParallelLinear` expose the same constructor option.

The communication entry points do not pad, unpad, or otherwise adjust token
counts. Callers must provide a token dimension that is valid for the TP
collective, including divisibility required by reduce-scatter.

LoRA column and row wrappers call the base layer's `prepare_input` and
`reduce_output` methods. This keeps the communication policy in the parallel
linear layer for both base and LoRA execution.

## Model migration

The attention-to-MoE paths in DeepSeek V2, Qwen3-Next/Qwen3.5, and DeepSeek
V3.2 now use the common input preparation and row-linear reduce-scatter. Kimi
K3 and DeepSeek V4 attention use the same path, and Kimi K3's sharded dense
MLP enables SP directly on its merged-column/row linear pair.

Some attention blocks fan one hidden-state tensor out to several projections,
and the first SP-enabled sparse layer can still receive a replicated input.
Those paths call `prepare_sequence_parallel_input` once and share the gathered
tensor. The helper is also used by `ColumnParallelLinear.prepare_input`, so the
collective policy and backend override remain centralized without performing
duplicate gathers.

Final hidden states, auxiliary hidden states, MTP inputs/outputs, pipeline
boundaries, and MoE dispatch collectives are not parallel-linear pairs. Their
explicit boundary gathers remain in model orchestration code.
