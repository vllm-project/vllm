# Sparse MLA with sequence parallelism

`AttentionConfig.sparse_mla_shard_q` distributes sparse MLA query tokens
across tensor-parallel ranks. Normally, projections produce all tokens for
each rank's heads. A query all-to-all changes this to local tokens with all
heads; an output all-to-all restores the original layout before the value
up-projection and output projection.

The NVIDIA DeepSeek-V3.2 / GLM-5.3 model path can start with local tokens when
`ParallelConfig.use_sequence_parallel_moe` is enabled and pipeline parallelism
is disabled. For eligible ShardQ steps, it omits the hidden-state all-gather
before attention and keeps the contiguous, padded SP partition:

This SP mode requires expert parallelism, both TP and DP greater than one,
and a supported expert all-to-all backend (for example,
`allgather_reducescatter`). TP alone does not enable it.

1. Each rank projects its local tokens. The Q up-projection weight is
   replicated; the key up-projection weight used to transform Q into latent
   space is gathered once after weight loading.
2. Each rank all-gathers the projected KV latent, positional K and indexer K.
   The existing fused normalization, RoPE and quantization kernel writes the
   complete replicated caches, preserving its original rounding behavior.
   Q normalization, indexer query projections and head weights remain local.
3. The indexer scores local query rows using their original request IDs and
   causal positions. Mixed batches include decode rows in this partition.
4. Sparse MLA consumes the local queries with all heads directly, without
   the query all-to-all.
5. The output all-to-all restores all tokens and local heads. The existing
   value up-projection, output projection and SP reduce-scatter follow.

The original prefill-token threshold and sparse-backend eligibility still
apply. Short batches, dense MHA steps, MTP layers and unsupported configurations
retain the existing path. Batches within the configured prefill CUDA graph
capture range also keep the original head-sharded projection activations and
query exchange, matching the layout captured during warmup. DCP, PCP and HiSparse are excluded by the ShardQ gate.
Other model implementations keep their original projection and query exchange.

Replicated Q and key up-projection weights increase weight memory. Contiguous
SP shards can also have less balanced causal indexer work than ShardQ's zigzag
partition. Removing the two activation exchanges does not guarantee a latency
improvement at every context length; benchmark with the intended TP degree,
batch mix and quantization. Pure decode fallback uses the replicated Q
projection followed by a local head slice.
