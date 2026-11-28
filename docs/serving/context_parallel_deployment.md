# Context Parallel Deployment

Context parallel mainly solves the problem of serving long context requests. As prefill and decode present quite different characteristics and have quite different SLO (service level objectives), we need to implement context parallel separately for them. The major considerations are:

- For long context prefill, we need to control the TTFT (time to first token) by amortizing the computation time of the prefill across query tokens.
- For long context decode, we need more space for KV cache to increase the batchsize (and hence the throughput).

## Prefill Context Parallel

During prefill, for a long request with `T` new tokens, we need to compute query/key/value tensors for these new tokens. Say we have `N` GPUs, we can split the request into `N` chunks, and each GPU computes one chunk of the query/key/value tensors.

Depending on the use case, there are two possible strategies:

1. Partial query, full key/value: If the request token length is moderately long (we can afford holding the full key/value tensors), and the goal is to accelerate the prefill (and amortize the computation time of the prefill across query tokens), then we can gather the key/value tensors from all GPUs and let each GPU compute the attention output corresponding to the query tokens of its chunk.
2. Partial query, partial key/value: If the request token length is too long, we cannot afford holding the full key/value tensors anymore, then we can only compute one chunk of query/key/value tensors for each GPU, and use techniques like [ring-attention](http://arxiv.org/abs/2310.01889) to send/recv key/value tensors chunk by chunk.

Both approaches are under active development.

### DeepSeek-V4.1-Flash on ROCm

ROCm supports eager prefill context parallelism for the text-only
DeepSeek-V4.1-Flash model:

```bash
VLLM_ROCM_USE_AITER=1 vllm serve deepseek-ai/DeepSeek-V4.1-Flash \
  --language-model-only \
  --tensor-parallel-size 2 \
  --prefill-context-parallel-size 2 \
  --no-enable-prefix-caching \
  --enforce-eager
```

This configuration requires PP=DP=DCP=1 and does not support microbatching,
speculative decoding, KV transfer, prefix caching, MegaMoE, or the ROCm
monokernel. TP and PCP multiply, so TP=2 with PCP=2 requires four GPUs.

### Overlapping the KV all-gather under DCP

Under DCP (`decode_context_parallel_size > 1`), each rank stores one DCP shard
of the KV cache, so MLA prefill layers all-gather historical context shards
before attention. `--dcp-kv-prefetch` moves these historical KV gathers off the
main stream onto dedicated zero-SM hardware copy-engine and auxiliary upconvert
streams over PyTorch symmetric memory, pulling peers in a conflict-free
staggered ring order and pipelining copy-engine DMA transfers with receiver
upconversion:

- **Dense MLA (`MLAAttention`) under TP + DCP (and PCP + DCP)**: overlaps
  chunked-context KV gathers across both chunks and layers (starting layer 0
  chunk 0 ahead of new-token projections and suffix attention) for BF16/FP16,
  `fp8_ds_mla`, and FP8 KV caches, including multi-request and multi-chunk
  batches.
- **Sparse MLA (`FlashMLASparseImpl`) under PCP + DCP** with `fp8_ds_mla`:
  transfers raw 656 B FP8 cache rows, upconverts on the receiver, and stages
  the current chunk's new rows locally. Layers that reuse a shared indexer
  top-k prefetch only the union of historical rows that top-k selects, using
  predicted per-peer union sizes with a single receive/top-up kernel (layer 0
  keeps the serial all-gather).

It acts in eager mode and under piecewise/breakable CUDA graphs, requires
symmetric memory across the DCP group, and falls back to the serial all-gather
for ineligible steps:

```bash
# TP + DCP (Dense MLA)
vllm serve <dense-mla-model> -tp 4 -dcp 4 --dcp-kv-prefetch

# PCP + DCP (Sparse MLA)
vllm serve <sparse-mla-model> -tp 1 -pcp 4 -dcp 4 --enable-expert-parallel \
    --dcp-kv-prefetch
```

For DeepSeek V3.2-style sparse attention under PCP + DCP,
`--pcp-dcp-indexer-k-replica` avoids repeated per-chunk DCP all-gathers and
de-interleaves of historical indexer K: every rank keeps a resident
whole-context FP8 K replica per indexer layer in token order and appends each
chunk's new K rows directly. The replica costs
`(max_model_len + 4096) * (index_head_dim + 4)` bytes per indexer layer on
every rank, taken from KV cache capacity. A request whose earlier rows are not
in the replica (such as a prefix cache hit or resumed request) rebuilds it once
from the sharded cache via the serial gather. It acts in eager mode and under
the default breakable CUDA graph, is disabled automatically when a KV connector
is configured, and combines independently with `--dcp-kv-prefetch`:

```bash
vllm serve <model> -tp 1 -pcp 4 -dcp 4 --enable-expert-parallel \
    --dcp-kv-prefetch --pcp-dcp-indexer-k-replica
```

Finally, `--pcp-new-row-overlap` moves each layer's PCP all-gather of newly
computed MLA KV (`kv_c`/`k_pe`) and indexer K rows off the main stream: the
fused norm/rope writes each rank's slice directly into a peer-mapped symmetric
buffer across the PCP group, and peer segments are pulled on an auxiliary stream
beside the projections that follow (via batched copy-engine calls in eager mode,
or one pull kernel recorded as a branch under a breakable CUDA graph). All three
options are independent and can be combined:

```bash
vllm serve <model> -tp 1 -pcp 4 -dcp 4 --enable-expert-parallel \
    --dcp-kv-prefetch --pcp-dcp-indexer-k-replica --pcp-new-row-overlap
```

## Decode Context Parallel

Due to the auto-regressive nature of decoding, every decoding step needs to compute a small amount of query tokens w.r.t. a large number of key/value tokens stored in the paged KV cache. The core of decode context parallel is how to shard the KV cache across GPUs.

`engine_client.vllm_config.cache_config.effective_attention_block_size` reports the
initialized full-attention block size in tokens, including DCP: a physical block of
16 tokens with DCP=4 represents 64 tokens. The same value is available through
gRPC `Control.GetServerInfo.effective_attention_block_size`. Physical block-size
fields retain their existing meaning. The new value is unavailable (`None` in
Python, absent in gRPC) if there is no common full-attention size, the scheduler
does not expose it, or an older engine omits it. gRPC also omits the value when
engine replicas disagree. Partial cache events carry their own actual size.

For a model with `H` kv-heads, a request with `T` tokens in the context needs to store `H * T` key/value tensors in the KV cache.

1. If one GPU can hold them all, and the performance is good enough, then no parallelization is needed.
2. If one GPU cannot hold them all, or we want to hold more requests in the KV cache, we can first shard the KV cache along the `H` dimension, that's the plain tensor parallel sharding. It's as simple as adding `-tp <num_gpus>` to the command line.
3. Since `H` is limited (determined by the model architecture), when we continue to increase the tensor parallel size, the KV cache for each GPU will be duplicated for `tp_size / H` times. Of course, duplication is not good for efficiency. Then we need to add decode context parallel to further shard the KV cache along the `T` dimension. This is as simple as adding `-dcp <size>` to the command line. Note that `size` does not increase the number of GPUs we need to launch, but just reduces the KV cache duplication. The dcp size should lie in the range of `[1, tp_size/H]`. With larger dcp size, the KV cache duplication is reduced, but the communication overhead increases.

Theoretically, it is possible to extend the dcp size beyond `tp_size / H` to further shard the KV cache and accelerate the decoding phase. However, since the number of query tokens is limited in decoding, it's unclear what should we do for the remaining `dcp_size - tp_size / H` GPUs for non-attention layers. For the sake of simplicity, dcp size is upper bounded by `tp_size / H`. If you want to further accelerate the decoding phase, you can consider increasing the `tp_size` first, and then increasing the dcp size.

Note that kv cache can grow during decoding, and the sharding strategy needs to be carefully implemented. We use an interleaving strategy to shard the KV cache along the `T` dimension, so that kv cache for future tokens can be naturally sharded along the `T` dimension. This is proposed by [Chao Hong from Moonshot](https://github.com/youzhedian), and also explained in details in [this paper](http://arxiv.org/abs/2507.07120).

Case study:

For DeepSeek-R1, we have 1 kv-head when MLA is enabled. The typical single-node deployment with `-tp 8` causes 8x KV cache duplication. We can consider adding `-dcp 8` to reduce the KV cache duplication.

For Kimi-K2, the architecture is similar to DeepSeek-R1, but with more parameters. When we deploy it with `-tp 16`, the KV cache duplication is 16x. We can add `-dcp 16` to completely remove the KV cache duplication, at the cost of more communication overhead. We can also add `-dcp 8` to reduce the KV cache duplication to 2x. Although it still duplicates the KV cache twice, the communication overhead is smaller since the DCP communication only happens inside one node.

For Qwen3-235B-A22B, we have 4 kv-heads. When we deploy it with `-tp 8`, the KV cache duplication is 2x. Then we can add `-dcp 2` to remove the KV cache duplication.

In short, for decode context parallel, try to increase `-tp` size until you get satisfactory performance, and then add `-dcp` to reduce the KV cache duplication.

Decode context parallel is supported in vLLM, for both MLA and GQA models. Some attention backends also support the combination of decode context parallel and MTP (multi-token prediction) to further accelerate the decoding phase.

## Technical Discussions

The main discussions happen in the `#sig-context-parallel` channel of [vLLM Slack](https://slack.vllm.ai/).
