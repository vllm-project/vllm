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

### DeepSeek-V4.1 CUDA graphs

DeepSeek-V4.1-Flash supports PCP CUDA graphs on NVIDIA CUDA using Model Runner V2
and `FLASHMLA_SPARSE_DSV41`. Select the graph mode explicitly:

- `FULL_AND_PIECEWISE` uses full graphs for decode and breakable piecewise graphs
  for prefill and mixed batches within the captured token sizes.
- `FULL_DECODE_ONLY` uses full graphs for decode; prefill and mixed batches run
  eagerly.
- `NONE` disables CUDA graphs. Pure `FULL` and `PIECEWISE` modes are not
  supported by this V4.1 PCP path.

V4.1 uses breakable CUDA graphs rather than `torch.compile` graph splitting.
Set `VLLM_USE_BREAKABLE_CUDAGRAPH=1` for piecewise modes. Compilation `mode=0`
(`NONE`) disables `torch.compile`, not CUDA graphs: `cudagraph_mode` controls
graph execution separately.

For example, a four-GPU TP1/PCP4 instance with fixed-length DSpark can use:

```bash
VLLM_USE_V2_MODEL_RUNNER=1 VLLM_USE_BREAKABLE_CUDAGRAPH=1 \
vllm serve deepseek-ai/DeepSeek-V4.1-Flash-1017 \
    --language-model-only --tokenizer-mode deepseek_v41 \
    --tensor-parallel-size 1 --prefill-context-parallel-size 4 \
    --enable-expert-parallel --all2all-backend allgather_reducescatter \
    --attention-backend FLASHMLA_SPARSE_DSV41 \
    --speculative-config '{"method":"dspark","num_speculative_tokens":5}' \
    --max-model-len 32768 --max-num-seqs 32 --max-num-batched-tokens 4096 \
    --enable-prefix-caching --swa-bounded-replay --no-async-scheduling \
    --compilation-config '{"mode":0,"cudagraph_mode":"FULL_AND_PIECEWISE","cudagraph_capture_sizes":[5,6,10,12,20,24,40,48,80,96,160,192,256,512,768,1024],"max_cudagraph_capture_size":1024}' \
    --cudagraph-metrics
```

Capture sizes are token counts. The example covers a 4096-token prefill split
across PCP4 with 1024 tokens per rank, plus padded DSpark draft and verification
buckets. Mixed and uneven batches can have larger local token counts; batches
outside the captured sizes run eagerly. The draft processes the restored
global batch, so do not divide its capture budget by the PCP size. Increase the
capture sizes when increasing the batch-token limit. Use `--cudagraph-metrics`
to check actual `FULL` and `PIECEWISE` execution counts; the configured mode
alone does not prove either path ran.

The example enables automatic prefix caching, which PCP bounded replay requires,
and SWA bounded replay. Large
piecewise graphs use an eager break to select the bounded decoder replay batch,
then run its matching inner graph when available; larger replay batches fall
back to eager execution. Keep `cudagraph_decoder_replay` enabled (the default)
to retain bounded trimming in piecewise graphs. Full graphs and piecewise
graphs below `decoder_replay_trim_threshold` keep the whole batch. Prefix-cache
hits and bounded trimming depend on the requests and are not guaranteed by
enabling the flags alone.

Omit `--speculative-config` for ordinary decoding. PCP continues to require
PP=DP=DCP=1, text-only input, synchronous scheduling, and no KV connector or
MegaMoE. Adaptive speculative verification is unsupported. Engram tables remain
TP-sharded and replicated across PCP ranks; account for their host memory when
choosing the TP/PCP split.

Speculative full-graph replay requires flattened or variable-length indexer queries,
including when a PCP rank owns no decode requests. On SM90, fixed DSpark with
5 speculative tokens uses the flattened path; native multi-query shapes such
as 1 or 3 speculative tokens are rejected for graph execution. Use eager mode
for those shapes.

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
