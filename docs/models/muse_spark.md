# Muse Spark Open

vLLM supports `meta-models/Muse-Spark-Open-816B-A42B-NVFP4` through
`MuseSparkForConditionalGeneration`, with text, image, video and audio inputs,
on NVIDIA and AMD ROCm GPUs. Serve it with `--reasoning-parser muse` and the
Muse [tool-call parser](../features/tool_calling.md). The
`Inferact/Muse-Spark-Open-DSpark` draft runs through
`--speculative-config '{"method": "dspark", ...}'`, and the
`RadixArk/Muse-Spark-Open-DFlash2` draft through
`--speculative-config '{"method": "dflash", ...}'` (see
[the DFlash2 section](#dflash2-draft)).

By default every batch uses the stock execution path. See
[ModelOpt](../features/quantization/modelopt.md) for the NVFP4 MoE backends.

## Latency-specialized decode

On SM100/SM103 GPUs with NVLink multicast, the opt-in `--monokernel` (the
same switch as
`--additional-config '{"decode_backend": "latency_specialized"}'`) runs decode
steps of one request verifying eight tokens through fused CuTe DSL kernels.
Other decode batches, prefill and mixed batches keep the stock path. An
unsupported deployment is refused at startup with
`ValueError: Muse Spark specialized decode does not support <reason>`. It needs:

- TP4, TP8 or TP16 with EP equal to TP, and data/pipeline parallel sizes of one.
- BF16 activations and logits, a BF16 or FP8 (E4M3, per-tensor scales) KV
  cache, `--block-size 16` and `--attention-backend FLASHINFER`.
- DSpark or DFlash2 with seven speculative tokens, and adaptive verification
  disabled. DSpark additionally needs probabilistic draft sampling
  (`draft_sample_method`); DFlash2's selector walk samples probabilistically
  by construction and takes no `draft_sample_method`.
- Compilation mode `NONE` and a CUDA graph mode whose decode component is
  `FULL`.
- None of LoRA, DBO, decode-context parallelism, batch-sharded sampling, sleep
  mode, KV transfer, watermarking, FP64 Gumbel sampling, the
  `deep_gemm_mega_moe` MoE backend or `VLLM_BATCH_INVARIANT`.

The router logits run on tensor cores within 1e-5 relative error of FP32, so
expert selection can differ on near-ties, and decoding is not batch-invariant.

When every request in a batch of up to eight is greedy, or samples with
`top_k` set between 1 and 64 (any temperature and `top_p`, `min_p` below 1),
verification is vocab-sharded: each TP rank keeps the top 64 logits of its
vocabulary shard and multicasts those to every rank over NVLink instead of
all-gathering full-vocabulary logits. The kept logits are masked as the
stock sampler masks them, in its order (temperature, `min_p`, top-k with
ties at the k-th logit going to the smaller token id, `top_p`), and the
drafts are verified as the stock rejection sampler verifies them. Greedy is
identical to stock; sampling follows the same distribution but not the same
random draws as the stock sampler, so a seeded request can differ between
the two paths. A disabled top-k, logprobs, structured output, penalties,
logit bias, allowed or bad words, a thinking budget or custom logits
processors send the batch to the stock path. If the candidates' buffer gets
no NVLink multicast mapping at start-up, every batch takes the stock path.

These options go in `--additional-config` and need the specialized decode
(`--monokernel`); without it they are refused at startup, as is a key that misspells one of them (such as `nvfp4_lmhead`).
Apart from `markov_bias_table` they are lossy: validate them with real
(not synthetic) rejection on your own quality metric.

| Option | Default | Effect |
| --- | --- | --- |
| `quantize_bf16_experts` | unset | `"nvfp4"` quantizes the checkpoint's BF16 first- and last-layer routed experts to NVFP4 at load (NVFP4 checkpoints). |
| `nvfp4_lm_head` | `false` | `true` runs the target and draft heads as NVFP4 W4A4 GEMMs for up to 128 rows. Activations beyond 512 saturate; larger projections use BF16. |
| `markov_bias_table` | `false` | Precomputes the DSpark Markov bias of the rank's vocabulary shard at load (about 9.5 GiB per rank at TP8, taken from the KV cache). Bit-identical draft tokens. |
| `draft_mlp_nvfp4` | unset | `{"scales": PREFIX}` runs the draft MLPs as NVFP4 W4A16 for 1-8 rows. Lossy. |

No `draft_mlp_nvfp4` scales are shipped. Serve once with
`{"calibrate": PREFIX}` and the other options you will deploy, send
representative prompts, wait for the next 10-second flush to
`PREFIX.tp<rank>.json`, then restart with `{"scales": PREFIX}`. A scale file
whose model revision, TP size or rank does not match is refused at load.

Example on two four-GPU GB300 nodes (node 1 adds `--node-rank 1 --headless`):

```bash
vllm serve meta-models/Muse-Spark-Open-816B-A42B-NVFP4 \
    --tensor-parallel-size 8 --enable-expert-parallel \
    --nnodes 2 --node-rank 0 --master-addr "$HEAD_ADDR" --master-port 29501 \
    --block-size 16 --attention-backend FLASHINFER \
    --reasoning-parser muse --tool-call-parser muse --enable-auto-tool-choice \
    --speculative-config '{"method": "dspark", "model": "Inferact/Muse-Spark-Open-DSpark", "num_speculative_tokens": 7, "draft_sample_method": "probabilistic", "rejection_sample_method": "block", "enable_adaptive_verification": false}' \
    --monokernel \
    --compilation-config '{"mode": 0, "decode_cudagraph_capture_request_sizes": [1, 2, 4, 8]}'
```

## DFlash2 draft

The `RadixArk/Muse-Spark-Open-DFlash2` draft is a block-diffusion proposer:
five conv-attention layers with a candidate selector propose all seven draft
tokens in parallel, where DSpark's Markov chain samples them sequentially. It
runs on the stock execution path and on the specialized decode alike: the
fused Markov loop is a hook only `DSparkSpeculator` asks for, and the DFlash2
speculator's own propose path (its model forward plus a Triton selector walk)
is CUDA-graph-captured with the rest of the step.

```bash
--speculative-config '{"method": "dflash", "model": "RadixArk/Muse-Spark-Open-DFlash2", "num_speculative_tokens": 7, "rejection_sample_method": "block"}'
```

On the same 75k-context agentic benchmark (4x B300, TP4+EP4, the
specialized decode, server-default sampling) the two drafters accept at
parity, and DFlash2's unfused draft costs about 0.9 ms more per step than
DSpark's fused draft:

| Draft position | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| DSpark | 0.650 | 0.418 | 0.281 | 0.190 | 0.130 | 0.078 | 0.051 |
| DFlash2 | 0.641 | 0.388 | 0.240 | 0.164 | 0.116 | 0.082 | 0.060 |

(mean acceptance 2.80 vs 2.69). On short coding prompts DFlash2 accepts
much more (mean 3.62), so acceptance is content-dependent: measure the
trade-off on your own traffic before switching drafters.
