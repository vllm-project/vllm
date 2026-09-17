# DeepSeek V4.1 dependency-preserving decoder tails

This experimental opt-in reduces long-prefill work after the model's last
shared-KV source layer. It targets throughput with multiple long prompts in a
scheduled batch. It is disabled by default.

## Dependency coverage

DeepSeek V4.1 Flash layers 21-39 reuse the encoder's global KV and have a
128-token causal sliding window. To preserve the final output and each layer's
trailing cache inputs, the first decoder layer needs at least
`128 + 18 * 127 = 2414` current-query rows per request. The implementation rounds
this up to 2560 and keeps that same suffix throughout all 19 layers.

The earliest retained rows can lack earlier context. Their influence cannot
reach the final token or any layer's trailing 128 cache inputs before the
suffix ends. Shorter queries retain their full history. This preserves the
causal dependency coverage across chunked prefill and decode; different GEMM
and MoE shapes can still change floating-point rounding and greedy outputs.

## Execution

Hidden and mHC states move directly to their new sequence-parallel owners once.
The model compacts the shared indexer row buffers and rebuilds private attention
metadata. Layers with the same original metadata reuse the compact metadata
while the layout is unchanged. Absolute positions and the encoder's full global
KV remain intact. Retained outputs are restored to the runner's sampling layout.

Compacting each layer saved more arithmetic but added repeated host planning,
metadata construction, small copies, and collective launches. Keeping a larger
fixed suffix amortizes those costs. Batches below 16384 scheduled tokens use the
full path: smaller tested batches were limited by launch overhead and did not
benefit from trimming. Requests for prompt logprobs also use the full path.

## Usage and current scope

Enable with `--additional-config '{"dsv41_exact_decoder_tail": true}'`.
The tested throughput configuration uses a 32768-token scheduling budget,
TP4/EP4 on GB200, model runner V2, normal decode graphs, Mega attention,
DeepGEMM MegaMoE, and host-offloaded Engram. For example:

```bash
VLLM_USE_V2_MODEL_RUNNER=1 vllm serve /data/models/DeepSeek-V4.1-Flash \
  --tokenizer-mode deepseek_v41 --language-model-only \
  --tensor-parallel-size 4 --enable-expert-parallel \
  --no-enable-prefix-caching --max-model-len 9216 \
  --max-num-seqs 64 --max-num-batched-tokens 32768 \
  --kv-cache-memory-bytes 4294967296 \
  --attention-config '{"indexer_kv_dtype":"fp8"}' \
  --kernel-config '{"moe_backend":"deep_gemm_mega_moe"}' \
  --engram-config '{"cpu_offload":true}' \
  --additional-config '{"dsv41_exact_decoder_tail":true}'
```

The initial implementation requires text-only MRV2, DP1/PP1, uniform-window
Mega attention in the suffix, and no context parallelism, DBO, speculative
decoding, prefix caching, LoRA, routed-expert export, sleep mode, or Torch
compilation. Decode and captured batches keep their normal execution path.
Performance depends on the scheduling budget and workload; fewer decoder rows
alone do not guarantee higher throughput.
