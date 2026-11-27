# DoSA sparse indexer

DoSA scores a seed, screens keys with certified bounds, and refines survivors
for DSA prefill. It is disabled by default. Enable it with
`--attention-config '{"indexer_dosa": true}'`. It uses the model's resolved
BF16 or FP32 logit dtype, configurable with `indexer_logits_dtype`.

The optimized path requires SM100-family GPUs, CUDA 13, FP8 index keys,
32 or 64 index heads, head dimension 128, and top-k 2048. Sparse contexts
range from 131,072 keys to 1,048,576, with a 131,073-key minimum for H32 FP32
and H64 BF16. Decode and unsupported prefill configurations use the existing
indexer. Eligible short BF16 prefills can use DeepSelect.

Selection matches native MX scores in the resolved dtype. Ties allow any valid
ID at the cutoff. Rejected rows recover exhaustively. Each execution slot
prepares workspace before KV-cache sizing and shares it across layers.

[PR measurements](https://github.com/vllm-project/vllm/pull/63381) and
[pinned historical reproduction](https://github.com/llm-sys/dosa-vllm/blob/11b1ac6020885aa86705f611b00348664e379e8f/docs/features/dosa.md).
