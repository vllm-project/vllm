# Last-layer FFN row pruning

`VLLM_ENABLE_LAST_LAYER_FFN_PRUNING=1` skips unused final-layer FFN rows during
prefill for dense Qwen3, Llama, and Transformers-backend Qwen2/Qwen2.5. It is
disabled by default. Attention and KV writes stay intact, weights do not change,
and the required token outputs stay mathematically equivalent. BF16 GEMM rounding
can still differ.

The original path remains for quantization, multimodal inputs, speculative
decoding, prompt logprobs, pooling, KV transfer, data/context parallelism,
microbatching, full CUDA graphs, and torch compilation. Prefill batches are
limited to 16 requests. LoRA, cached prefixes, chunked prefill, and TP/PP stay
supported; pipeline parallelism prunes only on the last stage.

Breakable graphs are keyed by the selected row count and captured on a side
stream. Compare runs by changing only this environment variable.
