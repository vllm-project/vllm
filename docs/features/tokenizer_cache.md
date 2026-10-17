# Segment tokenizer cache

`vllm serve MODEL --tokenizer-cache-gb 1` enables a CPU encoding cache with a
1 GiB budget per renderer/API process. The default is zero. This feature works
on the frontend before the inference engine and does not require a hardware
plugin. Multiple API processes each allocate their own cache.

The cache reuses text segments ending at added-token boundaries, so a growing
conversation can reuse prior turns. It is separate from the device KV prefix
cache. Token ids must match regular encoding; changing models never reuses this
cache or their KV states.

Supported tokenizers use standard HF fast ByteLevel BPE without automatic
prefix-space insertion. NFC normalization is supported when added tokens are
extracted before normalization (`normalized=false`); other normalizers and
context-sensitive added tokens disable the cache. Custom encoding/template
overrides retain their regular path.
Other tokenizers retain normal encoding. Offset requests, assistant masks,
tensor/dictionary chat outputs, and tokenizer-level truncation overflow retain
the existing path. Changing tokenizer vocabulary while serving is unsupported.

The budget accounts conservatively for retained text keys, token values and
cache entries. It is not a cap on the entire frontend RSS, temporary encoding
buffers, concurrent requests, or allocator overhead. Cache misses, tiny budgets
and unsupported tokenizers can reduce or eliminate the performance benefit.

Validate token equality against independent full encoding and compare cache
off/on with the same model, renderer worker count, request trace and KV cache
conditions. Separate cold tokenization from warmed CPU-cache measurements and
separate cold KV requests from device prefix hits. A generation completion
confirms transport and execution; task accuracy requires a task evaluation.
