# DSpark context-KV-only prefill

For disaggregated serving, a dedicated `kv_producer` can opt into
`"prefill_context_kv_only": true` in `--speculative-config` when using
`"method": "dspark"`. The consumer must use the same draft checkpoint and cache
configuration, with this option **disabled**.

The producer is only for the PD prefill leg; do not send standalone multi-token
generation requests directly to its API.

The producer loads the auxiliary-hidden-state projection, context normalization
and context key/value projections, but not draft query, output, MLP, embedding,
vocabulary, Markov or confidence-head weights. It inserts context KV and skips
draft query generation. The consumer runs the complete draft and reconstructs
its query-tail KV; target execution and verification are unchanged.

Supported drafts use `Qwen3DSparkModel` (GQA), with bias-free, non-quantized
BF16 draft weights, Model Runner V2, target/draft TP1 and PCP1/DCP1.
Target pipeline parallelism uses the DSpark PD pipeline path; the draft remains
local to the final pipeline stage. Draft CUDA graph capture is disabled on the
context-only producer. Existing full-draft deployments are unchanged when the
option is omitted.

This removes producer **weights and query computation**, not the draft context
cache. KV capacity and temporary long-prefill workspace still require separate
sizing. Matching producer/consumer layer names, cache dtype, block size, RoPE and
attention configuration remains necessary. Validate connector transfer and output
quality for the deployment; successful model loading alone does not establish
maximum request capacity.
