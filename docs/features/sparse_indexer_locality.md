# Sparse Indexer Overlap

An opt-in decode optimization for the sparse-attention indexer of Kimi K3.1 and
DeepSeek-V4.1 (DeepSeek Sparse Attention), off by default.

| Environment variable | Default | What it does |
| -------------------- | ------- | ------------ |
| `VLLM_INDEXER_OVERLAP` | `0` | Overlap the indexer's top-k with independent work of the same layer |

```bash
VLLM_INDEXER_OVERLAP=1 vllm serve <Kimi K3.1 or DeepSeek-V4.1 checkpoint> ...
```

## Indexer overlap (`VLLM_INDEXER_OVERLAP`)

Decode steps captured as FULL CUDA graphs reorder independent work of the same
layer around the indexer; the kernels and their inputs are unchanged, so the
outputs are bit-identical. Eager steps and piecewise CUDA graphs keep the stock
order.

- Kimi K3.1: the indexer runs on another stream while the layer issues the KV
  norm, the query projection, the output gate, the MLA cache insert and the
  query preparation; sparse attention waits for the kept tokens.
- DeepSeek-V4.1: the candidate source layer selects its candidate blocks, and
  expands them into the sparse block ids of its first consumer layer, on a side
  stream that the consumer joins. The main query projection and cache inserts
  run beside the indexer, and the indexer query beside the compressor and the
  index key.

The overlap is off with micro-batching (`--enable-dbo` or `--ubatch-size`), with
speculative decoding, and for Kimi K3.1 with an attention backend whose top-k
conversion does not go through the shared sparse MLA index group.
