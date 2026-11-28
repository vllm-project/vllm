# Hierarchical LM head

A hierarchical LM head scores trained cluster-routing vectors, selects the top-K
vocabulary clusters, then scores only their token rows. The shared
`HierarchicalLMHead` implementation fuses selected-weight gathering and scoring
with `torch.compile`, avoiding a materialized copy of selected embeddings.
The original vocabulary weights remain in their original order.

Gemma4 MTP and NemotronH MTP use this layer when their draft checkpoint enables
it. Vocabulary selection is approximate and can reduce draft acceptance; the
target model still verifies proposals using its full LM head.

## Checkpoint contract

The router must be trained for the specific draft head and stored with the draft
weights. Enable it in the draft checkpoint's `config.json`:

```json
{
  "use_ordered_embeddings": true,
  "num_centroids": 4096,
  "centroid_intermediate_top_k": 32
}
```

Include these tensors in the checkpoint, loaded through the normal model loader:

| Tensor | Shape | Meaning |
| --- | --- | --- |
| `masked_embedding.centroids.weight` | `[num_centroids, hidden_size]` | Finite floating-point routing vectors |
| `masked_embedding.token_ordering` | `[vocab_size]` | INT64 vocabulary permutation, grouped by cluster |

`hidden_size` is the draft head's input dimension. Clusters must be balanced:
`vocab_size` must be divisible by `num_centroids`. Reshaping `token_ordering` to
`[num_centroids, vocab_size / num_centroids]` defines each cluster's vocabulary
IDs. Top-K must be between one and `num_centroids`. Change top-K through the
draft checkpoint config; no additional speculative-config fields are needed.

With `use_ordered_embeddings` absent or false, the model retains its dense head.

## Serving

The official Gemma4 assistant already includes its router and configuration:

```bash
vllm serve google/gemma-4-E4B-it \
  --dtype bfloat16 --tensor-parallel-size 1 \
  --speculative-config '{
    "method": "mtp",
    "model": "google/gemma-4-E4B-it-assistant",
    "num_speculative_tokens": 7,
    "use_local_argmax_reduction": true
  }'
```

For a NemotronH MTP checkpoint containing its MTP weights and trained router:

```bash
vllm serve nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16 \
  --dtype bfloat16 --tensor-parallel-size 1 \
  --speculative-config '{
    "method": "mtp",
    "model": "/path/to/hierarchical-mtp-checkpoint",
    "num_speculative_tokens": 7,
    "use_local_argmax_reduction": true
  }'
```

NemotronH currently requires TP=1 and unquantized vocabulary weights shared with
the target. Gemma4 retains its existing embedding-sharing and vocabulary-weight
gathering behavior. The scorer requires floating-point vocabulary weights
matching the draft hidden-state dtype. Incompatible shapes and invalid vocabulary
permutations are rejected during loading.
