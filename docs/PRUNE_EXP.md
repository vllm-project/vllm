# What RIY Prunes — and What It Doesn't

## Pruned: Routed MoE Experts Only

RIY prunes **routed expert FFNs** — the per-token-selected feed-forward
networks inside Mixture-of-Experts layers. These are the weights addressed
by `FusedMoE` / `SharedFusedMoE` via `expert_map`:

- `w13_weight` (fused gate + up projection)
- `w2_weight` (down projection)
- Associated scales, zeros, biases if quantized

Each expert in each layer is a **unique FFN with unique weights**.
Expert 42 in layer 5 has completely different weights than expert 42
in layer 20. RIY prunes per `(layer, expert)` pair — different experts
can be pruned in different layers.

## Not Pruned

Everything else in the model remains untouched:

| Component | Why not pruned |
|-----------|---------------|
| **Shared experts** | Always active for every token — no routing, no selection. Critical for base quality. |
| **Router / Gate** | The linear layer that produces expert logits stays full-size. RIY adds a logit mask (`-inf` for pruned experts) but does not remove gate weights. |
| **DeltaNet / Gated Linear Attention** | These are attention mechanisms, not MoE experts. They process every token regardless of routing. |
| **Standard Attention** (Q/K/V/O) | Not part of MoE. Full attention layers are untouched. |
| **RMSNorm / LayerNorm** | Normalization layers — always active. |
| **Embeddings** | Token and position embeddings — always active. |
| **LM Head** | Output projection — always active. |
| **MTP layers** | Multi-Token Prediction layers have their own experts and attention. Not affected by the main model's prune profile. |
| **Vision encoder** | For VL models — completely separate from the MoE text backbone. |

## Memory Impact

For a typical MoE model, routed expert weights make up **80-95%** of
total model parameters. The remaining 5-20% (attention, norms, embeddings,
shared experts, gate) cannot be pruned.

This is why 20% expert pruning yields ~19% VRAM savings (not exactly 20%),
and 50% expert pruning yields ~40% savings.

### Example: Qwen3.5-397B-A17B INT4

| | Experts | Other | Total |
|---|---|---|---|
| Full model | ~90 GiB | ~8 GiB | ~98 GiB |
| 20% pruned | ~72 GiB | ~8 GiB | ~80 GiB |
| 50% pruned | ~45 GiB | ~8 GiB | ~53 GiB |

### Example: Qwen3.5-35B-A3B INT4

| | Experts | Other | Total |
|---|---|---|---|
| Full model | ~17 GiB | ~2.3 GiB | ~19.3 GiB |
| 50% pruned | ~8.5 GiB | ~2.3 GiB | ~11.6 GiB |

## How the Pruning Works

```
                        Router (full 512 logits)
                           |
                    logit_mask: -inf for pruned
                           |
                      Top-K selection
                     (pruned never selected)
                           |
                    expert_map[logical_id]
                     → physical_id (compact)
                     → -1 (pruned, skipped)
                           |
                   weights[physical_id]
                  (smaller tensor, VRAM saved)
```

1. **Gate produces logits** for all original experts (e.g., 512)
2. **Logit mask** adds `-inf` to pruned expert logits before Top-K
3. **Top-K selects** only from non-pruned experts
4. **expert_map** translates logical expert IDs to compact physical indices
5. **Weight tensors** are allocated with `local_num_experts` (smaller)
6. **Weight loader** skips pruned experts (`expert_map[id] == -1`)
