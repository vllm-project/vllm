# Fine-Grained Tensor Parallelism

`--fine-grained-tp-config` assigns module-specific tensor-parallelism overrides
along the `DP x PCP x TP` hierarchy (`TP` -> `PCP` -> `DP`) while keeping other
layers on their global parallel layouts.

Large-vocabulary MoE deployments often run with small global `TP` and wide `DP`
or `PCP`. Standard vocabulary parallelism shards `embed_tokens` and `lm_head`
only across `TP`, replicating both matrices on every `DP` and `PCP` rank.
Fine-grained TP expands vocabulary sharding orthogonally across `PCP * TP` or
`DP * PCP * TP` devices.

## Configuration

All knobs live under `--fine-grained-tp-config`:

| Field | Type | Default | Allowed Values (`P = PCP`, `D = DP`) | Description |
| --- | --- | --- | --- | --- |
| `embedding_tp_size` | `int` | `1` | `{1, P, D * P}` | Fine-grained TP multiplier on top of `TP` for token embeddings |
| `lmhead_tp_size` | `int` | `1` | `{1, P, D * P}` | Fine-grained TP multiplier on top of `TP` for the LM head |

Given $T = \text{tensor\_parallel\_size}$, $P = \text{prefill\_context\_parallel\_size}$,
and $D = \text{data\_parallel\_size}$, setting a module's `*_tp_size` to $F$
yields $T \cdot F$ contiguous vocabulary shards:

- `F = 1`: shard across `TP` only ($D_{\text{fg}} = 1, P_{\text{fg}} = 1$).
- `F = P`: shard across `PCP * TP` ($D_{\text{fg}} = 1, P_{\text{fg}} = P$).
- `F = D * P`: shard across `DP * PCP * TP` ($D_{\text{fg}} = D, P_{\text{fg}} = P$).

For example, in a `DP=4, PCP=2, TP=2` deployment:

- `*_tp_size = 1` shards across `TP=2` (`2` shards per layer).
- `*_tp_size = 2` shards across `PCP=2 * TP=2` (`4` shards per layer).
- `*_tp_size = 8` shards across `DP=4 * PCP=2 * TP=2` (`16` shards per layer).

For vocabulary size $V$, hidden size $H$, and element size $b$:

- Token embedding occupies approximately $\frac{V \cdot H \cdot b}{T \cdot F_{\text{emb}}}$ bytes per device.
- Untied LM head occupies approximately $\frac{V \cdot H \cdot b}{T \cdot F_{\text{lm}}}$ bytes per device.
- When `tie_word_embeddings` is enabled, `embedding_tp_size` and
  `lmhead_tp_size` must match so both layers share identical contiguous shards.

## Execution Model

Vocabulary shards are linearized over `TP x PCP x DP` with `TP` outer, `PCP`
middle, and `DP` inner:

$$\text{shard\_rank} = (t \cdot P_{\text{fg}} + p) \cdot D_{\text{fg}} + d, \qquad \text{shard\_world\_size} = T \cdot P_{\text{fg}} \cdot D_{\text{fg}}$$

### Token Embedding

1. **Row exchange over `PCP x DP`** ($P_{\text{fg}} \cdot D_{\text{emb}}$ ranks
   at fixed `TP` column): token IDs are gathered across independent `PCP` and
   `DP` row owners, looked up in each rank's local vocabulary shard, and
   reduce-scattered back to their owners, then cropped to the local token
   count. Under CUDA graphs or `PCP`-only sharding, ranks exchange the exact
   synchronized batch length without extra padding; in eager `DP` execution,
   the exchange span uses the step's synchronized `DP` token bound.
2. **Vocabulary sum over `TP`** ($T$ ranks): because `TP` peers hold identical
   token IDs, only the cropped local rows are all-reduced across `TP`.

### LM Head

Sampling runs after `PCP` restores global request hidden states, so ranks
within the same `DP` replica hold identical sampling rows while distinct `DP`
replicas hold independent request batches:

1. **`DP` exchange** (when $D_{\text{lm}} > 1$): hidden states are padded to
   the step-agreed `DP` request capacity, all-gathered across `DP`, projected
   against the local vocabulary shard, redistributed via `all_to_all_single`,
   and cropped back to each rank's active request rows. Idle `DP` ranks
   participate with zero-row slices so collectives stay aligned.
2. **`PCP` vocabulary gather** (when $P_{\text{lm}} > 1$): active request
   logits are all-gathered along the vocabulary dimension across `PCP`.
3. **`TP` vocabulary gather** (when $T > 1$): active request logits are
   gathered along the vocabulary dimension across `TP` in natural token-ID
   order.

## Limitations

- Requires unquantized vocabulary weights (`embed_tokens` and `lm_head`).
- Cross-`DP` sharding ($D_{\text{emb}} > 1$ or $D_{\text{lm}} > 1$) requires a
  MoE model.
- LoRA, speculative decoding, multimodal inputs, prompt embeddings,
  batch-sharded sampling, microbatching, and elastic EP are not supported when
  fine-grained vocabulary sharding across `DP` or `PCP` is active.
- Pooling runners are not supported when `lmhead_tp_size > 1`.
- When `lmhead_tp_size` expands into `DP` ($D_{\text{lm}} > 1$), prompt
  logprobs, local top-k/argmax vocabulary reductions, and async-scheduled
  pipeline parallelism are not supported. (`PCP * TP` sharding and
  embedding-only `DP` sharding keep sampling `DP`-local and continue to support
  prompt logprobs and local top-k/argmax reductions.)

## Examples

Shard vocabulary weights across `PCP * TP` (`2 * 2 = 4` shards per layer) in a
`DP=4, PCP=2, TP=2` deployment:

```bash
vllm serve MODEL \
    --data-parallel-size 4 \
    --prefill-context-parallel-size 2 \
    --tensor-parallel-size 2 \
    --enable-expert-parallel \
    --fine-grained-tp-config '{"embedding_tp_size": 2, "lmhead_tp_size": 2}'
```

Shard vocabulary weights across the full `DP * PCP * TP` domain (`4 * 2 * 2 = 16`
shards per layer):

```bash
vllm serve MODEL \
    --data-parallel-size 4 \
    --prefill-context-parallel-size 2 \
    --tensor-parallel-size 2 \
    --enable-expert-parallel \
    --fine-grained-tp-config.embedding-tp-size 8 \
    --fine-grained-tp-config.lmhead-tp-size 8
```
