# Reading aux outputs in the trainer

The reader library is the one supported way to read a rollout's aux outputs from the store. It is `oc_reader.py` under `tools/rl_aux_output/` on the implementation branch (README, Status). It depends on numpy and zstandard; `ml_dtypes` is optional and only needed to get bf16 hidden states as numbers.

It is for store mode; inline mode returns R3 rows with each completion (protocol §12). It reads manifests of the full-depth build only: a manifest from the earlier partial-capture build is rejected as a failed sample (`error` names it). This contract covers R3, the indexer and hidden states. Top-k logprobs are stored like hidden states (protocol §1); their reader support is part of the implementation branch and returns, per request, `[generated length, 128]` int32 ids and `[generated length, 128]` fp32 logprobs for positions `[hidden_start, n_rows)`. The sampling mask is not in the reader.

```python
sample = read_sample(get, manifest_key, n_rows, want=("r3", "indexer", "hidden"), strict=False,
                     exists=None, store=None)
```

| Argument | Meaning |
|---|---|
| `get(key) -> bytes or None` | Your store client's get. Return `None` (or empty bytes) for an absent key. |
| `exists`, `store` | The existence check for every object the manifest lists, in one batched call: `exists(keys) -> list` (1/True = present) if given, else `batch_is_exist` of the store `get` is bound to, else `store=` (its `batch_is_exist`, or `is_exist` per key). Objects are never downloaded to test existence; with none of these `read_sample` raises `ValueError`. |
| `manifest_key` | The `aux.manifest.key` of the response. Object keys come from the manifest's `key_prefix` and `tail_key` (older manifests: derived from this key, which then must not have `/` in the request id). |
| `n_rows` | `L - 1` for a sample of `L` tokens (prompt + generated). The manifest can cover more rows (async scheduling, truncated drafts); they are cut off. |
| `want` | The outputs you requested. A wanted output the manifest lacks is reported missing. |
| `manifest` | Optional: the manifest already fetched (dict or bytes). |
| `strict` | Raise `ReadError` instead of returning a failed or incomplete sample. With `strict=False` nothing in the stored data raises; it is reported in the sample. |

Read the manifest of the request whose tokens you train on. For a multi-turn trajectory trained as one sequence that is the last turn: its manifest covers every row from 0. A training sequence that is a shorter prefix (one turn, one segment of a compacted rollout) is read with its own request's manifest (protocol §7).

## What you get back

All arrays have `n_rows` rows; row `p` is position `p` (the position that produces the logits for token `p + 1`).

| Field | Type | Meaning |
|---|---|---|
| `ok` | bool | No error and every `missing.counts` entry is 0. |
| `error` | str or None | The request failed. Its manifest is absent (with an error marker: `error marker: ...`), or an object the manifest lists is absent (`listed object absent: <key>`) or malformed (an unknown codec, a header or size that does not fit the field). A failed store write fails the whole request; the sample is never delivered in part. A marker beside a manifest whose listed objects all exist is stale: it is ignored, and `stale_marker` is True. On error all arrays are `None`. |
| `missing.counts` | dict | Rows not delivered for reasons other than a failed request, per cause: `rows_short` (the manifest covers fewer than `n_rows` rows), `invalid_rows` (rows from the manifest's `invalid_from`, protocol §6.8), `r3_field` / `indexer_field` / `hidden_field` (a wanted output the manifest lacks), and rows that fail the R3 sanity check. |
| `missing.rows` | bool `[n]` | Rows some wanted output did not deliver. |
| `r3` | `[n, layers, k]` uint | Expert ids to replay, every MoE layer. Rows in `missing.rows` hold `r3_no_replay` (dtype max, 65535 for uint16). With one-byte ids 255 is a valid id: go by the mask, never by the value. |
| `r3_replay_mask` | bool `[n, layers]` | True on every cell of a delivered row, False on rows in `missing.rows`. |
| `indexer` | int32 `[n, idx_layers, topk]` | Top-k indices to replay, in engine order: ascending positions, `-1` pads last. `-1` is a pad entry inside a delivered cell; `-2` fills rows in `missing.rows`. |
| `indexer_layers` | list | Model layer of each indexer slot. |
| `indexer_replay_mask` | bool `[n, idx_layers]` | As for R3. |
| `hidden` | `[m, d]` | The `lm_head` input for positions `hidden_start + i`. Codec `raw`: the exact stored values in the manifest's dtype, float32 on DeepSeek-V4.1-Flash; for bf16 without `ml_dtypes` installed, uint16 bit patterns (view them as bfloat16 in torch). |
| `hidden_start` | int | First position with a hidden row: `P - 1` of this request. |
| `weight_version` | int64 `[n]` | Weight version of each row, from the manifest's segments; `-1` = unknown. |
| `version_unknown`, `version_unknown_count` | bool `[n]`, int | Delivered rows whose weight version is unknown. They stay replayable and do not make `ok` False. |

## Guarantees

- Every cell of a delivered row holds the value of the first computation of that position in the trajectory, at every layer, labelled with the row's weight version. Nothing is substituted, padded or guessed.
- A request whose store writes failed, or whose listed objects are absent or malformed, is reported through `error`, never as a partial sample.
- Every other row that was not delivered is in `missing.rows`, with a cause in `missing.counts`. On those rows every replay mask is False and the data holds the sentinel.
- Replayed R3 cells pass the sanity check (distinct ids, no sentinel for ids wider than one byte, no all-zero row); a row that fails is in `missing.rows`.
- Codecs are validated strictly; nothing is decoded on a guess. Indexer objects must carry the manifest's codec tag. Hidden states must name codec `raw`, and every chunk is checked against the field's dtype and row width before any size is computed from it. An unknown codec or a mismatch fails the request.
- Decoded indexer cells are byte-identical to the engine's: ascending positions, `-1` pads last.

## What the trainer must do

1. **Failed or incomplete sample.** If `not sample.ok`, drop the sample. Reads fail collectively: all ranks all-reduce a success flag and retry or skip the step together (protocol §7).
2. **Replay.** Replay every cell of a delivered row. For R3, compare expert sets; where the set is unchanged keep your own ids and weights. Hook replay where the model computes routing. Never gather at the sentinel.
3. **Versions.** Rows of different versions are not interchangeable. Use `weight_version` for staleness weighting or filtering. Rows in `version_unknown` can still be replayed, but their version must not feed importance weights or freshness filters: treat them as stale (worst-case version) or drop them from any version-dependent term.
4. **Hidden states** cover only this request's generated positions `[hidden_start, n_rows)`. For earlier turns read their own manifests. The rows are the exact `lm_head` input and can feed the LM head directly. Do not recompute the behavior logprob from them; use the inline sampled logprob.
5. **Release** the session (`<job>/<session>/`) after the step commits.

## Example

```python
from oc_reader import read_sample

def load(client, ref, tokens):
    s = read_sample(client.get, ref["key"], len(tokens) - 1, want=("r3", "indexer"))
    if not s.ok:
        return None                     # caller all-reduces and skips/retries together
    r3 = torch.from_numpy(s.r3.astype(np.int64))
    idx = torch.from_numpy(s.indexer)
    return r3, idx, torch.from_numpy(s.weight_version)
```
