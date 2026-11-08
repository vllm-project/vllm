# Reading aux outputs in the trainer

The reader library (`oc_reader.py`, to be added next to the connector) is the one supported way to read a rollout's aux outputs. It depends on
numpy and zstandard only.

```python
sample = read_sample(get, manifest_key, n_rows, want=("r3", "indexer", "hidden"), strict=False)
```

| Argument | Meaning |
|---|---|
| `get(key) -> bytes or None` | Your store client's get. Return `None` (or empty bytes) for an absent key. It is the only store call. |
| `manifest_key` | The `aux.manifest.key` of the response. Object keys come from the manifest's `key_prefix` and `tail_key` (older manifests: derived from this key, which then must not have `/` in the request id). |
| `n_rows` | `L - 1` for a sample of `L` tokens (prompt + generated). The manifest can cover more rows (async scheduling, truncated drafts); they are cut off. |
| `want` | The outputs you requested. A wanted output the manifest lacks is reported missing. |
| `manifest` | Optional: the manifest already fetched (dict or bytes). |
| `strict` | Raise `ReadError` instead of returning an incomplete sample. With `strict=False` nothing in the stored data (missing, malformed or mismatched objects) raises; it is reported in the sample. |

Read the manifest of the request whose tokens you train on. For a multi-turn trajectory that is
the last turn: its manifest covers every row from 0.

## What you get back

All arrays have `n_rows` rows; row `p` is position `p` (the position that produces the logits
for token `p + 1`).

| Field | Type | Meaning |
|---|---|---|
| `ok` | bool | No error and every `missing.counts` entry is 0. |
| `error` | str or None | Error marker found, or manifest absent. All arrays are `None`, every row is missing. |
| `missing.counts` | dict | Per cause: `r3_blocks`, `r3_tail`, `r3_cone_patches`, `r3_invalid_rows`, `r3_needed_missing`, `r3_field`, `indexer_blocks`, `indexer_tail`, `indexer_cone_patches`, `indexer_reported_missing`, `indexer_reported_holes`, `indexer_needed_missing`, `hidden_chunks`, `hidden_rows_short`, `rows_short`, `invalid_rows`, `*_field`, ... |
| `missing.rows` | bool `[n]` | Rows whose data was not delivered by some wanted output. |
| `r3` | `[n, layers, k]` uint | Expert ids to replay. Where not replayed the cell holds `r3_no_replay` (dtype max, 65535 for uint16). With one-byte ids 255 is a valid id: always go by the mask, never by the value. |
| `r3_replay_mask` | bool `[n, layers]` | True: replay this cell's ids. False: the cell holds the sentinel. |
| `r3_version` | int64 `[n, layers]` | Weight version of each cell. `-1` means *unknown version*: on a replayed cell (mask True) the cell is valid but its version was not recorded; on a non-replayed cell the value has no meaning. Always read it together with the mask. |
| `indexer` | int32 `[n, idx_layers, topk]` | Top-k indices to replay; `-1` is a pad entry inside a delivered cell, `-2` means not replayed. |
| `indexer_layers` | list | Model layer of each indexer slot. |
| `indexer_replay_mask`, `indexer_version` | `[n, idx_layers]` | As for R3. |
| `hidden` | float32 `[m, d]` | Dequantized `lm_head` input for positions `hidden_start + i`. NaN rows are missing. |
| `hidden_fp8`, `hidden_scale` | uint8 `[m, d]`, float32 `[m]` | The stored e4m3 codes and row scales, for decoding on the GPU. |
| `hidden_start` | int | First position with a hidden row: `P - 1` of this request. |
| `weight_version` | int64 `[n]` | Version of each row's shallow cells (from the manifest's segments); `-1` = unknown. |
| `version_unknown`, `version_unknown_count` | bool `[n]`, int | Delivered rows (not in `missing.rows`) whose weight version, or the version of one of their replayed cells, is unknown. They stay replayable and do not make `ok` False. |
| `first_deep_layer` | int or None | Set under partial capture: layers from here up come from cone patches. |
| `r3_needed_missing`, `indexer_needed_missing` | bool `[n, layers]` | Deep cells some sampled token of the trajectory depends on that are not replayed (never computed exactly, or held by no patch). Computed from the manifest's per-request `ends`, for the whole trajectory. Excludes rows already in `missing.rows`. `None` for manifests without `ends`. |
| `uncovered` | dict | The manifest's own per-request counts, for information. Older manifests without `ends` report them as `r3_uncovered` / `indexer_uncovered` in `missing.counts`. |

## Guarantees

- A cell whose replay mask is True holds the value of the first computation of that position in
  the trajectory, with its weight version. Nothing is substituted, padded or guessed.
- Every row that was not delivered is in `missing.rows`, with a cause in `missing.counts`. On those
  rows every replay mask is False and the data holds the sentinel.
- Deep cells (partial capture) come only from the cone patches, overlaid in list order, each up to
  its end, first written wins. The block columns of deep layers are never used.
- Replayed cells, shallow and deep, pass the R3 sanity check (distinct ids, no sentinel for ids
  wider than one byte); a row that fails is in `missing.rows`.
- Codec tags of indexer objects are checked against the manifest's codec; a mismatch is reported
  missing, not decoded. Indexer patches are decoded by their own `idx_codec`.
- Every needed deep cell that is not replayed is in `*_needed_missing` (or its row is missing).
- A patch whose header (layers, start, dtype, cell shape, coverage) does not fit the field is
  reported missing and fills nothing; its needed cells then show in `*_needed_missing`.

## What the trainer must do

1. **Failed or incomplete sample.** If `not sample.ok`, the default is to drop the sample (protocol
   §7). Reads fail collectively: all ranks all-reduce a success flag and retry or skip the step
   together.
2. **Masking instead of dropping.** Allowed only if your loss can exclude rows: drop the loss on
   every token `p + 1` with `missing.rows[p]`, and on every later token whose deep cells could
   depend on a lost row (up to `1 + (last_layer - first_deep_layer) * 127` rows later on
   DeepSeek-V4.1). `indexer_reported_*` and pre-v54 `*_uncovered` counts name no position; drop
   the sample for them.
3. **No-replay cells.** Where a replay mask is False on a row that is not missing, run the model's
   own routing (or indexer) for that cell, cell by cell. Never gather at the sentinel. Two kinds:
   - *not needed* (`*_needed_missing` False): under partial capture most deep cells of prompt
     rows; no sampled token depends on them, so routing them yourself costs nothing.
   - *needed but missing* (`*_needed_missing` True): a sampled token depends on a cell nobody
     delivered; your own routing there can differ from the rollout's. `ok` is False. Drop the
     sample, or mask the loss of every token `t + 1` with a needed-missing cell at layer `L` in
     rows `[t - need(L), t)`, `need(L) = 1 + (last_layer - L) * (window - 1)` (`r3_cone`
     fields of the manifest).
4. **Replay.** For replayed R3 cells, compare expert sets; where the set is unchanged keep your
   own ids and weights. Hook replay where the model computes routing.
5. **Versions.** Rows of different versions are not interchangeable. Use `weight_version` and the
   per-cell versions for staleness weighting or filtering. Rows in `version_unknown` can still be
   replayed, but their version must not feed importance weights or freshness filters: treat them
   as stale (worst-case version) or drop them from any version-dependent term.
6. **Hidden states** cover only this request's generated positions `[hidden_start, n_rows)`. For
   earlier turns read their own manifests. Do not recompute the behavior logprob from them; use
   the inline sampled logprob.
7. **Release** the session (`<job>/<session>/`) after the step commits.

## Example

```python
from oc_reader import read_sample

def load(client, ref, tokens):
    s = read_sample(client.get, ref["key"], len(tokens) - 1, want=("r3", "indexer"))
    if not s.ok:
        return None                     # caller all-reduces and skips/retries together
    r3 = torch.from_numpy(s.r3.astype(np.int64))
    r3_mask = torch.from_numpy(s.r3_replay_mask)          # False: route yourself
    idx = torch.from_numpy(s.indexer)
    idx_mask = torch.from_numpy(s.indexer_replay_mask)    # False: run your own indexer
    return r3, r3_mask, idx, idx_mask, torch.from_numpy(s.weight_version)
```
