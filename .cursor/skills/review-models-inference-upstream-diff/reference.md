# Reference: Models & Inference Diff Review

## Extract documented paths from the doc

```bash
rg -o '`vllm/[^`]+`|`examples/[^`]+`' \
  docs/cohere/code_notes/upstream-diff-models-and-inference.md \
  | tr -d '`' | sort -u
```

## Quick existence check

```bash
while IFS= read -r p; do
  if [ -e "$p" ]; then echo "OK   $p"
  else echo "MISS $p"; fi
done < paths.txt
```

## Find successor after upstream split (example: pooler)

```bash
# Old doc path
test -e vllm/model_executor/layers/pooler.py || echo "restructured"

# New package
ls vllm/model_executor/layers/pooler/

# Symbols that should survive a split
rg -n 'DispatchPooler|PoolerClass|PoolerHead' vllm/model_executor/layers/pooler/
```

## Section → primary symbols

| § | Grep anchors |
|---|-------------|
| 1 Pooling | `DispatchPooler`, `Pooler`, `pooler` task routing |
| 2 Reward | `Cohere2ForRewardModel`, `CohereForRewardModel`, `PoolingModel` |
| 3 EAGLE | `eagle_draft_model`, `EagleDraft`, draft weight prefix |
| 4 Guided decoding | `structural_tag`, `tool_grammar`, `get_text_model_name` |
| 5 Spec decode | `ngram-eagle`, `mean acceptance length`, `custom_mm` |
| 6 MoE | `SigmoidRenorm`, `token_choice_with_bias`, `norm_topk_prob` |
| 7 Perf | `compressed_tensors`, `marlin`, kernel benchmark MoE |
| 8 Hotspots | files listed in doc §8 |

## Common v0.21 restructures (watch list)

| Doc may say | Current branch often has |
|-------------|-------------------------|
| `pooler.py` | `pooler/` package (`abstract.py`, `seqwise/`, `tokwise/`, `special.py`) |
| `quantization/mxfp8.py` | `quantization/online/mxfp8.py` |
| `compressed_tensors_moe.py` | `compressed_tensors_moe/` package (`compressed_tensors_moe.py` router, `*_wna16_marlin.py`, …) |
| `experts_int8.py` monolith | `online/int8.py` + oracle backends |
| `commandr.py` only | also `cohere2_moe.py` for c5 MoE / LoRA |

## Split-file porting check (Step 2b)

Full procedure:
[`../_shared/split-file-porting-check.md`](../_shared/split-file-porting-check.md).

Quick MoE example (int4 regression class):

```bash
UPSTREAM_REF=v0.21.0
LEGACY=vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe.py
PKG=vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe

# Cohere hunks: legacy vs package
git diff "$UPSTREAM_REF" -- "$LEGACY" | grep -E 'cohere start|cohere end|# cohere' || true
git diff "$UPSTREAM_REF" -- "$PKG/" | grep -E 'cohere start|cohere end|# cohere' || true

# Who imports MoE dispatch?
rg -n 'compressed_tensors_moe import' vllm/model_executor/layers/quantization/compressed_tensors.py

# Duplicate entry points?
rg -l 'def get_moe_method|_normalize_weight_actorder' "$LEGACY" "$PKG/"
```

## Fork diff scope command

```bash
UPSTREAM_REF=v0.21.0  # confirm with user
git diff --name-status "$UPSTREAM_REF" -- \
  vllm/model_executor/models/commandr.py \
  vllm/model_executor/models/commandr_eagle.py \
  vllm/model_executor/models/cohere_reward.py \
  vllm/model_executor/layers/pooler \
  vllm/cohere/guided_decoding \
  vllm/v1/core/kv_cache_utils.py
```

## Doc update checklist

When updating `upstream-diff-models-and-inference.md`:

- [ ] Replace stale single-file paths with current package paths
- [ ] Update §8 hotspot list
- [ ] Add new sections for major undocumented customizations (if user approved)
- [ ] Refresh validation checklist commands/paths
- [ ] Keep section numbering contiguous or add a short "§N (new)" note
- [ ] Cross-link to `models-and-inference.md` for deep dives instead of duplicating
