# Self-Spec N-gram Usage Guide

## Overview

The `offline_inference_self_spec.py` script now supports two self-speculative decoding methods:

1. **`self_specs`** (baseline): Original self-speculative decoding
2. **`self_spec_ngram`** (new): Self-spec with n-gram draft proposals for faster accumulation

## Key Parameters

### Self-Spec Threshold
```bash
--num_speculative_tokens N
```
- **Purpose**: Number of tokens to accumulate before transitioning ACCUMULATING → VERIFYING
- **Default**: 8
- **Used by**: Both `self_specs` and `self_spec_ngram`
- **Example**: `--num_speculative_tokens 8` means accumulate 8 tokens, then verify all 8 with full KV

### N-gram Draft Tokens (NEW)
```bash
--ngram_draft_tokens N
```
- **Purpose**: Number of draft tokens proposed by n-gram per ACCUMULATING step
- **Default**: 3
- **Used by**: Only `self_spec_ngram`
- **Example**: `--ngram_draft_tokens 3` means n-gram proposes 3 tokens each step

### Advanced N-gram Window Size (Optional)
```bash
--prompt_lookup_max N   # Max n-gram window size (default: uses ngram_draft_tokens)
--prompt_lookup_min N   # Min n-gram window size (default: uses ngram_draft_tokens)
```

## Usage Examples

### Example 1: Baseline Self-Spec
```bash
python offline_inference_self_spec.py \
    --dataset_name debug \
    --num_prompts 10 \
    --enable_sspec \
    --sspec_method self_specs \
    --num_speculative_tokens 8
```

**Behavior:**
- Accumulates 1 token per step (no n-gram)
- Takes ~8 steps to reach threshold
- Then verifies all 8 tokens with full KV

### Example 2: Self-Spec with N-gram (Default)
```bash
python offline_inference_self_spec.py \
    --dataset_name debug \
    --num_prompts 10 \
    --enable_sspec \
    --sspec_method self_spec_ngram \
    --num_speculative_tokens 8 \
    --ngram_draft_tokens 3
```

**Behavior:**
- N-gram proposes 3 draft tokens per step
- With 50% acceptance: ~2 tokens accepted per step
- Takes ~4 steps to reach threshold (2x faster!)
- Then verifies all 8 tokens with full KV

### Example 3: Aggressive N-gram Drafting
```bash
python offline_inference_self_spec.py \
    --dataset_name debug \
    --num_prompts 10 \
    --enable_sspec \
    --sspec_method self_spec_ngram \
    --num_speculative_tokens 16 \
    --ngram_draft_tokens 5
```

**Behavior:**
- Higher threshold (16 tokens)
- More aggressive drafting (5 tokens per step)
- Potentially reaches threshold in ~3-5 steps

### Example 4: Custom N-gram Window
```bash
python offline_inference_self_spec.py \
    --dataset_name debug \
    --num_prompts 10 \
    --enable_sspec \
    --sspec_method self_spec_ngram \
    --num_speculative_tokens 8 \
    --prompt_lookup_min 2 \
    --prompt_lookup_max 4
```

**Behavior:**
- Variable n-gram window: 2-4 tokens
- Allows more flexible matching

## Parameter Relationship

```
┌─────────────────────────────────────────────────────────────┐
│ ACCUMULATING Phase                                           │
├─────────────────────────────────────────────────────────────┤
│ Step 1: N-gram proposes <ngram_draft_tokens> drafts         │
│         → Accept some (e.g., 2/3)                            │
│         → pending_output_tokens += accepted                  │
│                                                              │
│ Step 2: N-gram proposes <ngram_draft_tokens> drafts         │
│         → Accept some (e.g., 3/3)                            │
│         → pending_output_tokens += accepted                  │
│                                                              │
│ Step N: len(pending_output_tokens) >= <num_speculative_tokens>│
│         → TRANSITION TO VERIFYING                            │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ VERIFYING Phase                                              │
├─────────────────────────────────────────────────────────────┤
│ Verify all <num_speculative_tokens> with full KV            │
│ → Accept X tokens, reject rest                              │
│ → Back to ACCUMULATING with remaining tokens                │
└─────────────────────────────────────────────────────────────┘
```

## Tuning Recommendations

### For Repetitive Text (e.g., code, math)
```bash
--num_speculative_tokens 16    # Higher threshold (more accumulated tokens)
--ngram_draft_tokens 5         # Aggressive drafting
```
- N-gram works well on repetitive patterns
- Larger batches amortize verification cost

### For Creative Text (e.g., stories)
```bash
--num_speculative_tokens 8     # Lower threshold
--ngram_draft_tokens 2         # Conservative drafting
```
- N-gram acceptance may be lower
- Smaller batches reduce wasted computation

### For Mixed Workloads
```bash
--num_speculative_tokens 8     # Balanced
--ngram_draft_tokens 3         # Default
```
- Good starting point
- Monitor acceptance rate and adjust

## Performance Comparison

Run both methods side-by-side:

```bash
# Baseline
python offline_inference_self_spec.py \
    --dataset_name debug \
    --num_prompts 50 \
    --enable_sspec \
    --sspec_method self_specs \
    --num_speculative_tokens 8

# With n-gram
python offline_inference_self_spec.py \
    --dataset_name debug \
    --num_prompts 50 \
    --enable_sspec \
    --sspec_method self_spec_ngram \
    --num_speculative_tokens 8 \
    --ngram_draft_tokens 3
```

Compare:
- Generation time
- Mean acceptance length
- Acceptance rate per position

## Command Line Help

```bash
python offline_inference_self_spec.py --help
```

Key arguments:
```
  --enable_sspec                Enable self-speculative decoding
  --sspec_method {self_specs,self_spec_ngram}
                                Self-speculative method
  --num_speculative_tokens N    Self-spec threshold (default: 8)
  --ngram_draft_tokens N        N-gram drafts per step (default: 3)
  --prompt_lookup_max N         Max n-gram window (optional)
  --prompt_lookup_min N         Min n-gram window (optional)
```

## Expected Output

When running with `--sspec_method self_spec_ngram`, you should see:

```
======================================================================
SELF-SPECULATIVE DECODING CONFIGURATION
======================================================================
Method: self_spec_ngram
Self-spec threshold (ACCUMULATING → VERIFYING): 8 tokens
N-gram drafts per step: 3 tokens (max)
                        3 tokens (min)

Expected behavior:
  - During ACCUMULATING: N-gram proposes ~3 drafts/step
  - Reaches threshold in ~2 steps (with good acceptance)
  - Then VERIFYING: All 8 tokens verified with full KV
Sparse attention config: sink_size=32, recent_ratio=0.05
======================================================================
```

## Troubleshooting

### N-gram acceptance is very low
- Try reducing `--ngram_draft_tokens` to 1 or 2
- Check if your prompts have repetitive patterns
- Consider using baseline `self_specs` instead

### Generation is slower with n-gram
- Ensure `--ngram_draft_tokens` < `--num_speculative_tokens`
- Try reducing n-gram drafts
- Check acceptance metrics

### Memory issues
- Reduce `--num_speculative_tokens`
- Reduce `--max_num_seqs`
- Reduce `--max_num_batched_tokens`
