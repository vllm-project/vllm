# Adaptive Verifier Step Length (Verify Adaptive)

For parallel speculative decoding (e.g. DFlash), the drafter still proposes
`num_speculative_tokens` draft tokens every step (e.g. 15), but the **verifier
only checks a batch-adaptive subset** of those drafts on the next step. The
controller trades expected accepted length against measured verifier ITL using a
discrete marginal-gain scan.

## Enabling

```bash
vllm serve <target_model> \
  --spec-method dflash \
  --spec-model <dflash_draft_model> \
  --spec-tokens 15 \
  --speculative-adaptive-verify-config /path/to/verify_adaptive_config.json
```

Equivalent via the speculative JSON blob:

```bash
vllm serve ... \
  --speculative-config '{"method":"dflash","model":"...","num_speculative_tokens":15,"speculative_adaptive_verify_config":"/path/to/config.json"}'
```

- The controller is only constructed for parallel speculative methods:
  `method=dflash`, or `method=draft_model` with `parallel_drafting=true`
  (PARD). Other spec methods log a warning and ignore the path.
- Example JSON: `verify_adaptive_config.example.json` in this directory.
- Field reference: `VerifyAdaptiveConfig` in `verify_adaptive_config.py`.

## Runtime flow

```text
Step N
  start of execute_model
    → read cached draft_len[req] from step N-1
    → truncate scheduled_spec_decode_tokens
      (draft_len=0 removes the spec entry for that request)
  verifier forward (truncated query lengths)
  propose_draft_token_ids
    → DFlash still drafts full num_speculative_tokens
    → selected_probs[B, T] (softmax prob of each sampled draft token)
    → choose_query_lens_discrete → cache draft_len[req] for step N+1

Step N+1
  start of execute_model applies the cached draft_len …
```

## Cost model and warmup

Verifier latency is profiled as:

```text
cost = ITL(batch_size, sum_query_len)
sum_query_len = Σ_i query_lens[i] = B + Σ_i draft_lens[i]
```

After JIT / CUDA-graph warmup (`compile_or_warm_up_model`), `_adaptive_profile_run` fills
`cost_table[(bs, bs * ql)]` at discrete points:

- **Batch axis**: explicit `warmup_batch_sizes`, or step-2 from
  `min_warmup_batch_size` to `max_warmup_batch_size`.
- **Per-request query_len axis**: from `min_query_len_per_req` in steps of
  `query_len_step_per_req`, capped at `max_query_len_per_req` (default:
  `num_speculative_tokens + 1`).
- **Sequence context length**: controlled by `warmup_seq_lens` (default 1024).
  Set this to a value representative of your production sequence length so
  that FlashAttention kernel cost is realistic during profiling.

## Core algorithm (each decode step)

**Inputs**

- `probs[i][t]`: accept-probability proxy for active (post-prefill) sequence
  `i`, draft position `t+1` (softmax at the sampled token).
- `B`: full batch size (every sequence contributes one anchor token).
- Candidate `Q` values: profiled `sum_query_len` keys for `bs_key` (ceil of
  `B` on the batch axis), e.g. `32*2, 32*4, …, 32*16`.

**Marginal gains**

```text
m[i,t] = p[i,1] * p[i,2] * … * p[i,t]
```

Within each sequence, `m` is non-increasing, so a global top-`S` pick satisfies
the prefix constraint.

**For each candidate Q**

```text
S = Q - B
numerator = B + sum(top-S marginal gains)
denominator = cost_table[(bs_key, Q)]
score = numerator / denominator
```

Pick the `Q` with the best score, map the top-`best_S` marginals to per-request
`draft_len[i]`, and cache for the next step’s truncation.

**Relation to full draft length**

- DFlash always drafts `num_speculative_tokens` (e.g. 15).
- Adaptation only changes how many drafts the **verifier** checks next step
  (0–15).
- `query_len[i] = 1 + draft_len[i]` (1 anchor + drafts).

## Code map

| Module | Role |
|--------|------|
| `verify_adaptive_config.py` | JSON / dict configuration |
| `verify_adaptive_controller.py` | Cost table, algorithm, per-request cache |
| `llm_base_proposer.py` | `needs_draft_probs`, `_gather_selected_probs` |
| `gpu_model_runner.py` | Truncation, `process_draft_output`, profiling hook |
| `gpu_worker.py` | Calls profiling after warmup |

Full math and pseudocode: `核心算法说明.md` at the repository root.
