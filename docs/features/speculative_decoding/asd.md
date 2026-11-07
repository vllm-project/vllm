# Approximate Speculative Decoding

Approximate Speculative Decoding (ASD) changes greedy speculative verification
to accept a bounded number of non-greedy draft tokens. This can commit a longer
contiguous suffix already scored by the target. It requires no training, weight
changes, or additional target forward passes. See the
[ASD paper](https://arxiv.org/abs/2608.03447) and
[reference implementation](https://github.com/Kissmetothemoon/ASD).

ASD changes target greedy outputs. Its budget bounds cumulative local logit
regret along the realized history; it does not guarantee task accuracy, semantic
equivalence, or preservation of a sampling distribution.

Enable Model Runner V2 and explicitly select the approximate verifier:

```bash
VLLM_USE_V2_MODEL_RUNNER=1 vllm serve TARGET_MODEL \
    --speculative-config '{
        "method": "dspark",
        "model": "DRAFT_MODEL",
        "num_speculative_tokens": 7,
        "rejection_sample_method": "approximate",
        "asd_risk_budget": 8.0,
        "asd_max_regret_per_value": 0.25,
        "asd_max_relaxed_tokens_per_block": 2
    }'
```

Send greedy requests (`temperature=0`). Stochastic requests use ordinary
rejection sampling and do not consume the ASD ledger. The controls above match
the paper's Qwen operating point; evaluate them for each target, draft, and
workload before deployment. They are not a measured vLLM performance claim.

For a proposed token at position `i`, the local regret is the target maximum
logit minus the target logit for that token. A mismatch is accepted only if:

- its regret divided by the remaining proposal length is within the local gate;
- the accepted mismatches in this block remain within the block cap;
- the request's cumulative regret remains within its total budget.

Verification stops at the first infeasible position and emits the target greedy
recovery token. Fully accepted blocks receive the ordinary bonus token. The
existing accepted-length and cache rollback paths retain the contiguous prefix.
The GPU ledger follows request identity across batch reordering and preemption;
new requests start with zero spent regret. Zero budget, zero gate, or zero cap
uses strict token-ID verification, including tied maximum logits.

ASD requires the standard MRV2 model sampler and an existing MRV2 proposer.
Watermarking, custom model samplers, and batch sharded sampling are unsupported.
Masked or non-finite mismatch scores cannot be relaxed.
Regret is measured after the existing sampling processors and masks are applied.

For evaluation, measure strict-ASD-strict triplets with equal output-token counts,
then audit task quality separately with natural EOS. Report model and draft
revisions, hardware, concurrency, sampling settings, throughput, latency,
accepted length, output lengths, and task scores. A higher acceptance rate alone
does not establish an end-to-end speedup.
