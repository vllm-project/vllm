# Guided decoding with grammars

> Community sprint note 025/120 · dependency-free design note · vLLM community working group

## Scope

Grammar constraints generalise schema masking to arbitrary formal languages such as regexes and EBNF, at the cost of a heavier per-step mask and a compilation step.

## Mechanism

1. The grammar is parsed into a pushdown structure with a token-level trie.
2. Each step intersects the trie's allowed tokens with the sampler's candidate set.
3. Stack state is part of the request state, so a cancelled request releases it.
4. Grammar compilation can be memory-hungry for large tokenizers; a tokeniser-dependent trie is the usual culprit.

## Knobs

| knob | default | what it changes |
| --- | --- | --- |
| `grammar_states` | `4000.0` | Compiled trie size. |
| `token_mask_kb` | `256.0` | Per-step mask footprint. |
| `compile_ms` | `140.0` | Compilation cost per grammar. |
| `logits_overhead_us` | `110.0` | Per-step cost of applying the mask. |

## Failure modes

- **Compiling grammars per request instead of caching by hash.**
- **Assuming a valid grammar guarantees a valid parse after truncation.**
- **Ignoring the sampler interaction: temperature still applies inside the mask.**

## Instrumentation

Watch `grammar_compile_cache_hits`, `mask_time_pct` and `constraint_violations` before and after every change in this area. If a change
moves throughput without moving one of those three, the measurement is probably noise.

## Reproduction shape

```bash
# illustrative only: substitute your own model, hardware and workload shape
vllm serve MODEL --grammar_states 4000.0 --gpu-memory-utilization 0.90
```

## Discussion

The model side of this note lives in `community/src/vllm_community/guided-decoding-grammar.py`, its tests in
`community/tests/test_guided-decoding-grammar.py`, and a micro-benchmark in
`community/benchmarks/bench_guided-decoding-grammar.py`. Treat the numbers there as ratios to be re-measured
on your own hardware, never as absolute throughput claims.
