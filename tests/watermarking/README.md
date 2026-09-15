# Watermarking goldens

These goldens freeze the CPU reference path: the generator and detector for both
watermarking algorithms and the Philox PRF, over a deterministic logit fixture.
They are a backwards-compatibility check, not a check of GPU numerics. A
position whose context already occurred is sampled without the watermark, and
the detector skips it too; one skipped because its context is still partial is a
stand-in the detector does score; and a generation-side `max_history` limit
makes the detector skip positions that were watermarked.

## Regenerating

```bash
python -m tests.watermarking.generate_goldens
```

Run it from the repository root. `--check` regenerates in memory, prints every
differing field and exits 1 without writing; `--candidate ID` regenerates only
the named ids. Never hand-edit the JSON. The `environment` block records Python,
torch, platform and CPU capability, and is never compared.

## Comparison policy

`score` and `p_value` are stored as hexadecimal floats, which round-trip
exactly, yet they and `p_value_ratio` are compared with a relative tolerance of
1e-9 (`GOLDEN_FLOAT_RTOL`): the detector reaches them via `log1p`, `exp` and
`gammaincc`, which differ by a unit in the last place between libm builds.
Everything else is exact: tokens, counts, booleans, configuration, resolved
state, trace.

## Fixture

Logits are `((token_id * 37 + position * 17) mod logit_modulus - 56) /
logit_denominator`, plus `dominant_bias` on `dominant_token` where a fixture
sets one. `REPETITIVE_FIXTURE` makes contexts repeat during generation, and its
bias is chosen against a rule: every repetitive row changes at least 8 tokens
under `key ^ 1`, makes at least one deduplication skip that is not a
partial-context skip, and detects at no more than half its threshold.

Routing between key A and key B uses a frozen 64-value low-discrepancy sequence,
so alpha 0.1 sends about 10% of positions to key B. No value in it sits within
1e-6 of a candidate's realised routing boundary, and no row's p-value sits
within 1e-6 of its threshold, since `is_watermarked` is recomputed live. For the
two near-threshold rows, near means 0.5 to 0.9 of the row's own
`p_value_threshold`, the knob they set, since it cannot move the p-value. Prompt
tokens only enter the deduplication history, never a context.

## Not covered

The speculative-decoding draft and target schedule, deferred to the watermarking
hardening tracker (#56105), and GPU kernels, covered by the parity tests.

## Files

`golden_candidates.py` holds the fixture, the matrix, the loader and the
comparison helper; `generate_goldens.py` regenerates the data; `test_goldens.py`
checks that every candidate reproduces; `watermarking_goldens.json` is the
frozen data; `__init__.py` makes the directory a package. The other four files
are pre-existing unit tests.
