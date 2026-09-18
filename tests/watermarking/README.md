# Watermarking goldens

These goldens freeze the CPU reference path: the generator and detector for each
configured watermarking algorithm and PRF combination, over a deterministic
logit fixture. The current matrix uses the Philox PRF. Regeneration refuses an
incomplete algorithm/PRF matrix.
They are a backwards-compatibility check, not a check of GPU numerics. A
position whose context already occurred is sampled without the watermark, and
the detector skips it too; one skipped because its context is still partial is a
stand-in the detector does score; and a generation-side `max_history` limit
makes the detector skip positions that were watermarked.

## Updating the contract

A failing golden means the output contract changed and must be investigated.
Do not regenerate the snapshot merely to make the test pass. Updating the
existing contract is appropriate after review when a correctness fix preserves
reliable detection of generations emitted by the previous implementation, with
materially equivalent calibration. The size of the golden diff is useful
evidence, but compatibility is the deciding criterion.

If a fix prevents reliable detection of previous generations or materially
changes their calibration, introduce it as a new version of the scheme or PRF.
Generate and commit goldens for the new version, and keep the previous version's
detector and goldens alongside them so both contracts remain covered and
previous generations remain detectable.

When a reviewed compatible change requires updating the existing contract,
rebuild the snapshot from the repository root:

```bash
python -m tests.watermarking.generate_goldens
```

`--check` evaluates the current implementation in memory, prints every differing
field and exits 1 without writing; `--candidate ID` updates only the named ids.
Never hand-edit the JSON. The `environment` block records Python, torch, platform
and CPU capability, and is never compared.

## Comparison policy

`score` and `p_value` are stored as hexadecimal floats, which round-trip
exactly, yet they and `p_value_ratio` are compared with a relative tolerance of
1e-9 (`GOLDEN_FLOAT_RTOL`): the detector reaches them via `log1p`, `exp` and
`gammaincc`, which differ by a unit in the last place between libm builds.
Everything else is exact: tokens, counts, booleans, configuration, resolved
state, trace. Fixture-quality guards run in both pytest and the regeneration
script before it writes the file.

## Fixture

Logits are `((token_id * 37 + position * 17) mod logit_modulus - 56) /
logit_denominator`, plus `dominant_bias` on `dominant_token` where a fixture
sets one. `REPETITIVE_FIXTURE` makes contexts repeat during generation, and its
bias is chosen against a rule: every repetitive row changes at least 8 tokens
under `key ^ 1`, makes at least one deduplication skip that is not a
partial-context skip, and detects at no more than half its threshold.
`MIDDLE_FIXTURE` has paired candidates with generation and detection
deduplication enabled and disabled; the pair must produce different tokens and
different detector scored-token counts.

Routing between key A and key B uses a frozen 64-value low-discrepancy sequence,
so alpha 0.1 sends about 10% of positions to key B. No value in it sits within
1e-6 of a candidate's realised routing boundary, and no row's p-value sits
within 1e-6 of its threshold, since `is_watermarked` is recomputed live. For the
two near-threshold rows, near means 0.5 to 0.9 of the row's own
`p_value_threshold`, the knob they set, since it cannot move the p-value. Prompt
tokens only enter the deduplication history, never a context.

## Integration boundary

The goldens exercise the real `WatermarkConfig`, watermarker and PRF factories,
CPU watermarker implementations, CPU repeated-context mask, and detector math.
They supply deterministic logits, contexts, ordinary samples and dual-key routing
directly. For independent drift attribution, pytest scores the stored token ids
rather than feeding a newly generated sequence into the detector.

They do not instantiate the model runner or `GPUWatermarkSampler`, and therefore
do not cover model logits, top-k/top-p and temperature processing, request/batch
state, tokenizer or text round trips, CUDA kernels, distributed execution, or the
fresh-generation-to-detection coupling and speculative-decoding
draft/accept/recovery/bonus schedule. Those boundaries need sampler/kernel parity
tests and engine-level smoke or evaluation coverage; they should not be folded
into exact model-output goldens.

## Files

`golden_candidates.py` holds the fixture, the matrix, the loader and the
comparison helper; `generate_goldens.py` regenerates the data; `test_goldens.py`
checks that every candidate reproduces; `watermarking_goldens.json` is the
frozen data; `__init__.py` makes the directory a package. The other four files
are pre-existing unit tests.
