<!-- markdownlint-disable -->

## Purpose

Add **inference trace-replay**: a new `SamplingParams.trace_decode_token_ids`
field that forces the v1 sampler to emit a predetermined sequence of decode
token IDs, step by step, instead of sampling. Real logprobs and token ranks
are still computed from the unmodified logit distribution at each step.

**Motivation.** The goal is to make a request follow a *fixed* token trace and
return the logprobs the model assigns to that exact trace. This is useful for:

- **Quantifying differences between settings/backends.** Replaying the same
  trace under different configs (dtype, quantization, attention/MoE backend,
  TP size, etc.) and comparing per-token logprobs gives a direct, apples-to-apples
  measure of numerical divergence on identical token sequences.
- **RL training.** It lets you recompute logprobs for a rollout's exact tokens
  under the training engine, so the rollout↔train logprob gap can be measured
  and minimized — without re-sampling and drifting onto a different trajectory.

**Why decode-replay and not prefill-only scoring?** Computing logprobs by
prefilling the full (prompt + response) sequence does *not* faithfully
reproduce the decode-time distribution: many models run **different operators /
kernels in the prefill path vs. the decode path** (e.g. attention and MoE
kernels differ between the two phases). To measure the logprobs a model would
actually produce while *decoding*, the tokens must go through the decode path
one step at a time — which is exactly what trace-replay does.

### How it works

1. Set `SamplingParams.trace_decode_token_ids` to the list of decode token IDs
   to force.
2. The sampler injects each trace token **after** sampling
   (`Sampler._inject_trace_tokens` in `vllm/v1/sample/sampler.py`), so logprobs
   stay real — they are computed from the original logits, before injection.
3. `InputBatch` tracks the trace per request in parallel with `spec_token_ids`
   (`vllm/v1/worker/gpu_input_batch.py`); the per-request step index counts real
   output tokens, skipping the trailing `-1` async-scheduling placeholder.
4. `EngineCore.add_request` caps generation to the trace length and forces
   `ignore_eos`, so EOS tokens inside the trace do not halt generation early. It
   falls back to normal sampling (with a warning) for incompatible modes
   (speculative decoding, `n > 1`).

### Not a duplicate

No open PR addresses this — searched the repo for open PRs matching
`trace_decode_token_ids` and `trace`/`replay` in title (0 results). The closest
existing features are `logprob_token_ids` (only *scores* a fixed token set,
doesn't force the output sequence) and `spec_token_ids` (speculative decoding,
different purpose); neither forces a predetermined decode trace.

## Test Plan

```bash
# Unit tests (CPU-only, no GPU required)
.venv/bin/python -m pytest \
    tests/v1/sample/test_trace_replay.py \
    tests/v1/engine/test_engine_core_trace_replay.py -v

# Lint / type-check as in CI
pre-commit run --all-files
pre-commit run mypy-3.12 --all-files --hook-stage manual
```

- `tests/v1/sample/test_trace_replay.py` — `SamplingParams` field defaults,
  clone preservation, and validation (empty / negative / non-int rejected);
  sampler injection covering step 0, mixed per-request steps, past-end-of-trace
  (no-op), the async `-1` placeholder, and the unset/none path.
- `tests/v1/engine/test_engine_core_trace_replay.py` — engine-core admission:
  generation bounds capped to trace length + `ignore_eos` forced; fallback to
  normal sampling for speculative decoding and `n > 1`.
- Demo: `examples/generate/trace_replay_offline.py` — greedy-generate to capture
  decode tokens, then replay them and assert a token-by-token match while
  printing per-token logprobs.

## Test Result

```text
$ .venv/bin/python -m pytest tests/v1/sample/test_trace_replay.py \
      tests/v1/engine/test_engine_core_trace_replay.py
tests/v1/sample/test_trace_replay.py ..............                      [ 82%]
tests/v1/engine/test_engine_core_trace_replay.py ...                     [100%]
======================= 17 passed, 16 warnings in 0.14s ========================
```

`pre-commit run --all-files` and `mypy-3.12` (CI hook stage) pass on the changed
files (ruff-check, ruff-format, SPDX headers, forbidden-imports, etc.).

> ⚠️ Unit tests were run on a CPU-only box; the end-to-end demo
> (`examples/generate/trace_replay_offline.py`) should be run on a GPU to confirm
> token-by-token replay before merge.

> **AI assistance disclosure:** developed with AI assistance (Claude). The
> submitter has reviewed every changed line and is responsible for the change
> end-to-end.

---
<details>
<summary> Essential Elements of an Effective PR Description Checklist </summary>

- [x] The purpose of the PR, such as "Fix some issue (link existing issues this PR will resolve)".
- [x] The test plan, such as providing test command.
- [x] The test results, such as pasting the results comparison before and after, or e2e results
- [x] (Optional) The necessary documentation update, such as updating `supported_models.md` and `examples` for a new model.
</details>
