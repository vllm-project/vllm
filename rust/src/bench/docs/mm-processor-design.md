# Rust `vllm bench mm-processor` — Design (no code yet)

This document tracks the design discussion for a `vllm bench mm-processor`-style CLI
command for the Rust frontend. **No code is included yet**; this draft PR is opened to
align on scope before implementation.

See: https://github.com/vllm-project/vllm/issues/47601 and
https://github.com/vllm-project/vllm/issues/44280

## Motivation

- The Rust chat frontend currently has no timing instrumentation in its multimodal
  preprocessing path (`MultimodalModelInfo::prepare_multimodal` in
  `rust/src/chat/src/multimodal.rs`), unlike the Python renderer which records
  per-stage timings via `MultiModalTimingRegistry` + `TimingContext.record`
  (`vllm/multimodal/registry.py`, `vllm/multimodal/processing/context.py`) and
  reports them through `vllm bench mm-processor` (`vllm/benchmarks/mm_processor.py`).
- Benchmarking other models quickly requires a reusable preprocessing/e2e latency
  command for the Rust frontend.

### Initial benchmark numbers (Qwen/Qwen3-VL-8B-Instruct, CPU)

| Resolution | HF mean | Rust mean |
|---|---|---|
| 224x224, batch=1 | 0.961 ms | 0.451 ms |
| 640x480, batch=1 | 1.254 ms | 1.399 ms |
| 1024x768, batch=1 | 3.612 ms | 4.586 ms |

Rust results are from a small smoke run; full benchmark pending. Note the model's
`preprocessor_config.json` uses `shortest_edge=65536`/`longest_edge=16777216`, so even
224x224 inputs upscale to 256 tokens — keep in mind for fair cross-model comparisons.

## Proposed scope

### Phase 1 — Timing hooks in `vllm-chat`

- Request-id-keyed timing registry mirroring Python's `MultiModalTimingRegistry`
  (`vllm/multimodal/registry.py`).
- Instrument the preprocessing stages in `MultimodalModelInfo::prepare_multimodal`
  (`rust/src/chat/src/multimodal.rs`): media fetch, `processor.preprocess` (the
  `spawn_blocking` calls in `rust/src/chat/src/multimodal/image.rs`), prompt expansion,
  and total.
- Gated behind a flag; `stat()` clears; disabled is a no-op.

### Phase 2 — `vllm-bench mm-processor` subcommand (Rust)

Offline preprocessing-only mode: no server/engine/weights needed. Load model backends
via `load_model_backends`, generate N `random-mm` prompts (reusing the existing dataset
generation in `rust/src/bench/src/datasets/random_mm.rs`), run them through
`prepare_multimodal`, aggregate per-stage timing from the Phase 1 registry, and report
mean/median/std/P-s plus `--output-json`.

An HTTP/E2E mode is intentionally out of scope: E2E latency is dominated by generation
(delegated to the Python engine) and `vllm-bench --backend openai-chat` already covers
E2E serving benchmarks.

Flags mirroring Python: `--dataset-name random-mm`, `--num-prompts`, `--num-warmups`,
`--random-mm-*`, `--metric-percentiles`, `--output-json`.

## Open questions

1. Land Phase 1 (hooks) first, or both together?
2. Anything to add from the Rust frontend side?