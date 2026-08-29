# Rust `vllm bench mm-processor` — Design

This document tracks the design and implementation for a `vllm bench mm-processor`-style
CLI command for the Rust frontend.

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

## Implementation

### Phase 1 — Timing hooks in `vllm-chat`

- Request-id-keyed timing registry mirroring Python's `MultiModalTimingRegistry`
  (`vllm/multimodal/registry.py`): `TimingContext` + `MultiModalTimingRegistry` in
  `rust/src/chat/src/multimodal/timing.rs`.
- Instruments the preprocessing stages in `MultimodalModelInfo::prepare_multimodal`
  (`rust/src/chat/src/multimodal.rs`): `media_fetch`, `preprocess_image`,
  `preprocess_video`, `preprocess_audio`, `prompt_expansion`, and `preprocessor_total`.
- Gated behind `with_mm_processor_stats(enabled)`; `stat()` drains; disabled is a no-op.

### Phase 2 — `vllm-bench mm-processor` subcommand (Rust)

Implemented as `vllm_bench::run_mm_processor` plus the `MmProcessorArgs` CLI struct in
`rust/src/bench/src/mm_processor.rs`, wired through the `mm-processor` subcommand in
`rust/src/bench/src/main.rs`.

The benchmark spawns a managed headless Python engine (real weights, real encoder),
connects over the same handshake transport the serving frontend uses, then submits N
`random-mm` prompts (reusing the dataset generation in
`rust/src/bench/src/datasets/random_mm.rs`) through the full chat pipeline
(render -> markers -> media fetch -> processor -> engine encode/decode). Per-stage
timing is drained from the Phase 1 registry; results are reported as
mean/median/std/P-s plus `--output-json`.

Note (deviation from the original design): this is a full end-to-end path (a live
engine + generation), not an offline `prepare_multimodal`-only mode. An offline,
weights-free preprocessing-only mode is not implemented yet; E2E parity with Python's
`vllm bench mm-processor` was prioritized.

Concurrency is capped by `--max-concurrency` (default `1`), which matches Python's
serial `LLMEngine` driver for a like-for-like preprocessing/e2e latency comparison.
Set it higher to measure a concurrent serving-style workload.

Flags mirroring Python: `--model`, `--num-prompts` (default 10), `--num-warmups`
(default 1), `--random-mm-*`, `--metric-percentiles` (default 99),
`--output-json`, plus `--max-concurrency`.

## Open questions

1. Implement the offline `prepare_multimodal`-only mode (no engine/weights)?
2. Anything to add from the Rust frontend side?
