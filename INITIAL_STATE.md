# Initial Codebase Snapshot — 2025-12-08

A concise memory of the repository state before any modifications.

## Project identity
- Repository: `vllm` — high-throughput, memory-efficient LLM inference/serving.
- Packaging: `pyproject.toml` uses `setuptools-scm`; `vllm` is the top-level package with CLI entrypoint `vllm`.
- Python support `>=3.10,<3.14`; build deps include `cmake`, `ninja`, `torch==2.9.0`.

## User-facing entrypoints
- CLI (`vllm`): defined in `vllm/entrypoints/cli/main.py`; subcommands `serve`, `openai`, `bench`, `collect_env`, `run_batch`.
- OpenAI-compatible server: `vllm/entrypoints/openai/api_server.py` (also wired via `vllm serve`).
- Offline/SDK API: `vllm.entrypoints.llm.LLM` class (uses the V1 engine) with `generate`, `chat`, `embed`, `classify`, `score`, etc.; tokenization and LoRA helpers included.
- Version surface: `vllm/__init__.py` re-exports `LLM`, engine args/classes, sampling/pooling/output types; `__version__` from `vllm/_version.py` (falls back to `dev` if missing).

## Architecture notes (per docs/design/arch_overview.md)
- Central engine: `LLMEngine` / `AsyncLLMEngine` orchestrate input processing, scheduling, model execution, and output handling; `LLMEngine` is aliased to `vllm.v1.engine.llm_engine`.
- Workers: one process per accelerator; each hosts a model runner that owns the actual `torch.nn.Module` instance.
- Config: a single `VllmConfig` object carries all feature flags/options; constructor signature for models is standardized to `__init__(*, vllm_config: VllmConfig, prefix: str = "")` to enable sharding/quantization at init time.
- Design docs available under `docs/design/` (e.g., `arch_overview.md`, `paged_attention.md`, `plugin_system.md`, `hybrid_kv_cache_manager.md`, `optimization_levels.md`).

## Scheduler internals (`vllm/v1/core/sched/`)
- `Scheduler` (`scheduler.py`) implements `SchedulerInterface` and is used by the V1 engine. It owns the running/waiting queues, KV/encoder caches, LoRA limits, speculative decoding metadata, and connector hooks (KV and encoder cache transfer). Policies are FCFS or priority (`request_queue.py`).
- Scheduling loop: first advances already-running requests within a token budget (`max_num_batched_tokens`) while respecting `max_model_len` and optional `long_prefill_token_threshold`. It then pulls from the waiting queue, honors `max_num_seqs`, LoRA concurrency, structured-output FSM readiness, and optionally chunked prefill. Allocation uses `KVCacheManager.allocate_slots`; on failure it may preempt the lowest-priority running request (priority policy) and free KV/encoder cache before retrying.
- Encoder inputs: `_try_schedule_encoder_inputs` enforces encoder compute/cache budgets (also covers multimodal encoders) and can fall back to scheduling only decoder tokens if budgets are tight. Encoder cache allocations/frees are coordinated with `EncoderCacheManager` and optional `ECConnector`.
- KV transfer/offload: integrates with `KVConnector` for prefix cache hits, async KV loads (`WAITING_FOR_REMOTE_KVS`), and block invalidation recovery; publishes KV cache events and aggregates connector stats. `reset_prefix_cache` can preempt running requests (discarding async placeholders) to ensure cache reset, with optional connector reset.
- Outputs: builds `SchedulerOutput` carrying new/cached request data, scheduled token counts, spec-decoding tokens, encoder inputs, common prefix metadata, and preempted/finished IDs; `AsyncScheduler` adjusts placeholders for async decoding and caches tokens on completion. Stats (`make_stats`) include KV usage, prefix cache hit/miss, spec decode, and CUDA graph metrics.

## Scheduler SWOT (current understanding)
- Strengths: configurable FCFS/priority policy; integrates KV/encoder caches and multimodal budgets; spec decoding and structured-output aware; preemption path when KV blocks unavailable; observability hooks (KV events, stats, cudagraph, prefix-cache metrics); supports async KV loads and remote caches via connectors.
- Weaknesses: complexity/branchiness (KV connectors, encoder caches, spec decode, structured outputs) increases surface for subtle bugs; preemption logic differs by policy; edge cases around async placeholders and invalid block recovery; encoder-decoder and MM paths intertwined but less tested than text-only.
- Opportunities: tighten invariants/tests around async KV load/recovery and spec decoding; simplify or centralize budget checks (encoder + token) and LoRA concurrency; add more metrics on preemption and blocked reasons; document connector lifecycle/state machine.
- Threats: deadlocks or starvation if KV/encoder budgets exhaust with chunked prefills disabled; incorrect cache state after resets or connector failures; multi-engine/include_finished_set paths could drift; regressions when adding new connector backends or structured-output grammars.

## Package layout highlights (`vllm/`)
- `entrypoints/`: CLI, OpenAI server, Anthropic compatibility, pooling endpoints, tool server, SageMaker routes, renderers, utilities.
- `v1/`: current engine implementation.
  - `engine/`: async engine wrapper, coordinator, input/output processors, parallel sampling, detokenizer, exceptions.
  - `core/`: KV cache managers, scheduler (`sched/`), encoder cache, block pool.
  - `attention/`: multiple attention backends (FlashAttention, FlashInfer, linear, ROCm, Mamba, MLA, etc.).
  - `worker/`: GPU/CPU/XPU/TPU workers and model runners, ubatching utilities, block tables, structured outputs.
  - `executor/`: uniprocess/multiprocess executors, Ray executors, distributed helper utils.
  - `sample/`, `spec_decode/`, `structured_output/`, `kv_offload/`, `metrics/`, `outputs.py`, `request.py`, `utils.py`.
- Top-level (non-`vllm`): CUDA/C++ extensions under `csrc/`, build helpers in `cmake/`, Dockerfiles in `docker/`, benchmarks (`benchmarks/`), examples (`examples/`), docs site (`docs/`), tests (`tests/`), and various `requirements/*.txt`.

## Observed behaviors/defaults
- CLI lazily imports subcommands to avoid eager import issues; `vllm bench` forces CPU platform when platform is unspecified.
- LLM Python API defaults to disabling log stats; enforces `RequestOutputKind.FINAL_ONLY` when validating sampling params.
- Version helper warns and sets `__version__="dev"` if SCM metadata is unavailable.

## Testing and collateral
- Tests are extensive under `tests/` (entrypoints, models, kernels, quantization, distributed, tool use, v1 suites, etc.).
- Benchmarks include latency/throughput scripts and CUDA kernel microbenchmarks; examples cover offline/online usage and templates.

## Outstanding unknowns (not yet inspected)
- Exact contents of `vllm/config.py` and full worker implementation details.
- Current `_version.py` value (not read) and any local modifications to generated files.

I will keep this snapshot updated as the code evolves; ask for sections to expand or refresh as needed.

