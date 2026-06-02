<!-- markdownlint-disable MD024 -->
# Thinking Budget TPOT Overhead

> **Registry**: [`observability_matrix.md`](../observability_matrix.md) entry 4.5.1 |
> **Compatibility**: [`feature_matrix.md`](../feature_matrix.md) section Thinking Budget

Regression test that the thinking-token-budget sampling path adds at most 4%
median TPOT overhead versus an identical workload without
`thinking_token_budget`, across multiple budgets and concurrency levels.

<details>
<summary>Test case 1: TPOT overhead sweep (non-SD BLS)</summary>

## How it runs

1. `run_thinking_budget()` in [`run_tests.sh`](../../../../tests/cohere/scripts/run_tests.sh)
   calls `run_thinking_budget_overhead()` after the non-SD and SD correctness
   scripts; the overhead check can also run standalone via the
   `thinking_budget_overhead` test group.
   - [`tests/cohere/scripts/run_tests.sh` L417-438](../../../../tests/cohere/scripts/run_tests.sh)
   - [`tests/cohere/scripts/run_tests.sh` L441-461](../../../../tests/cohere/scripts/run_tests.sh)
2. Invokes `python3 tests/cohere/test_thinking_budget_overhead.py` with
   `--model ${ENGINES_DIR}/c5-3a30t_fp8`, the AIME JSONL dataset, and
   `--tp-size 1`.
   - [`tests/cohere/test_thinking_budget_overhead.py`](../../../../tests/cohere/test_thinking_budget_overhead.py)
3. Starts one `vllm serve` via `RemoteOpenAIServer` with test-specific CLI
   args (`--tensor-parallel-size`, `--trust-remote-code`,
   `--no-enable-prefix-caching`) and
   `VLLM_ENABLE_COHERE_AUTO_CONFIG=1` so Cohere hardware profiles fill in
   the remaining engine defaults (`reasoning-config`, chunked prefill, etc.).
   - [`tests/cohere/test_thinking_budget_overhead.py`](../../../../tests/cohere/test_thinking_budget_overhead.py)
   - [`tests/utils.py`](../../../../tests/utils.py)
   - [`vllm/cohere/auto_config.py`](../../../../vllm/cohere/auto_config.py)
4. For each thinking budget (`500`, `1000`) and concurrency (`1`, `8`, `32`,
   `64`), runs `vllm bench serve` twice against the live server: once without
   `thinking_token_budget` (baseline) and once with it in `--extra-body`.
   Parses **Median TPOT (ms)** from stdout.
   - [`tests/cohere/test_thinking_budget_overhead.py`](../../../../tests/cohere/test_thinking_budget_overhead.py)

## Checks

1. **Median TPOT overhead <= 4%** for every budget/concurrency pair:
   `100 * (tpot_with_budget - tpot_baseline) / tpot_baseline`.
   - `main()` via `run_overhead_sweep()`
2. Each **`vllm bench serve` run exits 0** with **zero failed requests**.
   - `_run_bench_serve()`

## Measurements

N/A — pass/fail regression only. Median TPOT values are printed to CI logs
but not uploaded to `ci_dump` or any reporting branch.

## Compatibility

Features from [Feature Matrix](../feature_matrix.md)
([Compatibility Sources](../feature_matrix.md#compatibility-sources)):

1. **Input**: Basic (compatible — AIME-style prompts from custom JSONL)
2. **Cohere Feature**: Thinking Budget (compatible); Speculative Decoding (not compatible — non-SD server)
3. **Model Architecture**: C5 Arch (compatible)
   - [`tests/cohere/test_thinking_budget_overhead.py`](../../../../tests/cohere/test_thinking_budget_overhead.py)
4. **Quantization**: FP8 (compatible)
5. **Hardware**: H100, B200, GB200, MI300x (compatible); A100 (not compatible)
   - [`tests/cohere/configs/runner_map.json`](../../../../tests/cohere/configs/runner_map.json) — `thinking_budget` group runners; no A100 entry
6. **vLLM Feature**: Chunked Prefill (compatible), CUDA Graphs (compatible); Prefix Caching (not compatible — explicitly disabled)
   - [`vllm/cohere/hardware_profiles.yaml`](../../../../vllm/cohere/hardware_profiles.yaml) — default profile enables chunked prefill and CUDA graphs

## Implementation

Primary test: [`tests/cohere/test_thinking_budget_overhead.py`](../../../../tests/cohere/test_thinking_budget_overhead.py)
Runtime path: [`vllm/v1/sample/thinking_budget_state.py`](../../../../vllm/v1/sample/thinking_budget_state.py)
CI entry: `run_thinking_budget_overhead()` in
[`tests/cohere/scripts/run_tests.sh`](../../../../tests/cohere/scripts/run_tests.sh)
Dispatcher: `thinking_budget` test group (also callable as `thinking_budget_overhead`).
Runner map: [`tests/cohere/configs/runner_map.json`](../../../../tests/cohere/configs/runner_map.json)

### Setup

1. `vllm serve` with `--tensor-parallel-size 1` (override via `--tp-size` /
   `TP` env), `--trust-remote-code`, `--no-enable-prefix-caching`.
2. Hardware profile defaults applied inside the spawned server via
   `VLLM_ENABLE_COHERE_AUTO_CONFIG=1` (`env_dict` on `RemoteOpenAIServer`).
   See [Hardware Profiles](../../code_notes/ci-and-automation.md#hardware-profiles).
3. Dataset: `tests/cohere/bee_eval_data/aime_2025_16samples.jsonl` (override
   via `--dataset-path` or `THINKING_BUDGET_OVERHEAD_DATASET`).
4. Sweep: budgets `[500, 1000]`, concurrencies `[1, 8, 32, 64]`,
   `num_prompts = concurrency * prompts_per_concurrency` (default 10),
   `output_len = budget + 3`, `--request-rate inf`, temperature `0.6`,
   top-p `0.95`.
5. Overhead limit: **4%** (override via `--max-overhead-pct` or
   `THINKING_BUDGET_MAX_OVERHEAD_PCT`).

</details>
