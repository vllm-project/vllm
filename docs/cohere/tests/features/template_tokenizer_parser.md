<!-- markdownlint-disable MD024 -->
# Template / Tokenizer / Parser Check

> **Registry**: [`observability_matrix.md`](../observability_matrix.md) section 4.4 (no asserted entries) |
> **Compatibility**: [`feature_matrix.md`](../feature_matrix.md) — *no entries* (investigation tool, no compatibility verified)

Diagnostic / investigation tool. Boots a `vllm serve` instance for
`c5-3a30t_fp8` twice (without and with the reasoning + tool parsers) and
logs the chat-template + tokenization + parser pipeline end-to-end so
engineers can compare templates side by side and surface chat-template /
parser mismatches (e.g. a template that puts `<|START_THINKING|>` into the
prompt instead of letting the model emit it, which silently breaks the
reasoning parser). **Not a correctness test** — fails only if a request to
the server errors out.

<details>
<summary>Test case 1: Chat template + tokenization + parser diagnostic</summary>

## How it runs

1. `run_template_tokenizer_parser_check()` builds two `vllm serve`
   invocations for `c5-3a30t_fp8` (TP=1, `--reasoning-config`,
   `--mm-processor-cache-type shm`, `--disable-log-stats`, optional
   `--chat-template ${CT_CHAT_TEMPLATE}`) and runs them in two passes:
   `no_parsers` (raw output — surfaces what the model raw-emits) and
   `with_parsers` (`--reasoning-parser cohere_command4
   --enable-auto-tool-choice --tool-call-parser cohere_command4` — surfaces
   how the parsers split that raw output).
   - [`tests/cohere/scripts/run_tests.sh` L651-738](../../../../tests/cohere/scripts/run_tests.sh)
2. For each pass the script
   [`tests/cohere/test_chat_template_rendering.py`](../../../../tests/cohere/test_chat_template_rendering.py)
   is invoked **directly with `python3` (not pytest)** — so no JUnit XML is
   produced and no `Test Report` row is rendered. See
   [Test Pipeline Integration](../../code_notes/ci-and-automation.md#7-test-pipeline-integration).
   - [`tests/cohere/scripts/run_tests.sh` L711-718](../../../../tests/cohere/scripts/run_tests.sh)
3. Inside the script, `run_one_pass()` runs **twice per parser pass** —
   once with `thinking_budget=None` and once with
   `thinking_budget=CT_THINKING_BUDGET` (default 2048) — so the chat
   template + reasoning interaction is observable both with and without TB.
   It loads 1 sample per task from the shared `TASK_CONFIG` (mmlupro,
   ocrbench, infovqa, mathvista, aime, mgsm, mbpp_plus, niah) and sends
   each through the same `send_sample()` path used by `test_bee_samples.py`.
   - [`tests/cohere/test_chat_template_rendering.py` L136-268](../../../../tests/cohere/test_chat_template_rendering.py)
   - [`tests/cohere/test_bee_samples.py`](../../../../tests/cohere/test_bee_samples.py) — `TASK_CONFIG`, `send_sample()`
4. For every sample the script logs four stages: raw OpenAI messages, the
   rendered prompt, tokenization (token IDs + `token_strs`), and the
   server's generation/reasoning. Both the rendered prompt and tokens come
   from the server (`POST /tokenize` returns IDs + `token_strs`;
   `POST /detokenize` reconstructs the prompt text from those same IDs),
   which is the same pipeline as `/v1/chat/completions` — no local
   re-tokenization that could diverge from what the model saw.
   - [`tests/cohere/test_chat_template_rendering.py` L90-133](../../../../tests/cohere/test_chat_template_rendering.py) — `server_tokenize`, `server_detokenize`
5. Dispatched via the `template_tokenizer_parser_check` test group (it is
   its own group in the dispatcher, expanded by both the
   `template_tokenizer_parser_check` and `all` features).
   - [`.github/workflows/dispatcher.yaml`](../../../../.github/workflows/dispatcher.yaml)
   - [`.github/scripts/dispatcher-set-matrix.js`](../../../../.github/scripts/dispatcher-set-matrix.js)

## Checks

1. **No correctness assertions.** The script returns nonzero only if any
   sample request to the server errored (HTTP failure, `/tokenize` /
   `/detokenize` failure, or generation error). There is **no** assertion
   on the rendered prompt, token IDs, parser output, `reasoning` content,
   or the `parser_signal` "leaked thinking tokens" diagnostic — those are
   reported to stdout and to the structured output JSON for human
   inspection only.
   - `main()` in [`tests/cohere/test_chat_template_rendering.py` L297-410](../../../../tests/cohere/test_chat_template_rendering.py)

## Measurements

N/A. The script writes structured per-sample logs to
`${OUTPUT_DIR}/chat_template_rendering_{no_parsers,with_parsers}.json` and
prints a stdout summary (including a "leaked thinking tokens" diagnostic
when `CT_PARSER_MODE=with_parsers`), but **no upload step** in
[`.github/workflows/test-pipeline.yaml`](../../../../.github/workflows/test-pipeline.yaml)
consumes these files — they remain on the runner for ad-hoc inspection
only.

## Compatibility

Features from [Feature Matrix](../feature_matrix.md)
([Compatibility Sources](../feature_matrix.md#compatibility-sources)):

Investigation tool with no correctness assertions — no compatibility is
*verified* by this tool, so it does not earn `T.<cat>.<feat>.<seq>` cells
in [`feature_matrix.md`](../feature_matrix.md). The classification below
records which features are **exercised** by the diagnostic (so the emitted
logs cover those axes) versus **not compatible** (the test group is not
mapped to that runner).

1. **Input**: Basic (exercised — mmlupro, aime, mbpp_plus), Long Context
   (exercised — niah), Multilingual (exercised — mgsm), Image (exercised —
   ocrbench, infovqa, mathvista)
   - Same `TASK_CONFIG` as [`tests/cohere/test_bee_samples.py`](../../../../tests/cohere/test_bee_samples.py)
2. **Cohere Feature**: Thinking Budget (exercised — second sub-pass per
   parser mode runs with `CT_THINKING_BUDGET`)
3. **Model Architecture**: C5 Arch (exercised — `c5-3a30t_fp8` only)
4. **Quantization**: FP8 (exercised — `c5-3a30t_fp8` only)
5. **Hardware**: H100, B200, GB200, MI300x (exercised); A100 (not compatible)
   - [`tests/cohere/configs/runner_map.json`](../../../../tests/cohere/configs/runner_map.json) — `template_tokenizer_parser_check` runners for H100/B200/GB200/MI300x; none for A100
6. **vLLM Feature**: Chunked Prefill (exercised), CUDA Graphs (exercised)
   - [`vllm/cohere/hardware_profiles.yaml`](../../../../vllm/cohere/hardware_profiles.yaml) — default profile enables both; `run_tests.sh` exports `VLLM_ENABLE_COHERE_AUTO_CONFIG=1` so the spawned `vllm serve` self-configures

## Implementation

Primary entry: [`tests/cohere/test_chat_template_rendering.py`](../../../../tests/cohere/test_chat_template_rendering.py)
CI entry: `run_template_tokenizer_parser_check()` in
[`tests/cohere/scripts/run_tests.sh` L651-738](../../../../tests/cohere/scripts/run_tests.sh)
Dispatcher: `template_tokenizer_parser_check` test group, expanded from
itself or from `all` (no parent feature group).
Runner map: [`tests/cohere/configs/runner_map.json`](../../../../tests/cohere/configs/runner_map.json)

### Setup

1. `vllm serve ${MODEL_DIR} --tensor-parallel-size 1
   --served-model-name ${MODEL_NAME} --disable-log-stats
   --mm-processor-cache-type shm --reasoning-config ${reasoning_json}
   [--chat-template ${CT_CHAT_TEMPLATE}]`. Pass 2 additionally appends
   `--reasoning-parser cohere_command4 --enable-auto-tool-choice
   --tool-call-parser cohere_command4`. The reasoning JSON is built from
   `START_THINKING_TOKEN` / `END_THINKING_TOKEN` in
   [`vllm/cohere/guided_decoding/cohere_constants.py`](../../../../vllm/cohere/guided_decoding/cohere_constants.py).
2. Hardware profile args applied automatically inside the spawned
   `vllm serve` process via `apply_cohere_auto_config` (`run_tests.sh`
   exports `VLLM_ENABLE_COHERE_AUTO_CONFIG=1`). See
   [Hardware Profiles](../../code_notes/ci-and-automation.md#hardware-profiles).
3. Two parser modes (`no_parsers`, `with_parsers`); each runs twice
   (`thinking_budget=None`, `thinking_budget=CT_THINKING_BUDGET`).
4. Env vars consumed by the script (defaults match `run_tests.sh`):
   `CT_MODEL_DIR`, `CT_MODEL_NAME`, `CT_BASE_URL`, `CT_DATA_DIR`,
   `CT_OUTPUT_DIR`, `CT_OUTPUT_SUFFIX` (`no_parsers` / `with_parsers`),
   `CT_PARSER_MODE` (same), `CT_THINKING_BUDGET` (default 2048), and
   optional `CT_CHAT_TEMPLATE` to override the served template for
   side-by-side template comparison.
5. 1 sample per task across the 8 `TASK_CONFIG` tasks, sent via the same
   `send_sample()` code path as `test_bee_samples.py` (single shared
   `AsyncOpenAI` client + `httpx.AsyncClient`).

</details>
