# MM + GG + TB (Multimodal Guided Generation with Thinking Budget)

> **Registry**: [`observability_matrix.md`](../observability_matrix.md) |
> **Compatibility**: [`feature_matrix.md`](../feature_matrix.md)

Validates that multimodal (vision) inference works correctly when guided
generation, thinking budget constraints, and speculative decoding are all
active simultaneously.

## Test File: `tests/cohere/test_gg_vision_spec_async.py`

Primary test entrypoint: [`tests/cohere/test_gg_vision_spec_async.py`](../../../../tests/cohere/test_gg_vision_spec_async.py)

### Checks

1. Model produces valid JSON output (structural tag constrained) when given
   image inputs with `thinking_token_budget` set to 500, 1000, and 5000.
2. Fewer than 6 out of 32 concurrent requests produce invalid JSON at each
   budget level.

### How it runs

1. `run_guided_generation` in [`run_tests.sh`](../../../../tests/cohere/scripts/run_tests.sh)
   invokes `test_gg_vision_spec_async.py` with `--mode speculative
   --thinking-budgets 500 1000 5000` using the C4 model
   (`c4-25a218t_fp8_eagle_l5`) and its EAGLE draft model at TP=4.
2. The test loads two fixture images (duck.jpg, lion.jpg) and constructs a
   vision prompt asking the model to describe them as a JSON object.
3. For each thinking budget, 32 concurrent requests are sent to the
   `AsyncLLM` engine with guided generation (xgrammar structural tags)
   and the specified `thinking_token_budget`.
4. Outputs are validated for JSON correctness via `validate_output`.

### Measurements

No CI-uploaded metrics. This is a pass/fail correctness test only.
The test does not emit benchmark artifacts or upload results to any
reporting branch.

### Compatibility

| Input | Basic | Long Context | Multilingual | Multi Turn | Image | Audio |
|---|---|---|---|---|---|---|
| | | | | | T.3.1.1 | |

| Cohere Feature | Speculative Decoding | Guided Generation | Thinking Budget | Melody |
|---|---|---|---|---|
| | T.3.1.1 | T.3.1.1 | T.3.1.1 | |

| Model Architecture | C3 Arch | C4 Arch | C5 Arch | Reward |
|---|---|---|---|---|
| | | T.3.1.1 | | |

| Quantization | BF16 | FP8 | MXFP8 | W4A16 | FP32 Logits |
|---|---|---|---|---|---|
| | | T.3.1.1 | | | |

| Hardware | A100 | H100 | B200 | GB200 | MI300x |
|---|---|---|---|---|---|
| | | T.3.1.1 | T.3.1.1 | T.3.1.1 | |

## Implementation

Primary test: [`tests/cohere/test_gg_vision_spec_async.py`](../../../../tests/cohere/test_gg_vision_spec_async.py)
CI entry: `run_guided_generation()` in
[`tests/cohere/scripts/run_tests.sh`](../../../../tests/cohere/scripts/run_tests.sh)
Dispatcher: `guided_generation` test group, expanded from `GG_TB`.
Runner map: [`tests/cohere/configs/runner_map.json`](../../../../tests/cohere/configs/runner_map.json)

### Setup

1. `AsyncLLM` engine with `--tensor-parallel-size 4`, `--reasoning-config`
   (thinking start/end tokens), `--structured-outputs-config xgrammar`,
   `--enable-prefix-caching false`.
2. Speculative decoding via EAGLE draft model (`c4-25a218t_fp8_eagle_l5/eagle`)
   with `--num-spec-tokens 3`, `--draft-tp 4`.
3. Hardware profile args applied via `VLLM_HARDWARE_PROFILE_ARGS`.
4. Fixture images loaded from `tests/cohere/fixtures/` (duck.jpg, lion.jpg).
5. Prompt constructed via `AutoProcessor.apply_chat_template` with image
   placeholders.
6. 32 concurrent async requests per thinking budget.

See also: [Feature Matrix](../feature_matrix.md)
