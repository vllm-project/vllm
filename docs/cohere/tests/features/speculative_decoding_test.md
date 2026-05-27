# Speculative Decoding Test

> **Registry**: [`observability_matrix.md`](../observability_matrix.md) |
> **Compatibility**: [`feature_matrix.md`](../feature_matrix.md)

Validates speculative decoding correctness via request cancellation sweeps,
exercising both SD and non-SD modes under concurrent load to ensure graceful
cancellation handling and output quality.

## Test File: `tests/cohere/test_request_cancellation.py`

Primary test entrypoint: [`tests/cohere/test_request_cancellation.py`](../../../../tests/cohere/test_request_cancellation.py)

### Checks

1. All completed (non-cancelled) requests pass **output quality validation**:
   no invalid JSON when guided generation is active, and `min_logprob` above
   threshold (`-50.0`).
2. **Doom loop detection** (excessive repetition ratio) emits a warning but
   does not fail the test.
3. **Gibberish detection** (long non-letter runs or 200+ repeated chars)
   emits a warning but does not fail the test. Downgraded from a hard
   failure because it intermittently trips the B200 nightly under
   speculative decoding; the warning is still surfaced in logs for triage.
4. The server handles concurrent request cancellation gracefully without
   crashes or hangs across concurrency levels `32` and `64`.

### How it runs

1. `run_speculative_decoding` in [`run_tests.sh`](../../../../tests/cohere/scripts/run_tests.sh)
   invokes `test_request_cancellation.py` with the BLS model (`c5-3a30t_fp8`)
   and EAGLE draft (`c5-3a30t_eagle_bf16`) at TP=1, sweeping `--num-requests 32 64`.
   - [`tests/cohere/scripts/run_tests.sh`](../../../../tests/cohere/scripts/run_tests.sh)
2. A second invocation runs the same sweep with `--disable-spec` (non-SD mode)
   to validate cancellation correctness without speculative decoding.
   - [`tests/cohere/scripts/run_tests.sh`](../../../../tests/cohere/scripts/run_tests.sh)
3. For each concurrency level, the test spawns N concurrent requests via the
   serving layer, cancels a subset mid-flight, and validates output quality
   on completed requests.
   - [`tests/cohere/test_request_cancellation.py`](../../../../tests/cohere/test_request_cancellation.py)

### Measurements

No CI-uploaded metrics. This is a pass/fail correctness test only.
The test does not emit benchmark artifacts or upload results to any
reporting branch.

### Compatibility

| Input | Basic | Long Context | Multilingual | Multi Turn | Image | Audio |
| --- | --- | --- | --- | --- | --- | --- |
| | T.4.1.1 | | | | | |

| Cohere Feature | Speculative Decoding | Guided Generation | Thinking Budget | Melody |
| --- | --- | --- | --- | --- |
| | T.4.1.1 | | | |

| Model Architecture | C3 Arch | C4 Arch | C5 Arch | Reward |
| --- | --- | --- | --- | --- |
| | | | T.4.1.1 | |

| Quantization | BF16 | FP8 | MXFP8 | W4A16 | FP32 Logits |
| --- | --- | --- | --- | --- | --- |
| | | T.4.1.1 | | | |

| Hardware | A100 | H100 | B200 | GB200 | MI300x |
| --- | --- | --- | --- | --- | --- |
| | | T.4.1.1 | T.4.1.1 | T.4.1.1 | |

## Implementation

Primary test: [`tests/cohere/test_request_cancellation.py`](../../../../tests/cohere/test_request_cancellation.py)
CI entry: `run_speculative_decoding()` in
[`tests/cohere/scripts/run_tests.sh`](../../../../tests/cohere/scripts/run_tests.sh)
Dispatcher: `speculative_decoding` test group.
Runner map: [`tests/cohere/configs/runner_map.json`](../../../../tests/cohere/configs/runner_map.json)

### Setup

1. vLLM serve with `--tensor-parallel-size 1`, BLS model (`c5-3a30t_fp8`).
2. SD mode: EAGLE draft model (`c5-3a30t_eagle_bf16`) with
   `--num-spec-tokens 3`, `--draft-tp 1`.
3. Non-SD mode: `--disable-spec` flag, same model without draft.
4. Concurrency sweep: `--num-requests 32 64`.
5. Quality validation thresholds: `repetition_ratio` (warning only),
   gibberish patterns (warning only), `min_logprob >= -50.0`, no invalid JSON.

See also: [Feature Matrix](../feature_matrix.md)
