# Speculative Decoding Test

> **Registry**: [`observability_matrix.md`](../observability_matrix.md) |
> **Compatibility**: [`feature_matrix.md`](../feature_matrix.md)

Validates speculative decoding quality and fp32-logits compatibility for both
C4 multimodal and C3 text speculative decoding scenarios.

## Test File: `tests/cohere/scripts/run_tests.sh`

Primary test entrypoint: [`tests/cohere/scripts/run_tests.sh`](../../../../tests/cohere/scripts/run_tests.sh)

### Checks

1. C4 multimodal speculative decoding meets the absolute acceptance-length gate (`AL=2.5`, tolerance `5%`) using `--custom-mm-prompts` (`check_absolute_acceptance_length`, `run_c4_spec_decode_case`): [`tests/cohere/scripts/run_tests.sh`](../../../../tests/cohere/scripts/run_tests.sh), [`examples/offline_inference/spec_decode.py`](../../../../examples/offline_inference/spec_decode.py)
2. C4 multimodal fp32-logits compatibility gate passes when acceptance-length drift between fp32 off/on runs is within `1%` (`c4_compatible` check): [`tests/cohere/scripts/run_tests.sh`](../../../../tests/cohere/scripts/run_tests.sh)
3. C3 speculative decoding meets the absolute acceptance-length gate (`AL=2.34`, tolerance `1%`) (`check_absolute_acceptance_length`, `run_c3_spec_decode_case`): [`tests/cohere/scripts/run_tests.sh`](../../../../tests/cohere/scripts/run_tests.sh)
4. C3 fp32-logits compatibility gate passes when acceptance-length drift between fp32 off/on runs is within `1%` (`c3_compatible` check): [`tests/cohere/scripts/run_tests.sh`](../../../../tests/cohere/scripts/run_tests.sh)
5. Each spec-decode run emits a parseable `mean acceptance length` metric (`extract_mean_acceptance_length`): [`tests/cohere/scripts/run_tests.sh`](../../../../tests/cohere/scripts/run_tests.sh), [`examples/offline_inference/spec_decode.py`](../../../../examples/offline_inference/spec_decode.py)

### How it runs

1. `run_speculative_decoding` first runs the C4 multimodal scenario (`c4-25a218t_fp8_eagle_l5` + draft `eagle`) with `VLLM_USE_LOGITS_FP32_COMPUTATION=0`, then checks absolute `AL` against `2.5` (`5%` tolerance): [`tests/cohere/scripts/run_tests.sh`](../../../../tests/cohere/scripts/run_tests.sh)
2. It reruns the same C4 multimodal scenario with `VLLM_USE_LOGITS_FP32_COMPUTATION=1` and enforces C4 fp32 compatibility (`AL on/off` delta <= `1%`): [`tests/cohere/scripts/run_tests.sh`](../../../../tests/cohere/scripts/run_tests.sh)
3. It then runs the C3 Command-A scenario (`command-a_fp8` + `command-a_fp8_draft`) with `VLLM_USE_LOGITS_FP32_COMPUTATION=0`, checking absolute `AL` against `2.34` (`1%` tolerance): [`tests/cohere/scripts/run_tests.sh`](../../../../tests/cohere/scripts/run_tests.sh)
4. Finally, it reruns C3 with `VLLM_USE_LOGITS_FP32_COMPUTATION=1` and enforces C3 fp32 compatibility (`AL on/off` delta <= `1%`): [`tests/cohere/scripts/run_tests.sh`](../../../../tests/cohere/scripts/run_tests.sh)
5. `spec_decode.py` computes and prints `mean acceptance length` from spec-decode metrics (`vllm:spec_decode_num_*`) for each run: [`examples/offline_inference/spec_decode.py`](../../../../examples/offline_inference/spec_decode.py)

See also: [Feature Matrix](../feature_matrix.md)
