# GSM8K Accuracy Evaluation

This directory owns the shared result and assertion contract for GSM8K tests.
It supports two explicit benchmark profiles:

- `isolated-v1` uses the in-repo prompt builder and last-number scorer. It can
  run against a vLLM server or an offline `LLM` and remains the fast path for
  most model and feature correctness tests.
- `lm-eval-v3` delegates prompting and scoring to lm-eval's version 3 `gsm8k`
  task. It supports offline vLLM and OpenAI-compatible server adapters.

The profiles are not score-equivalent. Tests must keep their existing profile
and baseline unless they are deliberately rebaselined with model evaluation
results.

## Usage

### Run tests with pytest (like buildkite)

```bash
pytest -s -v tests/evals/gsm8k/test_gsm8k_correctness.py \
    --config-list-file=configs/models-small.txt
```

### Run standalone evaluation script

```bash
# Start vLLM server first
vllm serve Qwen/Qwen2.5-1.5B-Instruct --port 8000

# Run evaluation
python tests/evals/gsm8k/gsm8k_eval.py --port 8000
```

## Configuration Format

Model configs in `configs/` directory use this YAML format:

```yaml
model_name: "Qwen/Qwen2.5-1.5B-Instruct"
accuracy_threshold: 0.54  # Minimum expected accuracy
num_questions: 1319       # Number of questions (default: full test set)
num_fewshot: 5            # Few-shot examples from train set
server_args: "--max-model-len 4096 --tensor-parallel-size 2 --moe-backend flashinfer_cutlass"  # Server arguments
env:                      # Environment variables (optional)
  VLLM_LOGGING_LEVEL: "DEBUG"
```

The `server_args` field accepts any arguments that can be passed to `vllm serve`.

The `env` field accepts a dictionary of environment variables to set for the server process.

## Shared test API

Both profiles return `GSM8KResult`. Accuracy gates should use
`assert_min_accuracy`, which reports the profile and metric and treats accuracy
improvements as passing:

```python
from tests.evals.gsm8k.gsm8k_eval import (
    assert_min_accuracy,
    evaluate_gsm8k_lm_eval,
)

result = evaluate_gsm8k_lm_eval(
    model="vllm",
    model_args="pretrained=Qwen/Qwen3-1.7B,max_model_len=4096",
)
assert_min_accuracy(result, expected=0.68, tolerance=0.03)
```

`gsm8k_platinum`, the generic multi-task `.buildkite/lm-eval-harness` jobs, and
uses of GSM8K questions as benchmark traffic are separate contracts and
intentionally do not use this API.
