# GSM8K Accuracy Evaluation

This directory contains a replacement for the lm-eval-harness GSM8K evaluation, using an isolated GSM8K script and vLLM server for better performance and control.

## Usage

### Run tests with pytest (like buildkite)

```bash
pytest -s -v tests/gsm8k/test_gsm8k_correctness.py \
    --config-list-file=configs/models-small.txt \
    --tp-size=1
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
max_model_len: 4096       # Model context length
```
