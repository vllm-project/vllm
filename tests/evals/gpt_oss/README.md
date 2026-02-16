# GPQA Evaluation using GPT-OSS

This directory contains GPQA evaluation tests using the GPT-OSS evaluation package and vLLM server.

## Usage

### Run tests with pytest (like buildkite)

```bash
# H200
pytest -s -v tests/evals/gpt_oss/test_gpqa_correctness.py \
    --config-list-file=configs/models-h200.txt

# B200
pytest -s -v tests/evals/gpt_oss/test_gpqa_correctness.py \
    --config-list-file=configs/models-b200.txt
```

## Configuration Format

Model configs in `configs/` directory use this YAML format:

```yaml
model_name: "openai/gpt-oss-20b"
metric_threshold: 0.568          # Minimum expected accuracy
reasoning_effort: "low"          # Reasoning effort level (default: "low")
server_args: "--tensor-parallel-size 2"  # Server arguments
startup_max_wait_seconds: 1800   # Max wait for server startup (default: 1800)
env:                             # Environment variables (optional)
  SOME_VAR: "value"
```

The `server_args` field accepts any arguments that can be passed to `vllm serve`.

The `env` field accepts a dictionary of environment variables to set for the server process.

## Adding New Models

1. Create a new YAML config file in the `configs/` directory
2. Add the filename to the appropriate `models-*.txt` file

## Tiktoken Encoding Files

The tiktoken encoding files required by the vLLM server are automatically downloaded from OpenAI's public blob storage on first run:

- `cl100k_base.tiktoken`
- `o200k_base.tiktoken`

Files are cached in the `data/` directory. The `TIKTOKEN_ENCODINGS_BASE` environment variable is automatically set to point to this directory when running evaluations.
