# GSM8K Accuracy Evaluation

This directory contains a replacement for the lm-eval-harness GSM8K evaluation, using an isolated GSM8K script and vLLM server for better performance and control.

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

## MoE token-dropping experiments

Set `VLLM_TEST_MOE_EXPERT_CAPACITY` in the evaluation YAML to limit assignments
per expert per source rank:

```yaml
env:
  VLLM_TEST_MOE_EXPERT_CAPACITY: "128"
```

Use a compatible modular MoE backend, such as `--moe-backend triton`.
Token dropping supports eager execution and CUDA graph capture/replay, but
not direct `torch.compile` tracing of the dropping function.
Unset the variable for the baseline; zero drops all
assignments, and negative values are rejected. Prepare/finalize backends that
do not support token dropping ignore the limit; monolithic kernels reject it.
Keep concurrency and scheduler batch limits fixed when comparing accuracy.

Set `VLLM_DEBUG_MOE_WORKSPACE=1` to log modular kernel workspace high-water
marks per kernel instance and GPU. `MOE_WORKSPACE` records logical buffer
bytes, requested bytes (accounting for output/workspace13 reuse), and unique
backing-storage bytes. Backing storage includes workspace-manager alignment
and retained allocations shared with other layers; do not sum per-layer peaks.
The five entries in `peaks_bytes` correspond to workspace13, workspace2,
output, requested, and backing bytes. These independent peaks include startup
profiling and warmup. They exclude backend-internal allocations, dispatch
buffers, and CUDA allocator reserved memory. Tracking uses tensor metadata
without GPU synchronization and logs only when a peak increases.
