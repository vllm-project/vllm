# EPD Correctness Test

This test verifies that EPD (Encoder-Prefill-Decode) disaggregation produces identical outputs to a baseline single instance.

## What It Tests

- **Baseline**: Single vLLM instance serving a multimodal model
- **EPD (1E+1PD)**: 1 Encoder + 1 Prefill-Decode instance
- **Baseline (1P+1D)**: 1 Prefill + 1 Decode instance
- **EPD (1E+1P+1D)**: 1 Encoder + 1 Prefill + 1 Decode instance

The test ensures that disaggregated encoding produces **identical** outputs to the baseline.

Note that currently PD disaggregation set up may give slightly different results from a single instance. Therefore, we need the result from 1P+1D as the baseline for 1E+1P+1D

Please refer to [Disaggregated Encoder Feature](../../../docs/features/disagg_encoder.md) for the detailed explanation for the EPD features.

## Files

- `run_epd_correctness_test.sh` - Main test script (starts all instances and runs tests)
- `test_epd_correctness.py` - Python test script (compares outputs)

## Usage

### Multimodal Prompts (Default)

```bash
cd vllm
./tests/v1/ec_connector/integration/run_epd_correctness_test.sh
```

This runs the test with actual multimodal (image) prompts.

### Text-Only Prompts

```bash
cd vllm
USE_MM_PROMPTS=0 ./tests/v1/ec_connector/integration/run_epd_correctness_test.sh
```

This runs a quick test with text-only prompts to verify the setup works.

### Custom Configuration

```bash
# Use specific GPUs
GPU_E=0 GPU_PD=1 GPU_P=1 GPU_D=2 bash ./tests/v1/ec_connector/integration/run_epd_correctness_test.sh

# Use specific ports
ENDPOINT_PORT=10001 bash ./tests/v1/ec_connector/integration/run_epd_correctness_test.sh

# Use specific model
MODEL="Qwen/Qwen2.5-VL-3B-Instruct" bash ./tests/v1/ec_connector/integration/run_epd_correctness_test.sh

# Use specific storage path
EC_SHARED_STORAGE_PATH="/tmp/my_ec_cache" bash ./tests/v1/ec_connector/integration/run_epd_correctness_test.sh
```

## How It Works

### Step 1: Baseline

1. Start single vLLM instance on GPU
2. Run test prompts (multimodal or text-only)
3. Save outputs to `.vllm_epd_baseline.txt`
4. Shutdown instance

### Step 2: EPD (1E + 1PD)

1. Clear encoder cache storage
2. Start instances and proxy
3. Run same test prompts
4. Assert outputs match baseline exactly
5. Shutdown instances

### Step 3: EPD (1E + 1P + 1D)

1. Clear encoder cache storage
2. Start instances and proxy
3. Run same test prompts
4. Assert outputs match baseline exactly
5. Shutdown instances

## Test Scenarios

### Multimodal Prompts (--use_mm_prompts)

Tests encoder cache transfer:

- Single image query
- Multiple images in one request
- Mixed image and text
- Image with detailed questions

### Text-Only Prompts (default)

Quick sanity check:

- Simple text queries
- Text-only explanations
- Verifies proxy routing works

## Expected Behavior

### ✅ Test Passes When

- All disagg outputs match baseline outputs exactly
- No errors during instance startup
- Encoder cache is properly saved and loaded
- Proxy correctly routes requests

### ❌ Test Fails When

- Outputs differ between baseline and disagg
- Server startup fails
- Encoder cache not found (should fallback to local execution)
- Proxy routing errors

## Notes

- The test uses deterministic generation (`temperature=0.0`, `seed=42`)
- Encoder cache should enable exact output reproduction
- Test cleans up all instances and cache files after completion
- Safe to run multiple times (idempotent)
- We setup the PD disagg part with NixlConnector. Please read details about EPD in `examples/online_serving/disaggregated_encoder/README.md`

## Requirements

- Multiple GPUs (3 for 1E+1P+1D, 2 for 1E+1PD, 1 for baseline)
    - 1E+1P+1D is runnable with 2 GPU by assign E and P on the same GPU now.
- Multimodal model (e.g., Qwen2.5-VL-3B-Instruct)
- Internet access (for accessing vllm test images)

## Debugging

### Check Logs

Logs and baseline output are saved in `/tmp/` by default.
Can be customized by changing the environment variables.

### Check Encoder Cache

```bash
# Verify cache files are created
ls -la $EC_SHARED_STORAGE_PATH/

# Should see directories with mm_hash names
# Each containing encoder_cache.safetensors
```

### Manual Testing

Run individual components:

```bash
# Baseline only
python test_epd_correctness.py \
    --service_url http://localhost:8000 \
    --model_name Qwen/Qwen2.5-VL-3B-Instruct \
    --mode baseline \
    --baseline_file test_output.txt \
    --use_mm_prompts

# Disagg only (requires baseline output file!)
python test_epd_correctness.py \
    --service_url http://localhost:8000 \
    --model_name Qwen/Qwen2.5-VL-3B-Instruct \
    --mode disagg \
    --baseline_file test_output.txt \
    --use_mm_prompts
```
