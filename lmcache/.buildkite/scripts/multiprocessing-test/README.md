# LMCache Multiprocessing Tests

This directory contains end-to-end integration tests for LMCache's multiprocessing functionality. These tests validate that LMCache correctly caches KV tensors and provides performance benefits when integrated with vLLM.

## Overview

The test suite runs two vLLM servers in parallel:
1. **vLLM with LMCache** - Connected to an LMCache multiprocessing server for KV caching
2. **vLLM baseline** - Standard vLLM without LMCache (for performance comparison)

Tests send workloads to both servers and verify:
- Correctness: LMCache produces identical outputs to baseline
- Performance: LMCache meets latency/throughput thresholds
- Caching behavior: Repeated requests benefit from cached KV tensors

## Quick Start

Run all tests:

```bash
./run-mp-test.sh
```

This will:
1. Build the Docker image
2. Launch LMCache and vLLM containers
3. Run all benchmark tests
4. Clean up containers (even on failure)

## Script Structure

```
multiprocessing-test/
├── run-mp-test.sh           # Main orchestrator - runs everything
├── common.sh                # Shared utilities (setup_venv, RESULTS_DIR)
│
├── build-mp-docker-image.sh # Builds lmcache/vllm-openai:test image
├── launch-containers.sh     # Starts LMCache + vLLM containers
├── wait-for-vllm.sh         # Waits for vLLM health endpoints
├── cleanup.sh               # Stops and removes containers
│
├── run-lm-eval.sh           # Test: lm_eval workload (correctness)
├── run-vllm-bench.sh        # Test: vllm bench serve (throughput)
├── run-long-doc-qa.sh       # Test: Long document QA (latency)
│
└── README.md                # This file
```

### Script Descriptions

| Script | Purpose |
|--------|---------|
| `run-mp-test.sh` | Main entry point. Orchestrates the entire test pipeline and ensures cleanup runs on exit. |
| `common.sh` | Shared utilities sourced by test scripts. Provides `setup_venv()` function and common variables (`RESULTS_DIR`, `BUILD_ID`, `VENV_DIR`). |
| `build-mp-docker-image.sh` | Builds the `lmcache/vllm-openai:test` Docker image from the repository's Dockerfile. |
| `launch-containers.sh` | Launches three containers: LMCache server, vLLM with LMCache, and baseline vLLM. |
| `wait-for-vllm.sh` | Polls health endpoints until both vLLM servers are ready. |
| `cleanup.sh` | Stops and removes all test containers. Captures logs before cleanup for debugging. |
| `run-lm-eval.sh` | Runs GSM8K evaluation twice to verify caching produces identical outputs. |
| `run-vllm-bench.sh` | Benchmarks throughput using `vllm bench serve` and compares LMCache vs baseline. |
| `run-long-doc-qa.sh` | Tests long-context caching with document QA workload and verifies latency thresholds. |

## Test Descriptions

### 1. LM-Eval Workload (`run-lm-eval.sh`)

**Purpose:** Verify that LMCache produces deterministic, correct outputs.

**How it works:**
1. Runs `lm_eval` with GSM8K task against vLLM+LMCache (first run populates cache)
2. Runs the same evaluation again (second run uses cached KV tensors)
3. Compares output samples from both runs

**Pass criteria:** Sample outputs from both runs must be identical.

### 2. vLLM Bench Serve (`run-vllm-bench.sh`)

**Purpose:** Verify LMCache doesn't significantly degrade throughput.

**How it works:**
1. Runs `vllm bench serve` with random prompts against baseline vLLM
2. Runs the same benchmark against vLLM+LMCache
3. Compares throughput metrics

**Pass criteria:**
- All prompts complete successfully
- LMCache throughput is within 5% of baseline

### 3. Long Document QA (`run-long-doc-qa.sh`)

**Purpose:** Verify LMCache provides latency benefits for long-context workloads.

**How it works:**
1. Runs long document QA benchmark against baseline vLLM
2. Runs the same benchmark against vLLM+LMCache
3. Compares TTFT (Time To First Token) and round-trip latency

**Pass criteria:**
- `query_ttft_per_prompt` ≤ 0.22s (loading 10K tokens)
- `query_round_time_per_prompt` ≤ 0.58s

## Expected Outputs

All test results are saved to a shared directory:

```
/tmp/lmcache_ci_results_${BUILD_ID}/
├── lm_eval/
│   ├── first_run/
│   │   └── samples_gsm8k_*.jsonl
│   └── second_run/
│       └── samples_gsm8k_*.jsonl
├── long_doc_qa/
│   ├── lmcache_result.json
│   ├── baseline_result.json
│   ├── lmcache_output.txt
│   └── baseline_output.txt
└── vllm_bench/
    ├── lmcache.json
    └── baseline.json
```

### Sample Output Files

**vllm_bench/lmcache.json:** (some fields are excluded)
```json
{
  "total_input_tokens": 500000,
  "completed": 50,
  "total_token_throughput": 12345.67
}
```

**long_doc_qa/lmcache_result.json:** (some fields are excluded)
```json
{
  "query_ttft_per_prompt": 0.15,
  "query_round_time_per_prompt": 0.45,
  "warmup_round_time_per_prompt": 2.1
}
```

## Configuration

All scripts support configuration via environment variables:

### Common Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `BUILD_ID` | `local_<timestamp>` | Unique identifier for this test run |
| `RESULTS_DIR` | `/tmp/lmcache_ci_results_${BUILD_ID}` | Directory for all test outputs |
| `VENV_DIR` | `<workspace>/.venv` | Python virtual environment path |
| `MODEL` | `Qwen/Qwen3-14B` | Model to use for testing |
| `VLLM_PORT` | `8000` | Port for vLLM with LMCache |
| `VLLM_BASELINE_PORT` | `9000` | Port for baseline vLLM |
| `LMCACHE_PORT` | `6555` | Port for LMCache server |
| `MAX_WAIT_SECONDS` | `300` | Timeout for vLLM startup |

### Test-Specific Variables

**run-lm-eval.sh:**
| Variable | Default | Description |
|----------|---------|-------------|
| `NUM_CONCURRENT` | `50` | Concurrent requests |
| `LIMIT` | `300` | Number of samples |

**run-vllm-bench.sh:**
| Variable | Default | Description |
|----------|---------|-------------|
| `NUM_PROMPTS` | `50` | Number of prompts |
| `RANDOM_INPUT_LEN` | `10000` | Input token length |
| `RANDOM_OUTPUT_LEN` | `1` | Output token length |
| `RANDOM_SEED` | `<timestamp>` | Seed for reproducibility |

**run-long-doc-qa.sh:**
| Variable | Default | Description |
|----------|---------|-------------|
| `DOCUMENT_LENGTH` | `10000` | Document length in tokens |
| `NUM_DOCUMENTS` | `30` | Number of documents |
| `OUTPUT_LEN` | `200` | Output length |
| `MAX_LMCACHE_TTFT` | `0.22` | Max allowed TTFT (seconds) |
| `MAX_LMCACHE_QUERY_ROUND_TIME` | `0.58` | Max allowed round time (seconds) |

## Adding a New Test

To add a new test to the suite:

### 1. Create Your Test Script

Create a new script (e.g., `run-my-test.sh`) following this template:

```bash
#!/bin/bash
# Brief description of your test

set -e
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source common utilities
source "$SCRIPT_DIR/common.sh"

# Configuration (use environment variables with defaults)
MY_PARAM="${MY_PARAM:-default_value}"

# Output directory (subdirectory of shared RESULTS_DIR)
MY_TEST_DIR="$RESULTS_DIR/my_test"

echo "=== My Test ==="
echo "Results dir: $MY_TEST_DIR"

# Create results directory
mkdir -p "$MY_TEST_DIR"

# Your test functions here
run_my_test() {
    # Implementation
    echo "Running test..."
}

verify_results() {
    # Verification logic
    # Return 0 on success, 1 on failure
    return 0
}

# Main execution
main() {
    # Setup venv with required packages
    setup_venv package1 package2
    
    # Run test
    run_my_test
    
    # Verify results
    if ! verify_results; then
        echo "❌ Test failed"
        exit 1
    fi
    
    echo "✅ Test passed"
}

main "$@"
```

### 2. Add to Orchestrator

Edit `run-mp-test.sh` to include your test:

```bash
# Step N: Run your test
echo "============================================"
echo "=== Step N: Running my test ==="
echo "============================================"
if ! "$SCRIPT_DIR/run-my-test.sh"; then
    echo "❌ my test failed"
    TEST_RESULT=1
    exit 1
fi
echo ""
```

### 3. Guidelines

- **Use `common.sh`**: Source it to get `setup_venv()`, `RESULTS_DIR`, `BUILD_ID`, etc.
- **Use subdirectories**: Store outputs in `$RESULTS_DIR/your_test_name/`
- **Exit codes**: Return 0 on success, non-zero on failure
- **Logging**: Use clear status indicators (✅, ❌) and descriptive messages
- **Cleanup**: Don't leave temporary files outside `RESULTS_DIR`
- **Configuration**: Use environment variables with sensible defaults
- **Comparison tests**: When comparing LMCache vs baseline, run baseline first to avoid any warm-up effects benefiting LMCache unfairly

### 4. Test Your Script

Run your script in isolation first:

```bash
# Start containers manually
./launch-containers.sh
./wait-for-vllm.sh

# Run your test
./run-my-test.sh

# Cleanup
./cleanup.sh
```

Then run the full suite:

```bash
./run-mp-test.sh
```

## Troubleshooting

### Container logs

If tests fail, check container logs:

```bash
docker logs lmcache-mp-test-<pid>
docker logs vllm-mp-test-<pid>
docker logs vllm-baseline-test-<pid>
```

The cleanup script automatically captures the last 50 lines of logs before removing containers.

### Common issues

1. **vLLM timeout**: Increase `MAX_WAIT_SECONDS` or check GPU availability
2. **Out of memory**: Reduce model size or adjust `--gpu-memory-utilization`
3. **Port conflicts**: Change `VLLM_PORT`, `VLLM_BASELINE_PORT`, or `LMCACHE_PORT`
4. **Model not found**: Ensure HuggingFace cache is mounted correctly

### Running individual tests

You can run tests individually after containers are up:

```bash
# Start infrastructure
./build-mp-docker-image.sh
./launch-containers.sh
./wait-for-vllm.sh

# Run specific test
./run-lm-eval.sh
# or
./run-vllm-bench.sh
# or
./run-long-doc-qa.sh

# Cleanup when done
./cleanup.sh
```

