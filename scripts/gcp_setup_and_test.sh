#!/usr/bin/env bash
# GCP Instance Setup & Test Script for KV Cache Tiering
# Usage: bash scripts/gcp_setup_and_test.sh
#
# Prerequisites:
#   - GCP instance with NVIDIA GPU (L4/A100/T4)
#   - CUDA drivers installed (deeplearning-platform images have this)
#   - This repo cloned on the instance

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
LOG_DIR="$REPO_DIR/test_results"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

mkdir -p "$LOG_DIR"

# ---------------------------------------------------------------------------
# Colors
# ---------------------------------------------------------------------------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log()  { echo -e "${GREEN}[$(date +%H:%M:%S)]${NC} $*"; }
warn() { echo -e "${YELLOW}[$(date +%H:%M:%S)] WARNING:${NC} $*"; }
err()  { echo -e "${RED}[$(date +%H:%M:%S)] ERROR:${NC} $*"; }

# ---------------------------------------------------------------------------
# Phase 0: Environment check
# ---------------------------------------------------------------------------
phase0_env_check() {
    log "=== Phase 0: Environment Check ==="

    log "Python version:"
    python3 --version

    log "CUDA availability:"
    if command -v nvidia-smi &>/dev/null; then
        nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
    else
        err "nvidia-smi not found. Are GPU drivers installed?"
        exit 1
    fi

    python3 -c "import torch; print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
}

# ---------------------------------------------------------------------------
# Phase 1: Install vllm in dev mode
# ---------------------------------------------------------------------------
phase1_install() {
    log "=== Phase 1: Install vLLM ==="
    cd "$REPO_DIR"

    if python3 -c "import vllm" 2>/dev/null; then
        log "vLLM already installed, skipping."
    else
        log "Installing vLLM in editable mode (this takes a few minutes)..."
        pip install -e ".[dev]" 2>&1 | tail -5
        log "vLLM installed."
    fi

    # Ensure test dependencies
    pip install pytest pytest-timeout tblib 2>&1 | tail -2
}

# ---------------------------------------------------------------------------
# Phase 2: Unit tests for our new modules
# ---------------------------------------------------------------------------
phase2_unit_tests() {
    log "=== Phase 2: Unit Tests ==="
    local test_log="$LOG_DIR/unit_tests_${TIMESTAMP}.log"

    cd "$REPO_DIR"

    log "Running tests/v1/kv_offload/ ..."
    set +e
    python3 -m pytest tests/v1/kv_offload/ -v --timeout=60 2>&1 | tee "$test_log"
    local exit_code=${PIPESTATUS[0]}
    set -e

    if [ "$exit_code" -eq 0 ]; then
        log "All unit tests PASSED."
    else
        err "Some unit tests FAILED (exit code $exit_code). See $test_log"
    fi
    return "$exit_code"
}

# ---------------------------------------------------------------------------
# Phase 3: Import smoke test -- make sure all modules load
# ---------------------------------------------------------------------------
phase3_import_test() {
    log "=== Phase 3: Import Smoke Test ==="

    python3 -c "
from vllm.v1.kv_offload.attention_manager import AttentionWeightedOffloadingManager
from vllm.v1.kv_offload.hybrid_manager import HybridOffloadingManager
from vllm.v1.kv_offload.instrumentation import AccessTracer, OffloadingMetrics
from vllm.v1.kv_offload.prefetcher import SequentialPrefetcher, FrequencyPrefetcher
from vllm.v1.kv_offload.score_estimator import (
    compute_block_scores_from_hidden_states,
    map_scores_to_block_hashes,
)
from vllm.v1.kv_offload.cpu import CPUOffloadingSpec
print('All modules imported successfully.')
"
    log "Import smoke test PASSED."
}

# ---------------------------------------------------------------------------
# Phase 4: Score estimator GPU test
# ---------------------------------------------------------------------------
phase4_score_estimator_gpu() {
    log "=== Phase 4: Score Estimator GPU Test ==="

    python3 -c "
import torch
from vllm.v1.kv_offload.score_estimator import compute_block_scores_from_hidden_states

# Simulate hidden states on GPU
hidden = torch.randn(64, 4096, device='cuda')
num_scheduled = {'req-1': 32, 'req-2': 32}
block_size = 16

scores = compute_block_scores_from_hidden_states(hidden, num_scheduled, block_size)

for req_id, block_scores in scores.items():
    print(f'  {req_id}: {len(block_scores)} blocks, scores={[round(s, 3) for s in block_scores]}')

assert len(scores) == 2, f'Expected 2 requests, got {len(scores)}'
assert len(scores['req-1']) == 2, f'Expected 2 blocks for req-1, got {len(scores[\"req-1\"])}'
assert len(scores['req-2']) == 2, f'Expected 2 blocks for req-2, got {len(scores[\"req-2\"])}'
assert all(s > 0 for s in scores['req-1']), 'Scores should be positive (L2 norms)'
print('Score estimator GPU test PASSED.')
"
    log "Score estimator GPU test PASSED."
}

# ---------------------------------------------------------------------------
# Phase 5: Offline inference with CPU offloading (LRU baseline)
# ---------------------------------------------------------------------------
phase5_offline_lru() {
    log "=== Phase 5: Offline Inference -- LRU Baseline ==="
    local test_log="$LOG_DIR/offline_lru_${TIMESTAMP}.log"

    python3 -c "
from vllm import LLM, SamplingParams

llm = LLM(
    model='facebook/opt-125m',
    kv_transfer_config={
        'kv_connector': 'OffloadingConnector',
        'kv_role': 'kv_both',
        'kv_connector_extra_config': {
            'cpu_bytes_to_use': 500 * int(1e6),
            'eviction_policy': 'lru',
        }
    },
    gpu_memory_utilization=0.5,
)

prompts = [
    'The future of artificial intelligence is',
    'In a galaxy far far away',
    'The key to effective machine learning is',
]
params = SamplingParams(temperature=0.8, max_tokens=32)
outputs = llm.generate(prompts, params)

for out in outputs:
    print(f'Prompt: {out.prompt[:40]}...')
    print(f'Output: {out.outputs[0].text[:80]}...')
    print()

print('LRU offline inference PASSED.')
" 2>&1 | tee "$test_log"
    log "LRU offline inference completed. See $test_log"
}

# ---------------------------------------------------------------------------
# Phase 6: Offline inference with attention-aware eviction
# ---------------------------------------------------------------------------
phase6_offline_attention() {
    log "=== Phase 6: Offline Inference -- Attention Eviction ==="
    local test_log="$LOG_DIR/offline_attention_${TIMESTAMP}.log"

    python3 -c "
from vllm import LLM, SamplingParams

llm = LLM(
    model='facebook/opt-125m',
    kv_transfer_config={
        'kv_connector': 'OffloadingConnector',
        'kv_role': 'kv_both',
        'kv_connector_extra_config': {
            'cpu_bytes_to_use': 500 * int(1e6),
            'eviction_policy': 'attention',
            'score_decay': 0.95,
        }
    },
    gpu_memory_utilization=0.5,
)

prompts = [
    'The future of artificial intelligence is',
    'In a galaxy far far away',
    'The key to effective machine learning is',
]
params = SamplingParams(temperature=0.8, max_tokens=32)
outputs = llm.generate(prompts, params)

for out in outputs:
    print(f'Prompt: {out.prompt[:40]}...')
    print(f'Output: {out.outputs[0].text[:80]}...')
    print()

print('Attention eviction offline inference PASSED.')
" 2>&1 | tee "$test_log"
    log "Attention eviction inference completed. See $test_log"
}

# ---------------------------------------------------------------------------
# Phase 7: Offline inference with hybrid eviction
# ---------------------------------------------------------------------------
phase7_offline_hybrid() {
    log "=== Phase 7: Offline Inference -- Hybrid Eviction ==="
    local test_log="$LOG_DIR/offline_hybrid_${TIMESTAMP}.log"

    python3 -c "
from vllm import LLM, SamplingParams

llm = LLM(
    model='facebook/opt-125m',
    kv_transfer_config={
        'kv_connector': 'OffloadingConnector',
        'kv_role': 'kv_both',
        'kv_connector_extra_config': {
            'cpu_bytes_to_use': 500 * int(1e6),
            'eviction_policy': 'hybrid',
            'attention_weight': 0.5,
            'recency_weight': 0.3,
            'frequency_weight': 0.2,
        }
    },
    gpu_memory_utilization=0.5,
)

prompts = [
    'The future of artificial intelligence is',
    'In a galaxy far far away',
    'The key to effective machine learning is',
]
params = SamplingParams(temperature=0.8, max_tokens=32)
outputs = llm.generate(prompts, params)

for out in outputs:
    print(f'Prompt: {out.prompt[:40]}...')
    print(f'Output: {out.outputs[0].text[:80]}...')
    print()

print('Hybrid eviction offline inference PASSED.')
" 2>&1 | tee "$test_log"
    log "Hybrid eviction inference completed. See $test_log"
}

# ---------------------------------------------------------------------------
# Phase 8: Offline inference with prefetching enabled
# ---------------------------------------------------------------------------
phase8_offline_prefetch() {
    log "=== Phase 8: Offline Inference -- Prefetching ==="
    local test_log="$LOG_DIR/offline_prefetch_${TIMESTAMP}.log"

    python3 -c "
from vllm import LLM, SamplingParams

llm = LLM(
    model='facebook/opt-125m',
    kv_transfer_config={
        'kv_connector': 'OffloadingConnector',
        'kv_role': 'kv_both',
        'kv_connector_extra_config': {
            'cpu_bytes_to_use': 500 * int(1e6),
            'eviction_policy': 'attention',
            'score_decay': 0.95,
            'enable_prefetching': True,
            'prefetch_lookahead': 2,
            'prefetch_max_pending': 8,
        }
    },
    gpu_memory_utilization=0.5,
)

prompts = [
    'The future of artificial intelligence is',
    'In a galaxy far far away',
    'The key to effective machine learning is',
]
params = SamplingParams(temperature=0.8, max_tokens=32)
outputs = llm.generate(prompts, params)

for out in outputs:
    print(f'Prompt: {out.prompt[:40]}...')
    print(f'Output: {out.outputs[0].text[:80]}...')
    print()

print('Prefetching offline inference PASSED.')
" 2>&1 | tee "$test_log"
    log "Prefetching inference completed. See $test_log"
}

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print_summary() {
    log "=========================================="
    log "  Test Results Summary"
    log "=========================================="
    log "Logs saved to: $LOG_DIR/"
    ls -la "$LOG_DIR"/*"${TIMESTAMP}"* 2>/dev/null || true
    log ""
    log "Next steps:"
    log "  1. Review logs for any errors"
    log "  2. Run benchmarks: python3 kv_cache_tiering/benchmarks/benchmark.py --help"
    log "  3. Share logs so we can fix any issues"
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
main() {
    log "Starting KV Cache Tiering test suite"
    log "Repo: $REPO_DIR"
    log "Logs: $LOG_DIR"
    log ""

    phase0_env_check
    phase1_install
    phase2_unit_tests  || true  # continue even if unit tests fail
    phase3_import_test
    phase4_score_estimator_gpu
    phase5_offline_lru
    phase6_offline_attention
    phase7_offline_hybrid
    phase8_offline_prefetch

    print_summary
}

main "$@"
