#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# End-to-end script: benchmark (before) -> MoE kernel tuning -> copy config ->
# re-benchmark (after) -> report speedup for TTFT/TPOT etc.
#
# Inputs:
#   - path to model directory (or HuggingFace model id)
#   - sizes (subset of 1 2 4 8); for each size we run both TP and EP. Size 1 runs once (tp=1, ep=1).
#
# Usage:
#   ./run_moe_tune_and_bench.sh --model-dir /path/to/model --sizes "1 2 4 8"
#
# Customize the placeholders below for your environment:
#   - VLLM_SERVE_CMD: command to start the vLLM server
#   - VLLM_BENCH_SERVE_CMD: command to run "vllm bench serve" (without --save-result/--result-dir/--result-filename; added by script)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
FUSED_MOE_CONFIGS_DIR="$REPO_ROOT/vllm/model_executor/layers/fused_moe/configs"
BENCHMARK_MOE="$SCRIPT_DIR/benchmark_moe.py"

# --- Placeholders: set these or pass via env ---
# MoE routing simulation strategy for benchmarking (used when starting the server).
export VLLM_MOE_ROUTING_SIMULATION_STRATEGY="${VLLM_MOE_ROUTING_SIMULATION_STRATEGY:-uniform_random}"

# Optional: command to wait for server ready (default: curl health)
WAIT_FOR_SERVER_CMD="${WAIT_FOR_SERVER_CMD:-}"
READY_CHECK_TIMEOUT="${READY_CHECK_TIMEOUT:-600}"
BENCH_READY_CHECK_TIMEOUT="${BENCH_READY_CHECK_TIMEOUT:-120}"
BENCH_WARMUP_RUN="${BENCH_WARMUP_RUN:-true}"
BENCH_WARMUP_NUM_PROMPTS="${BENCH_WARMUP_NUM_PROMPTS:-128}"
BENCH_MAIN_NUM_PROMPTS="${BENCH_MAIN_NUM_PROMPTS:-512}"

# Default port for vLLM server
PORT="${PORT:-8000}"

# --- Parse arguments ---
MODEL_DIR=""
SIZES=""
RESULT_DIR=""
DO_TUNE=true
DO_BENCH=true
TEST_MODE=false
TEST_BATCH_SIZE=128
ONLY_EP=false
EXTRA_MOE_ARGS=""
EXTRA_SERVE_ARGS=""
SERVE_PID=""

usage() {
  echo "Usage: $0 --model-dir <path> --sizes \"1 2 4 8\" [options]"
  echo ""
  echo "Required:"
  echo "  --model-dir PATH   Path to model directory (or HF model id)"
  echo "  --sizes LIST       Space-separated sizes, e.g. \"1 2 4 8\". For each size we run both TP and EP; size 1 runs once (tp=1, ep=1)."
  echo ""
  echo "Optional:"
  echo "  --result-dir PATH  Directory for before/after JSON results (default: ./moe_tune_bench_results_<timestamp>)"
  echo "  --no-tune          Skip MoE tuning step; only run before/after benchmarks (e.g. to compare existing configs)"
  echo "  --no-bench         Skip both before/after vllm bench serve runs (and server launches); useful for tune-only config sweeps"
  echo "  --test             Test mode: run benchmark_moe.py with a single batch size (default 128) to test end-to-end without full sweep"
  echo "  --test-batch-size N  With --test, use this batch size (default: 128)"
  echo "  --only-ep          For size > 1, run only EP path (tp=1, ep=size) and skip TP path (tp=size, ep=1)"
  echo "  --extra-moe-args   Extra args for benchmark_moe.py (e.g. --dtype fp8_w8a8 --trust-remote-code)"
  echo "  --extra-serve-args Extra args appended to VLLM_BENCH_SERVE_CMD"
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-dir)    MODEL_DIR="$2"; shift 2 ;;
    --sizes)        SIZES="$2";     shift 2 ;;
    --result-dir)   RESULT_DIR="$2"; shift 2 ;;
    --no-tune)      DO_TUNE=false;  shift ;;
    --no-bench)     DO_BENCH=false; shift ;;
    --test)         TEST_MODE=true; shift ;;
    --test-batch-size) TEST_BATCH_SIZE="$2"; shift 2 ;;
    --only-ep)      ONLY_EP=true; shift ;;
    --extra-moe-args)   EXTRA_MOE_ARGS="$2"; shift 2 ;;
    --extra-serve-args) EXTRA_SERVE_ARGS="$2"; shift 2 ;;
    -h|--help) usage ;;
    *) echo "Unknown option: $1"; usage ;;
  esac
done

if [[ -z "$MODEL_DIR" || -z "$SIZES" ]]; then
  echo "Missing required --model-dir or --sizes"
  usage
fi

if [[ "$DO_TUNE" == false && "$DO_BENCH" == false ]]; then
  echo "Nothing to do: both tuning and benchmarking are disabled (--no-tune --no-bench)."
  exit 1
fi

# Default commands use placeholders that are expanded later with eval.
# Users can override either command via environment variables.
if [[ -z "${VLLM_SERVE_CMD:-}" ]]; then
  VLLM_SERVE_CMD='vllm serve "${MODEL_DIR}" --tensor-parallel-size "${SERVE_TP_SIZE}" ${EP_SERVE_FLAG} --port "${PORT}" --no-enable-prefix-caching --load-format dummy --max-model-len 8192 --mm-encoder-attn-backend TORCH_SDPA'
fi

if [[ -z "${VLLM_BENCH_SERVE_CMD:-}" ]]; then
  VLLM_BENCH_SERVE_CMD='vllm bench serve \
    --base-url "http://127.0.0.1:${PORT}" \
    --model "${MODEL_DIR}" \
    --ignore-eos \
    --dataset-name random \
    --random-input-len 1000 \
    --random-output-len 100 \
    --num-prompts 512 \
    --max-concurrency 32'
fi

if [[ -z "$RESULT_DIR" ]]; then
  RESULT_DIR="./moe_tuning/moe_tune_bench_results_$(date +%Y%m%d_%H%M%S)"
fi
mkdir -p "$RESULT_DIR"
mkdir -p "$RESULT_DIR/logs"
echo "Results will be saved to: $RESULT_DIR"

# --- Helpers ---
cleanup_server() {
  if [[ -n "${SERVE_PID:-}" ]] && kill -0 "$SERVE_PID" 2>/dev/null; then
    echo "Cleaning up server process (pid=$SERVE_PID) ..."
    kill "$SERVE_PID" 2>/dev/null || true
    wait "$SERVE_PID" 2>/dev/null || true
  fi
  SERVE_PID=""
}

stop_server() {
  cleanup_server
}

# Expand known ${VAR} placeholders in command templates for logging/execution.
render_command_template() {
  local template="$1"
  local rendered="$template"
  rendered=${rendered//'${MODEL_DIR}'/${MODEL_DIR:-}}
  rendered=${rendered//'${MODEL_ID}'/${MODEL_ID:-}}
  rendered=${rendered//'${TP}'/${TP:-}}
  rendered=${rendered//'${EP}'/${EP:-}}
  rendered=${rendered//'${PORT}'/${PORT:-}}
  rendered=${rendered//'${SERVE_TP_SIZE}'/${SERVE_TP_SIZE:-}}
  rendered=${rendered//'${EP_SERVE_FLAG}'/${EP_SERVE_FLAG:-}}
  printf '%s' "$rendered"
}

# Always cleanup a running server on exit/signals.
trap cleanup_server EXIT INT TERM

run_serve_cmd() {
  local tp=$1
  local ep=$2
  local serve_tp_size="$tp"
  local ep_serve_flag=""
  if [[ "$ep" -gt 1 ]]; then
    serve_tp_size="$ep"
    ep_serve_flag="--enable-expert-parallel"
  fi

  export TP=$tp EP=$ep MODEL_DIR MODEL_ID PORT
  export SERVE_TP_SIZE="$serve_tp_size" EP_SERVE_FLAG="$ep_serve_flag"
  local cmd
  cmd=$(render_command_template "$VLLM_SERVE_CMD")
  echo "CMD(server): $cmd"
  # Replace this shell with the serve process so SERVE_PID is the real server pid.
  eval "exec $cmd"
}

wait_for_server() {
  local port=${1:-$PORT}
  local timeout=${2:-$READY_CHECK_TIMEOUT}
  if [[ -n "$WAIT_FOR_SERVER_CMD" ]]; then
    eval "$WAIT_FOR_SERVER_CMD"
    return
  fi
  local url="http://127.0.0.1:${port}/health"
  echo "Waiting for server at $url (timeout ${timeout}s) ..."
  for ((i=0; i<timeout; i+=2)); do
    if curl -s -o /dev/null -w "%{http_code}" "$url" 2>/dev/null | grep -q 200; then
      echo "Server is ready."
      return 0
    fi
    sleep 2
  done
  echo "Server did not become ready in time."
  return 1
}

run_bench_serve() {
  local result_name=$1
  local port=${2:-$PORT}
  local log_file="${3:-}"
  local cmd
  local warmup_file warmup_log
  local warmup_prompts="$BENCH_WARMUP_NUM_PROMPTS"
  local main_prompts="$BENCH_MAIN_NUM_PROMPTS"

  if ! [[ "$warmup_prompts" =~ ^[1-9][0-9]*$ ]]; then
    echo "Invalid BENCH_WARMUP_NUM_PROMPTS='$warmup_prompts' (must be positive integer)" >&2
    return 1
  fi
  if ! [[ "$main_prompts" =~ ^[1-9][0-9]*$ ]]; then
    echo "Invalid BENCH_MAIN_NUM_PROMPTS='$main_prompts' (must be positive integer)" >&2
    return 1
  fi

  export PORT=$port MODEL_DIR MODEL_ID TP EP

  # Optional warmup benchmark run (discarded result).
  if [[ "$BENCH_WARMUP_RUN" == "true" ]]; then
    warmup_file="${result_name}_warmup_discarded.json"
    cmd="$VLLM_BENCH_SERVE_CMD --save-result --result-dir \"$RESULT_DIR\" --result-filename \"${warmup_file}\" --host 127.0.0.1 --port \"$port\" --num-prompts \"$warmup_prompts\""
    if [[ -n "$EXTRA_SERVE_ARGS" ]]; then
      cmd="$cmd $EXTRA_SERVE_ARGS"
    fi
    cmd=$(render_command_template "$cmd")
    echo "CMD(bench warmup, discarded): $cmd"
    if [[ -n "$log_file" ]]; then
      warmup_log="${log_file%.log}_warmup_discarded.log"
      eval "$cmd" >"$warmup_log" 2>&1
    else
      eval "$cmd"
    fi
    echo "Completed warmup benchmark (discarded): $warmup_file"
  fi

  # Main benchmark run (saved as the official before/after file).
  cmd="$VLLM_BENCH_SERVE_CMD --save-result --result-dir \"$RESULT_DIR\" --result-filename \"${result_name}.json\" --host 127.0.0.1 --port \"$port\" --num-prompts \"$main_prompts\""
  if [[ -n "$EXTRA_SERVE_ARGS" ]]; then
    cmd="$cmd $EXTRA_SERVE_ARGS"
  fi
  cmd=$(render_command_template "$cmd")
  echo "CMD(bench main): $cmd"
  if [[ -n "$log_file" ]]; then
    eval "$cmd" >"$log_file" 2>&1
  else
    eval "$cmd"
  fi
}

# Extract metrics from a bench serve JSON result file
get_metric() {
  local file=$1
  local key=$2
  if [[ ! -f "$file" ]]; then
    echo ""
    return
  fi
  python3 -c "
import json
try:
    with open('$file') as f:
        d = json.load(f)
    print(d.get('$key', ''))
except Exception as e:
    print('')
" 2>/dev/null || echo ""
}

report_speedup() {
  local before_file=$1
  local after_file=$2
  local tp=$3
  local ep=$4
  local config_name="tp${tp}_ep${ep}"
  echo ""
  echo "========== Results for TP=$tp EP=$ep =========="
  if [[ ! -f "$before_file" ]]; then
    echo "Before file missing: $before_file"
    SUMMARY_ROWS+=("| \`${config_name}\` | N/A | N/A | N/A | N/A | N/A | N/A |")
    return
  fi
  if [[ ! -f "$after_file" ]]; then
    echo "After file missing: $after_file"
    SUMMARY_ROWS+=("| \`${config_name}\` | N/A | N/A | N/A | N/A | N/A | N/A |")
    return
  fi
  local mean_ttft_before mean_ttft_after mean_tpot_before mean_tpot_after
  local median_ttft_before median_ttft_after median_tpot_before median_tpot_after
  local req_throughput_before req_throughput_after
  mean_ttft_before=$(get_metric "$before_file" "mean_ttft_ms")
  mean_ttft_after=$(get_metric "$after_file" "mean_ttft_ms")
  mean_tpot_before=$(get_metric "$before_file" "mean_tpot_ms")
  mean_tpot_after=$(get_metric "$after_file" "mean_tpot_ms")
  median_ttft_before=$(get_metric "$before_file" "median_ttft_ms")
  median_ttft_after=$(get_metric "$after_file" "median_ttft_ms")
  median_tpot_before=$(get_metric "$before_file" "median_tpot_ms")
  median_tpot_after=$(get_metric "$after_file" "median_tpot_ms")
  req_throughput_before=$(get_metric "$before_file" "request_throughput")
  req_throughput_after=$(get_metric "$after_file" "request_throughput")
  echo "  mean_ttft_ms: before=$mean_ttft_before after=$mean_ttft_after"
  echo "  mean_tpot_ms: before=$mean_tpot_before after=$mean_tpot_after"
  echo "  median_ttft_ms: before=$median_ttft_before after=$median_ttft_after"
  echo "  median_tpot_ms: before=$median_tpot_before after=$median_tpot_after"
  echo "  request_throughput(req/s): before=$req_throughput_before after=$req_throughput_after"
  if [[ -n "$mean_ttft_before" && -n "$mean_ttft_after" && "$mean_ttft_before" != "0" ]]; then
    local ttft_pct
    ttft_pct=$(python3 -c "print(100.0 * (float('$mean_ttft_after') - float('$mean_ttft_before')) / float('$mean_ttft_before'))" 2>/dev/null || echo "N/A")
    echo "  TTFT change: ${ttft_pct}%"
  fi
  if [[ -n "$mean_tpot_before" && -n "$mean_tpot_after" && "$mean_tpot_before" != "0" ]]; then
    local tpot_pct
    tpot_pct=$(python3 -c "print(100.0 * (float('$mean_tpot_after') - float('$mean_tpot_before')) / float('$mean_tpot_before'))" 2>/dev/null || echo "N/A")
    echo "  TPOT change: ${tpot_pct}%"
  fi
  if [[ -n "$median_ttft_before" && -n "$median_ttft_after" && "$median_ttft_before" != "0" ]]; then
    local median_ttft_pct
    median_ttft_pct=$(python3 -c "print(100.0 * (float('$median_ttft_after') - float('$median_ttft_before')) / float('$median_ttft_before'))" 2>/dev/null || echo "N/A")
    echo "  Median TTFT change: ${median_ttft_pct}%"
  fi
  if [[ -n "$median_tpot_before" && -n "$median_tpot_after" && "$median_tpot_before" != "0" ]]; then
    local median_tpot_pct
    median_tpot_pct=$(python3 -c "print(100.0 * (float('$median_tpot_after') - float('$median_tpot_before')) / float('$median_tpot_before'))" 2>/dev/null || echo "N/A")
    echo "  Median TPOT change: ${median_tpot_pct}%"
  fi
  local md_median_ttft_before md_median_ttft_after md_median_tpot_before md_median_tpot_after
  local md_req_throughput_before md_req_throughput_after
  md_median_ttft_before=$(format_metric_for_md "$median_ttft_before")
  md_median_ttft_after=$(format_metric_for_md "$median_ttft_after")
  md_median_tpot_before=$(format_metric_for_md "$median_tpot_before")
  md_median_tpot_after=$(format_metric_for_md "$median_tpot_after")
  md_req_throughput_before=$(format_metric_for_md "$req_throughput_before")
  md_req_throughput_after=$(format_metric_for_md "$req_throughput_after")
  SUMMARY_ROWS+=("| \`${config_name}\` | ${md_median_ttft_before} | ${md_median_ttft_after} | ${md_median_tpot_before} | ${md_median_tpot_after} | ${md_req_throughput_before} | ${md_req_throughput_after} |")
  echo "=============================================="
}

format_metric_for_md() {
  local value=$1
  python3 -c "
import math
import sys
v = sys.argv[1].strip()
try:
    if v == '':
        raise ValueError('empty')
    n = float(v)
    if not math.isfinite(n):
        raise ValueError('not finite')
    print(f'{n:.2f}')
except Exception:
    print('N/A')
" "$value" 2>/dev/null || echo "N/A"
}

print_markdown_summary() {
  local summary_file="$RESULT_DIR/summary_median_ttft_tpot.md"
  {
    echo "| Config | Median TTFT Before (ms) | Median TTFT After (ms) | Median TPOT Before (ms) | Median TPOT After (ms) | Request Throughput Before (req/s) | Request Throughput After (req/s) |"
    echo "|---|---:|---:|---:|---:|---:|---:|"
    for row in "${SUMMARY_ROWS[@]}"; do
      echo "$row"
    done
  } > "$summary_file"

  echo ""
  echo "========== Markdown Summary =========="
  cat "$summary_file"
  echo ""
  echo "Saved markdown summary: $summary_file"
}

SUMMARY_ROWS=()

# --- Main loop: for each size, run TP and EP (size 1 runs once) ---
run_one() {
  local tp=$1
  local ep=$2
  echo "=============================================="
  echo " TP=$tp EP=$ep"
  echo "=============================================="

  TUNE_SAVE_DIR="$RESULT_DIR/tune_tp${tp}_ep${ep}"
  mkdir -p "$TUNE_SAVE_DIR"
  BEFORE_JSON="$RESULT_DIR/before_tp${tp}_ep${ep}.json"
  AFTER_JSON="$RESULT_DIR/after_tp${tp}_ep${ep}.json"
  RUN_LOG_DIR="$RESULT_DIR/logs/tp${tp}_ep${ep}"
  mkdir -p "$RUN_LOG_DIR"
  SERVER_BEFORE_LOG="$RUN_LOG_DIR/server_before.log"
  BENCH_BEFORE_LOG="$RUN_LOG_DIR/bench_before.log"
  TUNE_LOG="$RUN_LOG_DIR/tune.log"
  SERVER_AFTER_LOG="$RUN_LOG_DIR/server_after.log"
  BENCH_AFTER_LOG="$RUN_LOG_DIR/bench_after.log"
  export TP=$tp EP=$ep MODEL_DIR MODEL_ID="$MODEL_DIR" PORT

  if [[ "$DO_BENCH" == true ]]; then
    # 1) Start server
    echo "[1/6] Starting vLLM server (TP=$tp, EP=$ep) ... log: $SERVER_BEFORE_LOG"
    run_serve_cmd "$tp" "$ep" >"$SERVER_BEFORE_LOG" 2>&1 &
    SERVE_PID=$!
    wait_for_server "$PORT" "$READY_CHECK_TIMEOUT"

    # 2) Run bench serve (before)
    echo "[2/6] Running vllm bench serve (before tuning) ... log: $BENCH_BEFORE_LOG"
    run_bench_serve "before_tp${tp}_ep${ep}" "$PORT" "$BENCH_BEFORE_LOG" || true

    # 3) Stop server
    echo "[3/6] Stopping server ..."
    stop_server
  else
    echo "[1-3/6] Skipping pre-tune benchmark and server launch (--no-bench)"
  fi

  # 4) Run MoE tuning and copy config
  if [[ "$DO_TUNE" == true ]]; then
    echo "[4/6] Running benchmark_moe.py tuning ... log: $TUNE_LOG"
    EP_FLAG=""
    if [[ "$ep" -gt 1 ]]; then
      EP_FLAG="--enable-expert-parallel"
    fi
    # For EP runs, benchmark_moe uses --tp-size as the EP partition count.
    TUNE_TP_SIZE="$tp"
    if [[ "$ep" -gt 1 ]]; then
      TUNE_TP_SIZE="$ep"
    fi
    BATCH_ARG=""
    if [[ "$TEST_MODE" == true ]]; then
      echo "  (test mode: single batch size $TEST_BATCH_SIZE)"
      BATCH_ARG="--batch-size $TEST_BATCH_SIZE"
    fi
    TUNE_CMD="python3 \"$BENCHMARK_MOE\" --model \"$MODEL_DIR\" --tp-size \"$TUNE_TP_SIZE\" $EP_FLAG --tune --save-dir \"$TUNE_SAVE_DIR\" $BATCH_ARG $EXTRA_MOE_ARGS"
    echo "CMD(tune): $TUNE_CMD"
    # Run from repo root so Python can resolve vllm
    (cd "$REPO_ROOT" && eval "$TUNE_CMD") >"$TUNE_LOG" 2>&1 || {
        echo "Tuning failed for TP=$tp EP=$ep. See log: $TUNE_LOG"
        return 0
      }

    echo "Copying tuned config(s) to $FUSED_MOE_CONFIGS_DIR ..."
    for f in "$TUNE_SAVE_DIR"/*.json; do
      if [[ -f "$f" ]]; then
        cp -v "$f" "$FUSED_MOE_CONFIGS_DIR/"
      fi
    done
  else
    echo "[4/6] Skipping tuning (--no-tune)"
  fi

  if [[ "$DO_BENCH" == true ]]; then
    # 5) Restart server and run bench serve (after)
    echo "[5/6] Starting vLLM server again ... log: $SERVER_AFTER_LOG"
    run_serve_cmd "$tp" "$ep" >"$SERVER_AFTER_LOG" 2>&1 &
    SERVE_PID=$!
    wait_for_server "$PORT" "$BENCH_READY_CHECK_TIMEOUT"

    echo "[6/6] Running vllm bench serve (after tuning) ... log: $BENCH_AFTER_LOG"
    run_bench_serve "after_tp${tp}_ep${ep}" "$PORT" "$BENCH_AFTER_LOG" || true

    # Stop server
    stop_server

    # 6) Report speedup
    report_speedup "$BEFORE_JSON" "$AFTER_JSON" "$tp" "$ep"
  else
    echo "[5-6/6] Skipping post-tune benchmark and speedup report (--no-bench)"
  fi
}

for size in $SIZES; do
  if [[ "$size" -eq 1 ]]; then
    run_one 1 1
  else
    if [[ "$ONLY_EP" != true ]]; then
      run_one "$size" 1   # TP=size, EP=1
    fi
    run_one 1 "$size"   # TP=1, EP=size
  fi
done

[[ "$DO_BENCH" == true ]] && print_markdown_summary

echo "Done. Results in $RESULT_DIR"
