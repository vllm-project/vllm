#!/usr/bin/env bash
# Collect raw vLLM serving-performance data for the token-probe matrix.
#
# Matrix (by default):
#   ROUNDS x {base, mlp, attn} x {prefill off, prefill on}
#             x {1, 2, 4, 8, 16, 32 max concurrency}
#
# Service order is rotated by one position per round. Concurrency order is
# reversed on even rounds. Identical round/concurrency seeds stay paired
# across configurations while fixed time-order drift is spread across them.
#
# This script performs no cross-run aggregation. Every measured group keeps:
#   * result.json       - vllm bench serve --save-detailed output
#   * bench.stdout.log  - complete benchmark console output
#   * command.txt       - exact shell-escaped benchmark command
# Server logs and launch commands are kept once per service configuration.
# manifest.tsv is an execution index/status file, not a statistical summary.

set -uo pipefail

MODEL_PATH=${MODEL_PATH:-/home/admin/Ling-3.0-flash/}
MLP_PROBE_PATH=${MLP_PROBE_PATH:-/root/sing_probe_mlp_ling_flash/}
ATTN_PROBE_PATH=${ATTN_PROBE_PATH:-/root/sing_probe_attn_ling_flash/}

HOST=${HOST:-127.0.0.1}
PORT=${PORT:-30000}
TP_SIZE=${TP_SIZE:-4}
ROUNDS=${ROUNDS:-8}
INPUT_LEN=${INPUT_LEN:-8192}
OUTPUT_LEN=${OUTPUT_LEN:-1024}
CONCURRENCIES=${CONCURRENCIES:-"1 2 4 8 16 32"}
HEADS=${HEADS:-"base mlp attn"}
PREFILL_VALUES=${PREFILL_VALUES:-"0 1"}
PROMPTS_PER_CONCURRENCY=${PROMPTS_PER_CONCURRENCY:-6}
MIN_PROMPTS=${MIN_PROMPTS:-24}
MAX_PROMPTS=${MAX_PROMPTS:-96}
WARMUP_CONCURRENCY=${WARMUP_CONCURRENCY:-4}
WARMUP_PROMPTS=${WARMUP_PROMPTS:-12}
HEALTH_TIMEOUT_SECONDS=${HEALTH_TIMEOUT_SECONDS:-750}
CONTINUE_ON_ERROR=${CONTINUE_ON_ERROR:-0}
ENABLE_PREFIX_CACHE_RESET=${ENABLE_PREFIX_CACHE_RESET:-1}

PYTHON_BIN=${PYTHON_BIN:-/home/linjinzhen/miniconda3/bin/python}
VLLM_CLI_MODULE=${VLLM_CLI_MODULE:-vllm.entrypoints.cli.main}
COMPILATION_CONFIG=${COMPILATION_CONFIG:-'{"cudagraph_mode":"FULL"}'}
SERVER_EXTRA_ARGS=${SERVER_EXTRA_ARGS:-}
BENCH_EXTRA_ARGS=${BENCH_EXTRA_ARGS:-}
RUN_ROOT=${RUN_ROOT:-"$PWD/token_probe_raw_$(date +%Y%m%d_%H%M%S)"}

# Never route loopback serving traffic through a host-level proxy.
NO_PROXY="${NO_PROXY:+$NO_PROXY,}$HOST,localhost"
no_proxy="${no_proxy:+$no_proxy,}$HOST,localhost"
export NO_PROXY no_proxy

MANIFEST="$RUN_ROOT/manifest.tsv"
EXECUTION_ORDER="$RUN_ROOT/execution_order.tsv"
SERVER_PID=""

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*"
}

die() {
  log "ERROR: $*" >&2
  exit 1
}

shell_join() {
  local item
  printf '%q' "$1"
  shift
  for item in "$@"; do
    printf ' %q' "$item"
  done
  printf '\n'
}

is_positive_integer() {
  [[ $1 =~ ^[1-9][0-9]*$ ]]
}

stop_server() {
  if [[ -z ${SERVER_PID:-} ]]; then
    return 0
  fi

  if [[ $SERVER_PID =~ ^[1-9][0-9]*$ ]] && \
    kill -0 -- "-$SERVER_PID" 2>/dev/null; then
    log "Stopping server process group $SERVER_PID"
    kill -TERM -- "-$SERVER_PID" 2>/dev/null || true
    local attempt
    for attempt in $(seq 1 20); do
      kill -0 -- "-$SERVER_PID" 2>/dev/null || break
      sleep 1
    done
    if kill -0 -- "-$SERVER_PID" 2>/dev/null; then
      log "Server did not stop after 20 seconds; sending SIGKILL"
      kill -KILL -- "-$SERVER_PID" 2>/dev/null || true
    fi
    wait "$SERVER_PID" 2>/dev/null || true
  fi
  SERVER_PID=""
}

cleanup() {
  stop_server
}

on_signal() {
  local signal=$1
  trap - EXIT INT TERM
  cleanup
  if [[ $signal == INT ]]; then
    exit 130
  fi
  exit 143
}

trap cleanup EXIT
trap 'on_signal INT' INT
trap 'on_signal TERM' TERM

wait_until_healthy() {
  local deadline=$((SECONDS + HEALTH_TIMEOUT_SECONDS))
  while ((SECONDS < deadline)); do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      return 1
    fi
    if curl --noproxy '*' -fsS --max-time 5 \
      "http://$HOST:$PORT/health" >/dev/null 2>&1; then
      return 0
    fi
    sleep 5
  done
  return 1
}

port_is_available() {
  "$PYTHON_BIN" - "$HOST" "$PORT" <<'PY'
import socket
import sys

host = sys.argv[1]
port = int(sys.argv[2])
family = socket.AF_INET6 if ":" in host else socket.AF_INET
with socket.socket(family, socket.SOCK_STREAM) as sock:
    try:
        sock.bind((host, port))
    except OSError as exc:
        print(f"{host}:{port} is unavailable: {exc}", file=sys.stderr)
        raise SystemExit(1)
PY
}

flush_prefix_cache() {
  local output_file=$1
  local attempt http_status response=""
  : > "$output_file"
  if ((ENABLE_PREFIX_CACHE_RESET == 0)); then
    printf '{"skipped":true}\n' > "$output_file"
    return 0
  fi
  for ((attempt = 1; attempt <= 10; attempt++)); do
    http_status=$(curl --noproxy '*' -sS --max-time 30 \
      -o "$output_file" -w '%{http_code}' \
      -X POST "http://$HOST:$PORT/reset_prefix_cache") || true
    response=$(<"$output_file")
    if [[ $http_status == 200 && \
      $response =~ \"success\"[[:space:]]*:[[:space:]]*true ]]; then
      return 0
    fi
    if [[ $http_status == 404 ]]; then
      log "vLLM prefix-cache reset endpoint is unavailable;" \
        "launch with VLLM_SERVER_DEV_MODE=1 or set" \
        "ENABLE_PREFIX_CACHE_RESET=0" >&2
      return 1
    fi
    sleep 1
  done
  log "vLLM prefix-cache reset failed after 10 attempts" \
    "(HTTP $http_status): $response" >&2
  return 1
}

start_server() {
  local head=$1
  local prefill=$2
  local config_dir=$3
  local overlap=0
  local -a server_cmd extra_server_args

  server_cmd=(
    "$PYTHON_BIN" -m "$VLLM_CLI_MODULE" serve "$MODEL_PATH"
    --tensor-parallel-size "$TP_SIZE"
    --host "$HOST"
    --port "$PORT"
    --trust-remote-code
    --async-scheduling
    --compilation-config "$COMPILATION_CONFIG"
  )

  case "$head" in
    base)
      ;;
    mlp)
      overlap=1
      server_cmd+=(--probe-ckpt "$MLP_PROBE_PATH")
      ;;
    attn)
      overlap=1
      server_cmd+=(--probe-ckpt "$ATTN_PROBE_PATH")
      ;;
    *)
      log "Unknown head: $head" >&2
      return 1
      ;;
  esac

  if [[ -n $SERVER_EXTRA_ARGS ]]; then
    read -r -a extra_server_args <<< "$SERVER_EXTRA_ARGS"
    server_cmd+=("${extra_server_args[@]}")
  fi

  mkdir -p "$config_dir"
  {
    printf 'VLLM_SERVER_DEV_MODE=%q ' "$ENABLE_PREFIX_CACHE_RESET"
    printf 'VLLM_ENABLE_TOKEN_PROBE_PREFILL=%q ' "$prefill"
    printf 'VLLM_ENABLE_TOKEN_PROBE_OVERLAP=%q ' "$overlap"
    shell_join "${server_cmd[@]}"
  } > "$config_dir/server_command.txt"

  if ! port_is_available; then
    log "Port $HOST:$PORT is already occupied; refusing to launch vLLM" >&2
    return 1
  fi

  log "Launching head=$head prefill=$prefill"
  setsid env \
    VLLM_SERVER_DEV_MODE="$ENABLE_PREFIX_CACHE_RESET" \
    VLLM_ENABLE_TOKEN_PROBE_PREFILL="$prefill" \
    VLLM_ENABLE_TOKEN_PROBE_OVERLAP="$overlap" \
    "${server_cmd[@]}" > "$config_dir/server.log" 2>&1 < /dev/null &
  SERVER_PID=$!
  printf '%s\n' "$SERVER_PID" > "$config_dir/server.pid"

  if ! wait_until_healthy; then
    log "Server launch failed for head=$head prefill=$prefill" >&2
    tail -n 50 "$config_dir/server.log" >&2 || true
    stop_server
    return 1
  fi
}

run_benchmark() {
  local round=$1
  local head=$2
  local prefill=$3
  local concurrency=$4
  local num_prompts=$5
  local output_dir=$6
  local seed=$((round * 977 + concurrency))
  local result_json="$output_dir/result.json"
  local stdout_log="$output_dir/bench.stdout.log"
  local started_at ended_at rc status
  local -a bench_cmd extra_bench_args

  mkdir -p "$output_dir"
  rm -f "$result_json"

  bench_cmd=(
    "$PYTHON_BIN" -m "$VLLM_CLI_MODULE" bench serve
    --backend vllm
    --host "$HOST"
    --port "$PORT"
    --model "$MODEL_PATH"
    --trust-remote-code
    --dataset-name random
    --random-input-len "$INPUT_LEN"
    --random-output-len "$OUTPUT_LEN"
    --random-range-ratio 0.0
    --num-prompts "$num_prompts"
    --max-concurrency "$concurrency"
    --request-rate inf
    --seed "$seed"
    --ignore-eos
    --save-result
    --save-detailed
    --result-dir "$output_dir"
    --result-filename "result.json"
  )

  if [[ -n $BENCH_EXTRA_ARGS ]]; then
    read -r -a extra_bench_args <<< "$BENCH_EXTRA_ARGS"
    bench_cmd+=("${extra_bench_args[@]}")
  fi

  shell_join "${bench_cmd[@]}" > "$output_dir/command.txt"
  cat > "$output_dir/group.env" <<EOF
round=$round
head=$head
prefill_probe=$prefill
mtp=0
concurrency=$concurrency
num_prompts=$num_prompts
seed=$seed
input_len=$INPUT_LEN
output_len=$OUTPUT_LEN
EOF

  started_at=$(date --iso-8601=seconds)
  log "Measuring round=$round head=$head prefill=$prefill concurrency=$concurrency prompts=$num_prompts"
  set +e
  flush_prefix_cache "$output_dir/cache_reset.json"
  rc=$?
  if ((rc == 0)); then
    "${bench_cmd[@]}" > "$stdout_log" 2>&1
    rc=$?
  else
    printf 'Prefix-cache reset failed; benchmark was not started.\n' > "$stdout_log"
  fi
  set -e
  ended_at=$(date --iso-8601=seconds)

  if ((rc == 0)) && [[ -s $result_json ]]; then
    status=ok
  else
    status=failed
    log "Benchmark failed (rc=$rc): $output_dir" >&2
    tail -n 30 "$stdout_log" >&2 || true
  fi

  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$started_at" "$ended_at" "$round" "$head" "$prefill" 0 \
    "$concurrency" "$num_prompts" "$seed" "$status" "$rc" \
    "$result_json" "$stdout_log" >> "$MANIFEST"

  [[ $status == ok ]]
}

run_warmup() {
  local round=$1
  local config_dir=$2
  local seed=$((round * 977 + WARMUP_CONCURRENCY))
  local -a warmup_cmd

  warmup_cmd=(
    "$PYTHON_BIN" -m "$VLLM_CLI_MODULE" bench serve
    --backend vllm
    --host "$HOST"
    --port "$PORT"
    --model "$MODEL_PATH"
    --trust-remote-code
    --dataset-name random
    --random-input-len "$INPUT_LEN"
    --random-output-len "$OUTPUT_LEN"
    --random-range-ratio 0.0
    --num-prompts "$WARMUP_PROMPTS"
    --max-concurrency "$WARMUP_CONCURRENCY"
    --request-rate inf
    --seed "$seed"
    --ignore-eos
  )
  shell_join "${warmup_cmd[@]}" > "$config_dir/warmup_command.txt"
  if ! flush_prefix_cache "$config_dir/warmup_cache_reset.json"; then
    return 1
  fi
  "${warmup_cmd[@]}" > "$config_dir/warmup.stdout.log" 2>&1
}

validate_configuration() {
  command -v "$PYTHON_BIN" >/dev/null 2>&1 || \
    die "Python executable not found: $PYTHON_BIN"
  command -v curl >/dev/null 2>&1 || die "curl is required"
  "$PYTHON_BIN" -c "import vllm" >/dev/null 2>&1 || \
    die "vLLM cannot be imported by $PYTHON_BIN"
  is_positive_integer "$ROUNDS" || die "ROUNDS must be a positive integer"
  is_positive_integer "$TP_SIZE" || die "TP_SIZE must be a positive integer"
  is_positive_integer "$INPUT_LEN" || die "INPUT_LEN must be a positive integer"
  is_positive_integer "$OUTPUT_LEN" || die "OUTPUT_LEN must be a positive integer"
  [[ $ENABLE_PREFIX_CACHE_RESET == 0 || $ENABLE_PREFIX_CACHE_RESET == 1 ]] || \
    die "ENABLE_PREFIX_CACHE_RESET must be 0 or 1"
  if ((ENABLE_PREFIX_CACHE_RESET == 1)) && \
    [[ $HOST != 127.0.0.1 && $HOST != localhost && $HOST != ::1 ]]; then
    die "Prefix-cache reset enables vLLM development endpoints; use a" \
      "loopback HOST or set ENABLE_PREFIX_CACHE_RESET=0"
  fi
  [[ -e $MODEL_PATH ]] || die "MODEL_PATH does not exist: $MODEL_PATH"

  local head prefill concurrency
  for head in $HEADS; do
    case "$head" in
      base)
        ;;
      mlp)
        [[ -e $MLP_PROBE_PATH ]] || \
          die "MLP_PROBE_PATH does not exist: $MLP_PROBE_PATH"
        ;;
      attn)
        [[ -e $ATTN_PROBE_PATH ]] || \
          die "ATTN_PROBE_PATH does not exist: $ATTN_PROBE_PATH"
        ;;
      *)
        die "Unsupported HEADS entry: $head"
        ;;
    esac
  done
  for prefill in $PREFILL_VALUES; do
    [[ $prefill == 0 || $prefill == 1 ]] || \
      die "PREFILL_VALUES entries must be 0 or 1"
  done
  for concurrency in $CONCURRENCIES; do
    is_positive_integer "$concurrency" || die "Invalid concurrency: $concurrency"
  done
}

main() {
  # Model paths may exist only on the benchmark host, so validate at execution.
  validate_configuration
  mkdir -p "$RUN_ROOT"
  printf 'started_at\tended_at\tround\thead\tprefill_probe\tmtp\tconcurrency\tnum_prompts\tseed\tstatus\texit_code\tresult_json\tstdout_log\n' \
    > "$MANIFEST"
  printf 'round\tconfig_position\thead\tprefill_probe\tconcurrency_order\n' \
    > "$EXECUTION_ORDER"

  cat > "$RUN_ROOT/run.env" <<EOF
model_path=$MODEL_PATH
mlp_probe_path=$MLP_PROBE_PATH
attn_probe_path=$ATTN_PROBE_PATH
host=$HOST
port=$PORT
tp_size=$TP_SIZE
rounds=$ROUNDS
input_len=$INPUT_LEN
output_len=$OUTPUT_LEN
concurrencies=$CONCURRENCIES
heads=$HEADS
prefill_values=$PREFILL_VALUES
prompts_per_concurrency=$PROMPTS_PER_CONCURRENCY
min_prompts=$MIN_PROMPTS
max_prompts=$MAX_PROMPTS
python_bin=$PYTHON_BIN
vllm_cli_module=$VLLM_CLI_MODULE
compilation_config=$COMPILATION_CONFIG
async_scheduling=1
enable_prefix_cache_reset=$ENABLE_PREFIX_CACHE_RESET
config_order_policy=cyclic_rotation_by_round
concurrency_order_policy=ascending_odd_rounds_descending_even_rounds
EOF

  local round head prefill concurrency num_prompts config_dir output_dir
  local config config_count config_offset config_position i j tmp
  local -a configs base_concurrencies round_configs round_concurrencies
  local failures=0

  configs=()
  for head in $HEADS; do
    for prefill in $PREFILL_VALUES; do
      configs+=("$head:$prefill")
    done
  done
  read -r -a base_concurrencies <<< "$CONCURRENCIES"
  config_count=${#configs[@]}

  set -e
  for round in $(seq 1 "$ROUNDS"); do
    config_offset=$(((round - 1) % config_count))
    round_configs=()
    for ((i = 0; i < config_count; i++)); do
      round_configs+=("${configs[$(((config_offset + i) % config_count))]}")
    done

    round_concurrencies=("${base_concurrencies[@]}")
    if ((round % 2 == 0)); then
      for ((i = 0, j = ${#round_concurrencies[@]} - 1; i < j; i++, j--)); do
        tmp=${round_concurrencies[i]}
        round_concurrencies[i]=${round_concurrencies[j]}
        round_concurrencies[j]=$tmp
      done
    fi

    log "Round $round config order: ${round_configs[*]}"
    log "Round $round concurrency order: ${round_concurrencies[*]}"

    config_position=0
    for config in "${round_configs[@]}"; do
      config_position=$((config_position + 1))
      IFS=: read -r head prefill <<< "$config"
      printf '%s\t%s\t%s\t%s\t%s\n' \
        "$round" "$config_position" "$head" "$prefill" \
        "${round_concurrencies[*]}" >> "$EXECUTION_ORDER"

      config_dir=$(printf '%s/round_%02d/head_%s/prefill_%s' \
        "$RUN_ROOT" "$round" "$head" "$prefill")

      if ! start_server "$head" "$prefill" "$config_dir"; then
        failures=$((failures + 1))
        ((CONTINUE_ON_ERROR == 1)) && continue
        die "Stopping after server launch failure"
      fi

      if ! run_warmup "$round" "$config_dir"; then
        failures=$((failures + 1))
        log "Warmup failed: $config_dir" >&2
        stop_server
        ((CONTINUE_ON_ERROR == 1)) && continue
        die "Stopping after warmup failure"
      fi

      for concurrency in "${round_concurrencies[@]}"; do
        num_prompts=$((concurrency * PROMPTS_PER_CONCURRENCY))
        ((num_prompts < MIN_PROMPTS)) && num_prompts=$MIN_PROMPTS
        ((num_prompts > MAX_PROMPTS)) && num_prompts=$MAX_PROMPTS
        output_dir=$(printf '%s/concurrency_%03d' "$config_dir" "$concurrency")
        if ! run_benchmark "$round" "$head" "$prefill" \
          "$concurrency" "$num_prompts" "$output_dir"; then
          failures=$((failures + 1))
          ((CONTINUE_ON_ERROR == 1)) || die "Stopping after benchmark failure"
        fi
      done
      stop_server
    done
    log "Round $round complete"
  done

  log "Raw data collection complete: $RUN_ROOT"
  log "Failed launches/warmups/groups: $failures"
  ((failures == 0))
}

main "$@"
