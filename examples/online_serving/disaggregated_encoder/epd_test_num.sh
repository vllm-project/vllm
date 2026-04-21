#!/bin/bash
set -euo pipefail

# [Performance]: EPD Disaggregation Performance Testing Scripts #31961
# - kunxiongzhu: EPD performs even worse in this case...
# https://github.com/vllm-project/vllm/issues/31961#issuecomment-4007909605

MODE="${1:-}"

shift # 2>/dev/null || true   # shift if there is an argument

ENFORCE_EAGER=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --eager) ENFORCE_EAGER="--enforce-eager"; shift ;;
    *) break ;;   # stop at first non-global flag (mode-specific args)
  esac
done
# Now $@ contains only mode-specific args (e.g., --encoders 1 --pds 1)

############################################
# Global experiment config (edit here)
############################################
# MODEL="Qwen/Qwen2.5-VL-7B-Instruct"
MODEL="/mnt/nvme3n1/models/Qwen2.5-VL-7B-Instruct"
# MODEL="/mnt/nvme3n1/models/Qwen3-VL-4B-Instruct"

NUM_PROMPTS="${NUM_PROMPTS:-200}"
# REQUEST_RATES=(4 8 16 32)
# REQUEST_RATES=(32 16 8 4) # swap order
# REQUEST_RATES=(32) # swap order
REQUEST_RATES="${REQUEST_RATES:-32}"

# INPUT_LEN=400
INPUT_LEN="${INPUT_LEN:-2000}"
IMAGE_NUM="${IMAGE_NUM:-4}"
OUTPUT_LEN="${OUTPUT_LEN:-150}"

WIDTH="${WIDTH:-4096}"
HEIGHT="${HEIGHT:-2160}"

GPU_ALL="4,5,6,7"
GPU_E="4"
GPU_E2="5"
GPU_E3="6"
GPU_E4="7"

GPU_PD="4"
GPU_PD2="5"
GPU_PD3="6"
GPU_PD4="7"

BASELINE_PORT=8000

ENCODER_GPU_LIST=(4 5 6 7)
PD_GPU_LIST=(7 6 5 4)

# Base ports for encoders and PDs
ENCODER_BASE_PORT=16634
PD_BASE_PORT=16535
PROXY_PORT=14569

ME_PORT=18534
MPD_PORT=18535
MPD2_PORT=18536
MPD3_PORT=18537
MPROXY_PORT=18006

NUM_SEQ=64
# MAX_BATCHED_TOKENS_E=65536
MAX_BATCHED_TOKENS_E=114688
WAIT_TIMEOUT_SEC="${WAIT_TIMEOUT_SEC:-180}"

# EPD shared storage path
EC_SHARED_STORAGE_PATH="/workspace/hero/hero_EPD/test/vllm_ec_cache"
rm -rf "$EC_SHARED_STORAGE_PATH"
mkdir -p "$EC_SHARED_STORAGE_PATH"

# Optional: move temp/caches off /tmp (uncomment if /tmp fills up)
# export TMPDIR="/workspace/tmp"
# export TORCHINDUCTOR_CACHE_DIR="/workspace/torch_cache"
# mkdir -p "$TMPDIR" "$TORCHINDUCTOR_CACHE_DIR"

START_TIME=$(date +"%Y%m%d_%H%M%S")
# LOG_PATH="${LOG_PATH:-./logs}"

# Extract the last part of MODEL path (Qwen2.5-VL-7B-Instruct)
MODEL_NAME=$(basename "$MODEL")
# LOG_PATH="${LOG_PATH:-./logs}"

LOG_FOLDER="${LOG_FOLDER:-/workspace/hero/hero_EPD/test/epd_performance/log}"
LOG_PATH="${LOG_PATH:-${LOG_FOLDER}/epd_test_${MODEL_NAME}_${START_TIME}_mode${MODE}_CPU1_${WIDTH}_${HEIGHT}_newProxy_cuda13eager}"
mkdir -p $LOG_PATH


############################################
# Helpers
############################################
cleanup() {
  echo "[CLEANUP] Killing old vLLM/proxy processes..."
  pkill -9 -f "vllm serve" >/dev/null 2>&1 || true
  pkill -9 -f "disagg_epd_proxy.py" >/dev/null 2>&1 || true
}

wait_port() {
  local port="$1"
  local name="$2"
  local logfile="${3:-}"

  echo "[WAIT] Waiting for $name on port $port (timeout=${WAIT_TIMEOUT_SEC}s) ..."
  for ((i=1; i<=WAIT_TIMEOUT_SEC; i++)); do
    if (echo >/dev/tcp/127.0.0.1/"$port") >/dev/null 2>&1; then
      echo "[WAIT] $name is up."
      return 0
    fi
    sleep 1
  done
  echo "[ERROR] Timeout waiting for $name on port $port after ${WAIT_TIMEOUT_SEC}s"
  if [[ -n "$logfile" && -f "$logfile" ]]; then
    echo "[ERROR] Last 80 lines of $logfile:"
    tail -n 80 "$logfile" || true
  fi
  return 1
}

run_bench() {
  local port="$1"
  local tag="$2"

  for rps in "${REQUEST_RATES[@]}"; do
    echo "[$tag] Running benchmark: rps=${rps}, input_len=${INPUT_LEN}, image_num=${IMAGE_NUM}, out_len=${OUTPUT_LEN}, num_prompts=${NUM_PROMPTS}"

    vllm bench serve \
      --dataset-name random-mm \
      --model "$MODEL" \
      --endpoint /v1/chat/completions \
      --backend openai-chat \
      --host 127.0.0.1 \
      --port "$port" \
      --request-rate "$rps" \
      --num-prompts "$NUM_PROMPTS" \
      --random-input-len "$INPUT_LEN" \
      --random-range-ratio 0.0 \
      --random-mm-base-items-per-request "$IMAGE_NUM" \
      --random-mm-num-mm-items-range-ratio 0 \
      --random-mm-limit-mm-per-prompt '{"image":'"$IMAGE_NUM"',"video":0}' \
      --random-mm-bucket-config "{\"($WIDTH, $HEIGHT, 1)\": 1.0}" \
      --ignore-eos \
      --random-output-len "$OUTPUT_LEN" \
      --result-dir $LOG_PATH \
      --result-filename result_${tag}_rate${rps}_len${INPUT_LEN}_img${IMAGE_NUM}_out${OUTPUT_LEN}.json \
      > "${LOG_PATH}/${tag}_rate${rps}_len${INPUT_LEN}_img${IMAGE_NUM}num${NUM_PROMPTS}_out${OUTPUT_LEN}_${WIDTH}_${HEIGHT}.log" 2>&1

    echo "[$tag] Done: rps=${rps}"
  done
}

      # --save-result \
      # --save-detailed \

############################################
# Baseline (DP=4)
############################################
run_baseline() {
  cleanup

  echo "[BASELINE] Starting vLLM server (DP=4) on port $BASELINE_PORT ..."

  CUDA_VISIBLE_DEVICES="$GPU_ALL" \
  vllm serve "$MODEL" \
    --port "$BASELINE_PORT" \
    --gpu-memory-utilization 0.8 \
    --trust-remote-code \
    --data-parallel-size 4 \
    --max-num-seqs "$NUM_SEQ" \
    $ENFORCE_EAGER \
    > "${LOG_PATH}/baseline_server.log" 2>&1 &

  wait_port "$BASELINE_PORT" "baseline server" "baseline_server.log"
  run_bench "$BASELINE_PORT" "baseline"
  echo "[BASELINE] Finished."
}

# ==========================================
# run_epd – dynamic number of encoders and PDs
# ==========================================
run_epd() {
  local num_encoders=$1
  local num_pds=$2

  # Sanity checks
  if (( num_encoders < 1 || num_encoders > 4 )); then
    echo "ERROR: encoders must be 1..4"
    exit 1
  fi
  if (( num_pds < 1 || num_pds > 4 )); then
    echo "ERROR: PDs must be 1..4"
    exit 1
  fi

  cleanup

  # Assign GPUs from the list
  ENCODER_GPUS=("${ENCODER_GPU_LIST[@]:0:$num_encoders}")
  PD_GPUS=("${PD_GPU_LIST[@]:0:$num_pds}")

  # Generate ports
  ENCODER_PORTS=()
  for ((i=0; i<num_encoders; i++)); do
    ENCODER_PORTS+=($((ENCODER_BASE_PORT + i)))
  done
  PD_PORTS=()
  for ((i=0; i<num_pds; i++)); do
    PD_PORTS+=($((PD_BASE_PORT + i)))
  done

  echo "[EPD] Starting $num_encoders encoder(s) on GPUs: ${ENCODER_GPUS[*]}"
  for idx in "${!ENCODER_GPUS[@]}"; do
    gpu="${ENCODER_GPUS[$idx]}"
    port="${ENCODER_PORTS[$idx]}"
    CUDA_VISIBLE_DEVICES="$gpu" \
    vllm serve "$MODEL" \
      --gpu-memory-utilization 0.01 \
      --trust-remote-code \
      --port "$port" \
      --mm-encoder-only \
      --enable-request-id-headers \
      --no-enable-prefix-caching \
      --max-num-batched-tokens "$MAX_BATCHED_TOKENS_E" \
      --max-num-seqs "$NUM_SEQ" \
      $ENFORCE_EAGER \
      --ec-transfer-config '{
        "ec_connector": "ECExampleConnector",
        "ec_role": "ec_producer",
        "ec_connector_extra_config": {
          "shared_storage_path": "'"$EC_SHARED_STORAGE_PATH"'"
        }
      }' \
      > "${LOG_PATH}/epd_encoder_${idx}.log" 2>&1 &
  done

  echo "[EPD] Starting $num_pds PD consumer(s) on GPUs: ${PD_GPUS[*]}"
  for idx in "${!PD_GPUS[@]}"; do
    gpu="${PD_GPUS[$idx]}"
    port="${PD_PORTS[$idx]}"
    CUDA_VISIBLE_DEVICES="$gpu" \
    vllm serve "$MODEL" \
      --gpu-memory-utilization 0.8 \
      --trust-remote-code \
      --port "$port" \
      --enable-request-id-headers \
      --max-num-seqs "$NUM_SEQ" \
      $ENFORCE_EAGER \
      --ec-transfer-config '{
        "ec_connector": "ECExampleConnector",
        "ec_role": "ec_consumer",
        "ec_connector_extra_config": {
          "shared_storage_path": "'"$EC_SHARED_STORAGE_PATH"'"
        }
      }' \
      > "${LOG_PATH}/epd_pd_${idx}.log" 2>&1 &
  done

  # Wait for all encoders
  for idx in "${!ENCODER_PORTS[@]}"; do
    wait_port "${ENCODER_PORTS[$idx]}" "encoder_$idx" "epd_encoder_${idx}.log"
  done

  # Wait for all PDs
  for idx in "${!PD_PORTS[@]}"; do
    wait_port "${PD_PORTS[$idx]}" "pd_$idx" "epd_pd_${idx}.log"
  done

  # Build comma-separated URLs for proxy
  encoder_urls=$(printf "http://localhost:%s," "${ENCODER_PORTS[@]}" | sed 's/,$//')
  pd_urls=$(printf "http://localhost:%s," "${PD_PORTS[@]}" | sed 's/,$//')

  echo "[EPD] Starting Proxy on port $PROXY_PORT ..."
  python disagg_epd_proxy.py \
    --host 0.0.0.0 \
    --port "$PROXY_PORT" \
    --encode-servers-urls "$encoder_urls" \
    --prefill-servers-urls "disable" \
    --decode-servers-urls "$pd_urls" \
    > "${LOG_PATH}/epd_proxy.log" 2>&1 &

  wait_port "$PROXY_PORT" "proxy" "epd_proxy.log"
  run_bench "$PROXY_PORT" "epd"
  echo "[EPD] Finished."
}


############################################
# Entry
############################################
case "$MODE" in
  --baseline)
    run_baseline
    ;;
  --epd)
    # Parse optional --encoders and --pds
    NUM_ENCODERS=4
    NUM_PDS=4
    while [[ $# -gt 0 ]]; do
      case "$1" in
        --encoders) NUM_ENCODERS="$2"; shift 2 ;;
        --pds)      NUM_PDS="$2";      shift 2 ;;
        *) echo "Unknown option $1"; exit 1 ;;
      esac
    done
    run_epd "$NUM_ENCODERS" "$NUM_PDS"
    ;;
  --cleanup)
    cleanup
    ;;
  *)
    echo "Usage:"
    echo "  $0 --eager      # turn on eager enforce flag for EVERY instance"
    echo "  $0 --baseline   # start baseline (DP=4) and run benchmarks"
    echo "  $0 --epd        # start EPD (E + 3xPD + proxy) and run benchmarks"
    echo "  $0 --cleanup    # kill vLLM/proxy processes"
    exit 1
    ;;
esac
