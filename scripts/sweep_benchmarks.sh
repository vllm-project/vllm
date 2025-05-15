#!/bin/bash
set -e

# Usage: sweep_benchmarks.sh [NUM_NODES]
#
# Arguments:
#   NUM_NODES          (Optional) Number of nodes to use for the benchmark. Default: 1.
#
# Description:
#   This script automates benchmarking of vLLM across various configurations, including pipeline
#   and tensor parallelism, communication backends, and model settings. It supports both single-node
#   and multi-node setups, logging performance metrics for scalability and optimization analysis.
#
#   The script iterates over a list of server configurations (e.g., model length, parallelism settings)
#   and client configurations (e.g., input/output token lengths, concurrency levels). For each configuration:
#     - It launches a vLLM server with the specified settings.
#     - Runs client benchmarks to measure throughput, latency, and other metrics.
#     - Logs the results in a structured format for further analysis.
#
#   Key Features:
#     - Multi-node support with Ray cluster pre-checks.
#     - Automatic log management, including backups of previous logs.
#     - Progress bar for server startup and error handling for crashes or timeouts.
#     - Profiling support for detailed performance analysis.
#
# Example:
#   ./sweep_benchmarks.sh 2
#     - Runs the benchmark on 2 nodes with the configurations defined in the script.

NUM_NODES=${1:-1}
echo "Starting benchmark sweeper with NUM_NODES=${NUM_NODES}"

python3 -m pip install datasets

# Multi-node pre-check
if [ "$NUM_NODES" -gt 1 ]; then
  echo "[Warning] Multi-node mode: Ensure ray cluster is started via run_cluster.sh."

  # Verify Ray HPUs
  TOTAL_HPU=$((8 * NUM_NODES))
  if ray status | grep -q "0.0/${TOTAL_HPU}.0 HPU"; then
    echo "Ray cluster ready with ${TOTAL_HPU} HPUs."
  else
    echo "Ray cluster not ready; expected ${TOTAL_HPU} HPUs. Exiting."
    exit 1
  fi
fi

# Prepare logging
BASE_LOG_DIR=$(pwd)/logs/$(date +"%Y%m%d")/${NUM_NODES}-node
mkdir -p "$BASE_LOG_DIR"
SUMMARY_LOG=${BASE_LOG_DIR}/summary.log

# Build header
HEADER='nodes,pp_size,tp_size,comm_backend,kv_cache_dtype,partition,max_model_len,input_tokens,output_tokens,num_prompts,max_concurrency,client_concurrency,warmup,profile,mean_ttft,mean_tpot,total_throughput,output_throughput'
echo "$HEADER" | tee -a "$SUMMARY_LOG"

# Default client/server parameters
HOST=${HOST:-127.0.0.1}
PORT=${PORT:-8688}
MODEL_PATH=${MODEL_PATH:-/root/.cache/huggingface/DeepSeek-R1-BF16-w8afp8-dynamic-no-ste-G2}

# Server config list: max_len,num_prompts,max_conc,pp,tp,backend,warmup,profile,partition
KV=auto
WARMUP=true
PROFILE=false
server_config_list=(
  "16384,32,2,4,hccl,${KV},${WARMUP},${PROFILE},[32,29]"
)
# Client config list: input,output,num_prompts,conc
client_config_list=(
  "2048,2048,96,32"
  "4096,4096,96,32"
)

# Helper function for profiling
start_profile() {
  echo "Starting profiling..."
  sleep 5
  export no_proxy=localhost,${HOST},10.239.129.9
  curl -X POST http://${HOST}:${PORT}/start_profile
}

# Helper function for profiling
stop_profile() {
  echo "Stopping profiling..."
  sleep 5
  export no_proxy=localhost,${HOST},10.239.129.9
  curl -X POST http://${HOST}:${PORT}/stop_profile
}

# Function to back up benchmark logs while maintaining the nested folder structure
backup_benchmark_logs() {
  local CONFIG_LOG_DIR=$1
  local TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
  local BACKUP_DIR="${CONFIG_LOG_DIR}/backup_${TIMESTAMP}"

  # Find files with specific extensions and move them to the backup directory
  if find "$CONFIG_LOG_DIR" -type f \( -name "*.log" -o -name "*.txt" -o -name "*.json" -o -name "*.csv" \) | grep -q .; then
    echo "Backing up files from $CONFIG_LOG_DIR to $BACKUP_DIR"
    mkdir -p "$BACKUP_DIR"
    find "$CONFIG_LOG_DIR" -type f \( -name "*.log" -o -name "*.txt" -o -name "*.json" -o -name "*.csv" \) -exec bash -c '
      for file; do
        relative_path="${file#'"$CONFIG_LOG_DIR"'/}"
        mkdir -p "'"$BACKUP_DIR"'"/$(dirname "$relative_path")
        mv "$file" "'"$BACKUP_DIR"'"/"$relative_path"
      done
    ' bash {} +
  fi
}

for server_config in "${server_config_list[@]}"; do
  IFS=',' read -r MAX_MODEL_LEN MAX_CONCURRENCY PP_SIZE TP_SIZE COMM_BACKEND KV_CACHE_DTYPE DO_WARMUP DO_PROFILE PARTITION <<< "$server_config"
  PARTITION=$(echo $PARTITION | tr -d '[]')

  CONFIG_LOG_DIR=${BASE_LOG_DIR}
  CONFIG_LOG_DIR=${CONFIG_LOG_DIR}/kv_${KV_CACHE_DTYPE}
  CONFIG_LOG_DIR=${CONFIG_LOG_DIR}/tp${TP_SIZE}_pp${PP_SIZE}
  CONFIG_LOG_DIR=${CONFIG_LOG_DIR}/pp_comm_${COMM_BACKEND}
  CONFIG_LOG_DIR=${CONFIG_LOG_DIR}/warmup_${DO_WARMUP}
  CONFIG_LOG_DIR=${CONFIG_LOG_DIR}/profile_${DO_PROFILE}
  mkdir -p "$CONFIG_LOG_DIR"

  # Back up existing logs. Does not work intuitively as it backups older backups.
  # backup_benchmark_logs "$CONFIG_LOG_DIR"

  SERVER_LOG_PREFIX=mml${MAX_MODEL_LEN}_conc${MAX_CONCURRENCY}

  # Kill any existing OpenAI server processes if they exist
  echo "Checking for any previous servers..."
  if ps -ef | grep openai | grep -v grep > /dev/null; then
    echo "Killing previous OpenAI server processes..."
    ps -ef | grep openai | grep -v grep | awk '{print $2}' | xargs -r kill -9 2>/dev/null
    sleep 25
  else
    echo "No previous OpenAI server processes found."
  fi

  # Launch server
  if [ "$MAX_CONCURRENCY" == "1" ]; then
    PER_PP_CONCURRENCY=1
  else
    PER_PP_CONCURRENCY=$((MAX_CONCURRENCY / PP_SIZE))
  fi

  # Echo server configuration details
  echo "Starting server with the following configuration:"
  echo "  NUM_NODES: ${NUM_NODES}"
  echo "  MAX_MODEL_LEN: ${MAX_MODEL_LEN}"
  echo "  PER_PP_CONCURRENCY: ${PER_PP_CONCURRENCY}"
  echo "  TP_SIZE (Tensor Parallelism): ${TP_SIZE}"
  echo "  PP_SIZE (Pipeline Parallelism): ${PP_SIZE}"
  echo "  COMM_BACKEND: ${COMM_BACKEND}"
  echo "  PARTITION: ${PARTITION}"
  echo "  KV_CACHE_DTYPE: ${KV_CACHE_DTYPE}"
  echo "  WARMUP: ${DO_WARMUP}"
  echo "  PROFILE: ${DO_PROFILE}"
  echo "  HOST: ${HOST}"
  echo "  PORT: ${PORT}"
  echo "  MODEL_PATH: ${MODEL_PATH}"
  echo "  LOG_FILE: ${CONFIG_LOG_DIR}/server_${SERVER_LOG_PREFIX}.log"

  # Start server
  bash -x benchmark_server_param.sh ${NUM_NODES} ${MAX_MODEL_LEN} ${PER_PP_CONCURRENCY} \
    ${TP_SIZE} ${PP_SIZE} ${COMM_BACKEND} "${PARTITION}" ${KV_CACHE_DTYPE} ${WARMUP} \
    ${DO_PROFILE} ${HOST} ${PORT} ${MODEL_PATH} ${CONFIG_LOG_DIR} \
    > ${CONFIG_LOG_DIR}/server_${SERVER_LOG_PREFIX}.log 2>&1 &
  SERVER_LAUNCH_PID=$!

  # Wait for server startup
  connected=0
  pid_seen=0  # Switch to track if the server PID has been seen
  if [ "$DO_WARMUP" == "true" ]; then
    timeout=90000
  else
    timeout=9000
  fi
  interval=5
  start_time=$(date +%s)
  connected_info="Application startup complete"

  # Progress bar setup
  progress_bar_length=50  # Length of the progress bar
  echo -n "Waiting for server startup: ["
  for ((i = 0; i < progress_bar_length; i++)); do echo -n " "; done
  echo -n "]"
  echo -ne "\rWaiting for server startup: ["

  while :; do
    if [ "$pid_seen" -eq 1 ]; then
      # Check for successful startup message
      if grep -q "$connected_info" ${CONFIG_LOG_DIR}/server_${SERVER_LOG_PREFIX}.log; then
        connected=1
        break
      # Check for fatal errors in the server log
      elif grep -q "Fatal Python error" ${CONFIG_LOG_DIR}/server_${SERVER_LOG_PREFIX}.log; then
        connected=0
        echo "Server failed to launch with Fatal Python error, shutting down..."
        break
      elif grep -q "ValueError:" ${CONFIG_LOG_DIR}/server_${SERVER_LOG_PREFIX}.log; then
        connected=0
        echo "Server failed to launch with ValueError, shutting down..."
        break
      elif grep -q "RuntimeError:" ${CONFIG_LOG_DIR}/server_${SERVER_LOG_PREFIX}.log; then
        connected=0
        echo "Server failed to launch with RuntimeError, shutting down..."
        break
      elif grep -q "OSError:" ${CONFIG_LOG_DIR}/server_${SERVER_LOG_PREFIX}.log; then
        connected=0
        echo "Server failed to launch with OSError, shutting down..."
        break
      fi
    fi

    # Check for the server PID
    SERVER_PID=$(ps -ef | grep openai | grep -v grep | awk '{print $2}')
    if [ -n "$SERVER_PID" ]; then
      pid_seen=1  # Mark that the PID has been seen
    elif [ "$pid_seen" -eq 1 ]; then
      # If the PID was seen before but is no longer running, terminate early
      echo "Server process crashed after being detected. Exiting..."
      connected=0
      break
    fi

    # Update progress bar
    current_time=$(date +%s)
    elapsed_time=$((current_time - start_time))
    progress=$((elapsed_time * progress_bar_length / timeout))
    echo -ne "\rWaiting for server startup: ["
    for ((i = 0; i < progress_bar_length; i++)); do
      if [ "$i" -lt "$progress" ]; then
        echo -n "="
      else
        echo -n " "
      fi
    done
    echo -n "] $elapsed_time/$timeout seconds"

    # Check if the timeout has been exceeded
    if [ "$elapsed_time" -ge "$timeout" ]; then
      echo -e "\nServer startup timed out after $timeout seconds. Exiting..."
      connected=0
      break
    fi

    sleep $interval
  done

  # Finalize progress bar
  if [ "$connected" -eq 1 ]; then
    echo -e "\rWaiting for server startup: [$(printf '=%.0s' $(seq 1 $progress_bar_length))] Done after $elapsed_time/$timeout seconds!"
  fi

  if [ "$connected" -eq 0 ]; then
    echo "[Error] Server failed to start. Check logs. Continuing to next server configuration"
    # Make sure server is entirely torn down.
    kill -9 $SERVER_PID 2>/dev/null
    sleep 20
    continue
  fi

  for client_config in "${client_config_list[@]}"; do
    IFS=',' read -r INPUT_TOKENS OUTPUT_TOKENS NUM_PROMPTS CLIENT_CONCURRENCY <<< "$client_config"
    CLIENT_LOG_PREFIX=in${INPUT_TOKENS}_out${OUTPUT_TOKENS}_prompts${NUM_PROMPTS}_conc${CLIENT_CONCURRENCY}

    if ! [[ "$CLIENT_CONCURRENCY" =~ ^[0-9]+$ ]] || ! [[ "$MAX_CONCURRENCY" =~ ^[0-9]+$ ]]; then
      echo "[Error] CLIENT_CONCURRENCY or MAX_CONCURRENCY is not a valid integer. Skipping..."
      continue
    fi

    if [ $CLIENT_CONCURRENCY -gt $MAX_CONCURRENCY ]; then
      echo "Client concurrency ($CLIENT_CONCURRENCY) greater than max concurrency ($MAX_CONCURRENCY). Skipping..."
      continue
    fi

    # Run client benchmark
    source benchmark_client_param.sh
    if [ "$DO_WARMUP" != "true" ]; then
      test_benchmark_client_serving ${INPUT_TOKENS} ${OUTPUT_TOKENS} ${CLIENT_CONCURRENCY} ${NUM_PROMPTS} 0.8 ${HOST} ${PORT} ${MODEL_PATH} ${CONFIG_LOG_DIR} \
        | tee -a ${CONFIG_LOG_DIR}/warmup_${CLIENT_LOG_PREFIX}.log
    fi
    if [ "$DO_PROFILE" == "true" ]; then
      start_profile
    fi
    test_benchmark_client_serving ${INPUT_TOKENS} ${OUTPUT_TOKENS} ${CLIENT_CONCURRENCY} ${NUM_PROMPTS} 0.8 ${HOST} ${PORT} ${MODEL_PATH} ${CONFIG_LOG_DIR} \
      | tee -a ${CONFIG_LOG_DIR}/benchmark_${CLIENT_LOG_PREFIX}.log
    if [ "$DO_PROFILE" == "true" ]; then
      stop_profile
    fi

    # Collect metrics
    mean_ttft=$(grep 'Mean TTFT (ms):' ${CONFIG_LOG_DIR}/benchmark_${CLIENT_LOG_PREFIX}.log | tail -1 | awk '{print $NF}')
    mean_tpot=$(grep 'Mean TPOT (ms):' ${CONFIG_LOG_DIR}/benchmark_${CLIENT_LOG_PREFIX}.log | tail -1 | awk '{print $NF}')
    total_throughput=$(grep 'Total Token throughput (tok/s):' ${CONFIG_LOG_DIR}/benchmark_${CLIENT_LOG_PREFIX}.log | tail -1 | awk '{print $NF}')
    output_throughput=$(grep 'Output token throughput (tok/s):' ${CONFIG_LOG_DIR}/benchmark_${CLIENT_LOG_PREFIX}.log | tail -1 | awk '{print $NF}')

    # Build summary line
    row="${NUM_NODES},${PP_SIZE},${TP_SIZE},${COMM_BACKEND},${KV_CACHE_DTYPE},\"${PARTITION}\",${MAX_MODEL_LEN},${INPUT_TOKENS},${OUTPUT_TOKENS},${NUM_PROMPTS},${MAX_CONCURRENCY},${CLIENT_CONCURRENCY},${DO_WARMUP},${DO_PROFILE},${mean_ttft},${mean_tpot},${total_throughput},${output_throughput}"

    echo "$row" | tee -a "$SUMMARY_LOG"
  done

  # Teardown server
  echo "Tearing down server..."
  kill -9 $SERVER_PID
  sleep 20

done
