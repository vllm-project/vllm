#!/bin/bash
set -e

GIT_ROOT=$(git rev-parse --show-toplevel 2>/dev/null)
# Trap the SIGINT signal (triggered by Ctrl+C)
trap 'pids=$(jobs -pr); [ -n "$pids" ] && kill $pids' SIGINT SIGTERM EXIT

# Hosts / ports
PREFILL_HOST=${PREFILL_HOST:-"localhost"}
PREFILL_PORT=${PREFILL_PORT:-8100}
DECODE_HOST=${DECODE_HOST:-"localhost"}
DECODE_PORT=${DECODE_PORT:-8200}
PROXY_HOST=${PROXY_HOST:-"localhost"}
PROXY_PORT=${PROXY_PORT:-8192}
BASELINE_HOST=${BASELINE_HOST:-"localhost"}
BASELINE_PORT=${BASELINE_PORT:-9290}

# Model to run.
MODEL_NAME=${MODEL_NAME:-"/mnt/weka/data/pytorch/llama3/Meta-Llama-3-8B-Instruct/"}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-1024}
VLLM_GPU_MEMORY_UTILIZATION=0.8
MODEL_LEN=2048
max_num_batched_tokens=2048
max_num_seqs=16

OUTPUT_FILE="hpu_accuracy_test_outputs.txt"

start_etcd_and_mooncake() {
  etcd --listen-client-urls http://0.0.0.0:2379 --advertise-client-urls http://localhost:2379 > etcd.log 2>&1 &
  mooncake_master -enable_gc true -port 50001 &> mooncake_master.log &
  sleep 2
}

cleanup() {
  echo "Cleaning up..."
  sleep 2
  pkill -f etcd || true
  pkill -f mooncake_master || true
  pkill -f "vllm serve" || true
  pkill -f "disagg_proxy_demo.py" || true
  sleep 2
  echo "Cleanup complete."
}

wait_for_server() {
  local host=$1
  local port=$2
  timeout 1200 bash -c "
    until curl -s ${host}:${port}/v1/completions > /dev/null; do
      sleep 1
    done" && return 0 || return 1
}

launch_baseline() {
  BASELINE_BASE_CMD="
  HABANA_VISIBLE_DEVICES="0" \
  VLLM_USE_V1=0 \
  VLLM_SKIP_WARMUP=True \
  vllm serve $MODEL_NAME \
      --port $BASELINE_PORT \
      --seed 42 \
      --max-model-len $MODEL_LEN \
      --gpu-memory-utilization $VLLM_GPU_MEMORY_UTILIZATION \
      -tp 1 \
      --max-num-seqs $max_num_seqs \
      --trust-remote-code  \
      --disable-log-requests \
      --max-num-batched-tokens $max_num_batched_tokens \
      --use-padding-aware-scheduling \
      --dtype bfloat16 \
      --enforce-eager
  "
  echo ${BASELINE_BASE_CMD}      
  bash -c "${BASELINE_BASE_CMD}" &
}

launch_pd() {
  PREFILL_BASE_CMD="
  HABANA_VISIBLE_DEVICES="0" \
  MOONCAKE_CONFIG_PATH=./mooncake.json \
  VLLM_USE_V1=0 \
  VLLM_SKIP_WARMUP=True \
  vllm serve $MODEL_NAME \
      --port 8100 \
      --seed 42 \
      --max-model-len $MODEL_LEN \
      --gpu-memory-utilization $VLLM_GPU_MEMORY_UTILIZATION \
      -tp 1 \
      --max-num-seqs $max_num_seqs \
      --trust-remote-code  \
      --disable-log-requests \
      --max-num-batched-tokens $max_num_batched_tokens \
      --use-padding-aware-scheduling \
      --dtype bfloat16 \
      --enforce-eager \
      --kv-transfer-config '{\"kv_connector\":\"MooncakeStoreConnector\",\"kv_role\":\"kv_producer\"}'
  "


  DECODE_BASE_CMD="
  HABANA_VISIBLE_DEVICES="1" \
  MOONCAKE_CONFIG_PATH=./mooncake.json \
  VLLM_USE_V1=0 \
  VLLM_SKIP_WARMUP=True \
  vllm serve $MODEL_NAME \
      --port 8200 \
      --seed 42 \
      --max-model-len $MODEL_LEN \
      --gpu-memory-utilization $VLLM_GPU_MEMORY_UTILIZATION \
      -tp 1 \
      --max-num-seqs $max_num_seqs \
      --trust-remote-code  \
      --disable-log-requests \
      --max-num-batched-tokens $max_num_batched_tokens \
      --use-padding-aware-scheduling \
      --dtype bfloat16 \
      --enforce-eager \
      --kv-transfer-config '{\"kv_connector\":\"MooncakeStoreConnector\",\"kv_role\":\"kv_consumer\"}'
  "

  echo ${PREFILL_BASE_CMD}
  echo ${DECODE_BASE_CMD}
  sleep 2

  # execute on hosts
  bash -c "${PREFILL_BASE_CMD}" &
  bash -c "${DECODE_BASE_CMD}" &
  sleep 20
  wait_for_server ${PREFILL_HOST} ${PREFILL_PORT}
  sleep 1
  wait_for_server ${DECODE_HOST} ${DECODE_PORT}
  sleep 1
}

launch_pd_proxy(){
  PROXY_BASE_CMD="
  python3 ${GIT_ROOT}/examples/online_serving/disagg_examples/disagg_proxy_demo.py \
    --model $MODEL_NAME \
    --prefill localhost:8100 \
    --decode localhost:8200 \
    --port $PROXY_PORT"
  echo ${PROXY_BASE_CMD} 
  bash -c "${PROXY_BASE_CMD}" &
}

run_tests(){
  local service_url=$1
  local mode=$2
  python3 test_disagg_accuracy.py --service_url=${service_url} --model_name=$MODEL_NAME --mode=${mode} --file_name=${OUTPUT_FILE}
}


# run non-disagg. baseline & save outputs
launch_baseline
sleep 10
wait_for_server ${BASELINE_HOST} ${BASELINE_PORT}
run_tests "http://${BASELINE_HOST}:${BASELINE_PORT}" "baseline"
cleanup
sleep 10


# run disagg. & do exact-match with the outputs from baseline
start_etcd_and_mooncake
launch_pd
launch_pd_proxy
sleep 10
run_tests "http://${PROXY_HOST}:${PROXY_PORT}" "disagg"
echo "-----P/D success----"

rm ${OUTPUT_FILE}
cleanup

exit 0
