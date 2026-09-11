#!/bin/bash
# Prefill + decode started through `vllm-rs serve --grpc-port` (the only launcher that
# exposes the Rust frontend's gRPC control plane), then test_grpc_handshake.py checks the
# KV transfer discovery/handshake RPCs against the ZMQ side channel and runs a P/D request.
set -xe

PREFILL_GPU_ID="${PREFILL_GPU_ID:-0}"
DECODE_GPU_ID="${DECODE_GPU_ID:-1}"
MODEL="${MODEL:-Qwen/Qwen3-0.6B}"
KV_CONFIG='{"kv_connector":"NixlConnector","kv_role":"kv_both"}'
# The decoder may only use handshake metadata pushed by its frontend (fetched over the
# prefiller's control plane); the ZMQ fallback is disabled so a regression cannot hide.
DECODE_KV_CONFIG='{"kv_connector":"NixlConnector","kv_role":"kv_both","kv_connector_extra_config":{"handshake_transport":"grpc"}}'

PREFILL_PORT=8001
DECODE_PORT=8002
PROXY_PORT=8192
PREFILL_GRPC_PORT=50151
DECODE_GRPC_PORT=50152
PREFILL_SIDE_CHANNEL_PORT=5559
DECODE_SIDE_CHANNEL_PORT=6000

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
GIT_ROOT="${GIT_ROOT:-$(cd -- "${SCRIPT_DIR}/../../../.." && pwd -P)}"
VLLM_RS="${VLLM_RS:-$(python3 -c 'import os, vllm; print(os.path.dirname(vllm.__file__))')/vllm-rs}"
test -x "$VLLM_RS"

# grpcio-tools generates the Python stubs from rust/proto/control.proto. It runs in its
# own process: its bundled libprotobuf crashes when loaded next to torch.
python3 -c "import grpc, grpc_tools.protoc" 2>/dev/null || uv pip install --system grpcio grpcio-tools
CONTROL_PB_DIR="$(mktemp -d)"
python3 -m grpc_tools.protoc -I "${GIT_ROOT}/rust/proto" \
  --python_out="$CONTROL_PB_DIR" --grpc_python_out="$CONTROL_PB_DIR" control.proto
export CONTROL_PB_DIR

trap 'kill $(jobs -pr) 2>/dev/null; pkill -f "vllm.entrypoints.cli.main serve" || true' SIGINT SIGTERM EXIT

wait_for_server() {
  timeout 1200 bash -c "until curl -s localhost:$1/v1/models > /dev/null; do sleep 1; done"
}

start_instance() { # <gpu> <http port> <grpc port> <side channel port> <kv config>
  CUDA_VISIBLE_DEVICES=$1 VLLM_NIXL_SIDE_CHANNEL_PORT=$4 "$VLLM_RS" serve "$MODEL" \
    --host 127.0.0.1 --port "$2" --grpc-port "$3" \
    --enforce-eager --gpu-memory-utilization 0.2 --max-model-len 8192 \
    --kv-transfer-config "$5" &
}

start_instance "$PREFILL_GPU_ID" $PREFILL_PORT $PREFILL_GRPC_PORT $PREFILL_SIDE_CHANNEL_PORT "$KV_CONFIG"
start_instance "$DECODE_GPU_ID" $DECODE_PORT $DECODE_GRPC_PORT $DECODE_SIDE_CHANNEL_PORT "$DECODE_KV_CONFIG"
wait_for_server $PREFILL_PORT
wait_for_server $DECODE_PORT

python3 "${GIT_ROOT}/tests/v1/kv_connector/nixl_integration/toy_proxy_server.py" \
  --port $PROXY_PORT --prefiller-ports $PREFILL_PORT --decoder-ports $DECODE_PORT &
sleep 5

PREFILL_PORT=$PREFILL_PORT DECODE_PORT=$DECODE_PORT PROXY_PORT=$PROXY_PORT \
PREFILL_GRPC_PORT=$PREFILL_GRPC_PORT DECODE_GRPC_PORT=$DECODE_GRPC_PORT \
PREFILL_SIDE_CHANNEL_PORT=$PREFILL_SIDE_CHANNEL_PORT \
  python3 -m pytest -s -v "${GIT_ROOT}/tests/v1/kv_connector/nixl_integration/test_grpc_handshake.py"
