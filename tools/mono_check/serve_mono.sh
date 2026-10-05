#!/bin/bash
# Launch the OpenAI-compatible server (vllm serve) with the live GLM-5.2 MonoKernel.
# Needs a ROCm vLLM env with this branch. Foreground; wrap in timeout.
#   MODEL=<GLM-5.2 checkpoint> MODE=graphs|eager ARM=C|A PORT=8199 [POLL_LIMIT=20000000]
#   [EXTRA="..."] tools/mono_check/serve_mono.sh
# ARM=A serves vLLM as shipped (same flags, worker extension loaded for the dispatch recorder,
# but no MonoKernel install).
set -u
MODE=${MODE:-graphs}
ARM=${ARM:-C}
PORT=${PORT:-8199}
# ~19 s per mailbox wait (tolerates host skew); the fail-stop watch catches expiries
POLL_LIMIT=${POLL_LIMIT:-100000000}
MODEL=${MODEL:?set MODEL to a local GLM-5.2 checkpoint directory}
INDEXER_MODE=${INDEXER_MODE:-indexer_only}
export VLLM_ROCM_USE_AITER=1
export VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD=0
export VLLM_SERVER_DEV_MODE=1          # exposes POST /collective_rpc (mono_health)
if [ "$MODE" = graphs ]; then STEP_SYNC=false; else STEP_SYNC=true; fi
if [ "$ARM" = C ]; then
  export MONO_LIVE_PREINSTALL="{\"ckpt\": \"$MODEL\", \"layers\": [$(seq -s, 3 77)], \"sizes\": [1,2,4,5,6,8], \"poll_limit\": $POLL_LIMIT, \"indexer_mode\": \"$INDEXER_MODE\", \"step_sync\": $STEP_SYNC, \"max_model_len\": 4096}"
else
  unset MONO_LIVE_PREINSTALL
fi
ARGS=(--tensor-parallel-size 8 --max-model-len 4096 --gpu-memory-utilization 0.70 --max-num-seqs 8
      --trust-remote-code --kv-cache-dtype auto --block-size 16 --num-gpu-blocks-override 8192
      --served-model-name glm52 --port "$PORT" --disable-uvicorn-access-log
      --worker-extension-cls tools.mono_check.harness.serve_ext.MonoServeWorkerExtension)
if [ "$MODE" = graphs ]; then
  ARGS+=(--compilation-config '{"mode": 0, "cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [1,2,4,5,6,8]}')
else
  ARGS+=(--enforce-eager)
fi
REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO"
export PYTHONPATH=$REPO${PYTHONPATH:+:$PYTHONPATH}
echo "serve_mono: MODE=$MODE ARM=$ARM PORT=$PORT step_sync=$STEP_SYNC indexer_mode=$INDEXER_MODE poll_limit=$POLL_LIMIT"
exec vllm serve "$MODEL" "${ARGS[@]}" ${EXTRA:-}
