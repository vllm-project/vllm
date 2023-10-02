#!/bin/bash

GRPC_PORT=8001
MODEL=$1

if [ -z "$1" ]; then
  echo "Error: Model name is missing. Please provide a value."
  echo "The script can be run with the command \"./launch_triton_server.sh <model_name>\"."
  exit 1
fi

set -e

# Clone the Triton tutorials repository with the example vLLM deployment
# Build the server image with the necessary dependencies
mkdir -p triton && \
   cd triton && \
   rm -rf tutorials
git clone https://github.com/triton-inference-server/tutorials.git && \
   cd tutorials/Quick_Deploy/vLLM && \
   docker build -t tritonserver_vllm .

cd triton/tutorials/Quick_Deploy/vLLM
# Replace the model name in the engine arguments
JSON_FILE="model_repository/vllm/vllm_engine_args.json"
jq --arg new_model "$MODEL" '.model = $new_model' "$JSON_FILE"

docker run --gpus all -it --rm -p $GRPC_PORT:8001 --shm-size=1G \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    -v ${PWD}:/work -w /work tritonserver_vllm \
    tritonserver --model-store ./model_repository
    
