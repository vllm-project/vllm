#!/bin/bash
set -xe

# Models to run
MODELS=(
    "Qwen/Qwen3-0.6B"
)
MODELS=(
	"meta-llama/Llama-3.1-8B"
)

export VLLM_SKIP_WARMUP="true"
#export PT_HPU_LAZY_MODE=1

# Number of prefill and decode instances to create
NUM_PREFILL_INSTANCES=${NUM_PREFILL_INSTANCES:-1} # Default to 1
NUM_DECODE_INSTANCES=${NUM_DECODE_INSTANCES:-1}   # Default to 1
PREFILLER_TP_SIZE=${PREFILLER_TP_SIZE:-1}
DECODER_TP_SIZE=${DECODER_TP_SIZE:-1}

# Find the git repository root directory
#GIT_ROOT=$(git rev-parse --show-toplevel)
GIT_ROOT="/home/vllm-nixl/vllm"

#SMI_BIN=$(which nvidia-smi || which rocm-smi)

# Trap the SIGINT signal (triggered by Ctrl+C)
trap 'kill $(jobs -pr)' SIGINT SIGTERM EXIT

# Waits for vLLM to start.
wait_for_server() {
  local port=$1
  timeout 1200 bash -c "
    until curl -s localhost:${port}/v1/completions > /dev/null; do
      sleep 1  
    done" && return 0 || return 1
}

# Function to clean up previous instances
cleanup_instances() {
  echo "Cleaning up any running vLLM instances..."
  pkill -f "vllm serve" || true
  sleep 2
}

# Handle to get model-specific arguments for deepseek
get_model_args() {
  local model_name=$1
  local extra_args=""

  if [[ "$model_name" == "deepseek-ai/deepseek-vl2-tiny" ]]; then
    extra_args="--hf_overrides '{\"architectures\": [\"DeepseekVLV2ForCausalLM\"]}' --trust-remote-code"
  fi

  echo "$extra_args"
}

get_num_gpus() {
  if [[ "$SMI_BIN" == *"nvidia"* ]]; then
    echo "$($SMI_BIN --query-gpu=name --format=csv,noheader | wc -l)"
  else
    echo "$($SMI_BIN -l | grep GPU | wc -l)"
  fi
}

# Function to run tests for a specific model
run_tests_for_model() {
  local model_name=$1
  echo "================================"
  echo "Testing model: $model_name"
  echo "================================"

  # Get model-specific arguments
  local model_args=$(get_model_args "$model_name")

  # Arrays to store all hosts and ports
  PREFILL_HOSTS=()
  PREFILL_PORTS=()
  DECODE_HOSTS=()
  DECODE_PORTS=()

  # Start prefill instances
  for i in $(seq 0 $((NUM_PREFILL_INSTANCES-1))); do
    # Calculate GPU ID - we'll distribute across available GPUs
    #GPU_ID=$((i % $(get_num_gpus)))
    GPU_ID=2

    # Calculate port number (base port + instance number)
    PORT=$((8300 + i))
    # Calculate side channel port. Avoid clash with with TP workers. 
    SIDE_CHANNEL_PORT=$((6559 + i))

    echo "Starting prefill instance $i on GPU $GPU_ID, port $PORT"

    # Build the command with or without model-specific args
    BASE_CMD="CUDA_VISIBLE_DEVICES=$GPU_ID VLLM_NIXL_SIDE_CHANNEL_PORT=$SIDE_CHANNEL_PORT vllm serve $model_name \
    --port $PORT \
    --enforce-eager \
    --disable-log-requests \
    --gpu-memory-utilization 0.3 \
    --tensor-parallel-size $PREFILLER_TP_SIZE \
    --kv-transfer-config '{\"kv_connector\":\"NixlConnector\",\"kv_role\":\"kv_both\",\"kv_buffer_device\":\"cpu\"}'"

    if [ -n "$model_args" ]; then
    FULL_CMD="$BASE_CMD $model_args"
    else
    FULL_CMD="$BASE_CMD"
    fi

    eval "$FULL_CMD &"

    # Store host and port for proxy configuration
    PREFILL_HOSTS+=("localhost")
    PREFILL_PORTS+=($PORT)
  done

  # Start decode instances
  for i in $(seq 0 $((NUM_DECODE_INSTANCES-1))); do
    # Calculate GPU ID - we'll distribute across available GPUs, starting from after prefill GPUs
    #GPU_ID=$(((i + NUM_PREFILL_INSTANCES) % $(get_num_gpus)))
    GPU_ID=6
    # Calculate port number (base port + instance number)
    PORT=$((8400 + i))
    # Calculate side channel port
    SIDE_CHANNEL_PORT=$((5659 + i * $DECODER_TP_SIZE))

    echo "Starting decode instance $i on GPU $GPU_ID, port $PORT"

    # Build the command with or without model-specific args
    BASE_CMD="CUDA_VISIBLE_DEVICES=$GPU_ID VLLM_NIXL_SIDE_CHANNEL_PORT=$SIDE_CHANNEL_PORT vllm serve $model_name \
    --port $PORT \
    --enforce-eager \
    --disable-log-requests \
    --gpu-memory-utilization 0.3 \
    --tensor-parallel-size $DECODER_TP_SIZE \
    --kv-transfer-config '{\"kv_connector\":\"NixlConnector\",\"kv_role\":\"kv_both\",\"kv_buffer_device\":\"cpu\"}'"

    if [ -n "$model_args" ]; then
    FULL_CMD="$BASE_CMD $model_args"
    else
    FULL_CMD="$BASE_CMD"
    fi

    eval "$FULL_CMD &"

    # Store host and port for proxy configuration
    DECODE_HOSTS+=("localhost")
    DECODE_PORTS+=($PORT)
  done

  # Wait for all instances to start
  for PORT in "${PREFILL_PORTS[@]}"; do
    echo "Waiting for prefill instance on port $PORT to start..."
    wait_for_server $PORT
  done

  for PORT in "${DECODE_PORTS[@]}"; do
    echo "Waiting for decode instance on port $PORT to start..."
    wait_for_server $PORT
  done

  # Build the command for the proxy server with all the hosts and ports
  PROXY_CMD="python toy_proxy_server.py --port 9192"

  # Add all prefill hosts and ports
  PROXY_CMD+=" --prefiller-hosts ${PREFILL_HOSTS[@]}"
  PROXY_CMD+=" --prefiller-ports ${PREFILL_PORTS[@]}"

  # Add all decode hosts and ports
  PROXY_CMD+=" --decoder-hosts ${DECODE_HOSTS[@]}"
  PROXY_CMD+=" --decoder-ports ${DECODE_PORTS[@]}"

  # Start the proxy server
  echo "Starting proxy server with command: $PROXY_CMD"
  $PROXY_CMD &

  # Wait for the proxy to start
  sleep 10
  
# curl -X POST -s http://localhost:9192/v1/completions \
#	-H "Content-Type: application/json" \
#	-d '{
#	"model": "meta-llama/Llama-3.1-8B",
#	"prompt": "Mark Elliot Zuckerberg is an American businessman who co-founded the social media service Facebook and its parent company Meta Platforms, of which he is the chairman, chief executive officer, and controlling shareholder. Zuckerberg has been the subject of multiple lawsuits regarding the creation and ownership of the website as well as issues such as user privacy. Born in White Plains, New York, Zuckerberg briefly attended Harvard College, where he launched Facebook in February 2004 with his roommates Eduardo Saverin, Andrew McCollum, Dustin Moskovitz and Chris Hughes. Zuckerberg took the company public in May 2012 with majority shares. He became the worlds youngest self-made billionaire[a] in 2008, at age 23, and has consistently ranked among the worlds wealthiest individuals. According to Forbes, Zuckerbergs estimated net worth stood at US$221.2 billion as of May 2025, making him the second-richest individual in the world.[2]",
#	"max_tokens": 5,
#	"temperature": 0
#	}'
	sleep 5
	echo "--------------------===================-------------"
curl -X POST -s http://localhost:9192/v1/completions \
        -H "Content-Type: application/json" \
        -d '{
        "model": "meta-llama/Llama-3.1-8B",
        "prompt": "Mark Elliot Zuckerberg is an American businessman who co-founded the social media service Facebook and its parent company Meta Platforms, of which he is the chairman, chief executive officer, and controlling shareholder. Zuckerberg has been the subject of multiple lawsuits regarding the creation and ownership of the website as well as issues such as user privacy. Born in White Plains, New York, Zuckerberg briefly attended Harvard College, where he launched Facebook in February 2004 with his roommates Eduardo Saverin, Andrew McCollum, Dustin Moskovitz and Chris Hughes. Zuckerberg took the company public in May 2012 with majority shares. He became the worlds youngest self-made billionaire[a] in 2008, at age 23, and has consistently ranked among the worlds wealthiest individuals. According to Forbes, Zuckerbergs estimated net worth stood at US$221.2 billion as of May 2025, making him the second-richest individual in the world.[2] Intel opened its first international manufacturing facility in 1972, in Malaysia, which would host multiple Intel operations, before opening assembly facilities and semiconductor plants in Singapore and Jerusalem in the early 1980s, and manufacturing and development centers in China, India, and Costa Rica in the 1990s.[31] By the early 1980s, its business was dominated by DRAM chips. However, increased competition from Japanese semiconductor manufacturers had, by 1983, dramatically reduced the profitability of this market. The growing success of the IBM personal computer, based on an Intel microprocessor, was among factors that convinced Gordon Moore (CEO since 1975) to shift the companys focus to microprocessors and to change fundamental aspects of that business model. Moores decision to sole-source Intels 386 chip played into the companys continuing success.",
        "max_tokens": 5,
        "temperature": 0
        }'
 curl -X POST -s http://localhost:9192/v1/completions \
       -H "Content-Type: application/json" \
       -d '{
       "model": "meta-llama/Llama-3.1-8B",
       "prompt": ["This was a few months ago. It was my day off and the only thing I had to do was pick my girlfriend up from work at 9:00 pm. Other than that, I was free to loaf on the couch from morning to night, which is what I did. Around 8:00, I decided to shower before I left the house. Now, I have short hair that dries pretty quickly, but I am deeply vain about it, so I always dry it with the hairdryer right after I shower to ensure my hair doesnt get flat and weird. I never skip this step. So, I get out of the shower, start drying my hair... And then I wake up in bed. Its half an hour later. I feel like garbage, my entire body mysteriously hurts, and I am slowly realizing that I dont remember exiting the bathroom. My only clear thought is: oh shit, its 9:00! I have to pick up my girlfriend! Better shake myself awake. I dragged my aching carcass back to the bathroom, and this was when I noticed the massive blisters forming all over my hand. I was still pretty out of it, but I knew that this was a hospital visit kind of burn. My girlfriend then called to check in because I was running late and, despite my undoubtedly convincing argument that I was still perfectly fine to drive, she immediately knew something was wrong. She cabbed home and we got a ride to the ER. Turns out, I had my first ever seizure! It seems like during the seizure, I clenched the hairdryer in my fist and had it pointed at my other hand long enough to thoroughly cook it. The tissue loss is pretty deep in some areas and there was concerns about me retaining my mobility, but its been healing well so far.", 
       "Mark Elliot Zuckerberg is an American businessman who co-founded the social media service Facebook and its parent company Meta Platforms, of which he is the chairman, chief executive officer, and controlling shareholder. Zuckerberg has been the subject of multiple lawsuits regarding the creation and ownership of the website as well as issues such as user privacy. Born in White Plains, New York, Zuckerberg briefly attended Harvard College, where he launched Facebook in February 2004 with his roommates Eduardo Saverin, Andrew McCollum, Dustin Moskovitz and Chris Hughes. Zuckerberg took the company public in May 2012 with majority shares. He became the worlds youngest self-made billionaire[a] in 2008, at age 23, and has consistently ranked among the worlds wealthiest individuals. According to Forbes, Zuckerbergs estimated net worth stood at US$221.2 billion as of May 2025, making him the second-richest individual in the world.[2]"],
       "max_tokens": 2,
       "temperature": 0
       }'
  #sleep 10000
  # Run lm eval for this model
  echo "Running tests for $model_name"
  TEST_MODEL=$model_name python -m pytest -s -x test_accuracy.py

  # Clean up before running next model
  cleanup_instances
  sleep 3
}

# Run tests for each model
for model in "${MODELS[@]}"; do
  run_tests_for_model "$model"
done

echo "All tests completed!"
