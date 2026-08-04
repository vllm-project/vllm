#!/bin/bash

set -x

# Install lmcache
pip install -e .

# Pull the latest lmcache-vllm
cd ../lmcache-vllm
git pull

# Install requirements for benchmark scripts
cd ../benchmark
pip install -r ./benchmarks/multi_round_qa/requirements.txt

set +x

# Start the server
LMCACHE_TRACK_USAGE="false" lmcache_vllm serve mistralai/Mistral-7B-Instruct-v0.2 --disable-log-requests > lmcache_vllm.log 2>&1 &
echo "Waiting for service to start..."
timeout=90  # Timeout duration in seconds
elapsed=0   # Track elapsed time
until grep -q "Uvicorn running on" lmcache_vllm.log; do
  if [ $elapsed -ge $timeout ]; then
    echo "Timeout reached: Service did not start within $timeout seconds."
    exit 1
  fi
  sleep 10
  elapsed=$((elapsed + 10))
  echo "Waiting... ($elapsed seconds elapsed)"
done
echo "Service started successfully."

# Run benchmark scripts
python3 benchmarks/multi-round-qa.py \
    --num-users 10 \
    --num-rounds 5 \
    --qps 0.5 \
    --shared-system-prompt 1000 \
    --user-history-prompt 2000 \
    --answer-len 100 \
    --model mistralai/Mistral-7B-Instruct-v0.2 \
    --base-url http://localhost:8000/v1 \
    --time 300
    
