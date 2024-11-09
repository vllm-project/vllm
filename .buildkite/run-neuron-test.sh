#!/bin/bash

# This script build the Neuron docker image and run the API server inside the container.
# It serves a sanity check for compilation and basic model usage.
set -e

# Try building the docker image
aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin 763104351884.dkr.ecr.us-west-2.amazonaws.com

# prune old image and containers to save disk space, and only once a day
# by using a timestamp file in tmp.
if [ -f /tmp/neuron-docker-build-timestamp ]; then
    last_build=$(cat /tmp/neuron-docker-build-timestamp)
    current_time=$(date +%s)
    if [ $((current_time - last_build)) -gt 86400 ]; then
        docker system prune -f
        echo "$current_time" > /tmp/neuron-docker-build-timestamp
    fi
else
    date "+%s" > /tmp/neuron-docker-build-timestamp
fi

docker build -t neuron -f Dockerfile.neuron .

# Setup cleanup
remove_docker_container() { docker rm -f neuron || true; }
trap remove_docker_container EXIT
remove_docker_container

# Run the image
docker run --device=/dev/neuron0 --device=/dev/neuron1 --network host --name neuron neuron python3 -m vllm.entrypoints.api_server \
       --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --max-num-seqs 8 --max-model-len 128 --block-size 128 --device neuron --tensor-parallel-size 2 &

# Wait for the server to start
wait_for_server_to_start() {
    timeout=300
    counter=0

    while [ "$(curl -s -o /dev/null -w '%{http_code}' localhost:8000/health)" != "200" ]; do
        sleep 1
        counter=$((counter + 1))
        if [ $counter -ge $timeout ]; then
            echo "Timeout after $timeout seconds"
            break
        fi
    done
}
wait_for_server_to_start

# Test a simple prompt
curl -X POST -H "Content-Type: application/json" \
    localhost:8000/generate \
    -d '{"prompt": "San Francisco is a"}'
