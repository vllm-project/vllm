#!/bin/bash

# This script build the Neuron docker image and run the API server inside the container.
# It serves a sanity check for compilation and basic model usage.
set -e
set -v

image_name="neuron/vllm-ci"
container_name="neuron_$(tr -dc A-Za-z0-9 < /dev/urandom | head -c 10; echo)"

HF_CACHE="$(realpath ~)/huggingface"
mkdir -p "${HF_CACHE}"
HF_MOUNT="/root/.cache/huggingface"

NEURON_COMPILE_CACHE_URL="$(realpath ~)/neuron_compile_cache"
mkdir -p "${NEURON_COMPILE_CACHE_URL}"
NEURON_COMPILE_CACHE_MOUNT="/root/.cache/neuron_compile_cache"

# Try building the docker image
aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin 763104351884.dkr.ecr.us-west-2.amazonaws.com

# prune old image and containers to save disk space, and only once a day
# by using a timestamp file in tmp.
if [ -f /tmp/neuron-docker-build-timestamp ]; then
    last_build=$(cat /tmp/neuron-docker-build-timestamp)
    current_time=$(date +%s)
    if [ $((current_time - last_build)) -gt 86400 ]; then
        docker image prune -f
        docker system prune -f
        rm -rf "${HF_MOUNT:?}/*"
        rm -rf "${NEURON_COMPILE_CACHE_MOUNT:?}/*"
        echo "$current_time" > /tmp/neuron-docker-build-timestamp
    fi
else
    date "+%s" > /tmp/neuron-docker-build-timestamp
fi

docker build -t "${image_name}" -f Dockerfile.neuron .

# Setup cleanup
remove_docker_container() {
    docker image rm -f "${image_name}" || true;
}
trap remove_docker_container EXIT

# Run the image
docker run --rm -it --device=/dev/neuron0 --device=/dev/neuron1 --network host \
       -v "${HF_CACHE}:${HF_MOUNT}" \
       -e "HF_HOME=${HF_MOUNT}" \
       -v "${NEURON_COMPILE_CACHE_URL}:${NEURON_COMPILE_CACHE_MOUNT}" \
       -e "NEURON_COMPILE_CACHE_URL=${NEURON_COMPILE_CACHE_MOUNT}" \
       --name "${container_name}" \
       ${image_name} \
       /bin/bash -c "python3 /workspace/vllm/examples/offline_inference/neuron.py"
