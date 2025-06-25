#!/bin/bash
#
# Launch a Ray cluster inside Docker for vLLM inference.
# This script can start either a head node or a worker node, depending on the
# --head or --worker flag provided as the third positional argument.
#
# Usage:
# 1. Designate one machine as the head node and execute:
#    bash run_cluster.sh \
#         vllm/vllm-openai \
#         <head_node_ip> \
#         --head \
#         /abs/path/to/huggingface/cache \
#         -e VLLM_HOST_IP=<head_node_ip>
#
# 2. On every worker machine, execute:
#    bash run_cluster.sh \
#         vllm/vllm-openai \
#         <head_node_ip> \
#         --worker \
#         /abs/path/to/huggingface/cache \
#         -e VLLM_HOST_IP=<worker_node_ip>
#
# Keep each terminal session open. Closing a session stops the associated Ray
# node and thereby shuts down the entire cluster.
# Every machine must be reachable at the supplied IP address.
# Each worker requires a unique VLLM_HOST_IP value.

# Check for minimum number of required arguments.
if [ $# -lt 4 ]; then
    echo "Usage: $0 docker_image head_node_address --head|--worker path_to_hf_home [additional_args...]"
    exit 1
fi

# Extract the mandatory positional arguments and remove them from $@.
DOCKER_IMAGE="$1"
HEAD_NODE_ADDRESS="$2"
NODE_TYPE="$3"  # Should be --head or --worker.
PATH_TO_HF_HOME="$4"
shift 4

# Preserve any extra arguments so they can be forwarded to Docker.
ADDITIONAL_ARGS=("$@")

# Validate the NODE_TYPE argument.
if [ "${NODE_TYPE}" != "--head" ] && [ "${NODE_TYPE}" != "--worker" ]; then
    echo "Error: Node type must be --head or --worker"
    exit 1
fi

# Define a cleanup routine that removes the container when the script exits.
cleanup() {
    docker stop node
    docker rm node
}
trap cleanup EXIT

# Build the Ray start command based on the node role.
RAY_START_CMD="ray start --block"
if [ "${NODE_TYPE}" == "--head" ]; then
    RAY_START_CMD+=" --head --port=6379"
else
    RAY_START_CMD+=" --address=${HEAD_NODE_ADDRESS}:6379"
fi

# Launch the container with the assembled parameters.
docker run \
    --entrypoint /bin/bash \
    --network host \
    --name node \
    --shm-size 10.24g \
    --gpus all \
    -v "${PATH_TO_HF_HOME}:/root/.cache/huggingface" \
    "${ADDITIONAL_ARGS[@]}" \
    "${DOCKER_IMAGE}" -c "${RAY_START_CMD}"

#
# The container is named "node". To open a shell inside the container after
# launch, use:
#       docker exec -it node /bin/bash
#
# Then, you can execute vLLM commands on the Ray cluster as if it were a
# single machine, e.g. vllm serve ...
#
# To stop the container, use:
#       docker stop node
