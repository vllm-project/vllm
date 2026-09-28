#!/bin/bash
#
# Launch a Ray cluster inside Docker for vLLM inference.
#
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
# Each worker requires a unique VLLM_HOST_IP value.
# Keep each terminal session open. Closing a session stops the associated Ray
# node and thereby shuts down the entire cluster.
# Every machine must be reachable at the supplied IP address.
#
# The container is named "node-<random_suffix>". To open a shell inside
# a container after launch, use:
#       docker exec -it node-<random_suffix> /bin/bash
#
# Then, you can execute vLLM commands on the Ray cluster as if it were a
# single machine, e.g. vllm serve ...
#
# To stop the container, use:
#       docker stop node-<random_suffix>

# Check for minimum number of required arguments.
if [ $# -lt 4 ]; then
    echo "Usage: $0 docker_image head_node_ip --head|--worker path_to_hf_home [additional_args...]"
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

# Extract VLLM_HOST_IP from ADDITIONAL_ARGS (e.g. "-e VLLM_HOST_IP=...").
VLLM_HOST_IP=""
for ((i = 0; i < ${#ADDITIONAL_ARGS[@]}; i++)); do
    arg="${ADDITIONAL_ARGS[$i]}"
    case "${arg}" in
        -e)
            next="${ADDITIONAL_ARGS[$((i + 1))]:-}"
            if [[ "${next}" == VLLM_HOST_IP=* ]]; then
                VLLM_HOST_IP="${next#VLLM_HOST_IP=}"
                break
            fi
            ;;
        -eVLLM_HOST_IP=* | VLLM_HOST_IP=*)
            VLLM_HOST_IP="${arg#*=}"
            break
            ;;
    esac
done

# For the head node, HEAD_NODE_ADDRESS and VLLM_HOST_IP should be consistent.
if [[ "${NODE_TYPE}" == "--head" && -n "${VLLM_HOST_IP}" ]]; then
    if [[ "${VLLM_HOST_IP}" != "${HEAD_NODE_ADDRESS}" ]]; then
        echo "Warning: VLLM_HOST_IP (${VLLM_HOST_IP}) differs from head_node_ip (${HEAD_NODE_ADDRESS})."
        echo "Using VLLM_HOST_IP as the head node address."
        HEAD_NODE_ADDRESS="${VLLM_HOST_IP}"
    fi
fi

# Generate a unique container name with random suffix.
# Docker container names must be unique on each host.
# The random suffix allows multiple Ray containers to run simultaneously on the same machine,
# for example, on a multi-GPU machine.
CONTAINER_NAME="node-${RANDOM}"

# Define a cleanup routine that removes the container when the script exits.
# This prevents orphaned containers from accumulating if the script is interrupted.
cleanup() {
    docker stop "${CONTAINER_NAME}"
    docker rm "${CONTAINER_NAME}"
}
trap cleanup EXIT

# Build the Ray start command based on the node role.
# The head node manages the cluster and accepts connections on port 6379,
# while workers connect to the head's address.
RAY_START_CMD="ray start --block"
if [ "${NODE_TYPE}" == "--head" ]; then
    RAY_START_CMD+=" --head --node-ip-address=${HEAD_NODE_ADDRESS} --port=6379"
else

    RAY_START_CMD+=" --address=${HEAD_NODE_ADDRESS}:6379"
    if [ -n "${VLLM_HOST_IP}" ]; then
        RAY_START_CMD+=" --node-ip-address=${VLLM_HOST_IP}"
    fi
fi

# Launch the container with the assembled parameters.
# --network host: Allows Ray nodes to communicate directly via host networking
# --shm-size 10.24g: Increases shared memory
# --gpus all: Gives container access to all GPUs on the host
# -v HF_HOME: Mounts HuggingFace cache to avoid re-downloading models
docker run \
    --entrypoint /bin/bash \
    --network host \
    --name "${CONTAINER_NAME}" \
    --shm-size 10.24g \
    --gpus all \
    -v "${PATH_TO_HF_HOME}:/root/.cache/huggingface" \
    "${ADDITIONAL_ARGS[@]}" \
    "${DOCKER_IMAGE}" -c "${RAY_START_CMD}"
