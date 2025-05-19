#!/bin/bash

# Check for minimum number of required arguments
if [ $# -lt 4 ]; then
    echo "Usage: $0 docker_image head_node_address --head|--worker path_to_hf_home [--dashboard] [additional_args...]"
    exit 1
fi

# Assign the first three arguments and shift them away
DOCKER_IMAGE="$1"
HEAD_NODE_ADDRESS="$2"
NODE_TYPE="$3"  # Should be --head or --worker
PATH_TO_HF_HOME="$4"
shift 4

# Additional arguments are passed directly to the Docker command
ADDITIONAL_ARGS=("$@")

# Validate node type
if [ "${NODE_TYPE}" != "--head" ] && [ "${NODE_TYPE}" != "--worker" ]; then
    echo "Error: Node type must be --head or --worker"
    exit 1
fi

# Process additional arguments to check for --dashboard
DASHBOARD_ENABLED=false
NEW_ADDITIONAL_ARGS=()
for arg in "${ADDITIONAL_ARGS[@]}"; do
    if [ "$arg" == "--dashboard" ]; then
        DASHBOARD_ENABLED=true
    else
        NEW_ADDITIONAL_ARGS+=("$arg")
    fi
done
ADDITIONAL_ARGS=("${NEW_ADDITIONAL_ARGS[@]}")

# Define a function to cleanup on EXIT signal
cleanup() {
    docker stop node
    docker rm node
}
trap cleanup EXIT

# Command setup for head or worker node
RAY_START_CMD="ray start --block"
if [ "${NODE_TYPE}" == "--head" ]; then
    RAY_START_CMD+=" --head --port=6379"
    if [ "${DASHBOARD_ENABLED}" == "true" ]; then
        RAY_START_CMD+=" --dashboard-host=0.0.0.0 --include-dashboard=true"
    fi
else
    RAY_START_CMD+=" --address=${HEAD_NODE_ADDRESS}:6379"
fi

# Prepend pip install command if dashboard is enabled and it's the head node
if [ "${DASHBOARD_ENABLED}" == "true" ] && [ "${NODE_TYPE}" == "--head" ]; then
    RAY_START_CMD="pip install \"ray[default]\" && ${RAY_START_CMD}"
fi

# Run the docker command with the user specified parameters and additional arguments
docker run \
    --entrypoint /bin/bash \
    --network host \
    --name node \
    --shm-size 10.24g \
    --gpus all \
    -v "${PATH_TO_HF_HOME}:/root/.cache/huggingface" \
    "${ADDITIONAL_ARGS[@]}" \
    "${DOCKER_IMAGE}" -c "${RAY_START_CMD}"
