#!/bin/bash

# Check for minimum number of required arguments
Help() {
    # Display Help
    echo "Usage: $0 docker_image head_node_address --head|--worker path_to_hf_home"
    echo "       [-h] [-d hpu|gpu] [-c true|false] [-- additional_args..."]
}

if [ $# -lt 4 ]; then
    Help
    exit 1
fi

# Assign the first four arguments and shift them away
DOCKER_IMAGE="$1"
HEAD_NODE_ADDRESS="$2"
NODE_TYPE="$3"  # Should be --head or --worker
PATH_TO_HF_HOME="$4"
shift 4

PLATFORM="gpu"
CLEANUP_ON_EXIT="true"

# Get the options
while getopts hd:c: flag; do
    case $flag in
    h) # display Help
        Help
        exit
        ;;
    d) # get the device type
        PLATFORM=$OPTARG ;;
    c) # get TP value
        CLEANUP_ON_EXIT=$OPTARG ;;
    \?) # Invalid option
        echo "Error: Invalid option"
        Help
        exit
        ;;
    esac
done

# Shift the processed options and their arguments
shift $((OPTIND - 1))

# Additional arguments are passed directly to the Docker command
ADDITIONAL_ARGS=("$@")

# Validate node type
if [ "${NODE_TYPE}" != "--head" ] && [ "${NODE_TYPE}" != "--worker" ]; then
    echo "Error: Node type must be --head or --worker"
    exit 1
fi

# Define a function to cleanup on EXIT signal
cleanup() {
    docker stop node
    docker rm node
}
if [[ "$CLEANUP_ON_EXIT" == "true" ]]; then
    trap cleanup EXIT
fi

# Command setup for head or worker node
RAY_START_CMD="ray start --block"
if [ "${NODE_TYPE}" == "--head" ]; then
    RAY_START_CMD+=" --head --node-ip-address ${HEAD_NODE_ADDRESS} --port=6379"
else
    RAY_START_CMD+=" --address=${HEAD_NODE_ADDRESS}:6379"
fi

# Run the docker command with the user specified parameters and additional arguments
if [[ "$PLATFORM" == "hpu" ]]; then
    docker run \
        -td \
        --entrypoint /bin/bash \
        --network host \
        --ipc=host \
        --name node \
        --runtime=habana \
        -e HABANA_VISIBLE_DEVICES=all \
        -e GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME} \
        -e HCCL_SOCKET_IFNAME=${HCCL_SOCKET_IFNAME} \
        -v "${PATH_TO_HF_HOME}:/root/.cache/huggingface" \
        "${ADDITIONAL_ARGS[@]}" \
        "${DOCKER_IMAGE}" -c "${RAY_START_CMD}"
else
    docker run \
        --entrypoint /bin/bash \
        --network host \
        --name node \
        --shm-size 10.24g \
        --gpus all \
        -v "${PATH_TO_HF_HOME}:/root/.cache/huggingface" \
        "${ADDITIONAL_ARGS[@]}" \
        "${DOCKER_IMAGE}" -c "${RAY_START_CMD}"
fi
