#!/bin/bash

set -euox pipefail

# To detect ROCm
# Check multiple indicators:
if [ -e /dev/kfd ] || \
    [ -d /opt/rocm ] || \
    command -v rocm-smi &> /dev/null || \
    [ -n "${ROCM_HOME:-}" ]; then
    IS_ROCM=1
else
    IS_ROCM=0
fi

if [[ $# -lt 4 ]]; then
    echo "Usage: .buildkite/scripts/run-multi-node-test.sh WORKING_DIR NUM_NODES NUM_GPUS DOCKER_IMAGE COMMAND1 COMMAND2 ... COMMANDN"
    exit 1
fi

WORKING_DIR=$1
NUM_NODES=$2
NUM_GPUS=$3
DOCKER_IMAGE=$4

shift 4
COMMANDS=("$@")
if [ ${#COMMANDS[@]} -ne "$NUM_NODES" ]; then
    echo "The number of commands must be equal to the number of nodes."
    echo "Number of nodes: $NUM_NODES"
    echo "Number of commands: ${#COMMANDS[@]}"
    exit 1
fi

echo "List of commands"
for command in "${COMMANDS[@]}"; do
    echo "$command"
done


STARTED_CONTAINERS=()
NETWORK_CREATED=0
RUN_DIR=$(mktemp -d)
NETWORK_NAME="docker-net-${RUN_DIR##*/}"

start_network() {
    if docker network inspect docker-net > /dev/null 2>&1; then
        if [ "$(docker network inspect --format '{{len .Containers}}' docker-net)" != 0 ]; then
            echo "docker-net still has attached containers; refusing to reuse it." >&2
            return 1
        fi
        docker network rm docker-net
    fi
    docker network create --subnet=192.168.10.0/24 "$NETWORK_NAME"
    NETWORK_CREATED=1
}

start_nodes() {
    for node in $(seq 0 $(($NUM_NODES-1))); do
        DEVICE_LIST=""
        for node_gpu in $(seq 0 $(($NUM_GPUS - 1))); do
            DEVICE_NUM=$(($node * $NUM_GPUS + $node_gpu))
            DEVICE_LIST+=$(($DEVICE_NUM))
            if [ "$node_gpu" -lt $(($NUM_GPUS - 1)) ]; then
                DEVICE_LIST+=','
            fi
        done
        if [ "$IS_ROCM" -eq 1 ]; then
            GPU_DEVICES=(--device /dev/kfd --device /dev/dri -e "HIP_VISIBLE_DEVICES=${DEVICE_LIST}")
        else
            # The literal quotes around device=... are required by docker's
            # --gpus value parser when the device list itself contains commas.
            GPU_DEVICES=(--gpus "\"device=${DEVICE_LIST}\"")
        fi

        # start the container in detached mode
        # things to note:
        # 1. --shm-size=10.24gb is required. don't use --ipc=host
        # 2. pass HF_TOKEN to the container
        # 3. map the huggingface cache directory to the container
        # 3. assign ip addresses to the containers (head node: 192.168.10.10, worker nodes:
        #    starting from 192.168.10.11)
        CONTAINER_ID=$(docker run -d "${GPU_DEVICES[@]}" --shm-size=10.24gb -e HF_TOKEN \
            -v ~/.cache/huggingface:/root/.cache/huggingface --name "$NETWORK_NAME-node$node" \
            --network "$NETWORK_NAME" --ip 192.168.10.$((10 + $node)) --rm "$DOCKER_IMAGE" \
            /bin/bash -c "tail -f /dev/null")
        STARTED_CONTAINERS+=("$CONTAINER_ID")

        # organize containers into a ray cluster
        if [ "$node" -eq 0 ]; then
            # start the ray head node
            docker exec -d "$CONTAINER_ID" /bin/bash -c "ray start --head --port=6379 --block"
            # wait for the head node to be ready
            sleep 10
        else
            # start the ray worker nodes, and connect them to the head node
            docker exec -d "$CONTAINER_ID" /bin/bash -c "ray start --address=192.168.10.10:6379 --block"
        fi
    done

    # wait for the cluster to be ready
    sleep 10

    # print the cluster status
    docker exec "${STARTED_CONTAINERS[0]}" /bin/bash -c "ray status"
}

run_nodes() {
    # important: iterate in reverse order to start the head node last
    # we start the worker nodes first, in detached mode, and then start the head node
    # in the foreground, so that the output of the head node is visible in the buildkite logs
    for node in $(seq $(($NUM_NODES - 1)) -1 0); do
        DEVICE_LIST=""
        for node_gpu in $(seq 0 $(($NUM_GPUS - 1))); do
            DEVICE_NUM=$(($node * $NUM_GPUS + $node_gpu))
            DEVICE_LIST+=$(($DEVICE_NUM))
            if [ "$node_gpu" -lt $(($NUM_GPUS - 1)) ]; then
                DEVICE_LIST+=','
            fi
        done
        echo "Running node$node with GPU devices: $DEVICE_LIST"
        if [ "$node" -ne 0 ]; then
            docker exec -d "${STARTED_CONTAINERS[$node]}" /bin/bash -c "cd $WORKING_DIR ; ${COMMANDS[$node]}"
        else
            # Allocate a TTY (-t -i) for the foreground head node so its output
            # keeps ANSI color in the Buildkite log (see run-amd-test.sh).
            docker exec -t -i "${STARTED_CONTAINERS[$node]}" /bin/bash -c "cd $WORKING_DIR ; ${COMMANDS[$node]}"
        fi
    done
}
cleanup() {
    local status=$?
    local cleanup_status=0
    if [ "${#STARTED_CONTAINERS[@]}" -gt 0 ]; then
        for container_id in "${STARTED_CONTAINERS[@]}"; do
            docker stop "$container_id" || cleanup_status=$?
        done
    fi
    if [ "$NETWORK_CREATED" -eq 1 ]; then
        docker network rm "$NETWORK_NAME" || cleanup_status=$?
    fi
    rmdir "$RUN_DIR" || cleanup_status=$?
    if [ "$status" -ne 0 ]; then
        return "$status"
    fi
    return "$cleanup_status"
}
trap cleanup EXIT
start_network
start_nodes
run_nodes
