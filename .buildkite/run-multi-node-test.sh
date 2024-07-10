#!/bin/bash

set -euox pipefail

if [[ $# -lt 4 ]]; then
    echo "Please provide the working directory, number of nodes and GPU per node."
    exit 1
fi

WORKING_DIR=$1
NUM_NODES=$2
NUM_GPUS=$3
DOCKER_IMAGE=$4

shift 4
COMMANDS=("$@")
if [ ${#COMMANDS[@]} -ne $NUM_NODES ]; then
    echo "The number of commands must be equal to the number of nodes."
    echo "Number of nodes: $NUM_NODES"
    echo "Number of commands: ${#COMMANDS[@]}"
    exit 1
fi

echo "List of commands"
for command in "${COMMANDS[@]}"; do
    echo $command
done

start_network() {
    docker network create --subnet=192.168.10.0/24 docker-net
}

start_nodes() {
    for node in $(seq 0 $(($NUM_NODES-1))); do
        GPU_DEVICES='"device='
        for node_gpu in $(seq 0 $(($NUM_GPUS - 1))); do
            DEVICE_NUM=$(($node * $NUM_GPUS + $node_gpu))
            GPU_DEVICES+=$(($DEVICE_NUM))
            if [ $node_gpu -lt $(($NUM_GPUS - 1)) ]; then
                GPU_DEVICES+=','
            fi
        done
        GPU_DEVICES+='"'
        if [ $node -eq 0 ]; then
            docker run -d --gpus "$GPU_DEVICES" --name node$node --network docker-net --ip 192.168.10.$((10 + $node)) --rm $DOCKER_IMAGE /bin/bash -c "ray start --head && tail -f /dev/null"
        else
            docker run -d --gpus "$GPU_DEVICES" --name node$node --network docker-net --ip 192.168.10.$((10 + $node)) --rm $DOCKER_IMAGE /bin/bash -c "ray start --address=192.168.10.10:6379 && tail -f /dev/null"
        fi
        # echo "Starting node$node with GPU devices: $GPU_DEVICES"
    done
}

run_nodes() {
    # important: iterate in reverse order to start the head node last
    for node in $(seq $(($NUM_NODES - 1)) -1 0); do
        GPU_DEVICES='"device='
        for node_gpu in $(seq 0 $(($NUM_GPUS - 1))); do
            DEVICE_NUM=$(($node * $NUM_GPUS + $node_gpu))
            GPU_DEVICES+=$(($DEVICE_NUM))
            if [ $node_gpu -lt $(($NUM_GPUS - 1)) ]; then
                GPU_DEVICES+=','
            fi
        done
        GPU_DEVICES+='"'
        echo "Running node$node with GPU devices: $GPU_DEVICES"
        if [ $node -ne 0 ]; then
            docker exec -d node$node /bin/bash -c "cd $WORKING_DIR ; ${COMMANDS[$node]}"
        else
            docker exec node$node /bin/bash -c "cd $WORKING_DIR ; ${COMMANDS[$node]}"
        fi
    done
}
cleanup() {
    for node in $(seq 0 $(($NUM_NODES-1))); do
        docker stop node$node
    done
    docker network rm docker-net
}
trap cleanup EXIT
start_network
start_nodes
run_nodes

