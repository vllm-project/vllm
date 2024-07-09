#!/bin/bash

set -euox pipefail

if [[ $# -lt 3 ]]; then
    echo "Please provide the number of nodes and GPU per node."
    exit 1
fi

NUM_NODES=$1
NUM_GPUS=$2
DOCKER_IMAGE=$3

shift 3
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
        # echo "Starting node$node with GPU devices: $GPU_DEVICES"
        docker run -d --gpus "$GPU_DEVICES" --name node$node --network docker-net --ip 192.168.10.$((10 + $node)) --rm $DOCKER_IMAGE tail -f /dev/null
    done
}

run_nodes() {
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
        echo "Running node$node with GPU devices: $GPU_DEVICES"
        if [ $node -lt $(($NUM_NODES - 1)) ]; then
            docker exec -d node$node /bin/bash -c "${COMMANDS[$node]}"
        else
            docker exec node$node /bin/bash -c "${COMMANDS[$node]}"
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

