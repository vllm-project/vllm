#!/bin/bash

set -euox pipefail

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

start_network() {
    docker network create --subnet=192.168.10.0/24 docker-net
}

start_nodes() {
    for node in $(seq 0 $(($NUM_NODES-1))); do
        GPU_DEVICES='"device='
        for node_gpu in $(seq 0 $(($NUM_GPUS - 1))); do
            DEVICE_NUM=$(($node * $NUM_GPUS + $node_gpu))
            GPU_DEVICES+=$(($DEVICE_NUM))
            if [ "$node_gpu" -lt $(($NUM_GPUS - 1)) ]; then
                GPU_DEVICES+=','
            fi
        done
        GPU_DEVICES+='"'

        # start the container in detached mode
        # things to note:
        # 1. --shm-size=10.24gb is required. don't use --ipc=host
        # 2. pass HF_TOKEN to the container
        # 3. map the huggingface cache directory to the container
        # 3. assign ip addresses to the containers (head node: 192.168.10.10, worker nodes:
        #    starting from 192.168.10.11)
        docker run -d --gpus "$GPU_DEVICES" --shm-size=10.24gb -e HF_TOKEN \
            -v ~/.cache/huggingface:/root/.cache/huggingface --name "node$node" \
            --network docker-net --ip 192.168.10.$((10 + $node)) --rm "$DOCKER_IMAGE" \
            /bin/bash -c "tail -f /dev/null"

        # organize containers into a ray cluster
        if [ "$node" -eq 0 ]; then
            # start the ray head node
            docker exec -d "node$node" /bin/bash -c "ray start --head --port=6379 --block"
            # wait for the head node to be ready
            sleep 10
        else
            # start the ray worker nodes, and connect them to the head node
            docker exec -d "node$node" /bin/bash -c "ray start --address=192.168.10.10:6379 --block"
        fi
    done

    # wait for the cluster to be ready
    sleep 10

    # print the cluster status
    docker exec node0 /bin/bash -c "ray status"
}

run_nodes() {
    # important: iterate in reverse order to start the head node last
    # we start the worker nodes first, in detached mode, and then start the head node
    # in the foreground, so that the output of the head node is visible in the buildkite logs
    for node in $(seq $(($NUM_NODES - 1)) -1 0); do
        GPU_DEVICES='"device='
        for node_gpu in $(seq 0 $(($NUM_GPUS - 1))); do
            DEVICE_NUM=$(($node * $NUM_GPUS + $node_gpu))
            GPU_DEVICES+=$(($DEVICE_NUM))
            if [ "$node_gpu" -lt $(($NUM_GPUS - 1)) ]; then
                GPU_DEVICES+=','
            fi
        done
        GPU_DEVICES+='"'
        echo "Running node$node with GPU devices: $GPU_DEVICES"
        if [ "$node" -ne 0 ]; then
            docker exec -d "node$node" /bin/bash -c "cd $WORKING_DIR ; ${COMMANDS[$node]}"
        else
            docker exec "node$node" /bin/bash -c "cd $WORKING_DIR ; ${COMMANDS[$node]}"
        fi
    done
}
cleanup() {
    for node in $(seq 0 $(($NUM_NODES-1))); do
        docker stop "node$node"
    done
    docker network rm docker-net
}
trap cleanup EXIT
start_network
start_nodes
run_nodes

