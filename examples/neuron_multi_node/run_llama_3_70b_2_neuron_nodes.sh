#!/bin/bash

set -x #echo on

# Reference script to run multi node inference with Neuron.
# We can use this script as an example to run online inference using neuron.

# Pre-reqs:
# For this example we need a cluster of 2 nodes (trn1n/trn1) for tensor parallel as 64.

# Steps to run an example of multi node inference on neuron:
# Step 1: ssh to the Node1 and run this script with NEURON_RANK_ID set to 0.
#         This is the master/driver node and runs the server to accept inference requests.
#         Every cluster will have a single master node with NEURON_RANK_ID as 0.
# Step 2: ssh to the Node2 and run this script with NEURON_RANK_ID set to 1.
#         This is a worker node which loops forever to process requests broadcasted by
#         the master/driver node. All nodes in the cluster apart from master/driver node
#         are worker nodes. This step will be repeated across all nodes in the cluster
#         except master/driver with NEURON_RANK_ID incremented by 1.
#         Eg: For a cluster of 4 nodes, run step 2 for three nodes with NEURON_RANK_ID
#         as 1, 2 and 3 respectively.
# Step 3: Wait till the server starts on Node1 as part of Step 1.
# Step 4: ssh to the Node1 and post inference requests to get back the response.
#         Example :
#         time curl -X POST http://localhost:8080/generate \
#         -H "Content-Type: application/json" \
#         -d '{ "prompt": "The capital of France is", "top_k": 1 }'


# Use environment variables with defaults
if [ -z "$K8S_MASTER_ADDR" ]; then
	MASTER_ADDR=(`scontrol show hostnames $SLURM_JOB_NODELIST`)
else
	MASTER_ADDR=$K8S_MASTER_ADDR
fi
NEURON_RT_ROOT_COMM_ID=$MASTER_ADDR:8990
NEURON_RANK_ID=${K8S_NEURON_RANK_ID:-$SLURM_NODEID}
# 32 cores per node if for trn1/trn1n instances.
NEURON_LOCAL_TP=32
VLLM_HOST_IP=$MASTER_ADDR
VLLM_PORT=8989

# The master node’s <IP address>:<port>
export NEURON_RT_ROOT_COMM_ID
# Rank of the node.
export NEURON_RANK_ID
# The local tensor parallel degree on each node.
export NEURON_LOCAL_TP
# The master node’s <IP address>
export VLLM_HOST_IP
# Free port on the master used for inter process communications.
export VLLM_PORT

echo $NEURON_RT_ROOT_COMM_ID
echo $NEURON_RANK_ID
echo $NEURON_LOCAL_TP
echo $VLLM_HOST_IP
echo $VLLM_PORT

sudo modprobe -r neuron; sudo modprobe neuron
export FI_EFA_USE_DEVICE_RDMA=1
export FI_PROVIDER=efa

# Online inference with Neuron
python neuron_multi_node_runner.py \
    --model="meta-llama-3-1/Meta-Llama-3-1-70B" \
	--max-num-seqs=2 --max-model-len=128 --tensor-parallel-size=64 \
	--port=8080 --device="neuron"
