# This script provides an example to run DeepSeek V3 or
# R1 model with vLLM on multiple nodes with ray. 
# This script lists the individual steps to run on different nodes.
# Alternatively, you can use the `run_cluster.sh` wrapper.

########################################################
# Step 1: Prepare nodes and docker images
########################################################

# Assuming we have two 8xH100 nodes, each has 8 GPUs.

# Pull the vllm docker image on each node.
# Adjust the image version as desired.
docker pull vllm/vllm-openai:v0.8.4

# Optional: build your own docker image based on 
# the vllm-openai image.

# Download the DeepSeek V3 or R1 model checkpoint to each node,
# say /path/to/the/huggingface/home/on/this/node.
huggingface-cli download deepseek-ai/deepseek-v3 --local-dir /path/to/the/huggingface/home/on/this/node

########################################################
# Step 2: Start ray cluster
########################################################

# Pick one node as the head node and the other as the worker node.

# 2.1 Start the head node

# ssh to the head node and run the following command,
# replace the VLLM_HOST_IP with the actual IP address of the head node,
# adjust the environment variables as needed.
docker run -d --gpus all --privileged --ipc=host --network=host --shm-size 10.24g \
    -v /var/run/nvidia-topologyd:/var/run/nvidia-topologyd \
    -v /path/to/the/huggingface/home/on/this/node:/path/to/the/model/in/the/container \
    -e GLOO_SOCKET_IFNAME=eth0 -e NCCL_SOCKET_IFNAME=eth0 \
    -e NCCL_IB_HCA=mlx5_1:1,mlx5_2:1,mlx5_3:1,mlx5_4:1 \
    -e NCCL_IB_DISABLE=0 -e NCCL_IB_GID_INDEX=3 \
    -e CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    -e RAY_CGRAPH_submit_timeout=600 -e RAY_CGRAPH_get_timeout=600 \
    --entrypoint /bin/bash \
    -e VLLM_HOST_IP=192.168.0.1 --name ray_head vllm/vllm-openai:0.8.4 \
    -c "ray start --block --head --port=6379 --include-dashboard=True"

# 2.2 Start the worker node

# ssh to the worker node and run the following command,
# replace the VLLM_HOST_IP with the actual IP address of the worker node,
# point the ray IP address to the head node's IP address,
# adjust the environment variables as needed.
docker run -d --gpus all --privileged --ipc=host --network=host --shm-size 10.24g \
    -v /var/run/nvidia-topologyd:/var/run/nvidia-topologyd \
    -v /path/to/the/huggingface/home/on/this/node:/path/to/the/model/in/the/container \
    -e GLOO_SOCKET_IFNAME=eth0 -e NCCL_SOCKET_IFNAME=eth0 \
    -e NCCL_IB_HCA=mlx5_1:1,mlx5_2:1,mlx5_3:1,mlx5_4:1 \
    -e NCCL_IB_DISABLE=0 -e NCCL_IB_GID_INDEX=3 \
    -e CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    -e RAY_CGRAPH_submit_timeout=600 -e RAY_CGRAPH_get_timeout=600 \
    --entrypoint /bin/bash \
    -e VLLM_HOST_IP=192.168.0.2 --name ray_worker vllm/vllm-openai:0.8.4 \
    -c "ray start --block --address=192.168.0.1:6379"

########################################################
# Step 3: Start vLLM server
########################################################

# ssh to the head node and enter the container
docker exec -it ray_head bash

# start the vLLM server with PP=2 and TP=8,
# adjust the vllm arguments as desired..
VLLM_USE_V1=1 python3 -m vllm.entrypoints.openai.api_server \
    --model /path/to/the/model/in/the/container \
    --tensor-parallel-size 8 \
    --pipeline-parallel-size 2 \
    --gpu-memory-utilization 0.92 \
    --dtype auto \
    --distributed-executor-backend ray \
    --served-model-name deepseekv3 \
    --max-num-seqs 40 \
    --max-model-len 16384 \
    --enable-chunked-prefill \
    --enable-prefix-caching \
    --trust-remote-code
