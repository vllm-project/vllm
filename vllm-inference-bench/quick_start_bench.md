# vLLM Inference Benchmarks Quick Start


## Installation on vLLM docker image

To use vLLM with DeepEP and Perplexity kernels, we need to install the packages in the official vLLM docker container. We can use the containers from NVIDIA but the following scipt works seamlessly with images from vLLM. When starting the enroot container, use the --rw flag to make it exportable. 
Additional and up to date information on running MoE with expert parallelism here: https://docs.vllm.ai/en/latest/serving/expert_parallel_deployment.html#single-node-deployment

```bash
# make sure we are in the lustre fs to store the image 
enroot import docker://vllm/vllm-openai:latest
enroot create --name vllm-moe <image_name>.sqsh

enroot start \
    --mount /home/sshrestha:/home/sshrestha \
    --mount  /lustre/fsw/portfolios/hw/users/sshrestha/:/lustre/fsw/portfolios/hw/users/sshrestha/ \
    --root \
    --rw \
    vllm-moe

# install the packages from the vLLM repository 

cd <path_to_vllm>/tools/ep_kernels

# update CUDA_HOME env variable 
export CUDA_HOME=/usr/local/cuda

# for hopper
TORCH_CUDA_ARCH_LIST="9.0" bash install_python_libraries.sh
# for blackwell
TORCH_CUDA_ARCH_LIST="10.0" bash install_python_libraries.sh

# Additional steps for multi node development (instructions from vLLM)
# We need to do this in DFW, EOS should already be configured. (cannot use sudo or reboot the clusters ourselves)

sudo bash configure_system_drivers.sh
sudo reboot # Reboot is required to load the new driver

```

Once we have the container built, we can use this script from the login node to directly get into our container for interactive runs. 

```bash
srun -A  hw_nresearch_snoise --job-name setup:shell -N 1 --gpus-per-node=8 --partition=interactive --time=1:00:00 --container-image=/lustre/fsw/portfolios/hw/users/sshrestha/vllm-moe-v1.sqsh \
--container-mounts=$HOME:$HOME,/lustre/fsw/portfolios/hw/users/sshrestha/:/lustre/fsw/portfolios/hw/users/sshrestha/ \
 --pty /bin/bash -l 
```

We have two sets of benchmarking scripts 

+ **Serving Benchmarks**: Start a vLLM host server which loads the model and accepts requests. A vLLM client server sends requests to the server with configurable request sizes and measures the latency of prefill, decode and other metrics. This is the most realistic inference benchmark. 

+ **Offline Benchmarks**: Measure the prefill and decode latency using a custom inference script. This currently works with all dense models. MoE models work in progress. Known issue: the script gets stuck when using MoE models with expert parallel. 

## Things to configure before running the benchmarks (makes life easier)

Update huggingface transformer cache directory to the lustre fs. 

```bash
# create the directories if they do not exist
mkdir -p /lustre/fsw/portfolios/hw/users/sshrestha/.cache/huggingface/{hub,datasets,transformers}
mkdir -p /lustre/fsw/portfolios/hw/users/sshrestha/.cache/pip

# update the HF_CACHE environment variable 
export HF_HOME=/lustre/fsw/portfolios/hw/users/sshrestha/.cache/huggingface
export HF_HUB_CACHE=/lustre/fsw/portfolios/hw/users/sshrestha/.cache/huggingface/hub
export HF_DATASETS_CACHE=/lustre/fsw/portfolios/hw/users/sshrestha/.cache/huggingface/datasets
export TRANSFORMERS_CACHE=/lustre/fsw/portfolios/hw/users/sshrestha/.cache/huggingface/transformers
export PIP_CACHE_DIR=/lustre/fsw/portfolios/hw/users/sshrestha/.cache/pip
```

For some gated models, we need to create a huggingface account and request access for specific models like Llama-3,4 from huggingface's website. This should take < 1 hour, sometimes immediately. Once we have the model weight access, we need to login to our huggingface cli. 
Learn more about huggingface token here: https://huggingface.co/docs/hub/en/security-tokens

```bash
# if huggingface isn't installed 
pip install huggingface_hub

huggingface-cli login
# insert token when prompted
```

## Serving Benchmarks 

The model names or paths for all the benchmarks should be huggingface names. For example

Llama 4 scout: meta-llama/Llama-4-Scout-17B-16E-Instruct
DeepSeek-v2 : deepseek-ai/DeepSeek-V2-Chat-0628
DeepSeek-v3: deepseek-ai/DeepSeek-V3-0324

When used correctly, the following scripts will automatically download the weights during the first run and cache them in the HF_HOME. Depending on the model, network and file system traffic, the first run can take a significant amount of time to download the weights. 

### Single Node

This benchmark is currently interactive.
container-image=/lustre/fsw/portfolios/hw/users/sshrestha/vllm-moe-v1.sqsh

1. Start a node with the enroot container 

```bash
srun -A  hw_nresearch_snoise --job-name setup:shell -N 1 --gpus-per-node=8 --partition=interactive --time=1:00:00 --container-image=/lustre/fsw/portfolios/hw/users/sshrestha/vllm-moe-v1.sqsh \
--container-mounts=$HOME:$HOME,/lustre/fsw/portfolios/hw/users/sshrestha/:/lustre/fsw/portfolios/hw/users/sshrestha/ \
 --pty /bin/bash -l

``` 

2. Start the vLLM host server with different ALL2ALL backend

```bash
VLLM_ALL2ALL_BACKEND=pplx VLLM_USE_DEEP_GEMM=1 vllm serve $MODEL --trust-remote-code --tensor-parallel-size=$TENSOR_PARALLEL_SIZE --data-parallel-size=$DATA_PARALLEL_SIZE --enable-expert-parallel \
            --port $SERVER_PORT --max-model-len 16384 --gpu_memory_utilization=0.9 --api-server-count=8 &
or 

VLLM_ALL2ALL_BACKEND=deepep_low_latency VLLM_USE_DEEP_GEMM=1 vllm serve $MODEL --trust-remote-code --tensor-parallel-size=$TENSOR_PARALLEL_SIZE --data-parallel-size=$DATA_PARALLEL_SIZE --enable-expert-parallel \
            --port $SERVER_PORT --max-model-len 16384 --gpu_memory_utilization=0.9 --api-server-count=8 &

```

We can start the host server using the script : start_and_benchmark.sh

3. Start vLLM client server and measure inference benchmarks. 

We can change the input-len to control the prefill length and output-len to control the decode length and num-prompts to control the batch size.

```bash

vllm bench serve \
  --model $MODEL \
  --dataset-name random \
  --random-input-len 8000 \ 
  --random-output-len 1000 \
  --request-rate 10000 \
  --num-prompts 16 \
  --ignore-eos \
  --base-url http://localhost:8000 \
  --save-result \
  --append-result \
  --result-dir /home/sshrestha/workspace/vllm-distributed/vllm-inference-bench/benchmark_results \
  --result-filename llama-4-scout_tp_8.json

```

### Multi-Node

Get the primary node IP address

This benchmark is currently interactive. 

```bash
# Node 1 (Primary - handles incoming requests)
VLLM_ALL2ALL_BACKEND=deepep_low_latency VLLM_USE_DEEP_GEMM=1 \
    vllm serve deepseek-ai/DeepSeek-V3-0324 \
    --tensor-parallel-size 1 \               # TP size per node
    --enable-expert-parallel \               # Enable EP
    --data-parallel-size 16 \                # Total DP size across all nodes
    --data-parallel-size-local 8 \           # Local DP size on this node (8 GPUs per node)
    --data-parallel-address 192.168.1.100 \  # Replace with actual IP of Node 1
    --data-parallel-rpc-port 13345 \         # RPC communication port, can be any port as long as reachable by all nodes
    --api-server-count=8                     # Number of API servers for load handling (scaling this out to total ranks are recommended)

# Node 2 (Secondary - headless mode, no API server)
VLLM_ALL2ALL_BACKEND=deepep_low_latency VLLM_USE_DEEP_GEMM=1 \
    vllm serve deepseek-ai/DeepSeek-V3-0324 \
    --tensor-parallel-size 1 \               # TP size per node
    --enable-expert-parallel \               # Enable EP
    --data-parallel-size 16 \                # Total DP size across all nodes
    --data-parallel-size-local 8 \           # Local DP size on this node
    --data-parallel-start-rank 8 \           # Starting rank offset for this node
    --data-parallel-address 192.168.1.100 \  # IP of primary node (Node 1)
    --data-parallel-rpc-port 13345 \         # Same RPC port as primary
    --headless                               # No API server, worker only

```


Use the following scripts if we get errors on the commented scripts above. 

```bash
# Node 1
VLLM_ALL2ALL_BACKEND=deepep_low_latency VLLM_USE_DEEP_GEMM=1 vllm serve deepseek-ai/DeepSeek-V3-0324 --tensor-parallel-size 1 --enable-expert-parallel --data-parallel-size 16 --data-parallel-size-local 8 --data-parallel-address 10.65.27.215 --data-parallel-rpc-port 13345 --api-server-count=8 &

# Node 2
VLLM_ALL2ALL_BACKEND=deepep_low_latency VLLM_USE_DEEP_GEMM=1 vllm serve deepseek-ai/DeepSeek-V3-0324 --tensor-parallel-size 1 --enable-expert-parallel --data-parallel-size 16 --data-parallel-size-local 8 --data-parallel-start-rank 8 --data-parallel-address 10.65.27.215 --data-parallel-rpc-port 13345 --headless

```

Note: the multi-node scripts aren't fully tested because of DeepEP configuration. This guide is adpated from: https://docs.vllm.ai/en/latest/serving/expert_parallel_deployment.html#single-node-deployment. Once DeepEP is configured correctly, the multi-node command should run as expected. The --data-parallel-size controls the expert parallel size in these configurations and should equal to the total number of GPUs we want to use (if EP = Total GPUs). We need to update or increment the --data-parallel-start-rank in any additional nodes. 

## Offline Benchmarks 

The script for the offline benchmarking are in the vllm-distributed/vllm-inference-bench/benchmark_prefill_decode_v2.py. 

```bash
torchrun --nproc-per-node=8 benchmark_prefill_decode_v2.py --model meta-meta-llama/Llama-3.1-8B-Instruct --tensor-parallel-size 8 --pipeline-parallel-size 1 --batch-size 4 --input-length 512 --output-length 128 --data-path vllm-inference-bench/benchmark_results
```
or using sbatch

```bash
sbatch run_prefill_decode_benchmark.sh
```

This script auto generates "--batch-size" number of prompts of the context length "--input-length" and generates "--output-length" number of additional tokens. The results are saved in a csv file in the "--data-path". While serving benchmarks are more realistic, it is easier to work and debug offline scripts. The current known issues with the offline script is that it gets stuck for MoE models when using expert parallel. A lot of the new expert parallel and torchrun scripts are experimental in vLLM. Some additional work is needed to debug the offline script. In theory, this script should be easier to use with sbatch and scale to larger number of nodes/GPUs. 