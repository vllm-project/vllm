# Distributed Deployment Plan for Token Parallel vLLM

## Overview

This document provides a comprehensive plan for deploying vLLM with token parallelism across multiple nodes using various distributed computing environments, including SLURM, containers, and Ray. The plan covers deployment strategies, configuration examples, and best practices for scalable inference.

## Multi-Node Architecture Options

### Option 1: Ray-based Deployment (Recommended)
Ray provides the most mature multi-node support in vLLM with automatic resource management and fault tolerance.

### Option 2: Native Multi-Processing with SLURM
Direct PyTorch distributed deployment using SLURM for HPC environments.

### Option 3: Container-based Deployment
Using Docker/Singularity containers with orchestration (Kubernetes, Docker Swarm).

### Option 4: Hybrid Approaches
Combining Ray with containers or SLURM for different use cases.

## Ray-based Multi-Node Deployment

### Ray Cluster Setup

#### Manual Ray Cluster
```bash
# On head node
ray start --head --port=6379 --num-cpus=0 --num-gpus=8 \
    --dashboard-host=0.0.0.0 --dashboard-port=8265

# On worker nodes  
ray start --address=<HEAD_NODE_IP>:6379 --num-cpus=0 --num-gpus=8
```

#### Ray with Token Parallelism
```python
# Multi-node token parallel deployment
from vllm import LLM
import ray

# Initialize Ray cluster
ray.init(address="ray://<head_node_ip>:10001")

# Create LLM with token parallelism
llm = LLM(
    model="meta-llama/Llama-2-70b-hf",
    tensor_parallel_size=8,      # TP across 8 GPUs per node
    token_parallel_size=4,       # Token parallel across 4 nodes
    pipeline_parallel_size=2,    # Pipeline across 2 stages
    distributed_executor_backend="ray",
    max_model_len=4096,
    gpu_memory_utilization=0.9
)
```

#### Ray Cluster Configuration File
```yaml
# ray_cluster.yaml
cluster_name: vllm-token-parallel

max_workers: 4

head_node:
    instance_type: g4dn.12xlarge  # 4 T4 GPUs
    min_workers: 0
    max_workers: 0

worker_nodes:
    instance_type: p3.8xlarge     # 4 V100 GPUs
    min_workers: 3
    max_workers: 3
    worker_setup_commands:
        - pip install vllm torch

docker:
    image: "vllm/vllm-openai:latest"
    container_name: "vllm_worker"
    
provider:
    type: aws
    region: us-west-2
```

## SLURM-based Deployment

### SLURM Job Script for Token Parallelism
```bash
#!/bin/bash
#SBATCH --job-name=vllm-token-parallel
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:8
#SBATCH --time=04:00:00
#SBATCH --partition=gpu

# Set environment variables
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500
export WORLD_SIZE=$SLURM_NNODES
export RANK=$SLURM_PROCID

# Load modules
module load cuda/12.1
module load python/3.9

# Activate virtual environment
source vllm_env/bin/activate

# Start vLLM service on each node
srun python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-2-70b-hf \
    --tensor-parallel-size 8 \
    --token-parallel-size 4 \
    --pipeline-parallel-size 1 \
    --distributed-executor-backend mp \
    --host 0.0.0.0 \
    --port 8000
```

### Multi-Node SLURM with Ray
```bash
#!/bin/bash
#SBATCH --job-name=vllm-ray-cluster
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --time=04:00:00

# Start Ray head on first node
if [ $SLURM_PROCID -eq 0 ]; then
    ray start --head --port=6379 --num-gpus=8 \
        --dashboard-host=0.0.0.0 \
        --temp-dir=/tmp/ray_$SLURM_JOB_ID
    
    # Wait for head to initialize
    sleep 10
    
    # Start vLLM service
    python run_token_parallel_server.py
else
    # Worker nodes join cluster
    HEAD_NODE=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
    ray start --address=$HEAD_NODE:6379 --num-gpus=8
    
    # Keep worker alive
    sleep 3600
fi
```

## Container-based Deployment

### Docker Compose for Multi-Node Setup
```yaml
# docker-compose.yml
version: '3.8'

services:
  vllm-head:
    image: vllm/vllm-openai:latest
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - MASTER_ADDR=vllm-head
      - MASTER_PORT=29500
      - WORLD_SIZE=4
      - RANK=0
    ports:
      - "8000:8000"
      - "6379:6379"
    volumes:
      - ~/.cache/huggingface:/root/.cache/huggingface
      - ./configs:/configs
    command: >
      python -m vllm.entrypoints.openai.api_server
      --model meta-llama/Llama-2-70b-hf
      --tensor-parallel-size 2
      --token-parallel-size 4
      --distributed-executor-backend ray
      --host 0.0.0.0
    networks:
      - vllm-network

  vllm-worker-1:
    image: vllm/vllm-openai:latest
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - MASTER_ADDR=vllm-head
      - WORLD_SIZE=4
      - RANK=1
    depends_on:
      - vllm-head
    volumes:
      - ~/.cache/huggingface:/root/.cache/huggingface
    networks:
      - vllm-network

networks:
  vllm-network:
    driver: bridge
```

### Singularity/Apptainer Container Recipe
```bash
# vllm-token-parallel.def
Bootstrap: docker
From: nvcr.io/nvidia/pytorch:23.10-py3

%post
    # Install vLLM with token parallelism support
    pip install vllm
    pip install ray[default]
    
    # Set up environment
    mkdir -p /workspace
    cd /workspace

%environment
    export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    export NCCL_SOCKET_IFNAME=^docker0,lo
    export NCCL_IB_DISABLE=1

%runscript
    exec python -m vllm.entrypoints.openai.api_server "$@"
```

### Enroot Container Deployment
```bash
# Build container
enroot import docker://vllm/vllm-openai:latest
enroot create --name vllm-token-parallel vllm-openai+latest.sqsh

# SLURM script with enroot
#!/bin/bash
#SBATCH --job-name=vllm-enroot
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8

# Start container on each node
srun --container-image=vllm-token-parallel \
     --container-mounts=/scratch:/workspace \
     --container-workdir=/workspace \
     python -m vllm.entrypoints.openai.api_server \
     --model meta-llama/Llama-2-70b-hf \
     --tensor-parallel-size 8 \
     --token-parallel-size 4
```

## Kubernetes Deployment

### Token Parallel vLLM StatefulSet
```yaml
# vllm-token-parallel.yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: vllm-token-parallel
spec:
  serviceName: vllm-service
  replicas: 4
  selector:
    matchLabels:
      app: vllm-token-parallel
  template:
    metadata:
      labels:
        app: vllm-token-parallel
    spec:
      containers:
      - name: vllm
        image: vllm/vllm-openai:latest
        resources:
          limits:
            nvidia.com/gpu: 8
            memory: 200Gi
          requests:
            nvidia.com/gpu: 8
            memory: 100Gi
        env:
        - name: MASTER_ADDR
          value: "vllm-token-parallel-0.vllm-service"
        - name: WORLD_SIZE
          value: "4"
        - name: RANK
          valueFrom:
            fieldRef:
              fieldPath: metadata.labels['statefulset.kubernetes.io/pod-name']
        command:
        - python
        - -m
        - vllm.entrypoints.openai.api_server
        args:
        - --model=meta-llama/Llama-2-70b-hf
        - --tensor-parallel-size=8
        - --token-parallel-size=4
        - --host=0.0.0.0
        - --port=8000
        volumeMounts:
        - name: model-cache
          mountPath: /root/.cache/huggingface
        - name: shm
          mountPath: /dev/shm
      volumes:
      - name: shm
        emptyDir:
          medium: Memory
          sizeLimit: 32Gi
  volumeClaimTemplates:
  - metadata:
      name: model-cache
    spec:
      accessModes: ["ReadWriteOnce"]
      resources:
        requests:
          storage: 200Gi
```

## Network Configuration and Optimization

### Network Requirements
- **Bandwidth**: Minimum 100 Gbps interconnect (InfiniBand/Ethernet)
- **Latency**: <10μs between nodes for optimal token parallel performance
- **Topology**: Non-blocking network fabric preferred

### NCCL Optimization
```bash
# Environment variables for NCCL optimization
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=3
export NCCL_P2P_LEVEL=NVL
export NCCL_TOPO_PATHS=/path/to/topology.xml

# For token parallelism specific optimization
export NCCL_BUFFSIZE=8388608
export NCCL_MIN_NCHANNELS=16
export NCCL_MAX_NCHANNELS=32
```

## Monitoring and Debugging

### Ray Monitoring
```python
# Monitoring script for Ray cluster
import ray
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

@ray.remote
class TokenParallelMonitor:
    def get_cluster_status(self):
        return {
            'nodes': ray.nodes(),
            'resources': ray.cluster_resources(),
            'placement_groups': ray.util.placement_group_table()
        }

# Deploy monitor
monitor = TokenParallelMonitor.remote()
status = ray.get(monitor.get_cluster_status.remote())
```

### Performance Monitoring
```bash
# GPU utilization monitoring
nvidia-smi dmon -s pucvmet -d 1

# Network monitoring for multi-node communication
iftop -i eth0

# Memory monitoring
watch -n 1 'free -h && nvidia-smi --query-gpu=memory.used,memory.total --format=csv'
```

### Debugging Multi-Node Issues
```python
# Debugging script for token parallel deployment
import torch
import torch.distributed as dist
from vllm.distributed.parallel_state import (
    get_tknp_group, get_tknp_rank, get_tknp_world_size
)

def debug_token_parallel():
    print(f"Token Parallel Rank: {get_tknp_rank()}")
    print(f"Token Parallel World Size: {get_tknp_world_size()}")
    print(f"CUDA Device Count: {torch.cuda.device_count()}")
    print(f"Current Device: {torch.cuda.current_device()}")
    
    # Test communication
    tknp_group = get_tknp_group()
    test_tensor = torch.ones(1).cuda()
    dist.all_reduce(test_tensor, group=tknp_group.device_group)
    print(f"Communication test result: {test_tensor.item()}")
```

## Best Practices and Recommendations

### Hardware Recommendations
1. **GPU Selection**: 
   - V100/A100/H100 for production workloads
   - Minimum 4 GPUs per node for token parallelism benefits
   - NVLink connectivity within nodes

2. **Memory Configuration**:
   - 80GB+ GPU memory per node for 70B models
   - High-bandwidth system memory (DDR4-3200 or better)
   - Fast NVMe storage for model loading

3. **Network**:
   - InfiniBand HDR (200 Gbps) or Ethernet 100GbE
   - RDMA-capable network cards
   - Low-latency switches (cut-through switching)

### Software Recommendations
1. **Operating System**: Ubuntu 20.04/22.04 LTS
2. **CUDA**: Version 12.1 or later
3. **Driver**: Latest stable NVIDIA drivers (535+)
4. **Container Runtime**: Docker 24.0+ with nvidia-container-toolkit

### Configuration Best Practices
1. **Parallel Strategy Selection**:
   ```python
   # For different model sizes and node configurations
   
   # 7B model, 2 nodes, 8 GPUs each
   tensor_parallel_size = 4    # Within node
   token_parallel_size = 4     # Across nodes
   
   # 70B model, 4 nodes, 8 GPUs each  
   tensor_parallel_size = 8    # Within node
   token_parallel_size = 4     # Across nodes
   
   # 175B model, 8 nodes, 8 GPUs each
   tensor_parallel_size = 8    # Within node
   token_parallel_size = 8     # Across nodes
   pipeline_parallel_size = 2  # Pipeline stages
   ```

2. **Memory Management**:
   ```python
   # Optimal memory configuration
   gpu_memory_utilization = 0.85  # Leave headroom for token parallel communication
   max_num_seqs = batch_size // token_parallel_size
   max_num_batched_tokens = max_num_seqs * max_seq_len
   ```

3. **Batch Size Optimization**:
   - Start with batch size = token_parallel_size * 8
   - Increase gradually while monitoring memory usage
   - Monitor token parallel load balancing

## Testing and Validation

### Deployment Testing Script
```bash
#!/bin/bash
# test_deployment.sh

# Test different deployment scenarios
test_ray_deployment() {
    echo "Testing Ray deployment..."
    ray start --head --temp-dir=/tmp/ray_test
    python test_token_parallel.py --backend=ray
    ray stop
}

test_slurm_deployment() {
    echo "Testing SLURM deployment..."
    sbatch --wait test_slurm_job.sh
}

test_container_deployment() {
    echo "Testing container deployment..."
    docker-compose up -d
    sleep 30
    python test_api_endpoints.py
    docker-compose down
}

# Run all tests
test_ray_deployment
test_slurm_deployment  
test_container_deployment
```

### Load Testing
```python
# load_test.py
import asyncio
import aiohttp
import time

async def test_token_parallel_performance():
    """Test token parallel deployment performance"""
    
    # Test configuration
    num_requests = 100
    concurrent_requests = 20
    prompt_length = 512
    max_tokens = 100
    
    async with aiohttp.ClientSession() as session:
        start_time = time.time()
        
        tasks = []
        for i in range(num_requests):
            task = send_request(session, f"Request {i}", 
                              prompt_length, max_tokens)
            tasks.append(task)
            
            if len(tasks) >= concurrent_requests:
                await asyncio.gather(*tasks)
                tasks = []
        
        # Process remaining tasks
        if tasks:
            await asyncio.gather(*tasks)
            
        end_time = time.time()
        
    throughput = num_requests / (end_time - start_time)
    print(f"Throughput: {throughput:.2f} requests/second")

if __name__ == "__main__":
    asyncio.run(test_token_parallel_performance())
```

## Troubleshooting Guide

### Common Issues and Solutions

1. **NCCL Communication Errors**:
   ```bash
   # Check network connectivity
   nc -zv <other_node_ip> 29500
   
   # Test NCCL bandwidth
   /usr/local/cuda/samples/bin/bandwidthTest
   ```

2. **Memory Issues**:
   ```bash
   # Monitor GPU memory across nodes
   pdsh -w node[1-4] nvidia-smi
   
   # Check for memory fragmentation
   python -c "import torch; print(torch.cuda.memory_summary())"
   ```

3. **Load Balancing Issues**:
   ```python
   # Monitor token distribution across ranks
   def monitor_token_distribution():
       tknp_rank = get_tknp_rank()
       local_batch_size = get_local_batch_size()
       print(f"Rank {tknp_rank}: processing {local_batch_size} tokens")
   ```

## Conclusion

This distributed deployment plan provides comprehensive guidance for deploying vLLM with token parallelism across multiple nodes using various infrastructure options. The choice of deployment method depends on your specific requirements:

- **Ray**: Best for dynamic scaling and fault tolerance
- **SLURM**: Ideal for HPC environments with job scheduling
- **Containers**: Good for cloud-native and orchestrated deployments
- **Hybrid**: Combines benefits of multiple approaches

Success depends on proper hardware selection, network configuration, and following the best practices outlined in this document. 