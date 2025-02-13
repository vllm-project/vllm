# Multi-Node Setup Guide

## Steps for 2 Nodes

### 1. Environment Setup

#### a. Identical Software Stack
Ensure both nodes have the same software stack, including:
- Driver
- Docker
- VLLM version

#### b. Identical Model Path
Both nodes should load model weights from the same path, e.g.:
```
/mnt/disk0/model
```

#### c. Network Configuration
- Ensure both nodes are connected to the same switch/router.
- Example IP configuration:
  - Node 1: `192.168.1.100`
  - Node 2: `192.168.1.200`

#### d. HCCL Demo Test
Make sure the HCCL demo test passes using the assigned IPs.
##### Example Commands:
**Head Node:**
```bash
HCCL_COMM_ID=192.168.1.112:5555 python3 run_hccl_demo.py --test all_reduce --nranks 16 --loop 1000 --node_id 0 --size 32m --ranks_per_node 8
```

**Worker Node:**
```bash
HCCL_COMM_ID=192.168.1.112:5555 python3 run_hccl_demo.py --test all_reduce --nranks 16 --loop 1000 --node_id 1 --size 32m --ranks_per_node 8
```

#### e. Docker Environment Variables
Ensure the Docker container includes the following environment variables:
```bash
--env PT_HPU_ENABLE_LAZY_COLLECTIVES=true
--env PT_HPUGRAPH_DISABLE_TENSOR_CACHE=1
--env VLLM_HOST_IP=<your ip> # Host & node must be different
--env GLOO_SOCKET_IFNAME=<your network device>  # Host & node might be same or different
```

#### f. Verify Triton Version
Check that Triton is version **3.1.0**:
```bash
pip list | grep triton
```

---

### 2. Install VLLM
```bash
pip install vllm
```

---

### 3. Configure Multi-Node Script
Navigate to the VLLM scripts directory and edit `multi_nodes_source.sh`:
```bash
cd vllm/script
```

#### a. Adjust Environment Variables
Modify the following values as needed:
```bash
export VLLM_GPU_MEMORY_UTILIZATION=0.98
export VLLM_GRAPH_RESERVED_MEM=0.35
export VLLM_GRAPH_PROMPT_RATIO=0
```

#### b. Adjust Workload Parameters
Modify these parameters based on workload requirements:
```bash
max_num_batched_tokens=2048
max_num_seqs=256
input_min=1024
input_max=1024
output_max=1024
```

#### c. Apply Configuration on Both Nodes
Run the following command on both head and worker nodes:
```bash
source multi_nodes_source.sh
```

---

### 4. Start Ray Cluster

#### a. Start Ray on Head Node
```bash
ray start --head --port=<port number, e.g., 8850>
```

#### b. Start Ray on Worker Node
```bash
ray start --address='IP:port'
# Example:
ray start --address='192.168.1.112:8850'
```

---

### 5. Start Command Line on Head Node
Run the final command on the head node.

⚠️ **Ensure that the parameters in the command line match the settings in `multi_nodes_source.sh`.**


