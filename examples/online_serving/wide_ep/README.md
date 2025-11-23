# Wide-EP DeepSeek-V3 Benchmark

Minimal recipe for reproducing the Nebius-wide DeepSeek-V3 run (4 × 8xH200) with DeepGEMM + DeepEP inside vLLM. This is the config we ship in the accompanying `launch_server.sh`.

## Cluster Snapshot and Dependencies

- 4 nodes, each with 8xH200
- CUDA 12.8 + NVSHMEM 3.4.5
- Infinband GPUDirect RDMA enabled
- DeepGEMM `c9f8b34dcdacc20aa746b786f983492c51072870`
- DeepEP `92fe2deaec24bc92ebd9de276daa6ca9ed602ed4`
- vLLM v0.11.1rc5 (precompiled wheels)

```bash
VLLM_USE_PRECOMPILED=1 uv pip install \
  "git+https://github.com/vllm-project/vllm.git@v0.11.1rc5" \
  --extra-index-url https://download.pytorch.org/whl/cu128 \
  --index-strategy unsafe-best-match --prerelease=allow
```

## Base Environment

Set once per node before launching:

```bash
export VLLM_USE_DEEP_GEMM=1
export VLLM_ALL2ALL_BACKEND=deepep_low_latency
export VLLM_MOE_DP_CHUNK_SIZE=512
export VLLM_SKIP_P2P_CHECK=1
export VLLM_RANDOMIZE_DP_DUMMY_INPUTS=1
export NVIDIA_GDRCOPY=enabled
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_MOE_ROUTING_SIMULATION_STRATEGY=uniform_random
export NVSHMEM_QP_DEPTH=1512
export GLOO_SOCKET_IFNAME=eth0
```

## Launch

> This script is only for the benchmarking recipe; change knobs via the raw command if you need a different setup.

1. Pick a coordinator IP/port that all nodes can reach:

   ```bash
   export COORDINATOR_IP=<head-node-ip>
   export COORDINATOR_RPC_PORT=8888
   ```

2. On each node run either the script or the raw command below. Only argument that changes between nodes is `START_RANK`.

### Script (preferred for benchmark reproduction)

```bash
export START_RANK=<0|8|16|24>
./launch_server.sh
```
