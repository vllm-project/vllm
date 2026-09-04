# vLLM Prefill-Decode Disaggregation (PD Disagg) Setup Guide

This guide explains how to run vLLM in a **Prefill-Decode Disaggregated configuration** for DeepSeek-R1-0528-NVFP4 or Kimi-K2-Thinking-NVFP4 on GB200, where prefill and decode workloads run on separate nodes/instances, connected via a router.

## Configuration Summary

| Component | TP Size | Nodes | GPUs | Port |
|-----------|---------|-------|------|------|
| Prefill Instance 0 | 8 | 2 (master + worker) | 8 | 8087 |
| Prefill Instance 1 | 8 | 2 (master + worker) | 8 | 8087 |
| Decode Instance | 8 | 2 (master + worker) | 8 | 8087 |
| Router | - | 1 | 0 | 8123 |

**Total: 7 nodes, 24 GPUs** (4 GPUs/node)

---

## Environment Variables

### System Environment

```bash
export NVIDIA_GDRCOPY=1
export NVSHMEM_IB_ENABLE_IBGDA=1
export NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME={NETWORK_INTERFACE}
export UCX_IB_ROCE_REACHABILITY_MODE=local_subnet
export VLLM_SKIP_P2P_CHECK=1
export GLOO_SOCKET_IFNAME={NETWORK_INTERFACE}
export NCCL_SOCKET_IFNAME={NETWORK_INTERFACE}
export NCCL_CUMEM_ENABLE=1
export NCCL_MNNVL_ENABLE=1
export NCCL_NVLS_ENABLE=1
export NCCL_TIMEOUT=1800
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800
export HF_HOME={HF_CACHE_DIR}
```

### vLLM Environment

```bash
export VLLM_USE_FLASHINFER_MOE_FP4=1
export VLLM_USE_TRTLLM_RAGGED_DEEPSEEK_PREFILL=1
export VLLM_USE_NCCL_SYMM_MEM=1
```

### PD Disaggregation Environment (Required for all prefill/decode nodes)

```bash
export VLLM_NIXL_SIDE_CHANNEL_HOST=$(hostname -i)  # or {NODE_IP}
export VLLM_NIXL_SIDE_CHANNEL_PORT=5600
export VLLM_NIXL_ABORT_REQUEST_TIMEOUT=300
```

### Decode-Specific Environment (Optional, for performance tuning)

```bash
# Enable multi-stream for shared experts (beneficial for smaller batches)
export VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD=8192
```

---

## vLLM Serve Commands

### Common Arguments (Shared by all instances)

```bash
COMMON_ARGS="
--model {MODEL_PATH}
--kv-cache-dtype fp8
--tensor-parallel-size 1
--pipeline-parallel-size 1
--enable-expert-parallel
--data-parallel-rpc-port 13345
--max-model-len 4096
--data-parallel-size-local 4
--disable-uvicorn-access-log
--no-enable-prefix-caching
--port 8087
--trust_remote_code
--no-enable-chunked-prefill
--all2all-backend allgather_reducescatter
--data-parallel-hybrid-lb
--compilation_config.custom_ops+=+quant_fp8,+rms_norm,+rotary_embedding
--compilation_config.pass_config.fuse_attn_quant true
--compilation_config.pass_config.fuse_allreduce_rms true
--compilation_config.pass_config.eliminate_noops true
--async-scheduling
--kv-transfer-config '{\"kv_connector\":\"NixlConnector\",\"kv_role\":\"kv_both\",\"kv_load_failure_policy\":\"fail\"}'
"
```

### Prefill Instance Arguments

```bash
PREFILL_ARGS="
$COMMON_ARGS
--swap-space 16
--max-num-seqs 8
--enforce-eager
--gpu-memory-utilization 0.9
--max-num-batched-tokens 16384
"
```

### Decode Instance Arguments

```bash
DECODE_ARGS="
$COMMON_ARGS
--compilation-config '{\"cudagraph_mode\":\"FULL_DECODE_ONLY\"}'
--gpu-memory-utilization 0.9
--stream-interval 50
--max-num-seqs 512
--max-num-batched-tokens 4096
--max-cudagraph-capture-size 512
"
```

---

## Starting the Instances

### 1. Prefill Master (Node 0)

```bash
# Run on {PREFILL_MASTER_0_HOSTNAME}
vllm serve {MODEL_PATH} \
    $PREFILL_ARGS \
    --data-parallel-address {PREFILL_MASTER_0_HOSTNAME} \
    --data-parallel-size 8
```

### 2. Prefill Worker (Node 1, same instance as Master 0)

```bash
# Run on {PREFILL_WORKER_0_HOSTNAME}
vllm serve {MODEL_PATH} \
    $PREFILL_ARGS \
    --data-parallel-address {PREFILL_MASTER_0_HOSTNAME} \
    --data-parallel-start-rank 4 \
    --data-parallel-size 8
```

### 3. Additional Prefill Instance (Nodes 2-3)

Repeat steps 1-2 with `{PREFILL_MASTER_1_HOSTNAME}` and `{PREFILL_WORKER_1_HOSTNAME}`.

### 4. Decode Master

```bash
# Run on {DECODE_MASTER_HOSTNAME}
vllm serve {MODEL_PATH} \
    $DECODE_ARGS \
    --data-parallel-address {DECODE_MASTER_HOSTNAME} \
    --data-parallel-size 8
```

### 5. Decode Worker

```bash
# Run on {DECODE_WORKER_HOSTNAME}
vllm serve {MODEL_PATH} \
    $DECODE_ARGS \
    --data-parallel-address {DECODE_MASTER_HOSTNAME} \
    --data-parallel-start-rank 4 \
    --data-parallel-size 8
```

---

## Router Setup

The router distributes requests to prefill instances and routes KV cache transfers to decode instances.

### Build & Run Router

```bash
cd /vllm-workspace/router  # or your vllm router directory

RUST_LOG=warn cargo run --release -- \
    --policy round_robin \
    --vllm-pd-disaggregation \
    --max-concurrent-requests 9216 \
    --prefill http://{PREFILL_MASTER_0_HOSTNAME}:8087 \
    --prefill http://{PREFILL_WORKER_0_HOSTNAME}:8087 \
    --prefill http://{PREFILL_MASTER_1_HOSTNAME}:8087 \
    --prefill http://{PREFILL_WORKER_1_HOSTNAME}:8087 \
    --decode http://{DECODE_MASTER_HOSTNAME}:8087 \
    --decode http://{DECODE_WORKER_HOSTNAME}:8087 \
    --host 0.0.0.0 \
    --port 8123 \
    --intra-node-data-parallel-size 4
```

**Key Router Options:**
- `--vllm-pd-disaggregation`: Enable prefill-decode disaggregation mode
- `--prefill`: Prefill instance endpoints (can specify multiple)
- `--decode`: Decode instance endpoints (can specify multiple)
- `--intra-node-data-parallel-size`: Number of GPUs per node for hybrid load balancing

---

## Health Checks

Wait for all instances to be ready before starting the router:

```bash
# Check prefill instances
curl -s http://{PREFILL_MASTER_0_HOSTNAME}:8087/health
curl -s http://{PREFILL_WORKER_0_HOSTNAME}:8087/health
curl -s http://{PREFILL_MASTER_1_HOSTNAME}:8087/health
curl -s http://{PREFILL_WORKER_1_HOSTNAME}:8087/health

# Check decode instance
curl -s http://{DECODE_MASTER_HOSTNAME}:8087/health
curl -s http://{DECODE_WORKER_HOSTNAME}:8087/health

# Check router
curl -s http://{ROUTER_HOSTNAME}:8123/health
```

---

## Running Benchmarks

```bash
vllm bench serve \
    --model {MODEL_PATH} \
    --host {ROUTER_HOSTNAME} \
    --port 8123 \
    --dataset-name random \
    --ignore-eos \
    --num-prompts 5120 \
    --max-concurrency 2048 \
    --random-input-len 4096 \
    --random-output-len 2048 \
    --ready-check-timeout-sec 0 \
    --trust_remote_code
```

---

## Placeholder Reference

| Placeholder | Description |
|-------------|-------------|
| `{MODEL_PATH}` | Path to the model (e.g., `nvidia/DeepSeek-R1-0528-FP4-v2`) |
| `{NETWORK_INTERFACE}` | Network interface name (e.g., `eth0`, `enP22p3s0f1np1`) |
| `{HF_CACHE_DIR}` | Hugging Face cache directory |
| `{PREFILL_MASTER_0_HOSTNAME}` | Hostname/IP of prefill instance 0 master node |
| `{PREFILL_WORKER_0_HOSTNAME}` | Hostname/IP of prefill instance 0 worker node |
| `{PREFILL_MASTER_1_HOSTNAME}` | Hostname/IP of prefill instance 1 master node |
| `{PREFILL_WORKER_1_HOSTNAME}` | Hostname/IP of prefill instance 1 worker node |
| `{DECODE_MASTER_HOSTNAME}` | Hostname/IP of decode instance master node |
| `{DECODE_WORKER_HOSTNAME}` | Hostname/IP of decode instance worker node |
| `{ROUTER_HOSTNAME}` | Hostname/IP of router node |

