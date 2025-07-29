# Expert Parallel Deployment

vLLM supports Expert Parallelism (EP), which allows experts in Mixture-of-Experts (MoE) models to be deployed on separate GPUs, increasing locality, efficiency, and throughput overall.

EP is typically coupled with Data Parallelism (DP). While DP can be used independently of EP, EP is more efficient when used in conjunction with DP. You can read more about data parallelism [here](data_parallel_deployment.md).

## Prerequisites

Before using EP, you need to install the necessary dependencies. We are actively working on making this easier in the future:

1. **Install DeepEP and pplx-kernels**: Set up host environment following vLLM's guide for EP kernels [here](gh-file:tools/ep_kernels).
2. **Install DeepGEMM library**: Follow the [official instructions](https://github.com/deepseek-ai/DeepGEMM#installation).
3. **For disaggregated serving**: Install UCX and NIXL following the [script](gh-file:tools/install_nixl.sh).

### Backend Selection Guide

vLLM provides three communication backends for EP:

| Backend | Use Case | Features | Best For |
|---------|----------|----------|----------|
| `pplx` | Single node | Chunked prefill support | Development, best for intra-node deployments |
| `deepep_high_throughput` | Multi-node prefill | Grouped GEMM with continuous layout | High-throughput scenarios, prefill-dominated workloads |
| `deepep_low_latency` | Multi-node decode | CUDA graph support, masked layout | Low-latency scenarios, decode-dominated workloads |

## Single Node Deployment

!!! warning
    EP is an experimental feature. Argument names and default values may change in the future.

### Configuration

Enable EP by setting the `--enable-expert-parallel` flag. The EP size is automatically calculated as:

```
EP_SIZE = TP_SIZE × DP_SIZE
```

Where:
- `TP_SIZE`: Tensor parallel size (always 1 for now)
- `DP_SIZE`: Data parallel size
- `EP_SIZE`: Expert parallel size (computed automatically)

### Example Command

The following command serves a `DeepSeek-V3-0324` model with 1-way tensor parallel, 8-way (attention) data parallel, and 8-way expert parallel. The attention weights are replicated across all GPUs, while the expert weights are split across GPUs. It will work on a H200 (or H20) node with 8 GPUs. For H100, you can try to serve a smaller model or refer to the multi-node deployment section.

```bash
# Single node EP deployment with pplx backend
VLLM_ALL2ALL_BACKEND=pplx VLLM_USE_DEEP_GEMM=1 \
    vllm serve deepseek-ai/DeepSeek-V3-0324 \
    --tensor-parallel-size 1 \      # Tensor parallelism across 1 GPU
    --data-parallel-size 8 \         # Data parallelism across 8 processes
    --enable-expert-parallel         # Enable expert parallelism
```

## Multi-Node Deployment

For multi-node deployment, use the DeepEP communication kernel with one of two modes (see [Backend Selection Guide](#backend-selection-guide) above).

### Deployment Steps

1. **Run one command per node** - Each node requires its own launch command
2. **Configure networking** - Ensure proper IP addresses and port configurations
3. **Set node roles** - First node handles requests, additional nodes run in headless mode

### Example: 2-Node Deployment

The following example deploys `DeepSeek-V3-0324` across 2 nodes using `deepep_low_latency` mode:

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

### Key Configuration Notes

- **Headless mode**: Secondary nodes run with `--headless` flag, meaning all client requests are handled by the primary node
- **Rank calculation**: `--data-parallel-start-rank` should equal the cumulative local DP size of previous nodes
- **Load scaling**: Adjust `--api-server-count` on the primary node to handle higher request loads

### Network Configuration

!!! important "InfiniBand Clusters"
    On InfiniBand networked clusters, set this environment variable to prevent initialization hangs:
    ```bash
    export GLOO_SOCKET_IFNAME=eth0
    ```
    This ensures torch distributed group discovery uses Ethernet instead of InfiniBand for initial setup.

## Expert Parallel Load Balancer (EPLB)

While MoE models are typically trained so that each expert receives a similar number of tokens, in practice the distribution of tokens across experts can be highly skewed. vLLM provides an Expert Parallel Load Balancer (EPLB) to redistribute expert mappings across EP ranks, evening the load across experts.

### Configuration

Enable EPLB with the `--enable-eplb` flag.

!!! note "Model Support"
    Currently only DeepSeek V3 architecture is supported.

When enabled, vLLM collects load statistics with every forward pass and periodically rebalances expert distribution.

### EPLB Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--eplb-window-size` | Number of engine steps to track for rebalancing decisions | - |
| `--eplb-step-interval` | Frequency of rebalancing (every N engine steps) | - |
| `--eplb-log-balancedness` | Log balancedness metrics (avg tokens per expert ÷ max tokens per expert) | `false` |
| `--num-redundant-experts` | Additional global experts per EP rank beyond equal distribution | `0` |

### Expert Distribution Formula

- **Default**: Each EP rank has `NUM_TOTAL_EXPERTS ÷ NUM_EP_RANKS` experts
- **With redundancy**: Each EP rank has `(NUM_TOTAL_EXPERTS + NUM_REDUNDANT_EXPERTS) ÷ NUM_EP_RANKS` experts

### Example Command

Single node deployment with EPLB enabled:

```bash
# Single node with EPLB load balancing
VLLM_ALL2ALL_BACKEND=pplx VLLM_USE_DEEP_GEMM=1 vllm serve deepseek-ai/DeepSeek-V3-0324 \
    --tensor-parallel-size 1 \     # Tensor parallelism
    --data-parallel-size 8 \        # Data parallelism  
    --enable-expert-parallel \      # Enable EP
    --enable-eplb \                 # Enable load balancer
    --eplb-log-balancedness \       # Log balancing metrics
    --eplb-window-size 1000 \       # Track last 1000 engine steps
    --eplb-step-interval 3000       # Rebalance every 3000 steps
```

For multi-node deployment, add these EPLB flags to each node's command. We recommend setting `--num-redundant-experts` to 32 in large scale use cases so the most popular experts are always available.

## Disaggregated Serving (Prefill/Decode Split)

For production deployments requiring strict SLA guarantees for time-to-first-token and inter-token latency, disaggregated serving allows independent scaling of prefill and decode operations.

### Architecture Overview

- **Prefill Instance**: Uses `deepep_high_throughput` backend for optimal prefill performance
- **Decode Instance**: Uses `deepep_low_latency` backend for minimal decode latency  
- **KV Cache Transfer**: Connects instances via NIXL or other KV connectors

### Setup Steps

1. **Install KV Connector**: Install NIXL using the [installation script](gh-file:tools/install_nixl.sh)

2. **Configure Both Instances**: Add this flag to both prefill and decode instances `--kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both"}`

3. **Client Orchestration**: Use the client-side script below to coordinate prefill/decode operations. We are actively working on routing solutions.

### Client Orchestration Example

```python
from openai import OpenAI
import uuid

try:
    # 1: Set up clients for prefill and decode instances
    openai_api_key = "EMPTY"  # vLLM doesn't require a real API key
    
    # Replace these IP addresses with your actual instance addresses
    prefill_client = OpenAI(
        api_key=openai_api_key,
        base_url="http://192.168.1.100:8000/v1",  # Prefill instance URL
    )
    decode_client = OpenAI(
        api_key=openai_api_key,
        base_url="http://192.168.1.101:8001/v1",  # Decode instance URL  
    )
    
    # Get model name from prefill instance
    models = prefill_client.models.list()
    model = models.data[0].id
    print(f"Using model: {model}")

    # 2: Prefill Phase
    # Generate unique request ID to link prefill and decode operations
    request_id = str(uuid.uuid4())
    print(f"Request ID: {request_id}")
    
    prefill_response = prefill_client.completions.create(
        model=model,
        # Prompt must exceed vLLM's block size (16 tokens) for PD to work
        prompt="Write a detailed explanation of Paged Attention for Transformers works including the management of KV cache for multi-turn conversations",
        max_tokens=1,  # Force prefill-only operation
        extra_body={
            "kv_transfer_params": {
                "do_remote_decode": True,     # Enable remote decode
                "do_remote_prefill": False,   # This is the prefill instance
                "remote_engine_id": None,     # Will be populated by vLLM
                "remote_block_ids": None,     # Will be populated by vLLM
                "remote_host": None,          # Will be populated by vLLM
                "remote_port": None           # Will be populated by vLLM
            }
        },
        extra_headers={"X-Request-Id": request_id}
    )
    
    print("-" * 50)
    print("✓ Prefill completed successfully")
    print(f"Prefill response: {prefill_response.choices[0].text}")
    
    # 3: Decode Phase
    # Transfer KV cache parameters from prefill to decode instance
    decode_response = decode_client.completions.create(
        model=model,
        prompt="This prompt is ignored during decode",  # Original prompt not needed
        max_tokens=150,  # Generate up to 150 tokens
        extra_body={
            "kv_transfer_params": prefill_response.kv_transfer_params  # Pass KV cache info
        },
        extra_headers={"X-Request-Id": request_id}  # Same request ID
    )
    
    print("-" * 50)
    print("✓ Decode completed successfully")
    print(f"Final response: {decode_response.choices[0].text}")

except Exception as e:
    print(f"❌ Error during disaggregated serving: {e}")
    print("Check that both prefill and decode instances are running and accessible")
```
