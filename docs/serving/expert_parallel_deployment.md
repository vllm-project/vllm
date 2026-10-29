# Expert Parallel Deployment

vLLM supports Expert Parallelism (EP), which shards routed expert weights across Mixture-of-Experts (MoE) workers. For CPU workers, see [CPU Expert Parallelism](#cpu-expert-parallelism); the accelerator examples use the GPU backends described below.

EP is typically coupled with Data Parallelism (DP). While DP can be used independently of EP, EP is more efficient when used in conjunction with DP. You can read more about data parallelism [here](data_parallel_deployment.md).

## GPU Prerequisites {#prerequisites}

Before using the GPU EP backends below, you need to install the necessary dependencies. We are actively working on making this easier in the future:

1. **Install DeepEP**: Set up host environment following vLLM's guide for EP kernels [here](../../tools/ep_kernels).
2. **Install DeepGEMM library**: Follow the [official instructions](https://github.com/deepseek-ai/DeepGEMM#installation).
3. **For disaggregated serving**: Install `gdrcopy` by running the [`install_gdrcopy.sh`](../../tools/install_gdrcopy.sh) script (e.g., `install_gdrcopy.sh "${GDRCOPY_OS_VERSION}" "12.8" "x64"`). You can find available OS versions [here](https://developer.download.nvidia.com/compute/redist/gdrcopy/CUDA%2012.8/).

!!! note "NCCL version (CUDA 13+)"
    The `deepep_v2` backend requires NCCL >= 2.30.4. PyTorch ships an older
    NCCL, so you must upgrade it before building or running DeepEP. See the
    [EP kernels guide](../../tools/ep_kernels) for instructions.

### Backend Selection Guide

vLLM provides multiple communication backends for EP. Use `--all2all-backend` to select one:

| Backend | Use Case | Features | Best For |
| ------- | -------- | -------- | -------- |
| `allgather_reducescatter` | Default backend | Standard all2all using allgather/reducescatter primitives | General purpose, works with any EP+DP configuration |
| `deepep_high_throughput` | Multi-node prefill | Grouped GEMM with continuous layout, optimized for prefill | Prefill-dominated workloads, high-throughput scenarios |
| `deepep_low_latency` | Multi-node decode | CUDA graph support, masked layout, optimized for decode | Decode-dominated workloads, low-latency scenarios |
| `flashinfer_nvlink_one_sided` | MNNVL systems | FlashInfer's one-sided A2A strategy for multi-node NVLink | High-throughput workloads |
| `passthrough` | Any | Internal, not accepted on the CLI: bound automatically for MoE backends that dispatch and combine themselves, so no all-to-all runs in the framework | `--moe-backend flashinfer_moe_ep_*` (FlashInfer MoE-EP megakernels) |
| `flashinfer_nvlink_two_sided` | MNNVL systems | FlashInfer's two-sided A2A strategy for multi-node NVLink | Systems with NVLink across nodes |

## CPU Expert Parallelism

CPU EP combines data-parallel attention with sharded routed experts using
`--enable-expert-parallel --all2all-backend allgather_reducescatter`.
DP/EP communication uses **PyTorch Gloo over TCP/IP on both the same host and
across hosts**. Existing TP/PP communication selection is preserved. NCCL,
DeepEP, RDMA and InfiniBand are not required for this CPU path.

With `TP=PP=PCP=1` and `DP=EP=2`, attention weights and each engine's KV cache
remain local to its DP rank, while each rank loads its own routed expert
weights. The existing model, attention and expert selectors determine dtype,
activation, quantization and supported layouts. BF16 unquantized experts are
the reference example below, not a runtime restriction on other configurations.

### Compatibility and validation

Install a matching [CPU build](../getting_started/installation/cpu.md) on every
worker. Record the selected attention backend, expert implementation and local
expert count; CPU ISA alone does not prove a particular expert path was used.

| Configuration | Validation status | Capability or evidence boundary |
| --- | --- | --- |
| FP32/BF16 variable-size Gloo collectives, integer routing metadata, subgroup sizes 2 and 3 | Tested on native macOS arm64 and Linux amd64 emulation | Includes empty ranks, changed sizes, subgroup ordering and independent output storage. Exact collectives also pass between two containers with private IPC namespaces. |
| BF16 mapped and modular experts with DP2/EP2 or DP3/EP3, including input weighting and separate shared experts | Tested on native macOS arm64 and Linux amd64 emulation | Full-weight references check routing weights, local ownership, remote zeros and once-only shared output. |
| `axolotl-ai-co/tiny-mixtral-30m`, revision `99512191d2561cb631b0d3176e59ade2d76b287b`, BF16 vector experts, DP2/EP2 | Eager and real Inductor tested on native macOS arm64 | Public-server tests compare fixed-prefix logprobs, uneven requests, idle ranks, later waves and warm-cache restart. MRV1 was automatically selected because Triton is unavailable on macOS. This tiny checkpoint tests mechanics rather than trained-model quality. |
| The pinned tiny Mixtral checkpoint on Linux, including primary/headless containers | Not yet validated | Model-server startup is blocked before worker creation in the tested amd64 emulation environment. Actual default Linux runner and model comparisons remain required. |
| `ibm-research/PowerMoE-3b`, revision `13fcb5a98001438bed01cf1ac4b423751dc4c2ea`, one rank, eager, macOS BF16 vector experts | Loads and generates; numerical reference validation failed | The existing top-token helper passes, but a teacher-forced HF prompt-logprob check fails identically on the rebuilt borrowed baseline and integrated build. This is not trained-model EP correctness evidence. |
| AVX2 vector SWIGLUOAI experts with bias | Numerical regression check failed in Linux amd64 emulation | A small existing non-EP case exceeds the existing tolerance on the borrowed baseline and integrated build. This is distinct from a missing capability or an EP transport failure. |
| ZenTorch fused MoE kernel with EP | Known incompatible kernel | It lacks rank-local expert-map handling; existing CPU selection bypasses it for EP. |
| Inherited FP8/MXFP4 CPU expert paths | Target-specific regression validation pending | Existing capability checks require the appropriate native operator and CPU/OS ISA support; vector-only emulation does not validate AMX. |
| Default Linux compiled full-model execution and deployment on two independent Linux hosts | Not yet validated | macOS Inductor and two local containers do not establish these deployment claims. |

Configurations outside the reference tests are not automatically unsupported.
Use existing capability checks and their specific errors. Dynamic CPU EPLB is
separate work.

The refreshed macOS eager/compiled tests completed shutdown and restart with
no remaining server processes or forced kills. This does not establish
recovery from a failed collective or physical two-host behavior.

On the tested Docker setup, amd64-emulated model startup blocks while spawning
a subprocess after PyZMQ starts its networking threads. A minimal Python/PyZMQ
reproducer also blocks without vLLM; its native Linux arm64 control passes.
Collective and layer results from that emulation remain useful, while native
Linux x86-64 validation is still needed for MRV2 and target ISA claims.

### Same-host reference layout

Activate the CPU build's environment and first verify the pinned checkpoint on
one rank. Then run two local DP engines with EP:

```bash
export VLLM_CPU_KVCACHE_SPACE=1

.venv/bin/vllm serve axolotl-ai-co/tiny-mixtral-30m \
  --revision 99512191d2561cb631b0d3176e59ade2d76b287b \
  --dtype bfloat16 --enforce-eager \
  --max-model-len 512 --max-num-batched-tokens 512 --max-num-seqs 4 \
  --tensor-parallel-size 1 --pipeline-parallel-size 1 \
  --data-parallel-size 2 --data-parallel-size-local 2 \
  --data-parallel-backend mp --distributed-executor-backend mp \
  --enable-expert-parallel --all2all-backend allgather_reducescatter \
  --host 127.0.0.1 --port 8000
```

Size KV caches and CPU core bindings per engine. Leave capacity for the API,
scheduler and communication threads. Measure peak memory during loading and
compilation as well as inference: unequal gathers pad to the largest rank's
batch, and combine temporarily holds the global contribution tensor.

### Primary/headless layout across hosts

The following is a cross-host validation recipe; physical two-host acceptance
remains pending. Use the same vLLM revision, rebuilt CPU extension, PyTorch
dependencies and checkpoint revision on both workers. Substitute each host's
reachable private IP and private interface for these examples.

```bash
# Node A: primary API and DP rank 0.
export VLLM_HOST_IP=10.0.0.10
export GLOO_SOCKET_IFNAME=eth0
export VLLM_CPU_KVCACHE_SPACE=1

.venv/bin/vllm serve axolotl-ai-co/tiny-mixtral-30m \
  --revision 99512191d2561cb631b0d3176e59ade2d76b287b \
  --dtype bfloat16 --enforce-eager \
  --max-model-len 512 --max-num-batched-tokens 512 --max-num-seqs 4 \
  --tensor-parallel-size 1 --pipeline-parallel-size 1 \
  --data-parallel-size 2 --data-parallel-size-local 1 \
  --data-parallel-backend mp --distributed-executor-backend mp \
  --enable-expert-parallel --all2all-backend allgather_reducescatter \
  --data-parallel-address 10.0.0.10 --data-parallel-rpc-port 13345 \
  --cpu-distributed-timeout-seconds 300 --distributed-timeout-seconds 300 \
  --host 10.0.0.10 --port 8000
```

```bash
# Node B: headless DP rank 1.
export VLLM_HOST_IP=10.0.0.11
export GLOO_SOCKET_IFNAME=eth0
export VLLM_CPU_KVCACHE_SPACE=1

.venv/bin/vllm serve axolotl-ai-co/tiny-mixtral-30m \
  --revision 99512191d2561cb631b0d3176e59ade2d76b287b \
  --dtype bfloat16 --enforce-eager \
  --max-model-len 512 --max-num-batched-tokens 512 --max-num-seqs 4 \
  --tensor-parallel-size 1 --pipeline-parallel-size 1 \
  --data-parallel-size 2 --data-parallel-size-local 1 \
  --data-parallel-backend mp --distributed-executor-backend mp \
  --enable-expert-parallel --all2all-backend allgather_reducescatter \
  --data-parallel-address 10.0.0.10 --data-parallel-rpc-port 13345 \
  --cpu-distributed-timeout-seconds 300 --distributed-timeout-seconds 300 \
  --data-parallel-start-rank 1 --headless
```

This is internal load balancing: only Node A exposes the API. Do not add
`--data-parallel-rank`, which selects external load balancing, or the offline
example's `--dp-num-nodes`, which is not a `vllm serve` flag. These DP-only
launches do not use `--nnodes` to establish physical locality.

For compiled validation, remove `--enforce-eager` on both workers and unset
`VLLM_CPU_CI_ENV`; record the runner actually selected and verify real Inductor
execution. A forced MRV1 run or the CI-eager shortcut is not default compiled
validation.

### CPU network requirements

Follow the [inter-node security guidance](../usage/security.md#inter-node-communication):
worker traffic belongs on an isolated private network with direct peer TCP
reachability.

| Connection | Required access |
| --- | --- |
| API | Authorized clients reach Node A's API port; scope this separately from worker traffic. |
| DP startup | Node B reaches Node A's configured `--data-parallel-rpc-port`. |
| Gloo peers, stores and engine/coordinator endpoints | Allow dynamic internal TCP listeners and return traffic between the specific worker private IPs or allowlisted worker-group members. Opening only the DP startup port is insufficient. |
| Interface selection | `VLLM_HOST_IP` is that worker's reachable private address; `GLOO_SOCKET_IFNAME` selects its private NIC, not a port range. |

There is no supported `GLOO_PORT_RANGE` setting or fixed two-port firewall
recipe for this path. A small fixed internal port allowlist requires separate
port-allocation work. Pre-download matching model artifacts; a shared filesystem
is not required.

For container simulation, use two workers with distinct reachable container
addresses and private IPC namespaces on a bridge network. Containers on the
same Docker network can reach the dynamic peer ports without publishing them
to the host. A two-container run is useful functional evidence; it does not
validate independent host kernels, physical NICs or host firewalls. For actual
Linux multi-host containers, host networking is the initial deployment layout
to validate; publishing one or two ports is insufficient.

If startup stalls, first check advertised IPs, interfaces and peer TCP access.
If connections succeed, check corresponding collective sizes, tensor shapes,
dtypes and ordering. An idle engine must still participate in coordinated
dummy forwards. API readiness alone is not a numerical correctness check. Use
bounded group deadlines and restart the deployment after a collective failure.

## Single Node Deployment

### Configuration

Enable EP by setting the `--enable-expert-parallel` flag. The EP size is automatically calculated as:

```text
EP_SIZE = TP_SIZE × DP_SIZE
```

Where:

- `TP_SIZE`: Tensor parallel size
- `DP_SIZE`: Data parallel size
- `EP_SIZE`: Expert parallel size (computed automatically)

### Layer Behavior with EP Enabled

When EP is enabled, different layers in MoE models behave differently:

| Layer Type | Behavior | Parallelism Used |
| ---------- | -------- | ---------------- |
| **Expert (MoE) Layers** | Sharded across all EP ranks | Expert Parallel (EP) of size `TP × DP` |
| **Attention Layers** | Behavior depends on TP size | See below |

**Attention layer parallelism:**

- **When `TP = 1`**: Attention weights are **replicated** across all DP ranks (data parallelism)
- **When `TP > 1`**: Attention weights are **sharded** using tensor parallelism across TP ranks within each DP group

For example, with `TP=2, DP=4` (8 GPUs total):

- Expert layers form an EP group of size 8, with experts distributed across all GPUs
- Attention layers use TP=2 within each of the 4 DP groups

!!! note "Key Difference from Data Parallel Deployment"
    Without `--enable-expert-parallel`, MoE layers would use tensor parallelism (forming a TP group of size `TP × DP`), similar to dense models. With EP enabled, expert layers switch to expert parallelism, which can provide better efficiency and locality for MoE models.

### Example Command

The following command serves a `DeepSeek-V3-0324` model with 1-way tensor parallel, 8-way (attention) data parallel, and 8-way expert parallel. The attention weights are replicated across all GPUs, while the expert weights are split across GPUs. It will work on a H200 (or H20) node with 8 GPUs. For H100, you can try to serve a smaller model or refer to the multi-node deployment section.

```bash
# Single node EP deployment
vllm serve deepseek-ai/DeepSeek-V3-0324 \
    --tensor-parallel-size 1 \       # Tensor parallelism across 1 GPU
    --data-parallel-size 8 \         # Data parallelism across 8 processes
    --enable-expert-parallel         # Enable expert parallelism
```

## Multi-Node Deployment

For the GPU multi-node examples below, use the DeepEP communication kernel with one of two modes (see [Backend Selection Guide](#backend-selection-guide) above).

### Deployment Steps

1. **Run one command per node** - Each node requires its own launch command
2. **Configure networking** - Ensure proper IP addresses and port configurations
3. **Set node roles** - First node handles requests, additional nodes run in headless mode

### Example: 2-Node Deployment

The following example deploys `DeepSeek-V3-0324` across 2 nodes using `deepep_low_latency` mode:

```bash
# Node 1 (Primary - handles incoming requests)
vllm serve deepseek-ai/DeepSeek-V3-0324 \
    --all2all-backend deepep_low_latency \
    --tensor-parallel-size 1 \               # TP size per node
    --enable-expert-parallel \               # Enable EP
    --data-parallel-size 16 \                # Total DP size across all nodes
    --data-parallel-size-local 8 \           # Local DP size on this node (8 GPUs per node)
    --data-parallel-address 192.168.1.100 \  # Replace with actual IP of Node 1
    --data-parallel-rpc-port 13345 \         # RPC communication port, can be any port as long as reachable by all nodes
    --api-server-count=8                     # Number of API servers for load handling (scaling this out to # local ranks is recommended)

# Node 2 (Secondary - headless mode, no API server)
vllm serve deepseek-ai/DeepSeek-V3-0324 \
    --all2all-backend deepep_low_latency \
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

Dynamic CPU EPLB is separate from the CPU DP/EP path above. The following configuration applies to backends with EPLB support.

While MoE models are typically trained so that each expert receives a similar number of tokens, in practice the distribution of tokens across experts can be highly skewed. vLLM provides an Expert Parallel Load Balancer (EPLB) to redistribute expert mappings across EP ranks, evening the load across experts.

### Configuration

Enable EPLB with the `--enable-eplb` flag.

When enabled, vLLM collects load statistics with every forward pass and periodically rebalances expert distribution.

### EPLB Parameters

Configure EPLB with the `--eplb-config` argument, which accepts a JSON string. The available keys and their descriptions are:

| Parameter | Description | Default |
| --------- | ----------- | ------- |
| `window_size` | Number of engine steps to track for rebalancing decisions | 1000 |
| `step_interval` | Frequency of rebalancing (every N engine steps) | 3000 |
| `log_balancedness` | Log balancedness metrics (avg tokens per expert ÷ max tokens per expert) | `false` |
| `num_redundant_experts` | Additional global experts per EP rank beyond equal distribution | `0` |
| `use_async` | Use non-blocking EPLB for reduced latency overhead | `true` |
| `policy` | The policy type for expert parallel load balancing | `"default"` |
| `communicator` | Backend for expert weight transfers: `"torch_nccl"`, `"torch_gloo"`, `"pynccl"`, `"nixl"`,  or `null` (auto) | `null` |

For example:

```bash
vllm serve Qwen/Qwen3-30B-A3B \
  --enable-eplb \
  --eplb-config '{"window_size":1000,"step_interval":3000,"num_redundant_experts":2,"log_balancedness":true}'
```

??? tip "Prefer individual arguments instead of JSON?"

    ```bash
    vllm serve Qwen/Qwen3-30B-A3B \
            --enable-eplb \
            --eplb-config.window_size 1000 \
            --eplb-config.step_interval 3000 \
            --eplb-config.num_redundant_experts 2 \
            --eplb-config.log_balancedness true
    ```

### Expert Distribution Formula

- **Default**: Each EP rank has `NUM_TOTAL_EXPERTS ÷ NUM_EP_RANKS` experts
- **With redundancy**: Each EP rank has `(NUM_TOTAL_EXPERTS + NUM_REDUNDANT_EXPERTS) ÷ NUM_EP_RANKS` experts

### Memory Footprint Overhead

EPLB uses redundant experts that need to fit in GPU memory. This means that EPLB may not be a good fit for memory constrained environments or when KV cache space is at a premium.

This overhead equals `NUM_MOE_LAYERS * BYTES_PER_EXPERT * (NUM_TOTAL_EXPERTS + NUM_REDUNDANT_EXPERTS) ÷ NUM_EP_RANKS`.
For DeepSeekV3, this is approximately `2.4 GB` for one redundant expert per EP rank.

### Example Command

Single node deployment with EPLB enabled:

```bash
# Single node with EPLB load balancing
vllm serve deepseek-ai/DeepSeek-V3-0324 \
    --tensor-parallel-size 1 \       # Tensor parallelism
    --data-parallel-size 8 \         # Data parallelism
    --enable-expert-parallel \       # Enable EP
    --enable-eplb \                  # Enable load balancer
    --eplb-config '{"window_size":1000,"step_interval":3000,"num_redundant_experts":2,"log_balancedness":true}'
```

For multi-node deployment, add these EPLB flags to each node's command. We recommend setting `--eplb-config '{"num_redundant_experts":32}'` to 32 in large scale use cases so the most popular experts are always available.

## Advanced Configuration

### Performance Optimization

- **DeepEP kernels**: The `high_throughput` and `low_latency` kernels are optimized for disaggregated serving and may show poor performance for mixed workloads
- **Dual Batch Overlap**: Use `--enable-dbo` to overlap all-to-all communication with compute. See [Dual Batch Overlap](../design/dbo.md) for more details.
- **Async scheduling (experimental)**: Try `--async-scheduling` to overlap scheduling with model execution.

### Troubleshooting

- **`non-zero status: 7 cannot register cq buf`**: When using Infiniband/RoCE, make sure host VM and pods show `ulimit -l` "unlimited".
- **`init failed for transport: IBGDA`**: The InfiniBand GDA kernel modules are missing. Run `tools/ep_kernels/configure_system_drivers.sh` on each GPU node and reboot. Also fixes error `NVSHMEM API called before NVSHMEM initialization has completed`.
- **NVSHMEM peer disconnect**: Usually a networking misconfiguration. If deploying via Kubernetes, verify that every pod runs with `hostNetwork: true`, `securityContext.privileged: true` to access Infiniband.

### Benchmarking

- Use simulator flags `VLLM_MOE_ROUTING_SIMULATION_STRATEGY=uniform_random` and `VLLM_RANDOMIZE_DP_DUMMY_INPUTS=1` so token routing is balanced across EP ranks.

## Disaggregated Serving (Prefill/Decode Split)

For production deployments requiring strict SLA guarantees for time-to-first-token and inter-token latency, disaggregated serving allows independent scaling of prefill and decode operations.

### Architecture Overview

- **Prefill Instance**: Uses `deepep_high_throughput` backend for optimal prefill performance
- **Decode Instance**: Uses `deepep_low_latency` backend for minimal decode latency  
- **KV Cache Transfer**: Connects instances via NIXL or other KV connectors

### Setup Steps

1. **Install gdrcopy/ucx/nixl**: For maximum performance, run the [install_gdrcopy.sh](../../tools/install_gdrcopy.sh) script to install `gdrcopy` (e.g., `install_gdrcopy.sh "${GDRCOPY_OS_VERSION}" "12.8" "x64"`). You can find available OS versions [here](https://developer.download.nvidia.com/compute/redist/gdrcopy/CUDA%2012.8/). If `gdrcopy` is not installed, things will still work with a plain `pip install nixl`, just with lower performance. `nixl` and `ucx` are installed as dependencies via pip. For non-cuda platform to install nixl with non-cuda UCX build, run the [install_nixl_from_source_ubuntu.py](../../tools/install_nixl_from_source_ubuntu.py) script.

2. **Configure Both Instances**: Add this flag to both prefill and decode instances `--kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both"}`. Noted, you may also specify one or multiple NIXL_Backend. Such as: `--kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both", "kv_connector_extra_config":{"backends":["UCX", "GDS"]}}'`

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
                "remote_port": None,          # Will be populated by vLLM
            }
        },
        extra_headers={"X-Request-Id": request_id},
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
        extra_headers={"X-Request-Id": request_id},  # Same request ID
    )
    
    print("-" * 50)
    print("✓ Decode completed successfully")
    print(f"Final response: {decode_response.choices[0].text}")

except Exception as e:
    print(f"❌ Error during disaggregated serving: {e}")
    print("Check that both prefill and decode instances are running and accessible")
```

### Benchmarking

- To simulate the decode deployment of disaggregated serving, pass `--kv-transfer-config '{"kv_connector":"DecodeBenchConnector","kv_role":"kv_both"}'` to the `vllm serve` invocation. The connector populates KV cache with random values so decode can be profiled in isolation. Add `"kv_connector_extra_config":{"startup_fill":true}` to fill the whole KV cache once at startup instead of in the decode steps that admit new requests (see the connector docstring for its limitations).

- **CUDAGraph capture**: Use `--compilation_config '{"cudagraph_mode": "FULL_DECODE_ONLY"}'` to enable CUDA graph capture for decode only and save KV cache.
