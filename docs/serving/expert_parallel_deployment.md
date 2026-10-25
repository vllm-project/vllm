# Expert Parallel Deployment

vLLM supports Expert Parallelism (EP), which allows experts in Mixture-of-Experts (MoE) models to be deployed on separate GPUs, increasing locality, efficiency, and throughput overall.

EP is typically coupled with Data Parallelism (DP). While DP can be used independently of EP, EP is more efficient when used in conjunction with DP. You can read more about data parallelism [here](data_parallel_deployment.md).

## Prerequisites

Before using EP, you need to install the necessary dependencies. We are actively working on making this easier in the future:

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

For multi-node deployment, use the DeepEP communication kernel with one of two modes (see [Backend Selection Guide](#backend-selection-guide) above).

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

## Automatic External Elastic EP Bootstrap

With external load balancing, `--elastic-ep-auto-bootstrap` lets initial
participants and later append joiners use the same launch arguments. Only the
native DP rank varies between processes; vLLM does not interpret Kubernetes
labels or create capacity. Automatic bootstrap is disabled by default.

For an initial DP2 world that can grow to DP4, run this command for ranks 0 and 1,
with a separate GPU allocation and API endpoint for each process. On one host,
assign disjoint devices and distinct HTTP ports as described in
[External Load Balancing](data_parallel_deployment.md#external-load-balancing).

```bash
vllm serve Qwen/Qwen3-30B-A3B \
    --enable-expert-parallel \
    --enable-eplb \
    --eplb-config '{"communicator":"pynccl","use_async":false}' \
    --enable-elastic-ep \
    --elastic-ep-auto-bootstrap \
    --elastic-ep-initial-dp-size 2 \
    --elastic-ep-max-dp-size 4 \
    --data-parallel-size-local 1 \
    --data-parallel-rank "${DP_RANK}" \
    --data-parallel-address "${DP_MASTER_ADDRESS}" \
    --data-parallel-rpc-port 29550 \
    --tensor-parallel-size 1
```

To grow to DP4, send `POST /scale_elastic_ep` with
`{"new_data_parallel_size": 4}` concurrently to both existing API servers.
These requests wait for the resize: launch ranks 2 and 3 with the same command
while the requests are pending, rather than waiting for their responses first.
Leave `--elastic-ep-initial-dp-size 2` unchanged. It declares only the initial
world size, not the current topology. Both joiners discover target size 4
through rank zero's existing scale handshake before constructing their engine
configuration.

Automatic bootstrap requires `--elastic-ep-initial-dp-size` and rejects an
explicit `--data-parallel-size` (including `-dp` or a configuration-file value).
The initial-size option requires `--elastic-ep-auto-bootstrap`; it is not used
by manual launches. Do not set `VLLM_ELASTIC_EP_SCALE_UP_LAUNCH`; automatic
bootstrap selects the launch path.

Initial ranks form the complete initial world without synthetic resize requests.
Discovery reserves a rank or local rank range for one client and one handshake;
each subsequent engine handshake checks that reservation before initialization. Discovery is
not membership commit or serving readiness, and does not trigger a resize. The
external load balancer must still follow each endpoint's serving health.

This mode supports external MP DP with one API server/core client per process.
The default local rank count is one; see [Local Rank Ranges](#local-rank-ranges)
for multi-rank startup. Local routing is inferred automatically for multi-rank
clients; do not add explicit hybrid-LB or start-rank options. Headless startup
does not use this discovery path. Fault tolerance may be enabled with an FT-capable all-to-all backend, as
required by vLLM (`nixl_ep` or `deepep_low_latency`); automatic bootstrap does not perform
failure recovery or fill inactive slots in a committed topology. A rank removed
by a completed shrink may join a later expansion when the engine's handshake
expects it. Existing executor launch and
geometry validation remain unchanged. This option does not configure headless
TP workers or implement a new multi-node executor lifecycle.

Rank zero always starts a new world; replacing it requires external whole-world
lifecycle management. A replacement for an existing member cannot silently join:
without a matching initial or append handshake, startup fails or times out according to
`VLLM_ENGINE_READY_TIMEOUT_S`.

Manual external Elastic EP launch arguments remain supported without this flag.

### Local Rank Ranges

One client can bootstrap several local engines together. Set
`--data-parallel-rank` to its first global rank and
`--data-parallel-size-local` to its local count. Do not enable
`--data-parallel-multi-port-external-lb`: that option starts separate clients.

For an initial DP8 world with four local ranks per client and a maximum of DP12,
use the same template for each client, varying only `FIRST_DP_RANK` (0 or 4
initially, then 8 for an append joiner):

The same template is used for the native resize workflow below; only the first
rank varies between clients.

```bash
vllm serve Qwen/Qwen3-30B-A3B \
    --enable-expert-parallel --enable-eplb \
    --eplb-config '{"communicator":"pynccl","use_async":false}' \
    --enable-elastic-ep --elastic-ep-auto-bootstrap \
    --elastic-ep-initial-dp-size 8 --elastic-ep-max-dp-size 12 \
    --data-parallel-rank "${FIRST_DP_RANK}" \
    --data-parallel-size-local 4 \
    --data-parallel-address "${DP_MASTER_ADDRESS}" \
    --data-parallel-rpc-port 29550 --tensor-parallel-size 1
```

The client starting at rank 8 sends one discovery request for ranks 8–11. An
active native DP12 append handshake reserves the complete range atomically;
overlaps, ranges outside its expected participants and invalid counts are
rejected without partially reserving ranks. The client resolves DP12 before
constructing its engine configuration. Its existing process manager starts all
four engines, which share the reservation and receive the same native
rendezvous information. No client guesses the target from its last local rank.
The initial seed remains 8 and `--data-parallel-size` stays omitted.

Each process needs enough visible GPUs for its local engines; colocated clients
need disjoint devices and distinct HTTP endpoints. Multi-rank clients use the
existing local load-balancing client (`DPLBAsyncMPClient`), including its
request-to-engine tracking for cancellation. Runtime load balancing is hybrid
when the client owns only part of the world, or internal when it owns the
whole world; it is not single-rank external LB.

Without a rank hint, the client balances requests among its local engines.
To target one explicitly through the OpenAI HTTP API, set
`X-data-parallel-rank` to a **local** index from 0 through local count minus 1.
For a client whose first global rank is 8, a header value of 2 selects global
rank 10, not global rank 2. The original one-rank path stays direct and does
not acquire a supervisor or a local load balancer.

#### Resize Local Rank Ranges

Send one `POST /scale_elastic_ep` request per existing client, not per engine.
Send the requests concurrently, with the same `new_data_parallel_size`. Each
client coordinates preparation, draining and commit for all its owned engines
and waits for every local completion. Surviving clients keep their first rank,
local count and routing identities; only their global world metadata changes.

For EP8→EP12→EP8 with TP1 and four local ranks per client:

1. Start clients at first ranks 0 and 4 using the template above. Each owns four
   engines, forming the initial DP8 world.
2. Send `{"new_data_parallel_size":12}` concurrently to both clients. While
   these requests are pending, start a third client at first rank 8 using the
   unchanged initial-DP8 template. It discovers DP12 and starts ranks 8–11.
3. Wait for both resize requests to succeed and verify serving from all three
   clients. Local rank hints 0–3 can verify every owned engine independently.
4. Send `{"new_data_parallel_size":8}` concurrently to **all three** clients,
   including the retiring client at rank 8. Clients at 0 and 4 retain their
   original four engines. All engines of the rank-8 client retire; its API
   rejects inference while keeping `/health` and `POST /is_scaling_elastic_ep`
   available until the external lifecycle owner stops the process or Pod.

Wait for all three shrink requests to succeed before stopping the retired
client's process or Pod. Separately verify surviving clients' serving readiness;
a retired client's reachable health/status endpoints do not reopen inference.

World sizes must retain or retire whole existing client ranges. For example,
DP12→DP10 would split the rank-8 client's range and is rejected before existing
engines start reconfiguration. The maximum DP size is still a bound, not the
live target, and bootstrap discovery alone does not commit membership.

To drain requests before the collective pause, set
`VLLM_ELASTIC_EP_DRAIN_REQUESTS=1` on every client. A prepare/drain failure is
propagated to peers; a failed commit leaves inference admission closed because
the topology may be partially changed. Do not treat an HTTP response or a
completed control-plane status as a substitute for current serving health.

## Expert Parallel Load Balancer (EPLB)

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
