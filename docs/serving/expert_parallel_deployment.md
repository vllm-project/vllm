# Expert Parallel Deployment

vLLM supports expert parallelism (EP), which allows experts in Mixture-of-Experts (MoE) models to be deployed on a separate GPU, increasing locality, efficiency, and throughput overall.

EP is typically coupled with attention data parallelism (DP). While DP can be used independently of EP, EP is more efficient when used in conjunction with DP. You can read more about data parallelism [here](data_parallel_deployment.md).

In order to use EP, you need to install the necessary dependencies, we are actively working on making this easier in the future:

1. Install DeepEP and pplx-kernels, and set up host environment following vLLM's guide for EP kernels [here](https://github.com/vllm-project/vllm/tree/main/tools/ep_kernels).
2. Install DeepGEMM library following the [instructions](https://github.com/deepseek-ai/DeepGEMM#installation).
3. For Prefill/Decode (PD) disaggregated serving set up, install UCX and NIXL following the [script](https://github.com/vllm-project/vllm/blob/main/tools/install_nixl.sh).

## Enabling EP on Single Node

!!! warning
    Given EP is an experimental feature, the arguments names and default values may change in the future.

You can enable EP by setting the flag `--enable-expert-parallel` in the command line.
Currently, EP size is fixed to be the value of TP size times DP size: `EP_SIZE = TP_SIZE * DP_SIZE` (we expect to change this in the future).

The following command will serve a `DeepSeek-V3-0324` model with 1 way tensor parallel, 8 way data parallel, and 8 way expert parallel. The `pplx` backend is great for chunked prefill or single node setting where you do not wish to utilize disaggregated serving.

```bash
VLLM_ALL2ALL_BACKEND=pplx VLLM_USE_DEEP_GEMM=1 vllm serve deepseek-ai/DeepSeek-V3-0324 \
    --tensor-parallel-size 1 \
    --data-parallel-size 8 \
    --enable-expert-parallel \
```

## Enabling EP on Multi-Node

For multi-node deployment, we recommend using DeepEP communication kernel. It has two modes:

- `deepep_high_throughput`: This mode is suitable for high-throughput scenarios where prefill dominates. It does not support CUDA graph with padding and utilizes grouped gemm with continuous layout kernel.
- `deepep_low_latency`: This mode is suitable for low-latency scenarios where decode dominates. It supports CUDA graph with padding and utilizes grouped gemm with masked layout kernel.

When deploying in multi-node environment, you need to run one launch command per node. The following command will serve a `DeepSeek-V3-0324` model on 2 nodes, using `deepep_low_latency` mode:

```bash
# Node 1
VLLM_ALL2ALL_BACKEND=deepep_low_latency VLLM_USE_DEEP_GEMM=1 vllm serve deepseek-ai/DeepSeek-V3-0324 \
    --tensor-parallel-size 1 \
    --enable-expert-parallel \
    --data-parallel-size 16 \
    --data-parallel-size-local 8 \
    --data-parallel-address <node1_ip> \
    --data-parallel-rpc-port 13345 \
    --api-server-count=8 

# Node 2
VLLM_ALL2ALL_BACKEND=deepep_low_latency VLLM_USE_DEEP_GEMM=1 vllm serve deepseek-ai/DeepSeek-V3-0324 \
    --tensor-parallel-size 1 \
    --enable-expert-parallel \
    --data-parallel-size 16 \
    --data-parallel-size-local 8 \
    --data-parallel-start-rank 8 \
    --data-parallel-address <node1_ip> \
    --data-parallel-rpc-port 13345 \
    --headless
```

The main difference between the first node and subsequent nodes is the `data-parallel-start-rank` flag and the rest of the nodes being `--headless`.
Headless mode means all incoming requests should be handled by the first node.
You can configure `--api-server-count` to scale out the number of API servers to handle higher load.

!!! note
    If you are on a common Infiniband networked cluster, please set `export GLOO_SOCKET_IFNAME=eth0` to ensure the initial torch distributed group discovery is not leveraging IB.
    Without this flag, the initialization will hang.

## EPLB (Expert Parallel Load Balancer)

While MoE models are typically trained to have each experts receive similar amount of tokens; in practice, the distribution of tokens across experts can be very skewed.
vLLM provides a expert parallel load balancer (EPLB) to reshuffle the expert mapping across each EP ranks to even the load across experts.

You can enable it by setting the flag `--enable-eplb` flag. *Currently only DeepSeek V3 architecture is supported*. 
When this flag is turned on, vLLM will collect the load statistics with every forward pass and periodically rebalance them.

The following flags are available to configure the EPLB:
- `--eplb-window-size`: vLLM will keep track of the load statistics for the last `eplb-window-size` engine steps and use it for rebalancing decision.
- `--eplb-step-interval`: vLLM will rebalance the load every `eplb-step-interval` engine steps.
- `--eplb-log-balancedness`: This flag controls whether to log the balancedness of the load across experts. Balancedness is defined as the average tokens per expert divided by the maximum tokens for a given expert.
- `--num-redundant-experts`: This flag controls the number of global redundant experts to keep in the model. By default, each EP rank will only have `NUM_TOTAL_EXPERTS / NUM_EP_RANKS` experts. Setting this flag to a non-zero value will allow each EP rank to have `(NUM_TOTAL_EXPERTS + NUM_REDUNDANT_EXPERTS) / NUM_EP_RANKS` experts.

The following command will serve a `DeepSeek-V3-0324` with EPLB enabled on single node. For multi-node deployment, you can just add the flags to each node's `vllm serve` command.

```bash
VLLM_ALL2ALL_BACKEND=pplx VLLM_USE_DEEP_GEMM=1 vllm serve deepseek-ai/DeepSeek-V3-0324 \
    --tensor-parallel-size 1 \
    --data-parallel-size 8 \
    --enable-expert-parallel \
    --enable-eplb \
    --eplb-log-balancedness \
    --eplb-window-size 1000 \
    --eplb-step-interval 3000
```

## Enabling EP on Multi-Node with Disaggregated Serving

For production deployment where maintaining SLA for time-to-first-token latency and inter-token latency is critical, you can use disaggregated serving to scale out prefill and decode separately.

For prefill instance, you can use the `VLLM_ALL2ALL_BACKEND=deepep_high_throughput` mode to scale out prefill. For decode instance, you can use the `VLLM_ALL2ALL_BACKEND=deepep_low_latency` mode to scale out decode.

To connect the two instances, you need to configure vLLM to use a KV connector for KV cache transfer, for example, to use NIXL, install it with our script [here](https://github.com/vllm-project/vllm/blob/main/tools/install_nixl.sh) and add the following flag to both instances:

```bash
--kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both"}'
```

Once set up, you can perform client side orchestration using the following example script:

```python
from openai import OpenAI
import uuid

# 1: Set up your clients
openai_api_key = "EMPTY"
prefill_client = OpenAI(
    api_key=openai_api_key,
    base_url="http://IP_ADDRESS_OF_PREFILL_INSTANCE:8000/v1",
)
decode_client = OpenAI(
    api_key=openai_api_key,
    base_url="http://IP_ADDRESS_OF_DECODE_INSTANCE:8001/v1",
)
models = prefill_client.models.list()
model = models.data[0].id

# 2: Prefill
request_id = str(uuid.uuid4()) # maintain the same request_id for both prefill and decode
response = prefill_client.completions.create(
    model=model,
    # PD only works when the prompt is greater than vLLM's block size (16 tokens)
    prompt="A long prompt greater than 16 tokens",
    # Set this to force only prefill
    max_tokens=1,
    # This will be echoed back to pass to decode
    extra_body={"kv_transfer_params": {
        "do_remote_decode": True,
        "do_remote_prefill": False,
        "remote_engine_id": None,
        "remote_block_ids": None,
        "remote_host": None,
        "remote_port": None
    }},
    extra_headers={
        "X-Request-Id": request_id
    },
)
print("-" * 50)
print("Prefill results:")
print(response)


# 3: Decode
# Pass the kv_transfer_params from prefill to decode
decode_response = decode_client.completions.create(
    model=model,
    # This prompt is ignored when kv_transfer_params is present
    prompt="A robot may not injure a human being",
    extra_body={"kv_transfer_params": response.kv_transfer_params},
    extra_headers={
        "X-Request-Id": request_id
    },
)
print("-" * 50)
print("Decode results:")
print(decode_response)
```
