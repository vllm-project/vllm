An implementation of xPyD with dynamic scaling based on point-to-point communication, partly inspired by Dynamo.

# Detailed Design

## Overall Process
As shown in Figure 1, the overall process of this **PD disaggregation** solution is described through a request flow:  

1. The client sends an HTTP request to the Proxy/Router's `/v1/completions` interface.  
2. The Proxy/Router selects a **1P1D (1 Prefill instance + 1 Decode instance)** through either through round-robin or random selection, generates a `request_id` (rules to be introduced later), modifies the `max_tokens` in the HTTP request message to **1**, and then forwards the request to the **P instance**.  
3. Immediately afterward, the Proxy/Router forwards the **original HTTP request** to the **D instance**.  
4. The **P instance** performs **Prefill** and then **actively sends the generated KV cache** to the D instance (using **PUT_ASYNC** mode). The D instance's `zmq_addr` can be resolved through the `request_id`.  
5. The **D instance** has a **dedicated thread** for receiving the KV cache (to avoid blocking the main process). The received KV cache is saved into the **GPU memory buffer**, the size of which is determined by the vLLM startup parameter `kv_buffer_size`. When the GPU buffer is full, the KV cache is stored in the **local Tensor memory pool**.  
6. During the **Decode**, the D instance's main process retrieves the KV cache (transmitted by the P instance) from either the **GPU buffer** or the **memory pool**, thereby **skipping Prefill**.  
7. After completing **Decode**, the D instance returns the result to the **Proxy/Router**, which then forwards it to the **client**.

![image1](https://github.com/user-attachments/assets/fb01bde6-755b-49f7-ad45-48a94b1e10a7)

## Proxy/Router (Demo)

A simple HTTP service acts as the entry point for client requests and starts a background thread to listen for P/D instances reporting their HTTP IP and PORT, as well as ZMQ IP and PORT. It maintains a dictionary of `http_addr -> zmq_addr`. The `http_addr` is the IP:PORT for the vLLM instance's request, while the `zmq_addr` is the address for KV cache handshake and metadata reception.

The Proxy/Router is responsible for selecting 1P1D based on the characteristics of the client request, such as the prompt, and generating a corresponding `request_id`, for example:

```
cmpl-___prefill_addr_10.0.1.2:21001___decode_addr_10.0.1.3:22001_93923d63113b4b338973f24d19d4bf11-0
```

Currently, to quickly verify whether xPyD can work, a round-robin selection of 1P1D is used. In the future, it is planned to use a trie combined with the load status of instances to select appropriate P and D.

Each P/D instance periodically sends a heartbeat packet to the Proxy/Router (currently every 3 seconds) to register (i.e., report `http_addr -> zmq_addr`) and keep the connection alive. If an instance crashes and fails to send a ping for a certain period of time, the Proxy/Router will remove the timed-out instance (this feature has not yet been developed).

## KV Cache Transfer Methods

There are three methods for KVcache transfer: PUT, GET, and PUT_ASYNC. These methods can be specified using the `--kv-transfer-config` and `kv_connector_extra_config` parameters, specifically through the `send_type` field. Both PUT and PUT_ASYNC involve the P instance actively sending KVcache to the D instance. The difference is that PUT is a synchronous transfer method that blocks the main process, while PUT_ASYNC is an asynchronous transfer method. PUT_ASYNC uses a dedicated thread for sending KVcache, which means it does not block the main process. In contrast, the GET method involves the P instance saving the KVcache to the memory buffer after computing the prefill. The D instance then actively retrieves the computed KVcache from the P instance once it has allocated space for the KVcache.

Experimental results have shown that the performance of these methods, from highest to lowest, is as follows: PUT_ASYNC → GET → PUT.

## P2P Communication via ZMQ & NCCL

As long as the address of the counterpart is known, point-to-point KV cache transfer (using NCCL) can be performed, without being constrained by rank and world size. To support dynamic scaling (expansion and contraction) of instances with PD disaggregation. This means that adding or removing P/D instances does not require a full system restart.

Each P/D instance only needs to create a single `P2pNcclEngine` instance. This instance maintains a ZMQ Server, which runs a dedicated thread to listen on the `zmq_addr` address and receive control flow requests from other instances. These requests include requests to establish an NCCL connection and requests to send KVcache metadata (such as tensor shapes and data types). However, it does not actually transmit the KVcache data itself.

When a P instance and a D instance transmit KVcache for the first time, they need to establish a ZMQ connection and an NCCL group. For subsequent KVcache transmissions, this ZMQ connection and NCCL group are reused. The NCCL group consists of only two ranks, meaning the world size is equal to 2. This design is intended to support dynamic scaling, which means that adding or removing P/D instances does not require a full system restart. As long as the address of the counterpart is known, point-to-point KVcache transmission can be performed, without being restricted by rank or world size.

## NCCL Group Topology

Currently, only symmetric TP (Tensor Parallelism) methods are supported for KVcache transmission. Asymmetric TP and PP (Pipeline Parallelism) methods will be supported in the future. Figure 2 illustrates the 1P2D setup, where each instance has a TP (Tensor Parallelism) degree of 2. There are a total of 7 NCCL groups: three vLLM instances each have one NCCL group with TP=2. Additionally, the 0th GPU card of the P instance establishes an NCCL group with the 0th GPU card of each D instance. Similarly, the 1st GPU card of the P instance establishes an NCCL group with the 1st GPU card of each D instance.

![image2](https://github.com/user-attachments/assets/837e61d6-365e-4cbf-8640-6dd7ab295b36)

Each NCCL group occupies a certain amount of GPU memory buffer for communication, the size of which is primarily influenced by the `NCCL_MAX_NCHANNELS` environment variable. When `NCCL_MAX_NCHANNELS=16`, an NCCL group typically occupies 100MB, while when `NCCL_MAX_NCHANNELS=8`, it usually takes up 52MB. For large-scale xPyD configurations—such as DeepSeek's 96P144D—this implementation is currently not feasible. Moving forward, we are considering using RDMA for point-to-point communication and are also keeping an eye on UCCL.

## GPU Memory Buffer and Tensor Memory Pool

The trade-off in the size of the memory buffer is as follows: For P instances, the memory buffer is not required in PUT and PUT_ASYNC modes, but it is necessary in GET mode. For D instances, a memory buffer is needed in all three modes. The memory buffer for D instances should not be too large. Similarly, for P instances in GET mode, the memory buffer should also not be too large. The memory buffer of D instances is used to temporarily store KVcache sent by P instances. If it is too large, it will reduce the KVcache space available for normal inference by D instances, thereby decreasing the inference batch size and ultimately leading to a reduction in output throughput. The size of the memory buffer is configured by the parameter `kv_buffer_size`, measured in bytes, and is typically set to 5%～10% of the memory size.

If the `--max-num-seqs` parameter for P instances is set to a large value, due to the large batch size, P instances will generate a large amount of KVcache simultaneously. This may exceed the capacity of the memory buffer of D instances, resulting in KVcache loss. Once KVcache is lost, D instances need to recompute Prefill, which is equivalent to performing Prefill twice. Consequently, the time-to-first-token (TTFT) will significantly increase, leading to degraded performance.

To address the above issues, I have designed and developed a local Tensor memory pool for storing KVcache, inspired by the buddy system used in Linux memory modules. Since the memory is sufficiently large, typically in the TB range on servers, there is no need to consider prefix caching or using block-based designs to reuse memory, thereby saving space. When the memory buffer is insufficient, KVcache can be directly stored in the Tensor memory pool, and D instances can subsequently retrieve KVcache from it. The read and write speed is that of PCIe, with PCIe 4.0 having a speed of approximately 21 GB/s, which is usually faster than the Prefill speed. Otherwise, solutions like Mooncake and lmcache would not be necessary. The Tensor memory pool acts as a flood diversion area, typically unused except during sudden traffic surges. In the worst-case scenario, my solution performs no worse than the normal situation with a Cache store.

# Install vLLM

??? Commands

    ```shell
    # Enter the home directory or your working directory.
    cd /home

    # Download the installation package, and I will update the commit-id in time. You can directly copy the command.
    wget https://vllm-wheels.s3.us-west-2.amazonaws.com/9112b443a042d8d815880b8780633882ad32b183/vllm-1.0.0.dev-cp38-abi3-manylinux1_x86_64.whl

    # Download the code repository.
    git clone -b xpyd-v1 https://github.com/Abatom/vllm.git
    cd vllm

    # Set the installation package path.
    export VLLM_PRECOMPILED_WHEEL_LOCATION=/home/vllm-1.0.0.dev-cp38-abi3-manylinux1_x86_64.whl

    # installation
    pip install -e . -v
    ```

# Run xPyD

## Instructions
- The following examples are run on an A800 (80GB) device, using the Meta-Llama-3.1-8B-Instruct model.
- Pay attention to the setting of the `kv_buffer_size` (in bytes). The empirical value is 10% of the GPU memory size. This is related to the kvcache size. If it is too small, the GPU memory buffer for temporarily storing the received kvcache will overflow, causing the kvcache to be stored in the tensor memory pool, which increases latency. If it is too large, the kvcache available for inference will be reduced, leading to a smaller batch size and decreased throughput.
- For Prefill instances, when using non-GET mode, the `kv_buffer_size` can be set to 1, as Prefill currently does not need to receive kvcache. However, when using GET mode, a larger `kv_buffer_size` is required because it needs to store the kvcache sent to the D instance.
- You may need to modify the `kv_buffer_size` and `port` in the following commands (if there is a conflict).
- `PUT_ASYNC` offers the best performance and should be prioritized.
- The `--port` must be consistent with the `http_port` in the `--kv-transfer-config`.
- The `disagg_prefill_proxy_xpyd.py` script will use port 10001 (for receiving client requests) and port 30001 (for receiving service discovery from P and D instances).
- The node running the proxy must have `quart` installed.
- Supports multiple nodes; you just need to modify the `proxy_ip` and `proxy_port` in `--kv-transfer-config`.
- In the following examples, it is assumed that **the proxy's IP is 10.0.1.1**.

## Run 1P3D

### Proxy (e.g. 10.0.1.1)

```shell
cd {your vllm directory}/examples/online_serving/disagg_xpyd/
python3 disagg_prefill_proxy_xpyd.py &
```

### Prefill1 (e.g. 10.0.1.2 or 10.0.1.1)

??? Command

    ```shell
    VLLM_USE_V1=1 CUDA_VISIBLE_DEVICES=0 vllm serve {your model directory} \
        --host 0.0.0.0 \
        --port 20005 \
        --tensor-parallel-size 1 \
        --seed 1024 \
        --served-model-name base_model \
        --dtype float16 \
        --max-model-len 10000 \
        --max-num-batched-tokens 10000 \
        --max-num-seqs 256 \
        --trust-remote-code \
        --gpu-memory-utilization 0.9 \
        --disable-log-request \
        --kv-transfer-config \
        '{"kv_connector":"P2pNcclConnector","kv_role":"kv_producer","kv_buffer_size":"1e1","kv_port":"21001","kv_connector_extra_config":{"proxy_ip":"10.0.1.1","proxy_port":"30001","http_port":"20005","send_type":"PUT_ASYNC","nccl_num_channels":"16"}}' > /var/vllm.log 2>&1 &
    ```

### Decode1 (e.g. 10.0.1.3 or 10.0.1.1)

??? Command

    ```shell
    VLLM_USE_V1=1 CUDA_VISIBLE_DEVICES=1 vllm serve {your model directory} \
        --host 0.0.0.0 \
        --port 20009 \
        --tensor-parallel-size 1 \
        --seed 1024 \
        --served-model-name base_model \
        --dtype float16 \
        --max-model-len 10000 \
        --max-num-batched-tokens 10000 \
        --max-num-seqs 256 \
        --trust-remote-code \
        --gpu-memory-utilization 0.7 \
        --disable-log-request \
        --kv-transfer-config \
        '{"kv_connector":"P2pNcclConnector","kv_role":"kv_consumer","kv_buffer_size":"8e9","kv_port":"22001","kv_connector_extra_config":{"proxy_ip":"10.0.1.1","proxy_port":"30001","http_port":"20009","send_type":"PUT_ASYNC","nccl_num_channels":"16"}}' > /var/vllm.log 2>&1 &
    ```

### Decode2 (e.g. 10.0.1.4 or 10.0.1.1)

??? Command

    ```shell
    VLLM_USE_V1=1 CUDA_VISIBLE_DEVICES=2 vllm serve {your model directory} \
        --host 0.0.0.0 \
        --port 20003 \
        --tensor-parallel-size 1 \
        --seed 1024 \
        --served-model-name base_model \
        --dtype float16 \
        --max-model-len 10000 \
        --max-num-batched-tokens 10000 \
        --max-num-seqs 256 \
        --trust-remote-code \
        --gpu-memory-utilization 0.7 \
        --disable-log-request \
        --kv-transfer-config \
        '{"kv_connector":"P2pNcclConnector","kv_role":"kv_consumer","kv_buffer_size":"8e9","kv_port":"23001","kv_connector_extra_config":{"proxy_ip":"10.0.1.1","proxy_port":"30001","http_port":"20003","send_type":"PUT_ASYNC","nccl_num_channels":"16"}}' > /var/vllm.log 2>&1 &
    ```

### Decode3 (e.g. 10.0.1.5 or 10.0.1.1)

??? Command

    ```shell
    VLLM_USE_V1=1 CUDA_VISIBLE_DEVICES=3 vllm serve {your model directory} \
        --host 0.0.0.0 \
        --port 20008 \
        --tensor-parallel-size 1 \
        --seed 1024 \
        --served-model-name base_model \
        --dtype float16 \
        --max-model-len 10000 \
        --max-num-batched-tokens 10000 \
        --max-num-seqs 256 \
        --trust-remote-code \
        --gpu-memory-utilization 0.7 \
        --disable-log-request \
        --kv-transfer-config \
        '{"kv_connector":"P2pNcclConnector","kv_role":"kv_consumer","kv_buffer_size":"8e9","kv_port":"24001","kv_connector_extra_config":{"proxy_ip":"10.0.1.1","proxy_port":"30001","http_port":"20008","send_type":"PUT_ASYNC","nccl_num_channels":"16"}}' > /var/vllm.log 2>&1 &
    ```

## Run 3P1D

### Proxy (e.g. 10.0.1.1)

```shell
cd {your vllm directory}/examples/online_serving/disagg_xpyd/
python3 disagg_prefill_proxy_xpyd.py &
```

### Prefill1 (e.g. 10.0.1.2 or 10.0.1.1)

??? Command

    ```shell
    VLLM_USE_V1=1 CUDA_VISIBLE_DEVICES=0 vllm serve {your model directory} \
        --host 0.0.0.0 \
        --port 20005 \
        --tensor-parallel-size 1 \
        --seed 1024 \
        --served-model-name base_model \
        --dtype float16 \
        --max-model-len 10000 \
        --max-num-batched-tokens 10000 \
        --max-num-seqs 256 \
        --trust-remote-code \
        --gpu-memory-utilization 0.9 \
        --disable-log-request \
        --kv-transfer-config \
        '{"kv_connector":"P2pNcclConnector","kv_role":"kv_producer","kv_buffer_size":"1e1","kv_port":"21001","kv_connector_extra_config":{"proxy_ip":"10.0.1.1","proxy_port":"30001","http_port":"20005","send_type":"PUT_ASYNC","nccl_num_channels":"16"}}' > /var/vllm.log 2>&1 &
    ```

### Prefill2 (e.g. 10.0.1.3 or 10.0.1.1)

??? Command

    ```shell
    VLLM_USE_V1=1 CUDA_VISIBLE_DEVICES=1 vllm serve {your model directory} \
        --host 0.0.0.0 \
        --port 20009 \
        --tensor-parallel-size 1 \
        --seed 1024 \
        --served-model-name base_model \
        --dtype float16 \
        --max-model-len 10000 \
        --max-num-batched-tokens 10000 \
        --max-num-seqs 256 \
        --trust-remote-code \
        --gpu-memory-utilization 0.9 \
        --disable-log-request \
        --kv-transfer-config \
        '{"kv_connector":"P2pNcclConnector","kv_role":"kv_producer","kv_buffer_size":"1e1","kv_port":"22001","kv_connector_extra_config":{"proxy_ip":"10.0.1.1","proxy_port":"30001","http_port":"20009","send_type":"PUT_ASYNC","nccl_num_channels":"16"}}' > /var/vllm.log 2>&1 &
    ```

### Prefill3 (e.g. 10.0.1.4 or 10.0.1.1)

??? Command

    ```shell
    VLLM_USE_V1=1 CUDA_VISIBLE_DEVICES=2 vllm serve {your model directory} \
        --host 0.0.0.0 \
        --port 20003 \
        --tensor-parallel-size 1 \
        --seed 1024 \
        --served-model-name base_model \
        --dtype float16 \
        --max-model-len 10000 \
        --max-num-batched-tokens 10000 \
        --max-num-seqs 256 \
        --trust-remote-code \
        --gpu-memory-utilization 0.9 \
        --disable-log-request \
        --kv-transfer-config \
        '{"kv_connector":"P2pNcclConnector","kv_role":"kv_producer","kv_buffer_size":"1e1","kv_port":"23001","kv_connector_extra_config":{"proxy_ip":"10.0.1.1","proxy_port":"30001","http_port":"20003","send_type":"PUT_ASYNC","nccl_num_channels":"16"}}' > /var/vllm.log 2>&1 &
    ```

### Decode1 (e.g. 10.0.1.5 or 10.0.1.1)

??? Command

    ```shell
    VLLM_USE_V1=1 CUDA_VISIBLE_DEVICES=3 vllm serve {your model directory} \
        --host 0.0.0.0 \
        --port 20008 \
        --tensor-parallel-size 1 \
        --seed 1024 \
        --served-model-name base_model \
        --dtype float16 \
        --max-model-len 10000 \
        --max-num-batched-tokens 10000 \
        --max-num-seqs 256 \
        --trust-remote-code \
        --gpu-memory-utilization 0.7 \
        --disable-log-request \
        --kv-transfer-config \
        '{"kv_connector":"P2pNcclConnector","kv_role":"kv_consumer","kv_buffer_size":"8e9","kv_port":"24001","kv_connector_extra_config":{"proxy_ip":"10.0.1.1","proxy_port":"30001","http_port":"20008","send_type":"PUT_ASYNC","nccl_num_channels":"16"}}' > /var/vllm.log 2>&1 &
    ```

# Single request

```shell
curl -X POST -s http://10.0.1.1:10001/v1/completions \
-H "Content-Type: application/json" \
-d '{
    "model": "base_model",
    "prompt": "San Francisco is a",
    "max_tokens": 10,
    "temperature": 0
}'
```

# Benchmark

??? Command

    ```shell
    python3 benchmark_serving.py \
        --backend vllm \
        --model base_model \
        --tokenizer meta-llama/Llama-3.1-8B-Instruct \
        --dataset-name "random" \
        --host 10.0.1.1 \
        --port 10001 \
        --random-input-len 1024 \
        --random-output-len 1024 \
        --ignore-eos \
        --burstiness 100 \
        --percentile-metrics "ttft,tpot,itl,e2el" \
        --metric-percentiles "90,95,99" \
        --seed $(date +%s) \
        --trust-remote-code \
        --request-rate 3 \
        --num-prompts 1000
    ```

# Shut down

```shell
pgrep python | xargs kill -9 && pkill -f python
```

# Test data

## **Scenario 1**: 1K input & 1K output tokens, E2E P99 latency ~20s
- **1P5D (6×A800) vs vLLM (1×A800)**:
  - Throughput ↑7.2% (1085 → 6979/6)
  - ITL (P99) ↓81.3% (120ms → 22.9ms)
  - TTFT (P99) ↑26.8% (175ms → 222ms)
  - TPOT: No change

- **1P6D (7×A800) vs vLLM (1×A800)**:
  - Throughput ↑9.6% (1085 → 8329/7)
  - ITL (P99) ↓81.0% (120ms → 22.7ms)
  - TTFT (P99) ↑210% (175ms →543ms)
  - TPOT: No change

## **Scenario 2**: 1K input & 200 output tokens, E2E P99 latency ~4s
- **1P1D (2×A800) vs vLLM (1×A800)**:
  - Throughput ↑37.4% (537 → 1476/2)
  - ITL (P99) ↓81.8% (127ms → 23.1ms)
  - TTFT (P99) ↑41.8% (160ms → 227ms)
  - TPOT: No change

![testdata](https://github.com/user-attachments/assets/f791bfc7-9f3d-4e5c-9171-a42f9f4da627)
