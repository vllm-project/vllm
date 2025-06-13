# Install vLLM

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
- For Prefill instances, the `kv_buffer_size` can be set to 1, as Prefill currently does not need to receive kvcache.
- The `--port` must be consistent with the `http_port` in the `--kv-transfer-config`.
- `PUT_ASYNC` offers the best performance and should be prioritized.
- You may need to modify the `kv_buffer_size` and `port` in the following commands (if there is a conflict).
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
    --kv-transfer-config \
    '{"kv_connector":"P2pNcclConnector","kv_role":"kv_producer","kv_buffer_size":"1e1","kv_port":"21001","kv_connector_extra_config":{"proxy_ip":"10.0.1.1","proxy_port":"30001","http_port":"20005","send_type":"PUT_ASYNC"}}' > /var/vllm.log 2>&1 &
```
### Decode1 (e.g. 10.0.1.3 or 10.0.1.1)

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
    --kv-transfer-config \
    '{"kv_connector":"P2pNcclConnector","kv_role":"kv_consumer","kv_buffer_size":"8e9","kv_port":"22001","kv_connector_extra_config":{"proxy_ip":"10.0.1.1","proxy_port":"30001","http_port":"20009","send_type":"PUT_ASYNC"}}' > /var/vllm.log 2>&1 &
```
### Decode2 (e.g. 10.0.1.4 or 10.0.1.1)

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
    --kv-transfer-config \
    '{"kv_connector":"P2pNcclConnector","kv_role":"kv_consumer","kv_buffer_size":"8e9","kv_port":"23001","kv_connector_extra_config":{"proxy_ip":"10.0.1.1","proxy_port":"30001","http_port":"20003","send_type":"PUT_ASYNC"}}' > /var/vllm.log 2>&1 &
```
### Decode3 (e.g. 10.0.1.5 or 10.0.1.1)

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
    --kv-transfer-config \
    '{"kv_connector":"P2pNcclConnector","kv_role":"kv_consumer","kv_buffer_size":"8e9","kv_port":"24001","kv_connector_extra_config":{"proxy_ip":"10.0.1.1","proxy_port":"30001","http_port":"20008","send_type":"PUT_ASYNC"}}' > /var/vllm.log 2>&1 &
```

## Run 3P1D

### Proxy (e.g. 10.0.1.1)

```shell
cd {your vllm directory}/examples/online_serving/disagg_xpyd/
python3 disagg_prefill_proxy_xpyd.py &
```
### Prefill1 (e.g. 10.0.1.2 or 10.0.1.1)

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
    --kv-transfer-config \
    '{"kv_connector":"P2pNcclConnector","kv_role":"kv_producer","kv_buffer_size":"1e1","kv_port":"21001","kv_connector_extra_config":{"proxy_ip":"10.0.1.1","proxy_port":"30001","http_port":"20005","send_type":"PUT_ASYNC"}}' > /var/vllm.log 2>&1 &
```

### Prefill2 (e.g. 10.0.1.3 or 10.0.1.1)

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
    --kv-transfer-config \
    '{"kv_connector":"P2pNcclConnector","kv_role":"kv_producer","kv_buffer_size":"1e1","kv_port":"22001","kv_connector_extra_config":{"proxy_ip":"10.0.1.1","proxy_port":"30001","http_port":"20009","send_type":"PUT_ASYNC"}}' > /var/vllm.log 2>&1 &
```

### Prefill3 (e.g. 10.0.1.4 or 10.0.1.1)

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
    --kv-transfer-config \
    '{"kv_connector":"P2pNcclConnector","kv_role":"kv_producer","kv_buffer_size":"1e1","kv_port":"23001","kv_connector_extra_config":{"proxy_ip":"10.0.1.1","proxy_port":"30001","http_port":"20003","send_type":"PUT_ASYNC"}}' > /var/vllm.log 2>&1 &
```

### Decode1 (e.g. 10.0.1.5 or 10.0.1.1)

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
    --kv-transfer-config \
    '{"kv_connector":"P2pNcclConnector","kv_role":"kv_consumer","kv_buffer_size":"8e9","kv_port":"24001","kv_connector_extra_config":{"proxy_ip":"10.0.1.1","proxy_port":"30001","http_port":"20008","send_type":"PUT_ASYNC"}}' > /var/vllm.log 2>&1 &
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

```shell
python3 benchmark_serving.py \
    --backend vllm \
    --model base_model \
    --tokenizer auto \
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
