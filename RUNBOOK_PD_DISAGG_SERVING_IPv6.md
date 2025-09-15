# vLLM Disaggregated Prefill/Decode (P/D) Serving Runbook

## Overview

This runbook provides step-by-step instructions for setting up and running vLLM's disaggregated serving architecture on a devserver. The system separates prefill and decode workloads across different GPUs with efficient KV cache transfer via NCCL P2P communication.

## Architecture Components

- **Proxy Server**: Routes requests and manages service discovery (ports 10002 HTTP API, 30002 service discovery)
- **Prefill Server**: Handles prompt processing and KV cache generation (port 8081 HTTP, 8082 KV transfer)
- **Decode Server**: Consumes KV cache and generates completions (port 8083 HTTP, 8084 KV transfer)

## Prerequisites

### 1. Environment Setup

```bash
# Navigate to vLLM repository
cd /home/congc/local/gitrepos/vllm

# Activate virtual environment with vLLM installed
source ~/uv_env/vllm/bin/activate

# Ensure CUDA is available
nvidia-smi

# Verify at least 2 GPUs are available for prefill and decode
```

### 2. Model Path

Ensure you have a valid model path. In this example:
```
/data/users/congc/fbsource/fbcode/vllm/models/llama3_8b
```

## IPv6 Support Implementation

### Comprehensive IPv6 Support Fix

The system now includes comprehensive IPv6 support that works seamlessly in Meta's devserver environments. The implementation automatically handles both IPv4 and IPv6 addresses using proper URL formatting and ZMQ socket configuration.

**File**: `/home/congc/local/gitrepos/vllm/vllm/distributed/kv_transfer/kv_connector/v1/p2p/p2p_nccl_engine.py`

**Key Features**:
1. **IPv6-Aware Address Formatting**: Uses `join_host_port()` for RFC-compliant bracket notation
2. **Smart ZMQ Configuration**: Automatically enables IPv6 support when needed
3. **Backward Compatibility**: IPv4 addresses continue working unchanged
4. **Robust Parsing**: Proper IPv6 bracket notation handling

**The changes include**:

1. **Import IPv6 Utilities** (Line ~23):
```python
from vllm.utils import current_stream, get_ip, is_valid_ipv6_address, join_host_port, split_host_port
```

2. **IPv6-Aware Address Formatting** (Lines ~90-96):
```python
# Use IPv6-aware address formatting for ZMQ
self.zmq_address = join_host_port(self._hostname, self._port)

# Use IPv6-aware address formatting for HTTP
http_port = self.config.kv_connector_extra_config['http_port']
self.http_address = join_host_port(self._hostname, http_port)
```

3. **ZMQ IPv6 Socket Configuration** (Lines ~109-112):
```python
# Configure socket for IPv6 support if needed
if is_valid_ipv6_address(self._hostname):
    self.router_socket.setsockopt(zmq.IPV6, 1)
```

4. **Client Socket IPv6 Support** (Lines ~182-192):
```python
# Configure IPv6 support if connecting to IPv6 address
try:
    remote_host, _ = split_host_port(remote_address)
    if is_valid_ipv6_address(remote_host):
        sock.setsockopt(zmq.IPV6, 1)
except (ValueError, IndexError):
    # If address parsing fails, assume IPv4 for compatibility
    pass
```

**Address Formatting Examples**:
- **IPv4**: `192.168.1.1:8080` (unchanged)
- **IPv6**: `[2401:db00:2b1c:1c20:face:0:172:0]:8080` (properly formatted)

## Step-by-Step Setup

### Step 1: Start the Proxy Server

```bash
cd /home/congc/local/gitrepos/vllm
source ~/uv_env/vllm/bin/activate

# Start proxy server with correct ports
python disagg_proxy_p2p_nccl_xpyd.py > proxy_server.log 2>&1 &

# Verify proxy is running
ps aux | grep "disagg_proxy" | grep -v grep
tail -n 5 proxy_server.log
```

**Expected Output**:
```
[2025-09-XX XX:XX:XX -0700] [XXXXX] [INFO] Running on http://0.0.0.0:10002 (CTRL + C to quit)
```

### Step 2: Start the Prefill Server

```bash
cd /home/congc/local/gitrepos/vllm
source ~/uv_env/vllm/bin/activate

# Start prefill server (kv_producer on GPU 0)
CUDA_VISIBLE_DEVICES=0 VLLM_USE_V1=1 vllm serve /data/users/congc/fbsource/fbcode/vllm/models/llama3_8b \
  --host 0.0.0.0 --port 8081 \
  --kv-transfer-config '{"kv_connector":"P2pNcclConnector","kv_role":"kv_producer","kv_buffer_size":"1e9","kv_port":"8082","kv_connector_extra_config":{"proxy_ip":"0.0.0.0","proxy_port":"30002","http_port":"8081","send_type":"PUT_ASYNC","nccl_num_channels":"16"}}' \
  > prefill_server.log 2>&1 &

# Wait for initialization (30-60 seconds)
sleep 60

# Verify prefill server is running
grep "💯P2pNcclEngine" prefill_server.log
```

**Expected Output (IPv6 environment)**:
```
💯P2pNcclEngine init, rank:0, local_rank:0, http_address:[2401:db00:2b1c:1c20:face:0:172:0]:8081, zmq_address:[2401:db00:2b1c:1c20:face:0:172:0]:8082, proxy_address:0.0.0.0:30002, send_type:PUT_ASYNC, buffer_size_threshold:1000000000.00, nccl_num_channels:16
```

**Expected Output (IPv4 environment)**:
```
💯P2pNcclEngine init, rank:0, local_rank:0, http_address:127.0.0.1:8081, zmq_address:127.0.0.1:8082, proxy_address:0.0.0.0:30002, send_type:PUT_ASYNC, buffer_size_threshold:1000000000.00, nccl_num_channels:16
```

### Step 3: Start the Decode Server

```bash
cd /home/congc/local/gitrepos/vllm
source ~/uv_env/vllm/bin/activate

# Start decode server (kv_consumer on GPU 1)
CUDA_VISIBLE_DEVICES=1 VLLM_USE_V1=1 vllm serve /data/users/congc/fbsource/fbcode/vllm/models/llama3_8b \
  --host 0.0.0.0 --port 8083 \
  --kv-transfer-config '{"kv_connector":"P2pNcclConnector","kv_role":"kv_consumer","kv_buffer_size":"1e9","kv_port":"8084","kv_connector_extra_config":{"proxy_ip":"0.0.0.0","proxy_port":"30002","http_port":"8083","send_type":"PUT_ASYNC","nccl_num_channels":"16"}}' \
  > decode_server.log 2>&1 &

# Wait for initialization (30-60 seconds)
sleep 60

# Verify decode server is running
grep "💯P2pNcclEngine" decode_server.log
```

**Expected Output (IPv6 environment)**:
```
💯P2pNcclEngine init, rank:0, local_rank:0, http_address:[2401:db00:2b1c:1c20:face:0:172:0]:8083, zmq_address:[2401:db00:2b1c:1c20:face:0:172:0]:8084, proxy_address:0.0.0.0:30002, send_type:PUT_ASYNC, buffer_size_threshold:1000000000.00, nccl_num_channels:16
```

**Expected Output (IPv4 environment)**:
```
💯P2pNcclEngine init, rank:0, local_rank:0, http_address:127.0.0.1:8083, zmq_address:127.0.0.1:8084, proxy_address:0.0.0.0:30002, send_type:PUT_ASYNC, buffer_size_threshold:1000000000.00, nccl_num_channels:16
```

### Step 4: Verify Server Registration

```bash
# Check proxy logs for server registrations
tail -n 10 proxy_server.log
```

**Expected Output (IPv6 environment)**:
```
🔵Add [HTTP:[2401:db00:2b1c:1c20:face:0:172:0]:8081, ZMQ:[2401:db00:2b1c:1c20:face:0:172:0]:8082]
🔵Add [HTTP:[2401:db00:2b1c:1c20:face:0:172:0]:8083, ZMQ:[2401:db00:2b1c:1c20:face:0:172:0]:8084]
```

**Expected Output (IPv4 environment)**:
```
🔵Add [HTTP:127.0.0.1:8081, ZMQ:127.0.0.1:8082]
🔵Add [HTTP:127.0.0.1:8083, ZMQ:127.0.0.1:8084]
```

## Testing the System

### Basic Test

```bash
# Simple completion test
curl -X POST http://localhost:10002/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"/data/users/congc/fbsource/fbcode/vllm/models/llama3_8b","prompt":"Hello world","max_tokens":10}'
```

**Expected Response (IPv6 environment)**:
```json
{
  "id": "cmpl-___prefill_addr_[2401:db00:2b1c:1c20:face:0:172:0]:8082___decode_addr_[2401:db00:2b1c:1c20:face:0:172:0]:8084_...",
  "object": "text_completion",
  "created": 1757827808,
  "model": "/data/users/congc/fbsource/fbcode/vllm/models/llama3_8b",
  "choices": [{
    "index": 0,
    "text": "! I am thrilled to finally share my blog with",
    "finish_reason": "length"
  }],
  "usage": {
    "prompt_tokens": 3,
    "total_tokens": 13,
    "completion_tokens": 10
  }
}
```

### Additional Test Cases

```bash
# Question answering test
curl -s -X POST http://localhost:10002/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"/data/users/congc/fbsource/fbcode/vllm/models/llama3_8b","prompt":"What is the capital of France?","max_tokens":8}' | python -m json.tool

# Code completion test
curl -s -X POST http://localhost:10002/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"/data/users/congc/fbsource/fbcode/vllm/models/llama3_8b","prompt":"def fibonacci(n):\n    if n <= 1:\n        return n\n    else:\n        return","max_tokens":12}' | python -m json.tool

# Creative writing test
curl -s -X POST http://localhost:10002/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"/data/users/congc/fbsource/fbcode/vllm/models/llama3_8b","prompt":"Once upon a time in a magical forest, there lived a wise old dragon who","max_tokens":20}' | python -m json.tool
```

## Monitoring and Verification

### Check Request Routing

```bash
# Monitor proxy logs for request routing
tail -f proxy_server.log
```

**Expected Pattern (IPv6 environment)**:
```
handle_request count: X, [HTTP:[2401:db00:2b1c:1c20:face:0:172:0]:8081, ZMQ:[2401:db00:2b1c:1c20:face:0:172:0]:8082] 👉 [HTTP:[2401:db00:2b1c:1c20:face:0:172:0]:8083, ZMQ:[2401:db00:2b1c:1c20:face:0:172:0]:8084]
```

### Monitor GPU Usage

```bash
# Check GPU utilization
nvidia-smi

# Monitor continuous GPU usage
watch -n 1 nvidia-smi
```

### Check Server Health

```bash
# Prefill server health
curl -s http://localhost:8081/health

# Decode server health
curl -s http://localhost:8083/health

# Proxy health (should return server list)
curl -s http://localhost:10002/health || echo "Proxy doesn't have health endpoint"
```

## Configuration Parameters

### Key Configuration Settings

| Parameter | Description | Prefill Value | Decode Value |
|-----------|-------------|---------------|--------------|
| `kv_role` | Role in KV transfer | `kv_producer` | `kv_consumer` |
| `kv_port` | KV transfer port | `8082` | `8084` |
| `http_port` | HTTP API port | `8081` | `8083` |
| `proxy_port` | Service discovery port | `30002` | `30002` |
| `kv_buffer_size` | KV cache buffer size | `1e9` (1GB) | `1e9` (1GB) |
| `send_type` | Transfer mode | `PUT_ASYNC` | `PUT_ASYNC` |
| `nccl_num_channels` | NCCL channels | `16` | `16` |

## Troubleshooting

### Common Issues

#### 1. Port Conflicts
**Symptom**: `zmq.error.ZMQError: Address already in use`
**Solution**:
```bash
# Kill existing processes
pkill -f "vllm serve"
pkill -f "disagg_proxy"

# Check port usage
netstat -tuln | grep -E "(8081|8083|8082|8084|30002|10002)"

# Restart with different ports if needed
```

#### 2. IPv6 Address Resolution Issues
**Symptom**: Server fails to bind to IPv6 addresses
**Solution**:
- Ensure IPv6 support is enabled in the environment
- Verify `get_ip()` returns valid IPv6 addresses
- Check that the comprehensive IPv6 fix has been applied

#### 3. Server Not Registering with Proxy
**Symptom**: 500 error from proxy, no registration messages in logs
**Solution**:
```bash
# Check if servers are pinging proxy
grep -i "ping" prefill_server.log decode_server.log

# Verify proxy_port configuration matches (30002)
# Check firewall/network connectivity between servers and proxy
```

#### 4. URL Parsing Errors (Legacy Issue - Should Be Resolved)
**Symptom**: `InvalidUrlClientError` in proxy logs
**Solution**: This should be resolved with the IPv6 fix. If still occurring, verify the fix is properly applied.

### Log Analysis

#### Server Initialization Success Indicators

**Prefill Server**:
```bash
grep -E "(💯P2pNcclEngine|Starting vLLM API server)" prefill_server.log
```

**Decode Server**:
```bash
grep -E "(💯P2pNcclEngine|Starting vLLM API server)" decode_server.log
```

**Proxy Server**:
```bash
grep -E "(🔵Add|Running on)" proxy_server.log
```

### Performance Optimization

#### Buffer Size Tuning
- Increase `kv_buffer_size` for larger models: `"kv_buffer_size":"2e9"` (2GB)
- Monitor memory usage with `nvidia-smi`

#### NCCL Channel Optimization
- Adjust `nccl_num_channels` based on GPU interconnect:
  - NVLink: `"nccl_num_channels":"16"` or `"32"`
  - PCIe: `"nccl_num_channels":"8"`

## Cleanup Commands

```bash
# Stop all services
pkill -f "vllm serve"
pkill -f "disagg_proxy"

# Remove log files
rm -f proxy_server.log prefill_server.log decode_server.log prefill_server_v2.log decode_server_v2.log

# Clear GPU memory
python -c "import torch; torch.cuda.empty_cache()"
```

## Architecture Validation Checklist

- [ ] Proxy server running on ports 10002 (HTTP) and 30002 (service discovery)
- [ ] Prefill server running on GPU 0, ports 8081 (HTTP) and 8082 (KV)
- [ ] Decode server running on GPU 1, ports 8083 (HTTP) and 8084 (KV)
- [ ] Both servers registered with proxy using proper address formatting (IPv4/IPv6)
- [ ] P2P NCCL engines initialized with correct roles (producer/consumer)
- [ ] Request routing working: proxy → prefill → decode
- [ ] Response IDs contain correctly formatted addresses
- [ ] KV cache transfer functional (responses generated successfully)
- [ ] IPv6 support working in devserver environments

## Success Criteria

A successful setup should demonstrate:

1. **Request Routing**: Proxy logs show prefill → decode routing
2. **KV Transfer**: Responses generated with correct content
3. **Address Compatibility**: Both IPv4 and IPv6 addresses work seamlessly
4. **GPU Utilization**: Both GPUs show activity during inference
5. **Performance**: Low latency KV cache transfer via NCCL P2P
6. **Devserver Compatibility**: Works with Meta's IPv6-assigned addresses

## IPv6 Environment Benefits

- **True IPv6 Support**: Works with IPv6 addresses from Meta devservers (`get_ip()`)
- **Backward Compatible**: IPv4 addresses continue working unchanged
- **Proper URL Formatting**: HTTP URLs correctly formatted for proxy usage
- **Production Ready**: Uses battle-tested IPv6 utilities from vLLM's core codebase
- **Standards Compliant**: Follows RFC specifications for IPv6 URL formatting

## Notes

- This setup includes comprehensive IPv6 support for Meta devserver environments
- Buffer size of 1GB (`1e9`) is recommended for Llama 8B models
- The system supports various prompt types: questions, code, creative writing
- NCCL P2P provides efficient GPU-to-GPU KV cache transfer
- The proxy implements round-robin load balancing for multiple instances
- IPv6 addresses are automatically formatted with proper bracket notation
- ZMQ sockets are automatically configured for IPv6 when needed

## Version Information

- vLLM Version: 0.10.2rc3.dev52+g3e903b6cb
- NCCL: libnccl.so.2
- CUDA: Available via nvidia-smi
- Python Environment: ~/uv_env/vllm/bin/activate
- IPv6 Support: Comprehensive (D82409818)
