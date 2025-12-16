# Security

## Inter-Node Communication

All communications between nodes in a multi-node vLLM deployment are **insecure by default** and must be protected by placing the nodes on an isolated network. This includes:

1. PyTorch Distributed communications
2. KV cache transfer communications
3. Tensor, Pipeline, and Data parallel communications

### Configuration Options for Inter-Node Communications

The following options control internode communications in vLLM:

#### 1. **Environment Variables:**

- `VLLM_HOST_IP`: Sets the IP address for vLLM processes to communicate on

#### 2. **KV Cache Transfer Configuration:**

- `--kv-ip`: The IP address for KV cache transfer communications (default: 127.0.0.1)
- `--kv-port`: The port for KV cache transfer communications (default: 14579)

#### 3. **Data Parallel Configuration:**

- `data_parallel_master_ip`: IP of the data parallel master (default: 127.0.0.1)
- `data_parallel_master_port`: Port of the data parallel master (default: 29500)

### Notes on PyTorch Distributed

vLLM uses PyTorch's distributed features for some internode communication. For
detailed information about PyTorch Distributed security considerations, please
refer to the [PyTorch Security
Guide](https://github.com/pytorch/pytorch/security/policy#using-distributed-features).

Key points from the PyTorch security guide:

- PyTorch Distributed features are intended for internal communication only
- They are not built for use in untrusted environments or networks
- No authorization protocol is included for performance reasons
- Messages are sent unencrypted
- Connections are accepted from anywhere without checks

### Security Recommendations

#### 1. **Network Isolation:**

- Deploy vLLM nodes on a dedicated, isolated network
- Use network segmentation to prevent unauthorized access
- Implement appropriate firewall rules

#### 2. **Configuration Best Practices:**

- Always set `VLLM_HOST_IP` to a specific IP address rather than using defaults
- Configure firewalls to only allow necessary ports between nodes

#### 3. **Access Control:**

- Restrict physical and network access to the deployment environment
- Implement proper authentication and authorization for management interfaces
- Follow the principle of least privilege for all system components

### 4. **Restrict Domains Access for Media URLs:**

Restrict domains that vLLM can access for media URLs by setting
`--allowed-media-domains` to prevent Server-Side Request Forgery (SSRF) attacks.
(e.g. `--allowed-media-domains upload.wikimedia.org github.com www.bogotobogo.com`)

Also, consider setting `VLLM_MEDIA_URL_ALLOW_REDIRECTS=0` to prevent HTTP
redirects from being followed to bypass domain restrictions.

## Security and Firewalls: Protecting Exposed vLLM Systems

While vLLM is designed to allow unsafe network services to be isolated to
private networks, there are components—such as dependencies and underlying
frameworks—that may open insecure services listening on all network interfaces,
sometimes outside of vLLM's direct control.

A major concern is the use of `torch.distributed`, which vLLM leverages for
distributed communication, including when using vLLM on a single host. When vLLM
uses TCP initialization (see [PyTorch TCP Initialization
documentation](https://docs.pytorch.org/docs/stable/distributed.html#tcp-initialization)),
PyTorch creates a `TCPStore` that, by default, listens on all network
interfaces. This means that unless additional protections are put in place,
these services may be accessible to any host that can reach your machine via any
network interface.

**From a PyTorch perspective, any use of `torch.distributed` should be
considered insecure by default.** This is a known and intentional behavior from
the PyTorch team.

### Firewall Configuration Guidance

The best way to protect your vLLM system is to carefully configure a firewall to
expose only the minimum network surface area necessary. In most cases, this
means:

- **Block all incoming connections except to the TCP port the API server is
listening on.**

- Ensure that ports used for internal communication (such as those for
`torch.distributed` and KV cache transfer) are only accessible from trusted
hosts or networks.

- Never expose these internal ports to the public internet or untrusted
networks.

Consult your operating system or application platform documentation for specific
firewall configuration instructions.

## API Key Authentication Limitations

### Overview

The `--api-key` flag (or `VLLM_API_KEY` environment variable) provides authentication for vLLM's HTTP server, but **only for OpenAI-compatible API endpoints under the `/v1` path prefix**. Many other sensitive endpoints are exposed on the same HTTP server without any authentication enforcement.

**Important:** Do not rely exclusively on `--api-key` for securing access to vLLM. Additional security measures are required for production deployments.

### Protected Endpoints (Require API Key)

When `--api-key` is configured, the following `/v1` endpoints require Bearer token authentication:

- `/v1/models` - List available models
- `/v1/chat/completions` - Chat completions
- `/v1/completions` - Text completions
- `/v1/embeddings` - Generate embeddings
- `/v1/audio/transcriptions` - Audio transcription
- `/v1/audio/translations` - Audio translation
- `/v1/messages` - Anthropic-compatible messages API
- `/v1/responses` - Response management
- `/v1/score` - Scoring API
- `/v1/rerank` - Reranking API

### Unprotected Endpoints (No API Key Required)

The following endpoints **do not require authentication** even when `--api-key` is configured:

**Inference endpoints:**

- `/invocations` - SageMaker-compatible endpoint (routes to the same inference functions as `/v1` endpoints)
- `/inference/v1/generate` - Generate completions
- `/pooling` - Pooling API
- `/classify` - Classification API
- `/score` - Scoring API (non-`/v1` variant)
- `/rerank` - Reranking API (non-`/v1` variant)

**Operational control endpoints (always enabled):**

- `/pause` - Pause generation (causes denial of service)
- `/resume` - Resume generation
- `/scale_elastic_ep` - Trigger scaling operations

**Utility endpoints:**

- `/tokenize` - Tokenize text
- `/detokenize` - Detokenize tokens
- `/health` - Health check
- `/ping` - SageMaker health check
- `/version` - Version information
- `/load` - Server load metrics

**Tokenizer information endpoint (only when `--enable-tokenizer-info-endpoint` is set):**

This endpoint is **only available when the `--enable-tokenizer-info-endpoint` flag is set**. It may expose sensitive information such as chat templates and tokenizer configuration:

- `/tokenizer_info` - Get comprehensive tokenizer information including chat templates and configuration

**Development endpoints (only when `VLLM_SERVER_DEV_MODE=1`):**

These endpoints are **only available when the environment variable `VLLM_SERVER_DEV_MODE` is set to `1`**. They are intended for development and debugging purposes and should never be enabled in production:

- `/server_info` - Get detailed server configuration
- `/reset_prefix_cache` - Reset prefix cache (can disrupt service)
- `/reset_mm_cache` - Reset multimodal cache (can disrupt service)
- `/sleep` - Put engine to sleep (causes denial of service)
- `/wake_up` - Wake engine from sleep
- `/is_sleeping` - Check if engine is sleeping
- `/collective_rpc` - Execute arbitrary RPC methods on the engine (extremely dangerous)

**Profiler endpoints (only when `VLLM_TORCH_PROFILER_DIR` or `VLLM_TORCH_CUDA_PROFILE` are set):**

These endpoints are only available when profiling is enabled and should only be used for local development:

- `/start_profile` - Start PyTorch profiler
- `/stop_profile` - Stop PyTorch profiler

**Note:** The `/invocations` endpoint is particularly concerning as it provides unauthenticated access to the same inference capabilities as the protected `/v1` endpoints.

### Security Implications

An attacker who can reach the vLLM HTTP server can:

1. **Bypass authentication** by using non-`/v1` endpoints like `/invocations`, `/inference/v1/generate`, `/pooling`, `/classify`, `/score`, or `/rerank` to run arbitrary inference without credentials
2. **Cause denial of service** by calling `/pause` or `/scale_elastic_ep` without a token
3. **Access operational controls** to manipulate server state (e.g., pausing generation)
4. **If `--enable-tokenizer-info-endpoint` is set:** Access sensitive tokenizer configuration including chat templates, which may reveal prompt engineering strategies or other implementation details
5. **If `VLLM_SERVER_DEV_MODE=1` is set:** Execute arbitrary RPC commands via `/collective_rpc`, reset caches, put the engine to sleep, and access detailed server configuration

### Recommended Security Practices

#### 1. Minimize Exposed Endpoints

**CRITICAL:** Never set `VLLM_SERVER_DEV_MODE=1` in production environments. Development endpoints expose extremely dangerous functionality including:

- Arbitrary RPC execution via `/collective_rpc`
- Cache manipulation that can disrupt service
- Detailed server configuration disclosure

Similarly, never enable profiler endpoints (`VLLM_TORCH_PROFILER_DIR` or `VLLM_TORCH_CUDA_PROFILE`) in production.

**Be cautious with `--enable-tokenizer-info-endpoint`:** Only enable the `/tokenizer_info` endpoint if you need to expose tokenizer configuration information. This endpoint reveals chat templates and tokenizer settings that may contain sensitive implementation details or prompt engineering strategies.

#### 2. Deploy Behind a Reverse Proxy

The most effective approach is to deploy vLLM behind a reverse proxy (such as nginx, Envoy, or a Kubernetes Gateway) that:

- Explicitly allowlists only the endpoints you want to expose to end users
- Blocks all other endpoints, including the unauthenticated inference and operational control endpoints
- Implements additional authentication, rate limiting, and logging at the proxy layer

## Reporting Security Vulnerabilities

If you believe you have found a security vulnerability in vLLM, please report it following the project's security policy. For more information on how to report security issues and the project's security policy, please see the [vLLM Security Policy](https://github.com/vllm-project/vllm/blob/main/SECURITY.md).
