# Security

## Inter-Node Communication

All communications between nodes in a multi-node vLLM deployment are **insecure by default** and must be protected by placing the nodes on an isolated network. This includes:

1. PyTorch Distributed communications
2. KV cache transfer communications
3. Tensor, Pipeline, and Data parallel communications

### Configuration Options for Inter-Node Communications

The following options control inter-node communications in vLLM:

#### 1. **Environment Variables:**

- `VLLM_HOST_IP`: Sets the IP address for vLLM processes to communicate on

#### 2. **KV Cache Transfer Configuration:**

- `--kv-ip`: The IP address for KV cache transfer communications (default: 127.0.0.1)
- `--kv-port`: The port for KV cache transfer communications (default: 14579)

#### 3. **Data Parallel Configuration:**

- `data_parallel_master_ip`: IP of the data parallel master (default: 127.0.0.1)
- `data_parallel_master_port`: Port of the data parallel master (default: 29500)

### Notes on PyTorch Distributed

vLLM uses PyTorch's distributed features for some inter-node communication. For
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

**Operational control endpoints:**
- `/pause` - Pause generation (causes denial of service)
- `/resume` - Resume generation
- `/scale_elastic_ep` - Trigger scaling operations
- `/reset_prefix_cache` - Reset prefix cache
- `/reset_mm_cache` - Reset multimodal cache
- `/sleep` - Put engine to sleep
- `/wake_up` - Wake engine from sleep
- `/collective_rpc` - Internal RPC endpoint

**Utility endpoints:**
- `/tokenize` - Tokenize text
- `/detokenize` - Detokenize tokens
- `/tokenizer_info` - Get tokenizer information
- `/health` - Health check
- `/ping` - SageMaker health check
- `/version` - Version information
- `/load` - Server load metrics

**Note:** The `/invocations` endpoint is particularly concerning as it provides unauthenticated access to the same inference capabilities as the protected `/v1` endpoints.

### Security Implications

An attacker who can reach the vLLM HTTP server can:

1. **Bypass authentication** by using non-`/v1` endpoints like `/invocations`, `/inference/v1/generate`, `/pooling`, `/classify`, `/score`, or `/rerank` to run arbitrary inference without credentials
2. **Cause denial of service** by calling `/pause` or `/scale_elastic_ep` without a token
3. **Access operational controls** to manipulate server state via `/reset_prefix_cache`, `/sleep`, and other management endpoints

### Recommended Security Practices

One effective approach is to deploy vLLM behind a reverse proxy (such as nginx, Envoy, or a Kubernetes Gateway) that:

- Explicitly allowlists only the endpoints you want to expose to end users
- Blocks all other endpoints, including the unauthenticated inference and operational control endpoints

## Reporting Security Vulnerabilities

If you believe you have found a security vulnerability in vLLM, please report it following the project's security policy. For more information on how to report security issues and the project's security policy, please see the [vLLM Security Policy](https://github.com/vllm-project/vllm/blob/main/SECURITY.md).
