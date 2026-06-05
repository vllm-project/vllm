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

## Security Recommendations

### 1. **Network Isolation:**

- Deploy vLLM nodes on a dedicated, isolated network
- Use network segmentation to prevent unauthorized access
- Implement appropriate firewall rules

### 2. **Configuration Best Practices:**

- Always set `VLLM_HOST_IP` to a specific IP address rather than using defaults
- Configure firewalls to only allow necessary ports between nodes

### 3. **Access Control:**

- Restrict physical and network access to the deployment environment
- Implement proper authentication and authorization for management interfaces
- Follow the principle of least privilege for all system components

### 4. **Restrict Domains Access for Media URLs:**

Restrict domains that vLLM can access for media URLs by setting
`--allowed-media-domains` to prevent Server-Side Request Forgery (SSRF) attacks.
(e.g. `--allowed-media-domains upload.wikimedia.org github.com www.bogotobogo.com`)

This protection applies to both the online serving API (multimodal inputs) and
the **batch runner** (`vllm run-batch`), where `file_url` values in batch
transcription/translation requests are validated against the same allowlist.

Without domain restrictions, a malicious user could supply URLs that:

- **Target internal services**: Access internal network endpoints, cloud metadata
  services (e.g. `169.254.169.254`), or other services not intended to be
  publicly reachable (SSRF).
- **Consume excessive resources**: Point to extremely large files or slow
  endpoints, causing the server to download unbounded amounts of data and
  exhausting memory, disk, or network bandwidth.

By explicitly allowlisting only the domains you expect media to come from, you
significantly reduce the attack surface for these types of abuse.

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
- `/v1/chat/completions/batch` - Batch chat completions
- `/v1/chat/completions/render` - Render chat completion requests
- `/v1/completions` - Text completions
- `/v1/completions/render` - Render completion requests
- `/v1/embeddings` - Generate embeddings
- `/v1/audio/transcriptions` - Audio transcription
- `/v1/audio/translations` - Audio translation
- `/v1/messages` - Anthropic-compatible messages API
- `/v1/messages/count_tokens` - Count tokens for Anthropic messages
- `/v1/responses` - Create a response
- `/v1/responses/{response_id}` - Retrieve a response
- `/v1/responses/{response_id}/cancel` - Cancel a response
- `/v1/score` - Scoring API
- `/v1/rerank` - Reranking API
- `/v1/load_lora_adapter` - Load a LoRA adapter (can alter model behavior; only available when `--enable-lora` is set and `VLLM_ALLOW_RUNTIME_LORA_UPDATING=True`)
- `/v1/unload_lora_adapter` - Unload a LoRA adapter (can alter model behavior; only available when `--enable-lora` is set and `VLLM_ALLOW_RUNTIME_LORA_UPDATING=True`)

### Unprotected Endpoints (No API Key Required)

The following endpoints **do not require authentication** even when `--api-key` is configured:

**Inference endpoints:**

- `/invocations` - SageMaker-compatible endpoint (routes to the same inference functions as `/v1` endpoints)
- `/inference/v1/generate` - Generate completions
- `/generative_scoring` - Generative scoring API
- `/pooling` - Pooling API
- `/classify` - Classification API
- `/score` - Scoring API (non-`/v1` variant)
- `/rerank` - Reranking API (non-`/v1` variant)

**Operational control endpoints (only when `"generate"` task is supported):**

- `/pause` - Pause generation (causes denial of service)
- `/resume` - Resume generation
- `/is_paused` - Check if generation is paused
- `/scale_elastic_ep` - Trigger scaling operations
- `/is_scaling_elastic_ep` - Check if scaling is in progress
- `/init_weight_transfer_engine` - Initialize weight transfer engine for RLHF
- `/update_weights` - Update model weights (can alter model behavior)
- `/get_world_size` - Get distributed world size
- `/abort_requests` - Abort in-flight requests (only when `--tokens-only` is also set)

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
- `/reset_encoder_cache` - Reset encoder cache (can disrupt service)
- `/sleep` - Put engine to sleep (causes denial of service)
- `/wake_up` - Wake engine from sleep
- `/is_sleeping` - Check if engine is sleeping
- `/collective_rpc` - Execute arbitrary RPC methods on the engine (extremely dangerous)

**Profiler endpoints (only when profiling is enabled via `--profiler-config`):**

These endpoints are only available when profiling is enabled and should only be used for local development:

- `/start_profile` - Start PyTorch profiler
- `/stop_profile` - Stop PyTorch profiler

**Note:** The `/invocations` endpoint is particularly concerning as it provides unauthenticated access to the same inference capabilities as the protected `/v1` endpoints.

### Security Implications

An attacker who can reach the vLLM HTTP server can:

1. **Bypass authentication** by using non-`/v1` endpoints like `/invocations`, `/inference/v1/generate`, `/generative_scoring`, `/pooling`, `/classify`, `/score`, or `/rerank` to run arbitrary inference without credentials
2. **Cause denial of service** by calling `/pause`, `/scale_elastic_ep`, or `/abort_requests` without a token
3. **Access operational controls** to manipulate server state (e.g., pausing generation, updating model weights via `/update_weights`)
4. **If `--enable-tokenizer-info-endpoint` is set:** Access sensitive tokenizer configuration including chat templates, which may reveal prompt engineering strategies or other implementation details
5. **If `VLLM_SERVER_DEV_MODE=1` is set:** Execute arbitrary RPC commands via `/collective_rpc`, reset caches, put the engine to sleep, and access detailed server configuration

### Recommended Security Practices

#### 1. Minimize Exposed Endpoints

**CRITICAL:** Never set `VLLM_SERVER_DEV_MODE=1` in production environments. Development endpoints expose extremely dangerous functionality including:

- Arbitrary RPC execution via `/collective_rpc`
- Cache manipulation that can disrupt service
- Detailed server configuration disclosure

Similarly, never enable profiler endpoints in production.

**Be cautious with `--enable-tokenizer-info-endpoint`:** Only enable the `/tokenizer_info` endpoint if you need to expose tokenizer configuration information. This endpoint reveals chat templates and tokenizer settings that may contain sensitive implementation details or prompt engineering strategies.

#### 2. Deploy Behind a Reverse Proxy

The most effective approach is to deploy vLLM behind a reverse proxy (such as nginx, Envoy, or a Kubernetes Gateway) that:

- Explicitly allowlists only the endpoints you want to expose to end users
- Blocks all other endpoints, including the unauthenticated inference and operational control endpoints
- Implements additional authentication, rate limiting, and logging at the proxy layer

## Request Parameter Resource Limits

Certain API request parameters can have a large impact on resource consumption and may be abused to exhaust server resources. The `n` parameter in the `/v1/completions` and `/v1/chat/completions` endpoints controls how many independent output sequences are generated per request. A very large value causes the engine to allocate memory, CPU, and GPU time proportional to `n`, which can lead to out-of-memory conditions on the host and block the server from processing other requests.

To mitigate this, vLLM enforces a configurable upper bound on the `n` parameter via the `VLLM_MAX_N_SEQUENCES` environment variable (default: **16384**). Requests exceeding this limit are rejected before reaching the engine.

### Recommendations

- **Public-facing deployments:** Consider setting `VLLM_MAX_N_SEQUENCES` to a value appropriate for your workload (e.g., `64` or `128`) to limit the blast radius of a single request.
- **Reverse proxy layer:** In addition to vLLM's built-in limit, consider enforcing request body validation and rate limiting at your reverse proxy to further constrain abusive payloads.
- **Monitoring:** Monitor per-request resource consumption to detect anomalous patterns that may indicate abuse.

## Tool Server and MCP Security

vLLM supports connecting to external tool servers via the `--tool-server` argument. This enables models to call tools through the Responses API (`/v1/responses`). Tool server support works with all models — it is not limited to specific model architectures.

**Important:** No tool servers are enabled by default. They must be explicitly opted into via configuration.

### Built-in Demo Tools (GPT-OSS)

Passing `--tool-server demo` enables built-in demo tools that work with any model that supports tool calling. The tool implementations are not part of vLLM — they are provided by the separately installed [`gpt-oss`](https://github.com/openai/gpt-oss) package. vLLM provides thin wrappers that delegate to `gpt-oss`.

- **Code interpreter** (`python`): Python execution via Docker (via `gpt_oss.tools.python_docker`)
- **Web browser** (`browser`): Search via Exa API, requires `EXA_API_KEY` (via `gpt_oss.tools.simple_browser`)

#### Code Interpreter (Python Tool) Security Risks

The code interpreter executes model-generated code inside a Docker container. However, the container is **not configured with network isolation by default**. It inherits the host's Docker networking configuration (e.g., default bridge network or `--network=host`), which means:

- The container may be able to access the host network and LAN.
- Internal services reachable from the container may be exploited via SSRF (Server-Side Request Forgery).
- Cloud metadata services (e.g., `169.254.169.254`) may be accessible.
- If vulnerable internal services (such as `torch.distributed` endpoints) are reachable from the container, this could be used to attack them.

This is particularly concerning because the code being executed is generated by the model, which may be influenced by adversarial inputs (prompt injection).

#### Controlling Built-in Tool Availability

Built-in demo tools are controlled by two settings:

1. **`--tool-server demo`**: Enables the built-in demo tools (browser and Python code interpreter).

2. **`VLLM_GPT_OSS_SYSTEM_TOOL_MCP_LABELS`**: When built-in tools are requested via the `mcp` tool type in the Responses API, this comma-separated allowlist controls which tool labels are permitted. Valid values are:
   - `container` - Container tool
   - `code_interpreter` - Python code execution tool
   - `web_search_preview` - Web search/browser tool

   If this variable is not set or is empty, no built-in tools requested via MCP tool type will be enabled.

To disable the Python code interpreter specifically, omit `code_interpreter` from `VLLM_GPT_OSS_SYSTEM_TOOL_MCP_LABELS`.

**Consider a custom implementation**: The GPT-OSS Python tool is a reference implementation. For production deployments, consider implementing a custom code execution sandbox with stricter isolation guarantees. See the [GPT-OSS documentation](https://github.com/openai/gpt-oss?tab=readme-ov-file#python) for guidance.

## Dynamic LoRA Loading

vLLM supports dynamically loading and unloading LoRA adapters at runtime via the `/v1/load_lora_adapter` and `/v1/unload_lora_adapter` API endpoints. This functionality is **not enabled by default** — it requires both `--enable-lora` and the environment variable `VLLM_ALLOW_RUNTIME_LORA_UPDATING=True` to be set.

**Warning:** Dynamic LoRA loading is not a secure operation and should not be enabled in deployments exposed to untrusted clients. If you must enable dynamic LoRA loading, restrict access to the `/v1/load_lora_adapter` and `/v1/unload_lora_adapter` endpoints to trusted administrators only, using a reverse proxy or network-level access controls. Do not expose these endpoints to end users. For details on configuring LoRA adapters, see the [LoRA Adapters documentation](../features/lora.md).

## Cache Directory Security

vLLM assumes that its cache directories are **private and trusted**. Cache contents are loaded without cryptographic integrity verification, including formats that support arbitrary code execution. If an untrusted user or process can write to vLLM's cache directories, they may be able to crash vLLM or cause it to execute arbitrary code.

**Do not share vLLM cache directories with untrusted users or mount them from untrusted storage.** Treat the cache directory with the same care as the vLLM installation itself.

### Cache Directory Configuration

Most cache paths default to subdirectories under a single root. Changing `VLLM_CACHE_ROOT` changes the default location for all features that inherit from it. When `torch.compile` caching is enabled (the default), vLLM also redirects `TRITON_CACHE_DIR` into this tree. If compile caching is disabled, Triton falls back to its own default location (`~/.triton/cache`).

| Environment Variable | Default | Description |
| --- | --- | --- |
| `VLLM_CACHE_ROOT` | `~/.cache/vllm` | Base cache directory. Respects `XDG_CACHE_HOME` if set. All paths below inherit from this unless explicitly overridden. |
| *(torch.compile)* | `$VLLM_CACHE_ROOT/torch_compile_cache/` | Compilation cache for AOT-compiled models, Inductor graphs, and Triton kernels. Controlled by `VLLM_DISABLE_COMPILE_CACHE` (set to `1` to disable). |
| `VLLM_ASSETS_CACHE` | `$VLLM_CACHE_ROOT/assets/` | Downloaded assets (e.g., tokenizer files). |
| `VLLM_XLA_CACHE_PATH` | `$VLLM_CACHE_ROOT/xla_cache/` | XLA/TPU compilation cache. |
| `VLLM_MEDIA_CACHE` | *(disabled)* | Optional cache for downloaded media (images, video, audio). Not enabled unless explicitly set. |

### Recommendations

- **Restrict file permissions** on `VLLM_CACHE_ROOT` (and any other cache directories used by dependencies, such as `~/.triton` if compile caching is disabled) so that only the vLLM process owner can read and write to them.
- **Do not copy cache contents from untrusted sources.** If you distribute cache artifacts between environments, ensure they originate from a trusted build pipeline.
- **Container deployments:** If mounting cache directories into containers, ensure the volume source is trusted.

## FIPS Compatibility

FIPS compliance depends on many factors, so a vLLM deployment is not automatically FIPS compliant. Recent changes have improved vLLM's *tolerance* of FIPS-enabled hosts — that is, avoiding crashes when non-approved algorithms are blocked — but tolerance is not the same as compliance. Whether a deployment satisfies FIPS requirements depends on the host operating system, the OpenSSL provider backing Python's `hashlib` and `ssl` modules, and which optional dependencies are installed.

### FIPS-relevant configuration

Operators running vLLM on FIPS-enabled hosts should select FIPS-approved algorithms via the following knobs:

- **Multimodal input hashing** — `VLLM_MM_HASHER_ALGORITHM` defaults to `blake3`, which is not FIPS-approved. Set it to `sha256` or `sha512` in FIPS-enabled environments.
- **Prefix-cache hashing** — set `--prefix-caching-hash-algo` (config field `prefix_caching_hash_algo`) to `sha256` or `sha256_cbor`. The `xxhash` and `xxhash_cbor` options are not FIPS-approved.
- **TLS ciphers** — use `--ssl-ciphers` to restrict the API server's TLS handshake to FIPS-approved cipher suites that match your environment's policy.

### Automatic fallback for non-security MD5 use

vLLM uses MD5 in a few places to derive non-security cache keys (for example, configuration hashes). These call sites pass `usedforsecurity=False` and additionally fall back to SHA-256 when the underlying OpenSSL provider refuses MD5 outright (see `safe_hash()` in `vllm/utils/hashing.py`). No user action is required; this behavior is documented so that auditors and security reviewers can identify the MD5 references and understand their purpose.

### Dependencies that provide non-FIPS hash implementations

Some dependencies expose hash implementations that are not FIPS-approved. vLLM only invokes them when the corresponding algorithm is selected, but operators with strict cryptographic controls may want to ensure the code paths are not exercised — and, where policy requires, that the packages themselves are absent:

- `blake3` — currently listed in `requirements/common.txt`, so a standard install pulls it in. It is imported lazily and only used when `VLLM_MM_HASHER_ALGORITHM=blake3` (the default). Setting `VLLM_MM_HASHER_ALGORITHM` to `sha256` or `sha512` is sufficient to keep the non-FIPS code path dormant. If your policy additionally forbids the package being present, uninstall it after `pip install` (`pip uninstall blake3`); vLLM will continue to function as long as `VLLM_MM_HASHER_ALGORITHM` is set to a non-blake3 value.
- `xxhash` — a true optional dependency (not in `requirements/common.txt`). It is only imported when an `xxhash`-based prefix-cache algorithm is selected. Leave it uninstalled and select a `sha256`-based prefix-cache algorithm.

### Beyond hashing: other FIPS considerations

Hashing is the area where vLLM has explicit FIPS-aware code, but a FIPS-compliant deployment depends on several factors that sit outside vLLM itself. Operators should evaluate the following with their platform and security teams:

- **Host crypto provider.** Python's `hashlib` and `ssl` modules are FIPS-aware only when Python is linked against a FIPS-validated OpenSSL (or equivalent) provider supplied by the host OS. vLLM inherits whatever provider the host configures — it does not bundle one.
- **API server TLS.** TLS termination for the OpenAI-compatible API server uses the host's OpenSSL via Python's `ssl` module. Restrict the cipher suite with `--ssl-ciphers` to match your environment's FIPS policy, and ensure server certificates are issued with FIPS-approved algorithms and key sizes.
- **Outbound HTTPS.** Model and asset downloads (for example, via `huggingface_hub`) use the same host TLS stack. The same provider/cipher considerations apply.
- **Inter-node communication is unencrypted by default.** As described in [Inter-Node Communication](#inter-node-communication), PyTorch Distributed, KV-cache transfer, and data-parallel channels do not encrypt traffic. FIPS environments that require FIPS-approved cryptography for data in transit must provide that protection externally — for example, via an mTLS sidecar or IPsec terminated by a FIPS-validated module — since vLLM's internal channels cannot satisfy the requirement on their own. Network isolation alone is not cryptography and does not meet a "FIPS-approved cryptography for data in transit" requirement, though it remains a useful defense-in-depth measure.
- **Dependencies that bundle their own OpenSSL.** Some Python wheels statically link OpenSSL builds that fail the kernel FIPS self-test on FIPS-enabled hosts (`FATAL FIPS SELFTEST FAILURE`). `opencv-python-headless` is a known example; other manylinux wheels may behave similarly. Audit your installed wheels for bundled crypto libraries when troubleshooting FIPS startup failures.
- **Accelerator and ML libraries.** PyTorch, CUDA, cuDNN, NCCL, and similar components have their own crypto and FIPS posture independent of vLLM. NVIDIA publishes FIPS-validated builds for some libraries; vLLM does not pin to those builds, so selecting and validating them is the operator's responsibility.
- **What is *not* a FIPS concern in vLLM.** Random number generation used for token sampling (Python/NumPy/PyTorch RNGs) is not a cryptographic use and is out of scope for FIPS. Pickled cache artifacts are a separate security concern covered under [Cache Directory Security](#cache-directory-security).

In short: the configuration knobs above let vLLM avoid non-approved algorithms, and the automatic fallbacks let it run without crashing on FIPS-enabled hosts. End-to-end FIPS compliance, however, is a property of the full deployment — host OS, crypto provider, transitive dependencies, and network architecture — not of vLLM alone.

## Reporting Security Vulnerabilities

If you believe you have found a security vulnerability in vLLM, please report it following the project's security policy. For more information on how to report security issues and the project's security policy, please see the [vLLM Security Policy](https://github.com/vllm-project/vllm/blob/main/SECURITY.md).
