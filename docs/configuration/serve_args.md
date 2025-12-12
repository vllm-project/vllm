# Server Arguments

The `vllm serve` command is used to launch the OpenAI-compatible server.

## CLI Arguments

The `vllm serve` command is used to launch the OpenAI-compatible server.
To see the available options, take a look at the [CLI Reference](../cli/README.md)!

## Configuration file

You can load CLI arguments via a [YAML](https://yaml.org/) config file.
The argument names must be the long form of those outlined [above](serve_args.md).

For example:

```yaml
# config.yaml

model: meta-llama/Llama-3.1-8B-Instruct
host: "127.0.0.1"
port: 6379
uvicorn-log-level: "info"
```

To use the above config file:

```bash
vllm serve --config config.yaml
```

!!! note
    In case an argument is supplied simultaneously using command line and the config file, the value from the command line will take precedence.
    The order of priorities is `command line > config file values > defaults`.
    e.g. `vllm serve SOME_MODEL --config config.yaml`, SOME_MODEL takes precedence over `model` in config file.

## Warmup Configuration

vLLM supports running warmup requests before the server reports healthy via the `/health` endpoint. This ensures consistent performance from the first production request by pre-warming JIT compilation, memory allocations, and other caches.

### Usage

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct --warmup-config warmup.json
```

### Configuration Format

The warmup configuration is a JSON file with the following structure:

```json
{
  "concurrency": 4,
  "requests": [
    {
      "endpoint": "/v1/chat/completions",
      "payload": {
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 32
      },
      "count": 5
    },
    {
      "endpoint": "/v1/completions",
      "payload": {
        "prompt": "The quick brown fox",
        "max_tokens": 16
      },
      "count": 3
    }
  ]
}
```

| Field | Required | Default | Description |
|-------|----------|---------|-------------|
| `concurrency` | No | 1 | Maximum number of parallel warmup requests |
| `requests` | Yes | - | List of warmup request configurations |
| `requests[].endpoint` | Yes | - | API endpoint path (e.g., `/v1/chat/completions`) |
| `requests[].payload` | Yes | - | Request body (model is auto-filled if not specified) |
| `requests[].count` | No | 1 | Number of times to repeat this request |

### Supported Endpoints

- `/v1/chat/completions` - Chat completions (LLMs)
- `/v1/completions` - Text completions (LLMs)
- `/v1/embeddings` - Embeddings (embedding models)
- `/pooling` - Pooling (pooling models)
- `/classify` - Classification (classification models)
- `/score`, `/v1/score` - Scoring (cross-encoder models)
- `/rerank`, `/v1/rerank`, `/v2/rerank` - Re-ranking (cross-encoder models)

### Health Endpoint Behavior

During warmup, the `/health` endpoint returns HTTP 503 (Service Unavailable). Once warmup completes, it returns HTTP 200. This integrates with Kubernetes readiness probes to ensure traffic is only routed to fully warmed instances.

!!! tip
    Start with a small number of warmup requests (e.g., 3-5) and adjust based on your latency requirements. The goal is to warm up critical code paths without significantly delaying server startup.
