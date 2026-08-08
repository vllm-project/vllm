# JarvisLabs

vLLM can be deployed on [JarvisLabs.ai](https://jarvislabs.ai/), a GPU cloud platform for AI workloads.

## Prerequisites

- A JarvisLabs account
- A GPU instance with enough VRAM for the model you want to serve
- Optional: the [JarvisLabs CLI](https://docs.jarvislabs.ai/cli/), configured with `jl setup`

You can check current GPU availability in the dashboard or with:

```bash
jl gpus
```

## Launching a GPU Instance

Launch a GPU container or VM from the JarvisLabs dashboard. Containers are the simplest path for a public API endpoint because port `6006` is exposed by default.

You can do the same with the CLI:

```bash
jl create --gpu H100 --template pytorch --storage 100 --name vllm-server
```

If you prefer a custom port, expose it when creating the container:

```bash
jl create --gpu H100 --template pytorch --storage 100 --http-ports 8000 --name vllm-server
```

If you want full VM access instead of a container, select an available GPU VM from the dashboard or check availability with `jl gpus`, then run:

```bash
jl create --gpu <gpu-type> --vm --storage 100 --name vllm-vm
```

SSH into the instance:

```bash
jl ssh <machine-id>
```

## Starting vLLM

Create a Python environment and install vLLM:

```bash
uv venv --python 3.12 --seed --managed-python
source .venv/bin/activate
uv pip install vllm --torch-backend=auto
```

Start the OpenAI-compatible server on port `6006`:

```bash
vllm serve google/gemma-4-26B-A4B-it \
    --host 0.0.0.0 \
    --port 6006 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.90
```

!!! note
    Use `--host 0.0.0.0` so the server is reachable through the JarvisLabs API endpoint or VM public IP.

## Accessing the Endpoint

For containers, JarvisLabs maps port `6006` to a public API endpoint. In the dashboard, open the instance details and copy the API endpoint.

You can also retrieve the endpoint with the CLI:

```bash
jl get <machine-id>
```

Use the copied endpoint as the base URL.

For VMs, use the public IP from the instance details or `jl get <machine-id>`. VMs use public IP access directly and do not use `--http-ports`; use `http://<public-ip>:6006` as the base URL.

## Verifying the Deployment

Once the server is ready, test the health endpoint:

```bash
curl https://<jarvislabs-api-endpoint>/health
```

Then send a chat completion request:

```bash
curl https://<jarvislabs-api-endpoint>/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "google/gemma-4-26B-A4B-it",
        "messages": [
            {"role": "user", "content": "Say hello."}
        ],
        "max_tokens": 40,
        "temperature": 0
    }'
```

```json
{
    "id": "chatcmpl-a378e0dee141fe7a",
    "object": "chat.completion",
    "model": "google/gemma-4-26B-A4B-it",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "Hello! How can I help you today?"
            },
            "finish_reason": "stop"
        }
    ],
    "usage": {
        "prompt_tokens": 16,
        "completion_tokens": 10,
        "total_tokens": 26
    }
}
```

## Troubleshooting

- **Endpoint is not responding**: The model may still be downloading, compiling, or loading. Check the vLLM logs and wait for vLLM to finish startup.
- **Connection fails**: Make sure vLLM was started with `--host 0.0.0.0` and port `6006`.
- **Port mismatch**: If you use a custom port, verify that the vLLM `--port` value matches the HTTP port exposed in JarvisLabs.
- **Out of memory**: Use a smaller model, select a GPU with more VRAM, or reduce the served context length.
