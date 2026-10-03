# AI Badgr

vLLM can be run through [AI Badgr](https://aibadgr.com/), a Compute API
for AI workloads. AI Badgr can provision a capped vLLM endpoint and expose an
OpenAI-compatible server URL.

## Prerequisites

- An AI Badgr account with compute access
- A Badgr API key (`BADGR_API_KEY`) for CLI and API requests
- A model name or path to serve with vLLM

## Launching vLLM

With the [Badgr CLI](https://aibadgr.com/gpu/launch?template=vllm), serve a
model directly:

```bash
badgr serve meta-llama/Llama-3.1-8B-Instruct --max-cost 5
```

Or use the preconfigured vLLM template:

```bash
badgr serve template vllm --max-cost 5
```

To serve a different model through the template, pass `MODEL` as an environment
variable:

```bash
badgr serve template vllm \
    --max-cost 5 \
    --env MODEL=<model-name>
```

For gated models, such as Meta Llama, Google Gemma, or Mistral models, pass
`HF_TOKEN`:

```bash
badgr serve template vllm \
    --max-cost 5 \
    --env MODEL=meta-llama/Llama-3.1-8B-Instruct \
    --env HF_TOKEN=$HF_TOKEN
```

AI Badgr Compute can also run this as a capped `model.serve` job through
`POST https://aibadgr.com/v1/jobs`, with status, logs, teardown, billing, and
receipts.

The vLLM template defaults to `meta-llama/Llama-3.1-8B-Instruct` and starts the
OpenAI-compatible server on port 8000. The preconfigured template command is
equivalent to:

```bash
python -m vllm.entrypoints.openai.api_server \
    --model <model-name> \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 1
```

!!! note

    Use `--host 0.0.0.0` when customizing the startup command so the server is
    reachable through the endpoint URL shown in the CLI output. You can also use
    `vllm serve <model-name> --host 0.0.0.0 --port 8000` for custom images that
    prefer the newer vLLM CLI.

Stop a deployment with `badgr down <deployment-id>`.

## Troubleshooting

If the endpoint is not ready yet, common causes include:

- **Model still loading** — Large models take time to download and load into GPU
  memory. Check progress with `badgr logs <deployment-id>`.
- **Wrong host binding** — Ensure the server binds to `0.0.0.0`, not
  `127.0.0.1`.
- **Out of GPU memory** — The model may be too large for the allocated GPU.
  Try a larger GPU or increase `--tensor-parallel-size` for multi-GPU setups.

## Verifying the Deployment

After the deployment starts, use the endpoint URL from the CLI output to test
the OpenAI-compatible server. AI Badgr waits for vLLM to be ready on
`/v1/models`, so check that endpoint first:

!!! console "Command"

    ```bash
    curl https://<deployment-url>/v1/models
    ```

Then, send a chat completion request:

!!! console "Command"

    ```bash
    curl https://<deployment-url>/v1/chat/completions \
        -H "Content-Type: application/json" \
        -d '{
            "model": "<model-name>",
            "messages": [
                {"role": "user", "content": "Hello, how are you?"}
            ],
            "max_tokens": 50
        }'
    ```

!!! console "Response"

    ```json
    {
        "id": "chat-abc123",
        "object": "chat.completion",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "I'm doing well, thank you for asking! How can I help you today?"
                },
                "index": 0,
                "finish_reason": "stop"
            }
        ]
    }
    ```

You can also check the server health endpoint:

```bash
curl https://<deployment-url>/health
```
