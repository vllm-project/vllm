# AI Badgr

vLLM can be run through [AI Badgr](https://aibadgr.com/), a Compute API
for AI workloads. AI Badgr provides a preconfigured vLLM launch template for
starting an OpenAI-compatible vLLM server as a capped compute job.

## Prerequisites

- An AI Badgr account with access to compute jobs
- A model name or path to serve with vLLM
- A Badgr API key for authenticated endpoint requests
- The [AI Badgr vLLM launch template](https://aibadgr.com/gpu/launch?template=vllm)

## Launching vLLM

AI Badgr Compute can also run this as a capped compute job through
`POST https://aibadgr.com/v1/jobs`, with status, logs, outputs, teardown,
billing, and receipts. CLI users can launch the template from the AI Badgr
console or with the Badgr CLI:

```bash
badgr serve template vllm --max-cost 5
```

To serve a different model, pass it as an environment variable:

```bash
badgr serve template vllm \
    --max-cost 5 \
    --env MODEL=<model-name>
```

For gated models, such as Meta Llama, Google Gemma, or Mistral models, provide
a Hugging Face token through `HF_TOKEN` or `BADGR_HF_TOKEN`:

```bash
badgr serve template vllm \
    --max-cost 5 \
    --env MODEL=meta-llama/Llama-3.1-8B-Instruct \
    --env HF_TOKEN=$HF_TOKEN
```

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
    reachable through the endpoint URL exposed by AI Badgr. You can also use
    `vllm serve <model-name> --host 0.0.0.0 --port 8000` for custom images that
    prefer the newer vLLM CLI.

## Verifying the Deployment

After the deployment starts, use the endpoint URL shown in the CLI output or AI
Badgr console to test the OpenAI-compatible server. AI Badgr waits for vLLM to
be ready on `/v1/models`, so check that endpoint first:

!!! console "Command"

    ```bash
    curl https://<deployment-url>/v1/models \
        -H "Authorization: Bearer $BADGR_API_KEY"
    ```

Then, send a chat completion request:

!!! console "Command"

    ```bash
    curl https://<deployment-url>/v1/chat/completions \
        -H "Content-Type: application/json" \
        -H "Authorization: Bearer $BADGR_API_KEY" \
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
