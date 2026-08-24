# Queue Architecture Local Development Guide

This directory contains a local development stack for the vLLM queue-based architecture, which demonstrates request queuing and async processing through a proxy.

## Architecture Overview

The stack consists of four services:

- **valkey**: Message broker for request queuing
- **mockvllm**: Mock vLLM server (simulates the LLM backend)
- **sidecar**: Consumer that processes requests from the Valkey queue and forwards them to mockvllm
- **proxy**: API gateway that accepts requests and enqueues them to Valkey

## Running the Stack

### Prerequisites

- Docker and Docker Compose installed
- Port 18001 (proxy), 18000 (mockvllm), and 16379 (valkey) available on your machine

### Start the Stack

From this directory, run:

```bash
docker compose up --build
```

This command will:
1. Build all four services from their Dockerfiles
2. Start Valkey with health checks
3. Start the mock vLLM server
4. Start the sidecar consumer
5. Start the proxy API gateway

Wait for all services to be healthy (you should see logs indicating successful startup).

## Testing the Stack

The proxy listens on `http://localhost:18001` and exposes an OpenAI-compatible API endpoint at `/v1/chat/completions`.

### Non-Streaming Request

Send a non-streaming request (the request is queued, processed, and the full response is returned):

```bash
curl -X POST http://localhost:18001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mock-model",
    "messages": [
      {
        "role": "user",
        "content": "Hello, how are you?"
      }
    ],
    "stream": false
  }'
```

Expected response: A JSON object containing the model's completion.

### Streaming Request

Send a streaming request (the response is streamed back as Server-Sent Events):

```bash
curl -X POST http://localhost:18001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mock-model",
    "messages": [
      {
        "role": "user",
        "content": "Hello, how are you?"
      }
    ],
    "stream": true
  }'
```

Expected response: A stream of JSON objects (one per line), each containing a partial completion chunk.

## Stopping the Stack

To stop all services:

```bash
docker compose down
```

To stop and remove all volumes (including Redis data):

```bash
docker compose down -v
```

## Troubleshooting

- **Port already in use**: If port 18001, 18000, or 16379 is already in use, either stop the conflicting service or modify the port mappings in `docker-compose.yaml`.
- **Services not starting**: Check the logs with `docker compose logs <service-name>` (e.g., `docker compose logs proxy`).
- **Requests timing out**: Ensure all services are healthy by running `docker compose ps` and checking the STATUS column.
