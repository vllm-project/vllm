# Nebius Serverless AI

[Nebius Serverless AI](https://docs.nebius.com/serverless/overview) runs GPU containers as persistent HTTP endpoints or finite jobs. This guide deploys the vLLM OpenAI-compatible server as a single-GPU Endpoint using the Nebius CLI.

The Endpoint keeps running until you stop or delete it. This recipe uses a regular GPU instance; it does not configure autoscaling or automatic scale-to-zero.

## Prerequisites

- A Nebius project with permission to create and delete Serverless Endpoints, and sufficient GPU, VM, disk, and networking quotas.
- The [Nebius CLI](https://docs.nebius.com/cli/install), installed and authenticated for that project.
- A subnet in `eu-north1` with outbound access to Docker Hub and Hugging Face. The example uses the `gpu-l40s-a` platform and `1gpu-8vcpu-32gb` preset; check availability in your project before starting.
- `curl`, `jq`, and `openssl` on your local machine.

The example model, `Qwen/Qwen3-0.6B`, does not require a Hugging Face token. Other models may require accepting their license and supplying a token through Nebius [secret environment variables](https://docs.nebius.com/serverless/endpoints/manage).

## Create the Endpoint

The commands below were tested with Nebius CLI `0.12.265`. Set your project and subnet IDs explicitly. Use the same CLI profile throughout the guide.

```bash
export PROJECT_ID="<project-id>"
export SUBNET_ID="<subnet-id>"
export ENDPOINT_NAME="vllm-qwen-$(openssl rand -hex 6)"
export AUTH_TOKEN="$(openssl rand -hex 32)"
export MODEL_ID="Qwen/Qwen3-0.6B"
export MODEL_REVISION="c1899de289a04d12100db370d81485cdf75e47ca"
# vllm/vllm-openai:v0.19.1, Linux/amd64 image digest.
export VLLM_IMAGE="vllm/vllm-openai@sha256:89c1d0629d377daa3f7f369cbea6167a7b48ea89aaacd12555e2b0b2f7f740d3"

nebius ai endpoint create \
    --parent-id "$PROJECT_ID" \
    --subnet-id "$SUBNET_ID" \
    --name "$ENDPOINT_NAME" \
    --image "$VLLM_IMAGE" \
    --container-command vllm \
    --args "serve $MODEL_ID --revision $MODEL_REVISION --tokenizer-revision $MODEL_REVISION --host 0.0.0.0 --port 8000 --tensor-parallel-size 1 --max-model-len 4096 --gpu-memory-utilization 0.8" \
    --platform gpu-l40s-a \
    --preset 1gpu-8vcpu-32gb \
    --container-port 8000/http \
    --auth token --token "$AUTH_TOKEN" \
    --disk-size 250Gi --shm-size 16Gi \
    --public=false --preemptible=false \
    --retries 1
```

The image and model revisions are pinned for reproducibility. When changing the image, check its CUDA/driver requirements against the selected Nebius platform. The model downloads into the container disk on startup; this example does not configure persistent model storage.

The Endpoint token is separate from your Nebius CLI credentials. Keep it private. Nebius token authentication requires exactly one HTTP port. This example relies on authentication at the managed HTTPS URL; it does not set a second vLLM API key. Access from inside the subnet must be restricted to trusted clients.

A public VM IP is not required to use the managed HTTPS URL. The subnet still needs outbound connectivity for image and model downloads.

Save the ID of the Endpoint you created:

```bash
export ENDPOINT_ID="$(nebius ai endpoint get-by-name \
    --parent-id "$PROJECT_ID" --name "$ENDPOINT_NAME" \
    --format jsonpath='{.metadata.id}')"
```

If creation times out or your terminal disconnects, use the same project and name to find the Endpoint before attempting another creation. A local timeout does not mean that provisioning was cancelled.

## Check readiness

Inspect the state and recent logs:

```bash
nebius ai endpoint get "$ENDPOINT_ID"
nebius ai endpoint logs "$ENDPOINT_ID" --tail 100 --timestamps
```

Wait for `RUNNING`, then select the managed HTTPS URL. The URL can appear while the Endpoint is still `STARTING`, when ingress may return HTTP 404. The `jq` expression requires exactly one HTTPS URL and preserves its scheme:

```bash
ENDPOINT_URL="$(nebius ai endpoint get "$ENDPOINT_ID" --format json \
    | jq -er '[.status.public_endpoints[]? | select(startswith("https://"))]
        | if length == 1 then .[0] else error("Expected one HTTPS URL") end')"
export ENDPOINT_URL="${ENDPOINT_URL%/}"

curl --fail-with-body --silent --show-error --max-time 10 \
    "$ENDPOINT_URL/health" -H "Authorization: Bearer $AUTH_TOKEN"
curl --fail-with-body --silent --show-error --max-time 10 \
    "$ENDPOINT_URL/v1/models" -H "Authorization: Bearer $AUTH_TOKEN" | jq
```

`RUNNING` alone does not establish that the model has finished loading. Wait for `/health` to succeed and `/v1/models` to list `Qwen/Qwen3-0.6B` before sending a chat request. While waiting, inspect logs instead of creating another Endpoint. If startup does not succeed within your chosen time or spending limit, delete the Endpoint and investigate the error.

## Send a chat request

```bash
curl --fail-with-body --silent --show-error --max-time 120 \
    "$ENDPOINT_URL/v1/chat/completions" \
    -H "Authorization: Bearer $AUTH_TOKEN" \
    -H 'Content-Type: application/json' \
    -d '{
        "model": "Qwen/Qwen3-0.6B",
        "messages": [{"role": "user", "content": "Say hello in one short sentence."}],
        "max_tokens": 128,
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": false}
    }' | jq
```

Expect a chat completion containing an assistant message. The Qwen-specific template option disables thinking for this short example; it is not a universal option for other models.

For streaming, add `"stream": true` to the JSON and use `curl -N` without piping into `jq`. A successful stream contains completion chunks followed by `data: [DONE]`. For an OpenAI-compatible client, use `${ENDPOINT_URL}/v1` as the base URL and the Endpoint token as the API key.

Verify that the same inference URL rejects requests without the token:

```bash
curl --silent --show-error --max-time 10 --output /dev/null \
    --write-out '%{http_code}\n' "$ENDPOINT_URL/v1/models"
```

Expect HTTP `401` or `403`. Do not share an Endpoint that accepts unauthenticated inference unexpectedly.

## Stop or delete the Endpoint

To stop serving while preserving the Endpoint configuration:

```bash
nebius ai endpoint stop "$ENDPOINT_ID"
nebius ai endpoint get "$ENDPOINT_ID"
```

Let the synchronous stop command complete and confirm `STOPPED` before starting again. To start it again, run `nebius ai endpoint start "$ENDPOINT_ID"`, retrieve its current URL, and repeat the readiness checks. Starting again can require fresh capacity and new image/model downloads; do not rely on the container disk as persistent storage.

When finished, delete the Endpoint and confirm that getting the same ID returns `NotFound`:

```bash
nebius ai endpoint delete "$ENDPOINT_ID"
nebius ai endpoint get "$ENDPOINT_ID"
unset AUTH_TOKEN
```

Nebius deletes the managed VM and container disk with the Endpoint. Separately mounted storage has its own lifecycle and billing. Closing your terminal or interrupting a request does not stop the Endpoint. If deletion fails, keep the ID and retry cleanup; a generic connection or authentication error is not confirmation of deletion.

## Troubleshooting

| Symptom | Check |
| --- | --- |
| `PROVISIONING` or `NotEnoughResources` | Inspect quota, region, platform and preset. Capacity availability can differ from quota. |
| `RUNNING`, but HTTP 502/503 | Check model-download and engine-start logs, subnet egress, host binding and port 8000. |
| HTTP 401/403 | Use the Endpoint token, not the Nebius control-plane token. |
| CUDA or out-of-memory error | Check image/driver compatibility, the model's memory requirements and maximum context length. |
| Model not found | Verify the requested model ID and pinned revision. |
| Interrupted stream | Inspect ingress and server logs. A partial response should not be automatically replayed as a new generation. |

See the [Nebius vLLM cookbook example](https://github.com/nebius/serverless-ai-cookbook/tree/main/inference/vllm-endpoint) and [Endpoint management documentation](https://docs.nebius.com/serverless/endpoints/manage) for further details.
