# Control Plane

[Control Plane](https://controlplane.com) runs the vLLM OpenAI-compatible server from its template catalog, on a built-in NVIDIA GPU (A10G or T4) or on your own GPU cluster registered as a BYOK location. The template serves one Hugging Face model, protects the API with a key, and caches the model on a persistent volume so restarts do not download it again.

## Prerequisites

- A Control Plane org with GPU quota in a location that offers it (ask Control Plane support), or a BYOK location backed by NVIDIA GPU nodes with driver 580 or newer.
- A GVC with exactly one location (each location runs its own copy).
- The [`cpln` CLI](https://docs.controlplane.com/cli-reference/installation), logged in.

## Deploy

Create the API key clients will send as a Bearer token:

```bash
export VLLM_API_KEY=$(openssl rand -hex 32)
printf '%s' "$VLLM_API_KEY" | cpln secret create-opaque --name vllm-api-key --encoding plain -f -
```

Install the template, choosing the model:

```bash
cpln helm install my-vllm oci://ghcr.io/controlplane-com/templates/vllm \
  --version 1.1.0 --gvc GVC_NAME \
  --set apiKey.secretName=vllm-api-key \
  --set model.name=Qwen/Qwen2.5-1.5B-Instruct
```

The first start pulls the image and downloads the model, which takes several minutes; `cpln workload get-deployments my-vllm-vllm --gvc GVC_NAME` shows when it is ready.

For a gated model (Llama, Gemma), accept its license on Hugging Face, store a read token in another opaque secret and set `huggingFace.tokenSecretName`.

## Send a request

The server is private by default. From your machine, open a tunnel:

```bash
cpln port-forward my-vllm-vllm 8000:8000 --gvc GVC_NAME
```

Then call it with the key:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer $VLLM_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model": "Qwen/Qwen2.5-1.5B-Instruct", "messages": [{"role": "user", "content": "Hello"}]}'
```

Other workloads in the same GVC reach it at `http://my-vllm-vllm.GVC_NAME.cpln.local:8000/v1`. Public access, replicas, tensor parallelism and BYOK cluster setup are template values; see the [template documentation](https://docs.controlplane.com/template-catalog/templates/vllm?utm_source=github&utm_medium=readme&utm_campaign=upstream-vllm).
