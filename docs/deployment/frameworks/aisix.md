# AISIX AI Gateway

[AISIX](https://github.com/api7/aisix) is an Apache-2.0-licensed open-source AI
gateway maintained by API7.ai. It can place caller authentication and a stable
model alias in front of vLLM's OpenAI-compatible server without changing the
vLLM deployment.

This guide keeps the vLLM server credential at the gateway boundary.
Applications use a separate AISIX caller key and request the alias
`local-chat`; AISIX sends the vLLM credential and the served model name
upstream.

## Prerequisites

- A vLLM server that AISIX can reach. Follow the
  [vLLM installation guide](../../getting_started/installation/README.md) for your
  accelerator or CPU platform.
- Docker to run the open-source AISIX gateway.
- `curl` to verify the request path.

The AISIX–vLLM request path below was validated with vLLM 0.29.0 and AISIX
1.2.0. Choose a vLLM-supported chat model that fits your hardware.

## Start vLLM

Set a credential for the OpenAI-compatible API and start vLLM with a stable
served model name:

```bash
export VLLM_API_KEY="replace-with-a-vllm-key"

vllm serve HuggingFaceTB/SmolLM2-135M-Instruct \
  --served-model-name smollm2 \
  --api-key "$VLLM_API_KEY"
```

In a second terminal, export the same upstream credential and confirm that
vLLM is reachable from the machine where AISIX will run:

```bash
export VLLM_API_KEY="replace-with-the-same-vllm-key"

curl -sS http://127.0.0.1:8000/v1/models \
  -H "Authorization: Bearer $VLLM_API_KEY"
```

If AISIX runs in Docker while vLLM runs on the host, use an address that
resolves from the container, such as
`http://host.docker.internal:8000/v1`. Docker Desktop provides this hostname.
The AISIX startup command below adds the equivalent host mapping for Linux
Docker Engine. For services on the same Docker or Kubernetes network, use the
vLLM service name instead.

## Configure AISIX

Create a working directory:

```bash
mkdir aisix-vllm
cd aisix-vllm
```

Create `resources.yaml`:

```yaml
_format_version: "1"

provider_keys:
  - display_name: vllm-local
    provider: vllm
    adapter: openai
    api_key: ${VLLM_API_KEY}
    api_base: "${VLLM_API_BASE}"

models:
  - display_name: local-chat
    provider: vllm
    model_name: smollm2
    provider_key: vllm-local

api_keys:
  - display_name: local-client
    key_env: CALLER_API_KEY
    allowed_models:
      - local-chat
```

The caller-facing `display_name` can remain stable when the underlying vLLM
model changes. The `model_name` must match a model ID returned by the vLLM
`/v1/models` endpoint. The caller key and the vLLM server key are deliberately
different credentials.

Create `config.yaml`:

```yaml
resources_file: /etc/aisix/resources.yaml

proxy:
  addr: "0.0.0.0:3000"

admin:
  enabled: false
```

Export the values referenced by the resource file:

```bash
export VLLM_API_KEY="replace-with-the-same-vllm-key"
export VLLM_API_BASE="http://host.docker.internal:8000/v1"
export CALLER_API_KEY="replace-with-a-caller-key"
```

Validate the resources before starting the gateway:

```bash
docker run --rm \
  -v "$(pwd):/etc/aisix:ro" \
  -e VLLM_API_KEY \
  -e VLLM_API_BASE \
  -e CALLER_API_KEY \
  --entrypoint /usr/local/bin/aisix \
  ghcr.io/api7/aisix:1.2.0 \
  validate --resources /etc/aisix/resources.yaml
```

Start AISIX:

```bash
docker run -d --rm --name aisix-vllm \
  --add-host=host.docker.internal:host-gateway \
  -v "$(pwd):/etc/aisix:ro" \
  -e VLLM_API_KEY \
  -e VLLM_API_BASE \
  -e CALLER_API_KEY \
  -p 3000:3000 \
  ghcr.io/api7/aisix:1.2.0
```

## Verify the request path

With AISIX running in the background, list the models available to the caller:

```bash
curl -sS http://127.0.0.1:3000/v1/models \
  -H "Authorization: Bearer $CALLER_API_KEY"
```

The response should contain the caller-facing `local-chat` alias. It does not
expose the vLLM credential or the upstream model name.

Send a non-streaming request:

```bash
curl -sS http://127.0.0.1:3000/v1/chat/completions \
  -H "Authorization: Bearer $CALLER_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "local-chat",
    "messages": [
      {"role": "user", "content": "Reply with one short sentence."}
    ],
    "max_tokens": 32
  }'
```

For streaming, add `"stream": true` to the request body and use `curl -N`.
AISIX relays the vLLM SSE response incrementally and ends the stream with
`data: [DONE]`.

An invalid caller key is rejected at the AISIX boundary with HTTP `401`. The
application does not need the credential configured with vLLM. In the tested
refused-connection scenario, AISIX 1.2.0 returned HTTP `502`. After vLLM was
available again, a subsequent request succeeded without changing the AISIX
resources.

## Security boundary

AISIX caller authentication protects traffic that enters through the gateway.
It does not make a publicly reachable vLLM service private. The
[vLLM security guide](../../usage/security.md#api-key-authentication-limitations)
explains that `--api-key` protects specific path prefixes rather than every
server endpoint. Keep vLLM on a private network and apply network policy or
another control to administrative and diagnostic paths as needed.

For endpoint-specific behavior and passthrough options, see the
[AISIX vLLM provider guide](https://docs.api7.ai/ai-gateway/providers/vllm).

## Clean up

Stop AISIX with `docker stop aisix-vllm`. Because the container was started
with `--rm`, Docker removes it after it stops. Stop the vLLM process separately
when you no longer need the model server.
