# Runcrate

[Runcrate](https://www.runcrate.ai/?utm_source=vllm&utm_medium=directory&utm_campaign=gpu_pricing) rents GPU virtual machines by the hour, from RTX A6000 and L40S up to H100, H200, B200 and B300. This guide starts a single-GPU instance with the `runcrate` CLI, runs the vLLM OpenAI-compatible server on it in the official Docker image, and serves it at an HTTPS address protected by a vLLM API key.

## Prerequisites

- A Runcrate account with credit, and an API key from **Dashboard > Keys**.
- The [Runcrate CLI](https://www.runcrate.ai/docs/cli/installation?utm_source=vllm&utm_medium=directory&utm_campaign=gpu_pricing):

    ```bash
    curl -sSL https://runcrate.ai/install.sh | sh
    ```

- `curl`, `jq` and `openssl` on your local machine.

The CLI reads the API key from `RUNCRATE_API_KEY`; `runcrate login` signs in through the browser instead. The HTTPS link step below calls the REST API, so it needs the key either way:

```bash
export RUNCRATE_API_KEY="rc_live_..."
```

The example model, `Qwen/Qwen3-8B`, does not require a Hugging Face token and fits on one 48 GB GPU at BF16 with room for the KV cache.

## Start an instance

List the machines that have the GPU you want, with their price per hour:

```bash
runcrate instances types --gpu A6000
```

Create the instance with the `ubuntu-inference` image, which comes with Docker and the NVIDIA container runtime. `--max-runtime` stops it after four hours, so a forgotten server does not keep billing:

```bash
runcrate instances create --name vllm-server --gpu A6000 \
    --template ubuntu-inference --max-runtime 4h
```

Wait until `runcrate instances status vllm-server` reports `running`. This usually takes a few minutes.

## Start the server

Generate a key for vLLM's own `--api-key` check, then start the [vLLM Docker image](../docker.md) in the background. `runcrate ssh` runs the command on the instance over SSH, with no key setup:

```bash
export VLLM_API_KEY="$(openssl rand -hex 32)"

runcrate ssh vllm-server -- "docker run -d --name vllm --gpus all --ipc=host \
    -p 8000:8000 \
    -v /root/.cache/huggingface:/root/.cache/huggingface \
    -e VLLM_ENABLE_CUDA_COMPATIBILITY=1 \
    vllm/vllm-openai:latest \
    Qwen/Qwen3-8B --max-model-len 8192 --api-key $VLLM_API_KEY"
```

!!! note

    `VLLM_ENABLE_CUDA_COMPATIBILITY=1` lets the image run on a host driver older than the image's CUDA version; see [Running on Systems with Older CUDA Drivers](../../getting_started/installation/gpu.md#running-on-systems-with-older-cuda-drivers). Check the driver with `runcrate ssh vllm-server -- nvidia-smi`: the A6000 used for this guide reports CUDA 12.8. Leave the variable out when the driver already supports the image's CUDA version.

The image is about 33 GB, so the first start spends a few minutes on the pull before the model download. Follow the start-up in the container log:

```bash
runcrate ssh vllm-server -- "docker logs --tail 20 vllm"
```

The server is ready when the log shows `Application startup complete.`

## Serve it over HTTPS

An HTTPS link serves one port of the instance at `https://<port>-<instance>.runcrate.app` with a valid certificate, and needs no firewall rule. Create a public link for port 8000; vLLM's `--api-key` still rejects requests without the key:

```bash
INSTANCE_ID="$(runcrate instances info vllm-server --json | jq -r .id)"

export VLLM_URL="$(curl --fail-with-body -sS -X POST \
    "https://www.runcrate.ai/api/v1/instances/$INSTANCE_ID/links" \
    -H "Authorization: Bearer $RUNCRATE_API_KEY" \
    -H "Content-Type: application/json" \
    -d '{"port": 8000, "access": "public"}' | jq -r .data.url)"

echo "$VLLM_URL"
```

To accept requests only from known addresses, send `{"port": 8000, "access": "ips", "sources": ["203.0.113.7"]}` instead.

## Send a request

!!! console "Command"

    ```bash
    curl "$VLLM_URL/v1/chat/completions" \
        -H "Authorization: Bearer $VLLM_API_KEY" \
        -H "Content-Type: application/json" \
        -d '{
            "model": "Qwen/Qwen3-8B",
            "messages": [{"role": "user", "content": "Say hello in one short sentence."}],
            "max_tokens": 64,
            "chat_template_kwargs": {"enable_thinking": false}
        }' | jq -r '.choices[0].message.content'
    ```

!!! console "Response"

    ```text
    Hello! How can I assist you today?
    ```

The `chat_template_kwargs` option turns off Qwen3's thinking mode for this short example; other models do not need it.

Any OpenAI-compatible client works with `$VLLM_URL/v1` as the base URL and `$VLLM_API_KEY` as the API key:

```python
import os

from openai import OpenAI

client = OpenAI(
    base_url=os.environ["VLLM_URL"] + "/v1",
    api_key=os.environ["VLLM_API_KEY"],
)

response = client.chat.completions.create(
    model="Qwen/Qwen3-8B",
    messages=[{"role": "user", "content": "Say hello in one short sentence."}],
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
)
print(response.choices[0].message.content)
```

A request without the key returns HTTP `401`:

```bash
curl -s -o /dev/null -w '%{http_code}\n' "$VLLM_URL/v1/models"
```

## Larger models

For a model that does not fit on one GPU, create the instance with several GPUs on the same machine, for example `--gpu H100:4`, and add `--tensor-parallel-size 4` after the model name in the `docker run` command. Gated models such as Llama also need a Hugging Face token: add `-e HF_TOKEN=<token>` to `docker run`.

## Clean up

Deleting the instance stops billing and removes its HTTPS link:

```bash
runcrate instances delete vllm-server
```

## Troubleshooting

| Symptom | Check |
| --- | --- |
| HTTP 502 from the link | The image is still pulling or the model is still loading. Check `docker logs vllm` for `Application startup complete.` |
| HTTP 401 | Send `Authorization: Bearer $VLLM_API_KEY`, the key given to `--api-key`. |
| CUDA out of memory | Lower `--max-model-len`, or choose a GPU with more memory. |
