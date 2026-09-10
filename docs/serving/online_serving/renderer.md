# Renderer APIs

Our renderer API is designed to disaggregate the render phase(preprocessing) and enable a token-in / token-out API server.

- GPU-less deployment of frontend: Allow preprocessing (tokenization, MM input processing) and postprocessing (detokenization, tool call parsing, reasoning parsing) to run without GPU.
- Disaggregated tokenization: Support use cases such as llm-d, Dynamo, and custom frontends that need to leverage vLLM's preprocessing logic without running the full inference engine.
- Tokens-in / tokens-out engine: Make the engine a pure token-in / token-out service, decoupled from request preprocessing.

The dedicated `vllm launch render` server always exposes the `/render` and
`/derender` endpoints when `VLLM_ENABLE_SCALE_OUT_ENDPOINTS` is unset or set to
`1`. An explicit value of `0` conflicts with the renderer command and is
rejected at startup.

Scale-out endpoints, including `/render`, `/derender`, and
`/inference/v1/generate`, are disabled by default on a standard inference
server. To expose them with `vllm serve`, opt in explicitly:

```bash
VLLM_ENABLE_SCALE_OUT_ENDPOINTS=1 vllm serve <model>
```

## API Reference

- [Completions Render API](renderer.md) (`/v1/completions/render`)
    - Render completion requests
- [Chat Completions Render API](renderer.md) (`/v1/chat/completions/render`)
    - Render chat completions
- [Responses Render API](renderer.md) (`/v1/responses/render`)
    - Render a self-contained Responses request

The Responses render endpoint uses the same prompt construction as
`/v1/responses` and returns one token-in `GenerateRequest`. It is stateless:
inline history is supported, but `previous_response_id` is not. Callers must
resolve stored response state and include the resulting history in the request
before rendering.

For multimodal requests, the `GenerateRequest` contains the model-processed
multimodal payload, which can be substantially larger than the source image or
video. The caller must forward that payload unchanged to the generation service
and provision transport limits and memory accordingly.

```bash
curl http://localhost:8000/v1/responses/render \
    -H "Content-Type: application/json" \
    -d '{
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "input": "Explain prefix caching in one sentence.",
        "max_output_tokens": 32
    }'
```

For the post processing counterpart that turns generated token IDs back into OpenAI compatible responses, see the [Derenderer APIs](derenderer.md).

## Multimodal Render Features

Multimodal render responses include a `features` object with per-modality
hashes, placeholder ranges, and serialized processor data. When the model
exposes placeholder-metadata or `keep_on_cpu` fields (for example
`image_grid_thw`), the response also includes `mm_metadata`. Each
`mm_metadata` entry is a base64-encoded `MultiModalKwargsItem` containing
only those fields, not encoder inputs such as `pixel_values`.

The arrays in `mm_hashes`, `mm_placeholders`, `kwargs_data`, and
`mm_metadata` use the same per-modality item order. Downstream workers
should split those fields:

- Encode requests keep `kwargs_data`.
- Prefill requests may omit `kwargs_data` and send `mm_metadata` only when
  `ec_transfer_params` is also set, so embeddings are loaded by the EC
  connector. Omitting `kwargs_data` without `ec_transfer_params` is rejected.
- Legacy clients that ignore `mm_metadata` and keep sending `kwargs_data`
  continue to work.

## Example

The example below shows how a disaggregated encode / prefill coordinator can
split a multimodal render response. The render step returns both
`kwargs_data` (encoder tensors plus metadata) and `mm_metadata` (metadata
only). Encode keeps the full payload; prefill drops `kwargs_data` after the
EC connector has published embeddings.

```python
import httpx

MODEL = "Qwen/Qwen3-VL-2B-Instruct"
RENDER = "http://localhost:8100"  # vllm launch render ...
ENCODE = "http://localhost:8200"  # encode worker
PREFILL = "http://localhost:8300"  # prefill worker

chat_request = {
    "model": MODEL,
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": "<data-url>"}},
                {"type": "text", "text": "Describe this image."},
            ],
        }
    ],
}

with httpx.Client(timeout=120.0) as client:
    # 1. Render: preprocess into token IDs and multimodal features.
    render_response = client.post(
        f"{RENDER}/v1/chat/completions/render", json=chat_request
    ).json()

    features = render_response["features"]
    # features["kwargs_data"]["image"][0]  -> pixel_values + image_grid_thw
    # features["mm_metadata"]["image"][0]  -> image_grid_thw only

    # 2. Encode: send full kwargs_data so the encoder can run vision towers.
    encode_response = client.post(
        f"{ENCODE}/inference/v1/generate",
        json={
            "token_ids": render_response["token_ids"],
            "features": {
                "mm_hashes": features["mm_hashes"],
                "mm_placeholders": features["mm_placeholders"],
                "kwargs_data": features["kwargs_data"],
            },
            "sampling_params": {"max_tokens": 1},
        },
    ).json()
    ec_transfer_params = encode_response["ec_transfer_params"]

    # 3. Prefill: omit kwargs_data; load embeddings via EC connector.
    prefill_response = client.post(
        f"{PREFILL}/inference/v1/generate",
        json={
            "token_ids": render_response["token_ids"],
            "features": {
                "mm_hashes": features["mm_hashes"],
                "mm_placeholders": features["mm_placeholders"],
                "mm_metadata": features["mm_metadata"],
            },
            "ec_transfer_params": ec_transfer_params,
            "sampling_params": {"max_tokens": 64},
        },
    ).json()

print(prefill_response["choices"][0]["token_ids"])
```

Single-process clients can keep passing the full render response to
`/inference/v1/generate` unchanged; `mm_metadata` is optional and ignored
when `kwargs_data` is present.

### Payload shape

#### Render response

`/v1/chat/completions/render` returns both `kwargs_data` and `mm_metadata`.
The arrays share the same per-modality item order. Base64 blobs are truncated
below for readability.

```json
{
  "token_ids": [151644, 872],
  "features": {
    "mm_hashes": {"image": ["abc123..."]},
    "mm_placeholders": {"image": [{"offset": 0, "length": 256}]},
    "kwargs_data": {
      "image": ["<base64 MultiModalKwargsItem: pixel_values + image_grid_thw>"]
    },
    "mm_metadata": {
      "image": ["<base64 MultiModalKwargsItem: image_grid_thw only>"]
    }
  }
}
```

Forward `kwargs_data` to the encode worker. Keep `mm_metadata` for prefill.

#### Prefill request

Prefill omits `kwargs_data` and sends `mm_metadata` with
`ec_transfer_params` from the encode response:

```json
{
  "token_ids": [151644, 872],
  "features": {
    "mm_hashes": {"image": ["abc123..."]},
    "mm_placeholders": {"image": [{"offset": 0, "length": 256}]},
    "mm_metadata": {
      "image": ["<base64 MultiModalKwargsItem: image_grid_thw only>"]
    }
  },
  "ec_transfer_params": {
    "ec_items": [{"mm_hash": "abc123...", "peer_host": "10.0.0.1"}]
  },
  "sampling_params": {"max_tokens": 64}
}
```
