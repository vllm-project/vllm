# Embedding Usages

Embedding models are a class of machine learning models designed to transform unstructured data—such as text, images, or audio—into a structured numerical representation known as an embedding.

## Summary

- Model Usage: (sequence) embedding
- Pooling Task: `embed`
- Offline APIs:
    - `LLM.embed(...)`
    - `LLM.encode(..., pooling_task="embed")`
    - `LLM.score(...)`
- Online APIs:
    - [Cohere Embed API](embed.md#cohere-embed-api) (`/v2/embed`)
    - [Openai-compatible Embeddings API](embed.md#openai-compatible-embeddings-api) (`/v1/embeddings`)
    - Pooling API (`/pooling`)

The primary distinction between (sequence) embedding and token embedding lies in their output granularity: (sequence) embedding produces a single embedding vector for an entire input sequence, whereas token embedding generates an embedding for each individual token within the sequence.

Many embedding models support both (sequence) embedding and token embedding. For further details on token embedding, please refer to [this page](token_embed.md).

## Typical Use Cases

### Embedding

The most basic use case of embedding models is to embed the inputs, e.g. for RAG.

### Pairwise Similarity

You can compute pairwise similarity scores to build a similarity matrix using the [Score API](scoring.md).

## Supported Models

--8<-- "docs/models/pooling_models/supported_models.inc.md:embed-models"

## Offline Inference

### Pooling Parameters

The following [pooling parameters][vllm.PoolingParams] are supported.

```python
--8<-- "vllm/pooling_params.py:common-pooling-params"
--8<-- "vllm/pooling_params.py:embed-pooling-params"
```

### `LLM.embed`

The [embed][vllm.LLM.embed] method outputs an embedding vector for each prompt.

```python
from vllm import LLM

llm = LLM(model="intfloat/e5-small", runner="pooling")
(output,) = llm.embed("Hello, my name is")

embeds = output.outputs.embedding
print(f"Embeddings: {embeds!r} (size={len(embeds)})")
```

A code example can be found here: [examples/offline_inference/basic/embed.py](../../../examples/basic/offline_inference/embed.py)

### `LLM.encode`

The [encode][vllm.LLM.encode] method is available to all pooling models in vLLM.

Set `pooling_task="embed"` when using `LLM.encode` for embedding Models:

```python
from vllm import LLM

llm = LLM(model="intfloat/e5-small", runner="pooling")
(output,) = llm.encode("Hello, my name is", pooling_task="embed")

data = output.outputs.data
print(f"Data: {data!r}")
```

### `LLM.score`

The [score][vllm.LLM.score] method outputs similarity scores between sentence pairs.

All models that support embedding task also support using the score API to compute similarity scores by calculating the cosine similarity of two input prompt's embeddings.

```python
from vllm import LLM

llm = LLM(model="intfloat/e5-small", runner="pooling")
(output,) = llm.score(
    "What is the capital of France?",
    "The capital of Brazil is Brasilia.",
)

score = output.outputs.score
print(f"Score: {score}")
```

## Online Serving

### OpenAI-Compatible Embeddings API

Our Embeddings API is compatible with [OpenAI's Embeddings API](https://platform.openai.com/docs/api-reference/embeddings);
you can use the [official OpenAI Python client](https://github.com/openai/openai-python) to interact with it.

Code example: [examples/pooling/embed/openai_embedding_client.py](../../../examples/pooling/embed/openai_embedding_client.py)

#### Completion Parameters

The following Classification API parameters are supported:

??? code

    ```python
    --8<-- "vllm/entrypoints/pooling/base/protocol.py:pooling-common-params"
    --8<-- "vllm/entrypoints/pooling/base/protocol.py:completion-params"
    --8<-- "vllm/entrypoints/pooling/base/protocol.py:encoding-params"
    --8<-- "vllm/entrypoints/pooling/base/protocol.py:embed-params"
    ```

The following extra parameters are supported:

??? code

    ```python
    --8<-- "vllm/entrypoints/pooling/base/protocol.py:pooling-common-extra-params"
    --8<-- "vllm/entrypoints/pooling/base/protocol.py:completion-extra-params"
    --8<-- "vllm/entrypoints/pooling/base/protocol.py:encoding-extra-params"
    --8<-- "vllm/entrypoints/pooling/base/protocol.py:embed-extra-params"
    ```

#### Chat Parameters

For chat-like input (i.e. if `messages` is passed), the following parameters are supported:

??? code

    ```python
    --8<-- "vllm/entrypoints/pooling/base/protocol.py:pooling-common-params"
    --8<-- "vllm/entrypoints/pooling/base/protocol.py:chat-params"
    --8<-- "vllm/entrypoints/pooling/base/protocol.py:encoding-params"
    --8<-- "vllm/entrypoints/pooling/base/protocol.py:embed-params"
    ```

these extra parameters are supported instead:

??? code

    ```python
    --8<-- "vllm/entrypoints/pooling/base/protocol.py:pooling-common-extra-params"
    --8<-- "vllm/entrypoints/pooling/base/protocol.py:chat-extra-params"
    --8<-- "vllm/entrypoints/pooling/base/protocol.py:encoding-extra-params"
    --8<-- "vllm/entrypoints/pooling/base/protocol.py:embed-extra-params"
    ```

#### Examples

If the model has a [chat template](../../serving/openai_compatible_server.md#chat-template), you can replace `inputs` with a list of `messages` (same schema as [Chat API](../../serving/openai_compatible_server.md#chat-api))
which will be treated as a single prompt to the model. Here is a convenience function for calling the API while retaining OpenAI's type annotations:

??? code

    ```python
    from openai import OpenAI
    from openai._types import NOT_GIVEN, NotGiven
    from openai.types.chat import ChatCompletionMessageParam
    from openai.types.create_embedding_response import CreateEmbeddingResponse

    def create_chat_embeddings(
        client: OpenAI,
        *,
        messages: list[ChatCompletionMessageParam],
        model: str,
        encoding_format: Union[Literal["base64", "float"], NotGiven] = NOT_GIVEN,
    ) -> CreateEmbeddingResponse:
        return client.post(
            "/embeddings",
            cast_to=CreateEmbeddingResponse,
            body={"messages": messages, "model": model, "encoding_format": encoding_format},
        )
    ```

##### Multi-modal inputs

You can pass multi-modal inputs to embedding models by defining a custom chat template for the server
and passing a list of `messages` in the request. Refer to the examples below for illustration.

=== "VLM2Vec"

    To serve the model:

    ```bash
    vllm serve TIGER-Lab/VLM2Vec-Full --runner pooling \
      --trust-remote-code \
      --max-model-len 4096 \
      --chat-template examples/pooling/embed/template/vlm2vec_phi3v.jinja
    ```

    !!! important
        Since VLM2Vec has the same model architecture as Phi-3.5-Vision, we have to explicitly pass `--runner pooling`
        to run this model in embedding mode instead of text generation mode.

        The custom chat template is completely different from the original one for this model,
        and can be found here: [examples/pooling/embed/template/vlm2vec_phi3v.jinja](../../../examples/pooling/embed/template/vlm2vec_phi3v.jinja)

    Since the request schema is not defined by OpenAI client, we post a request to the server using the lower-level `requests` library:

    ??? code

        ```python
        from openai import OpenAI
        client = OpenAI(
            base_url="http://localhost:8000/v1",
            api_key="EMPTY",
        )
        image_url = "https://vllm-public-assets.s3.us-west-2.amazonaws.com/vision_model_images/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"

        response = create_chat_embeddings(
            client,
            model="TIGER-Lab/VLM2Vec-Full",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_url}},
                        {"type": "text", "text": "Represent the given image."},
                    ],
                }
            ],
            encoding_format="float",
        )

        print("Image embedding output:", response.data[0].embedding)
        ```

=== "DSE-Qwen2-MRL"

    To serve the model:

    ```bash
    vllm serve MrLight/dse-qwen2-2b-mrl-v1 --runner pooling \
      --trust-remote-code \
      --max-model-len 8192 \
      --chat-template examples/pooling/embed/template/dse_qwen2_vl.jinja
    ```

    !!! important
        Like with VLM2Vec, we have to explicitly pass `--runner pooling`.

        Additionally, `MrLight/dse-qwen2-2b-mrl-v1` requires an EOS token for embeddings, which is handled
        by a custom chat template: [examples/pooling/embed/template/dse_qwen2_vl.jinja](../../../examples/pooling/embed/template/dse_qwen2_vl.jinja)

    !!! important
        `MrLight/dse-qwen2-2b-mrl-v1` requires a placeholder image of the minimum image size for text query embeddings. See the full code
        example below for details.

Full example: [examples/pooling/embed/vision_embedding_online.py](../../../examples/pooling/embed/vision_embedding_online.py)

### Cohere Embed API

Our API is also compatible with [Cohere's Embed v2 API](https://docs.cohere.com/reference/embed) which adds support for some modern embedding feature such as truncation, output dimensions, embedding types, and input types. This endpoint works with any embedding model (including multimodal models).

#### Cohere Embed API request parameters

| Parameter | Type | Required | Description |
| --------- | ---- | -------- | ----------- |
| `model` | string | Yes | Model name |
| `input_type` | string | No | Prompt prefix key (model-dependent, see below) |
| `texts` | list[string] | No | Text inputs (use one of `texts`, `images`, or `inputs`) |
| `images` | list[string] | No | Base64 data URI images |
| `inputs` | list[object] | No | Mixed text and image content objects |
| `embedding_types` | list[string] | No | Output types (default: `["float"]`) |
| `output_dimension` | int | No | Truncate embeddings to this dimension (Matryoshka) |
| `truncate` | string | No | `END`, `START`, or `NONE` (default: `END`) |

#### Text embedding

```bash
curl -X POST "http://localhost:8000/v2/embed" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Snowflake/snowflake-arctic-embed-m-v1.5",
    "input_type": "query",
    "texts": ["Hello world", "How are you?"],
    "embedding_types": ["float"]
  }'
```

??? console "Response"

    ```json
    {
      "id": "embd-...",
      "embeddings": {
        "float": [
          [0.012, -0.034, ...],
          [0.056, 0.078, ...]
        ]
      },
      "texts": ["Hello world", "How are you?"],
      "meta": {
        "api_version": {"version": "2"},
        "billed_units": {"input_tokens": 12}
      }
    }
    ```

#### Mixed text and image inputs

For multimodal models, you can embed images by passing base64 data URIs. The `inputs` field accepts a list of objects with mixed text and image content:

```bash
curl -X POST "http://localhost:8000/v2/embed" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "google/siglip-so400m-patch14-384",
    "inputs": [
      {
        "content": [
          {"type": "text", "text": "A photo of a cat"},
          {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBOR..."}}
        ]
      }
    ],
    "embedding_types": ["float"]
  }'
```

#### Embedding types

The `embedding_types` parameter controls the output format. Multiple types can be requested in a single call:

| Type | Description |
| ---- | ----------- |
| `float` | Raw float32 embeddings (default) |
| `binary` | Bit-packed signed binary |
| `ubinary` | Bit-packed unsigned binary |
| `base64` | Little-endian float32 encoded as base64 |

```bash
curl -X POST "http://localhost:8000/v2/embed" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Snowflake/snowflake-arctic-embed-m-v1.5",
    "input_type": "query",
    "texts": ["What is machine learning?"],
    "embedding_types": ["float", "binary"]
  }'
```

??? console "Response"

    ```json
    {
      "id": "embd-...",
      "embeddings": {
        "float": [[0.012, -0.034, ...]],
        "binary": [[42, -117, ...]]
      },
      "texts": ["What is machine learning?"],
      "meta": {
        "api_version": {"version": "2"},
        "billed_units": {"input_tokens": 8}
      }
    }
    ```

#### Truncation

The `truncate` parameter controls how inputs exceeding the model's maximum sequence length are handled:

| Value | Behavior |
| ----- | --------- |
| `END` (default) | Keep the first tokens, drop the end |
| `START` | Keep the last tokens, drop the beginning |
| `NONE` | Return an error if the input is too long |

#### Input type and prompt prefixes

The `input_type` field selects a prompt prefix to prepend to each text input. The available values
depend on the model:

- **Models with `task_instructions` in `config.json`**: The keys from the `task_instructions` dict are
  the valid `input_type` values and the corresponding value is prepended to each text.
- **Models with `config_sentence_transformers.json` prompts**: The keys from the `prompts` dict are
  the valid `input_type` values. For example, `Snowflake/snowflake-arctic-embed-xs` defines `"query"`,
  so setting `input_type: "query"` prepends `"Represent this sentence for searching relevant passages: "`.
- **Other models**: `input_type` is not accepted and will raise a validation error if passed.

## More examples

More examples can be found here: [examples/pooling/embed](../../../examples/pooling/embed)

## Features

### Supported Features

#### Enable/disable normalize

You can enable or disable normalize via `use_activation`.

#### Matryoshka Embeddings

[Matryoshka Embeddings](https://sbert.net/examples/sentence_transformer/training/matryoshka/README.html#matryoshka-embeddings) or [Matryoshka Representation Learning (MRL)](https://arxiv.org/abs/2205.13147) is a technique used in training embedding models. It allows users to trade off between performance and cost.

!!! warning
    Not all embedding models are trained using Matryoshka Representation Learning. To avoid misuse of the `dimensions` parameter, vLLM returns an error for requests that attempt to change the output dimension of models that do not support Matryoshka Embeddings.

    For example, setting `dimensions` parameter while using the `BAAI/bge-m3` model will result in the following error.

    ```json
    {"object":"error","message":"Model \"BAAI/bge-m3\" does not support matryoshka representation, changing output dimensions will lead to poor results.","type":"BadRequestError","param":null,"code":400}
    ```

##### Manually enable Matryoshka Embeddings

There is currently no official interface for specifying support for Matryoshka Embeddings. In vLLM, if `is_matryoshka` is `True` in `config.json`, you can change the output dimension to arbitrary values. Use `matryoshka_dimensions` to control the allowed output dimensions.

For models that support Matryoshka Embeddings but are not recognized by vLLM, manually override the config using `hf_overrides={"is_matryoshka": True}` or `hf_overrides={"matryoshka_dimensions": [<allowed output dimensions>]}` (offline), or `--hf-overrides '{"is_matryoshka": true}'` or `--hf-overrides '{"matryoshka_dimensions": [<allowed output dimensions>]}'` (online).

Here is an example to serve a model with Matryoshka Embeddings enabled.

```bash
vllm serve Snowflake/snowflake-arctic-embed-m-v1.5 --hf-overrides '{"matryoshka_dimensions":[256]}'
```

##### Offline Inference

You can change the output dimensions of embedding models that support Matryoshka Embeddings by using the dimensions parameter in [PoolingParams][vllm.PoolingParams].

```python
from vllm import LLM, PoolingParams

llm = LLM(
    model="jinaai/jina-embeddings-v3",
    runner="pooling",
    trust_remote_code=True,
)
outputs = llm.embed(
    ["Follow the white rabbit."],
    pooling_params=PoolingParams(dimensions=32),
)
print(outputs[0].outputs)
```

A code example can be found here: [examples/pooling/embed/embed_matryoshka_fy_offline.py](../../../examples/pooling/embed/embed_matryoshka_fy_offline.py)

##### Online Inference

Use the following command to start the vLLM server.

```bash
vllm serve jinaai/jina-embeddings-v3 --trust-remote-code
```

You can change the output dimensions of embedding models that support Matryoshka Embeddings by using the dimensions parameter.

```bash
curl http://127.0.0.1:8000/v1/embeddings \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
    "input": "Follow the white rabbit.",
    "model": "jinaai/jina-embeddings-v3",
    "encoding_format": "float",
    "dimensions": 32
  }'
```

Expected output:

```json
{"id":"embd-5c21fc9a5c9d4384a1b021daccaf9f64","object":"list","created":1745476417,"model":"jinaai/jina-embeddings-v3","data":[{"index":0,"object":"embedding","embedding":[-0.3828125,-0.1357421875,0.03759765625,0.125,0.21875,0.09521484375,-0.003662109375,0.1591796875,-0.130859375,-0.0869140625,-0.1982421875,0.1689453125,-0.220703125,0.1728515625,-0.2275390625,-0.0712890625,-0.162109375,-0.283203125,-0.055419921875,-0.0693359375,0.031982421875,-0.04052734375,-0.2734375,0.1826171875,-0.091796875,0.220703125,0.37890625,-0.0888671875,-0.12890625,-0.021484375,-0.0091552734375,0.23046875]}],"usage":{"prompt_tokens":8,"total_tokens":8,"completion_tokens":0,"prompt_tokens_details":null}}
```

An OpenAI client example can be found here: [examples/pooling/embed/openai_embedding_matryoshka_fy_client.py](../../../examples/pooling/embed/openai_embedding_matryoshka_fy_client.py)

### Removed Features

#### Remove `normalize` from PoolingParams

We have already removed `normalize` from PoolingParams, use `use_activation` instead.
