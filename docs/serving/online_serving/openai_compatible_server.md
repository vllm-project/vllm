# OpenAI-Compatible Server

vLLM provides an HTTP server that implements OpenAI's [Completions API](https://platform.openai.com/docs/api-reference/completions), [Chat API](https://platform.openai.com/docs/api-reference/chat), and more! This functionality lets you serve models and interact with them using an HTTP client.

## Supported APIs

We currently support the following OpenAI APIs:

- [Completions API](#completions-api) (`/v1/completions`)
    - Only applicable to [text generation models](../../models/generative_models.md).
    - *Note: `suffix` parameter is not supported.*
- [Responses API](#responses-api) (`/v1/responses`)
    - Only applicable to [text generation models](../../models/generative_models.md).
- [Chat Completions API](#chat-api) (`/v1/chat/completions`)
    - Only applicable to [text generation models](../../models/generative_models.md) with a [chat template](../online_serving/README.md#chat-template).
    - *Note: `user` parameter is ignored.*
    - *Note:* Setting the `parallel_tool_calls` parameter to `false` ensures vLLM only returns zero or one tool call per request. Setting it to `true` (the default) allows returning more than one tool call per request. There is no guarantee more than one tool call will be returned if this is set to `true`, as that behavior is model dependent and not all models are designed to support parallel tool calls.
- [Embeddings API](../../models/pooling_models/embed.md#openai-compatible-embeddings-api) (`/v1/embeddings`)
    - Only applicable to [embedding models](../../models/pooling_models/embed.md).
- [Transcriptions API](./speech_to_text.md#transcriptions-api) (`/v1/audio/transcriptions`)
    - Only applicable to [Automatic Speech Recognition (ASR) models](../../models/supported_models.md#transcription).
- [Translation API](./speech_to_text.md#translations-api) (`/v1/audio/translations`)
    - Only applicable to [Automatic Speech Recognition (ASR) models](../../models/supported_models.md#transcription).

## Completions API

In your terminal, you can [install](../../getting_started/installation/README.md) vLLM, then start the server with the [`vllm serve`](../../configuration/serve_args.md) command. (You can also use our [Docker](../../deployment/docker.md) image.)

```bash
vllm serve NousResearch/Meta-Llama-3-8B-Instruct \
  --dtype auto \
  --api-key token-abc123
```

To call the server, in your preferred text editor, create a script that uses an HTTP client. Include any messages that you want to send to the model. Then run that script. Below is an example script using the [official OpenAI Python client](https://github.com/openai/openai-python).

??? code

    ```python
    from openai import OpenAI
    client = OpenAI(
        base_url="http://localhost:8000/v1",
        api_key="token-abc123",
    )

    completion = client.chat.completions.create(
        model="NousResearch/Meta-Llama-3-8B-Instruct",
        messages=[
            {"role": "user", "content": "Hello!"},
        ],
    )

    print(completion.choices[0].message)
    ```

!!! tip
    vLLM supports some parameters that are not supported by OpenAI, `top_k` for example.
    You can pass these parameters to vLLM using the OpenAI client in the `extra_body` parameter of your requests, i.e. `extra_body={"top_k": 50}` for `top_k`.

!!! important
    By default, the server applies `generation_config.json` from the Hugging Face model repository if it exists. This means the default values of certain sampling parameters can be overridden by those recommended by the model creator.

    To disable this behavior, please pass `--generation-config vllm` when launching the server.

## Extra Parameters

vLLM supports a set of parameters that are not part of the OpenAI API.
In order to use them, you can pass them as extra parameters in the OpenAI client.
Or directly merge them into the JSON payload if you are using HTTP call directly.

```python
completion = client.chat.completions.create(
    model="NousResearch/Meta-Llama-3-8B-Instruct",
    messages=[
        {"role": "user", "content": "Classify this sentiment: vLLM is wonderful!"},
    ],
    extra_body={
        "structured_outputs": {"choice": ["positive", "negative"]},
    },
)
```

## Extra HTTP Headers

Only `X-Request-Id` HTTP request header is supported for now. It can be enabled
with `--enable-request-id-headers`.

??? code

    ```python
    completion = client.chat.completions.create(
        model="NousResearch/Meta-Llama-3-8B-Instruct",
        messages=[
            {"role": "user", "content": "Classify this sentiment: vLLM is wonderful!"},
        ],
        extra_headers={
            "x-request-id": "sentiment-classification-00001",
        },
    )
    print(completion._request_id)

    completion = client.completions.create(
        model="NousResearch/Meta-Llama-3-8B-Instruct",
        prompt="A robot may not injure a human being",
        extra_headers={
            "x-request-id": "completion-test",
        },
    )
    print(completion._request_id)
    ```

## API Reference

### Completions API

Our Completions API is compatible with [OpenAI's Completions API](https://platform.openai.com/docs/api-reference/completions);
you can use the [official OpenAI Python client](https://github.com/openai/openai-python) to interact with it.

Code example: [examples/basic/online_serving/openai_completion_client.py](../../../examples/basic/online_serving/openai_completion_client.py)

#### Extra parameters

The following [sampling parameters](../../api/README.md#inference-parameters) are supported.

??? code

    ```python
    --8<-- "vllm/entrypoints/openai/completion/protocol.py:completion-sampling-params"
    ```

The following extra parameters are supported:

??? code

    ```python
    --8<-- "vllm/entrypoints/openai/completion/protocol.py:completion-extra-params"
    ```

### Chat API

Our Chat API is compatible with [OpenAI's Chat Completions API](https://platform.openai.com/docs/api-reference/chat);
you can use the [official OpenAI Python client](https://github.com/openai/openai-python) to interact with it.

We support both [Vision](https://platform.openai.com/docs/guides/vision)- and
[Audio](https://platform.openai.com/docs/guides/audio?audio-generation-quickstart-example=audio-in)-related parameters;
see our [Multimodal Inputs](../../features/multimodal_inputs.md) guide for more information.

- *Note: `image_url.detail` parameter is not supported.*

Code example: [examples/basic/online_serving/openai_chat_completion_client.py](../../../examples/basic/online_serving/openai_chat_completion_client.py)

#### Extra parameters

The following [sampling parameters](../../api/README.md#inference-parameters) are supported.

??? code

    ```python
    --8<-- "vllm/entrypoints/openai/chat_completion/protocol.py:chat-completion-sampling-params"
    ```

The following extra parameters are supported:

??? code

    ```python
    --8<-- "vllm/entrypoints/openai/chat_completion/protocol.py:chat-completion-extra-params"
    ```

### Responses API

Our Responses API is compatible with [OpenAI's Responses API](https://platform.openai.com/docs/api-reference/responses);
you can use the [official OpenAI Python client](https://github.com/openai/openai-python) to interact with it.

Code example: [examples/tool_calling/openai_responses_client_with_tools.py](../../../examples/tool_calling/openai_responses_client_with_tools.py)

#### Extra parameters

The following extra parameters in the request object are supported:

??? code

    ```python
    --8<-- "vllm/entrypoints/openai/responses/protocol.py:responses-extra-params"
    ```

The following extra parameters in the response object are supported:

??? code

    ```python
    --8<-- "vllm/entrypoints/openai/responses/protocol.py:responses-response-extra-params"
    ```

### llm_sign response signatures (experimental)

!!! warning
    `llm_sign` support is experimental and subject to change. The `llm_sign`
    protocol is currently in beta; breaking changes may happen at any time,
    and production use is not recommended.

vLLM can attach `llm_sign` metadata to OpenAI-compatible Chat Completions and
Responses API responses. The metadata lets downstream clients verify that the
visible response body was signed by the provider certificate key instead of
being silently modified by an intermediate relay.

This integration is disabled by default. When it is disabled, vLLM does not add
the `llm_sign` field.

Install `llm-sign` alongside vLLM and point vLLM at the TLS certificate and
private key used to identify the provider:

```bash
pip install llm-sign

export VLLM_LLM_SIGN_ENABLED=1
export VLLM_LLM_SIGN_CERTFILE=/path/to/fullchain.pem
export VLLM_LLM_SIGN_KEYFILE=/path/to/privkey.pem

vllm serve meta-llama/Meta-Llama-3-8B-Instruct
```

When enabled, vLLM signs non-streaming `/v1/chat/completions` responses and
final `/v1/responses` response objects, then attaches the artifact under the
`llm_sign` field. vLLM-only request and response extensions are excluded from
the signed OpenAI-compatible payload.

Clients can verify signed responses with the `llm_sign` client helpers:

```python
from llm_sign.client import (
    verify_openai_response,
    verify_openai_responses_response,
)

chat_result = verify_openai_response(chat_response)
responses_result = verify_openai_responses_response(responses_response)
```
