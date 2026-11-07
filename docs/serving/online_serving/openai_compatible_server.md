# OpenAI-Compatible Server

vLLM provides an HTTP server that implements OpenAI's [Completions API](https://platform.openai.com/docs/api-reference/completions), [Chat API](https://platform.openai.com/docs/api-reference/chat), and more! This functionality lets you serve models and interact with them using an HTTP client.

!!! warning "API key authentication does not protect every endpoint"
    The `--api-key` option (or `VLLM_API_KEY` environment variable) only
    authenticates requests to endpoints under the `/v1`, `/v2`, `/inference`, and
    `/cohere` path prefixes. Other endpoints on the same HTTP server are
    **not** authenticated — most notably `/invocations`, which exposes the same
    inference capabilities as the `/v1` endpoints. Do not rely on `--api-key`
    alone to secure vLLM. See
    [API Key Authentication Limitations](../../usage/security.md#api-key-authentication-limitations)
    for the full list of protected and unprotected endpoints and recommended
    hardening, such as deploying behind a reverse proxy.

## Supported APIs

We currently support the following OpenAI APIs:

- [Completions API](#completions-api) (`/v1/completions`)
    - Only applicable to [text generation models](../../models/generative_models.md).
    - *Note: `suffix` parameter is not supported.*
- [Chat Completions API](#chat-api) (`/v1/chat/completions`)
    - Only applicable to [text generation models](../../models/generative_models.md) with a [chat template](../online_serving/README.md#chat-template).
    - *Note: `user` parameter is ignored.*
    - *Note:* Setting the `parallel_tool_calls` parameter to `false` ensures vLLM only returns zero or one tool call per request. Setting it to `true` (the default) allows returning more than one tool call per request. There is no guarantee more than one tool call will be returned if this is set to `true`, as that behavior is model dependent and not all models are designed to support parallel tool calls.
- [Chat Completions batch API](#chat-api) (`/v1/chat/completions/batch`)
- [Responses API](#responses-api) (`/v1/responses`, `/v1/responses/{response_id}`, `/v1/responses/{response_id}/cancel`)
    - Only applicable to [text generation models](../../models/generative_models.md).
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

The `X-Request-Id` HTTP request header can be enabled with
`--enable-request-id-headers`.

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

The Completions, Chat Completions, and Responses APIs also support the
`X-Vllm-Priority` request header. Its value must be an integer and overrides
the `priority` value in the JSON request body. Non-zero priorities require the
server to use priority scheduling.

```python
completion = client.chat.completions.create(
    model="NousResearch/Meta-Llama-3-8B-Instruct",
    messages=[{"role": "user", "content": "Hello!"}],
    extra_headers={"X-Vllm-Priority": "-10"},
)
```

## Reverse proxies

Both frontends use `FORWARDED_ALLOW_IPS` to decide which proxies can supply
the client address through `X-Forwarded-For`. Set it to a comma-separated
list of proxy IPs or CIDR networks:

```bash
FORWARDED_ALLOW_IPS="127.0.0.1,10.0.0.0/8" VLLM_USE_RUST_FRONTEND=1 \
  vllm serve Qwen/Qwen3-0.6B
```

The Rust frontend defaults to `127.0.0.1,::1`, as does Uvicorn 0.53 and later.
An empty value disables proxy trust. Use `*` only if all connections come
through trusted proxies that remove client-supplied `X-Forwarded-For` headers
before they set their own. If a proxy appends to the header, list its IP
instead. The [Nginx example](../../deployment/nginx.md) uses this append behavior.

The Rust frontend combines repeated `X-Forwarded-For` fields in order and selects the
rightmost untrusted IP, or the leftmost IP if all are trusted. IPv4 and IPv6
addresses may include a port; use brackets for IPv6 with a port. An explicit
port is preserved; otherwise it is `0`. An invalid selected address or port
leaves the socket address unchanged.

For Rust, write networks with a prefix length, such as `10.0.0.0/8`.
Dotted netmasks such as `10.0.0.0/255.0.0.0` are not supported.

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

With `--enable-prompt-tokens-details`, `/v1/chat/completions/batch` sums `cached_tokens`, `created_cache_tokens` and per-modality `multimodal_tokens` across conversations in `usage.prompt_tokens_details`; an unreported cache count makes that total `null`, and disabling the flag leaves the details `null`.

With `--reasoning-parser`, `/v1/chat/completions/batch` sums `reasoning_tokens` across conversations in `usage.completion_tokens_details`, as `/v1/chat/completions` does for one conversation, also when `include_reasoning` is `false`.

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

To get prompt token IDs before generation, use
[`/v1/responses/render`](renderer.md#get-responses-prompt-token-ids). It preprocesses
a self-contained Responses request without inference and returns the prompt
token IDs, so a separate `/tokenize` call is not needed.

#### Sampling parameters

In addition to the OpenAI request fields, the Responses API forwards vLLM
[sampling parameters](../../api/README.md#inference-parameters) such as `top_k`,
`min_p`, `repetition_penalty`, `seed`, `stop`, and `ignore_eos` to the engine.

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
