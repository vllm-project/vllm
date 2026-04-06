# OpenAI-Compatible Server

vLLM provides an HTTP server that implements OpenAI's [Completions API](https://platform.openai.com/docs/api-reference/completions), [Chat API](https://platform.openai.com/docs/api-reference/chat), and more! This functionality lets you serve models and interact with them using an HTTP client.

In your terminal, you can [install](../getting_started/installation/README.md) vLLM, then start the server with the [`vllm serve`](../configuration/serve_args.md) command. (You can also use our [Docker](../deployment/docker.md) image.)

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

## Supported APIs

We currently support the following OpenAI APIs:

- [Completions API](#completions-api) (`/v1/completions`)
    - Only applicable to [text generation models](../models/generative_models.md).
    - *Note: `suffix` parameter is not supported.*
- [Responses API](#responses-api) (`/v1/responses`)
    - Only applicable to [text generation models](../models/generative_models.md).
- [Chat Completions API](#chat-api) (`/v1/chat/completions`)
    - Only applicable to [text generation models](../models/generative_models.md) with a [chat template](../serving/openai_compatible_server.md#chat-template).
    - *Note: `user` parameter is ignored.*
    - *Note:* Setting the `parallel_tool_calls` parameter to `false` ensures vLLM only returns zero or one tool call per request. Setting it to `true` (the default) allows returning more than one tool call per request. There is no guarantee more than one tool call will be returned if this is set to `true`, as that behavior is model dependent and not all models are designed to support parallel tool calls.
- [Embeddings API](../models/pooling_models/embed.md#openai-compatible-embeddings-api) (`/v1/embeddings`)
    - Only applicable to [embedding models](../models/pooling_models/embed.md).
- [Transcriptions API](#transcriptions-api) (`/v1/audio/transcriptions`)
    - Only applicable to [Automatic Speech Recognition (ASR) models](../models/supported_models.md#transcription).
- [Translation API](#translations-api) (`/v1/audio/translations`)
    - Only applicable to [Automatic Speech Recognition (ASR) models](../models/supported_models.md#transcription).
- [Realtime API](#realtime-api) (`/v1/realtime`)
    - Only applicable to [Automatic Speech Recognition (ASR) models](../models/supported_models.md#transcription).

In addition, we have the following custom APIs:

- [Tokenizer API](#tokenizer-api) (`/tokenize`, `/detokenize`)
    - Applicable to any model with a tokenizer.
- [pooling API](../models/pooling_models/README.md#pooling-api) (`/pooling`)
    - Applicable to all [pooling models](../models/pooling_models/README.md).
- [Classification API](../models/pooling_models/classify.md#classification-api) (`/classify`)
    - Only applicable to [classification models](../models/pooling_models/classify.md).
- [Cohere Embed API](../models/pooling_models/embed.md#cohere-embed-api) (`/v2/embed`)
    - Compatible with [Cohere's Embed API](https://docs.cohere.com/reference/embed)
    - Works with any [embedding model](../models/pooling_models/embed.md#supported-models), including multimodal models.
- [Score API](../models/pooling_models/scoring.md#score-api) (`/score`, `/v1/score`)
    - Applicable to [score models](../models/pooling_models/scoring.md) (cross-encoder, bi-encoder, late-interaction).
- [Generative Scoring API](#generative-scoring-api) (`/generative_scoring`)
    - Applicable to [CausalLM models](../models/generative_models.md) (task `"generate"`).
    - Computes next-token probabilities for specified `label_token_ids`.
- [Rerank API](../models/pooling_models/scoring.md#rerank-api) (`/rerank`, `/v1/rerank`, `/v2/rerank`)
    - Implements [Jina AI's v1 rerank API](https://jina.ai/reranker/)
    - Also compatible with [Cohere's v1 & v2 rerank APIs](https://docs.cohere.com/v2/reference/rerank)
    - Jina and Cohere's APIs are very similar; Jina's includes extra information in the rerank endpoint's response.

## Chat Template

In order for the language model to support chat protocol, vLLM requires the model to include
a chat template in its tokenizer configuration. The chat template is a Jinja2 template that
specifies how roles, messages, and other chat-specific tokens are encoded in the input.

An example chat template for `NousResearch/Meta-Llama-3-8B-Instruct` can be found [here](https://llama.com/docs/model-cards-and-prompt-formats/meta-llama-3/#prompt-template-for-meta-llama-3)

Some models do not provide a chat template even though they are instruction/chat fine-tuned. For those models,
you can manually specify their chat template in the `--chat-template` parameter with the file path to the chat
template, or the template in string form. Without a chat template, the server will not be able to process chat
and all chat requests will error.

```bash
vllm serve <model> --chat-template ./path-to-chat-template.jinja
```

vLLM community provides a set of chat templates for popular models. You can find them under the [examples](../../examples) directory.

With the inclusion of multi-modal chat APIs, the OpenAI spec now accepts chat messages in a new format which specifies
both a `type` and a `text` field. An example is provided below:

```python
completion = client.chat.completions.create(
    model="NousResearch/Meta-Llama-3-8B-Instruct",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Classify this sentiment: vLLM is wonderful!"},
            ],
        },
    ],
)
```

Most chat templates for LLMs expect the `content` field to be a string, but there are some newer models like
`meta-llama/Llama-Guard-3-1B` that expect the content to be formatted according to the OpenAI schema in the
request. vLLM provides best-effort support to detect this automatically, which is logged as a string like
*"Detected the chat template content format to be..."*, and internally converts incoming requests to match
the detected format, which can be one of:

- `"string"`: A string.
    - Example: `"Hello world"`
- `"openai"`: A list of dictionaries, similar to OpenAI schema.
    - Example: `[{"type": "text", "text": "Hello world!"}]`

If the result is not what you expect, you can set the `--chat-template-content-format` CLI argument
to override which format to use.

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

## Offline API Documentation

The FastAPI `/docs` endpoint requires an internet connection by default. To enable offline access in air-gapped environments, use the `--enable-offline-docs` flag:

```bash
vllm serve NousResearch/Meta-Llama-3-8B-Instruct --enable-offline-docs
```

## API Reference

### Completions API

Our Completions API is compatible with [OpenAI's Completions API](https://platform.openai.com/docs/api-reference/completions);
you can use the [official OpenAI Python client](https://github.com/openai/openai-python) to interact with it.

Code example: [examples/basic/online_serving/openai_completion_client.py](../../examples/basic/online_serving/openai_completion_client.py)

#### Extra parameters

The following [sampling parameters](../api/README.md#inference-parameters) are supported.

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
see our [Multimodal Inputs](../features/multimodal_inputs.md) guide for more information.

- *Note: `image_url.detail` parameter is not supported.*

Code example: [examples/basic/online_serving/openai_chat_completion_client.py](../../examples/basic/online_serving/openai_chat_completion_client.py)

#### Extra parameters

The following [sampling parameters](../api/README.md#inference-parameters) are supported.

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

Code example: [examples/online_serving/openai_responses_client_with_tools.py](../../examples/online_serving/openai_responses_client_with_tools.py)

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

### Transcriptions API

Our Transcriptions API is compatible with [OpenAI's Transcriptions API](https://platform.openai.com/docs/api-reference/audio/createTranscription);
you can use the [official OpenAI Python client](https://github.com/openai/openai-python) to interact with it.

!!! note
    To use the Transcriptions API, please install with extra audio dependencies using `pip install vllm[audio]`.

Code example: [examples/online_serving/openai_transcription_client.py](../../examples/online_serving/openai_transcription_client.py)

NOTE: beam search is currently supported in the transcriptions endpoint for encoder-decoder multimodal models, e.g., whisper, but highly inefficient as work for handling the encoder/decoder cache is actively ongoing. This is an active point of ongoing optimization and will be handled properly in the very near future.

#### API Enforced Limits

Set the maximum audio file size (in MB) that VLLM will accept, via the
`VLLM_MAX_AUDIO_CLIP_FILESIZE_MB` environment variable. Default is 25 MB.

#### Uploading Audio Files

The Transcriptions API supports uploading audio files in various formats including FLAC, MP3, MP4, MPEG, MPGA, M4A, OGG, WAV, and WEBM.

**Using OpenAI Python Client:**

??? code

    ```python
    from openai import OpenAI

    client = OpenAI(
        base_url="http://localhost:8000/v1",
        api_key="token-abc123",
    )

    # Upload audio file from disk
    with open("audio.mp3", "rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            model="openai/whisper-large-v3-turbo",
            file=audio_file,
            language="en",
            response_format="verbose_json",
        )

    print(transcription.text)
    ```

**Using curl with multipart/form-data:**

??? code

    ```bash
    curl -X POST "http://localhost:8000/v1/audio/transcriptions" \
      -H "Authorization: Bearer token-abc123" \
      -F "file=@audio.mp3" \
      -F "model=openai/whisper-large-v3-turbo" \
      -F "language=en" \
      -F "response_format=verbose_json"
    ```

**Supported Parameters:**

- `file`: The audio file to transcribe (required)
- `model`: The model to use for transcription (required)
- `language`: The language code (e.g., "en", "zh") (optional)
- `prompt`: Optional text to guide the transcription style (optional)
- `response_format`: Format of the response ("json", "text") (optional)
- `temperature`: Sampling temperature between 0 and 1 (optional)

For the complete list of supported parameters including sampling parameters and vLLM extensions, see the [protocol definitions](https://github.com/vllm-project/vllm/blob/main/vllm/entrypoints/openai/protocol.py#L2182).

**Response Format:**

For `verbose_json` response format:

??? code

    ```json
    {
      "text": "Hello, this is a transcription of the audio file.",
      "language": "en",
      "duration": 5.42,
      "segments": [
        {
          "id": 0,
          "seek": 0,
          "start": 0.0,
          "end": 2.5,
          "text": "Hello, this is a transcription",
          "tokens": [50364, 938, 428, 307, 275, 28347],
          "temperature": 0.0,
          "avg_logprob": -0.245,
          "compression_ratio": 1.235,
          "no_speech_prob": 0.012
        }
      ]
    }
    ```
Currently “verbose_json” response format doesn’t support no_speech_prob.

#### Extra Parameters

The following [sampling parameters](../api/README.md#inference-parameters) are supported.

??? code

    ```python
    --8<-- "vllm/entrypoints/openai/speech_to_text/protocol.py:transcription-sampling-params"
    ```

The following extra parameters are supported:

??? code

    ```python
    --8<-- "vllm/entrypoints/openai/speech_to_text/protocol.py:transcription-extra-params"
    ```

### Translations API

Our Translation API is compatible with [OpenAI's Translations API](https://platform.openai.com/docs/api-reference/audio/createTranslation);
you can use the [official OpenAI Python client](https://github.com/openai/openai-python) to interact with it.
Whisper models can translate audio from one of the 55 non-English supported languages into English.
Please mind that the popular `openai/whisper-large-v3-turbo` model does not support translating.

!!! note
    To use the Translation API, please install with extra audio dependencies using `pip install vllm[audio]`.

Code example: [examples/online_serving/openai_translation_client.py](../../examples/online_serving/openai_translation_client.py)

#### Extra Parameters

The following [sampling parameters](../api/README.md#inference-parameters) are supported.

```python
--8<-- "vllm/entrypoints/openai/speech_to_text/protocol.py:translation-sampling-params"
```

The following extra parameters are supported:

```python
--8<-- "vllm/entrypoints/openai/speech_to_text/protocol.py:translation-extra-params"
```

### Realtime API

The Realtime API provides WebSocket-based streaming audio transcription, allowing real-time speech-to-text as audio is being recorded.

!!! note
    To use the Realtime API, please install with extra audio dependencies using `uv pip install vllm[audio]`.

#### Audio Format

Audio must be sent as base64-encoded PCM16 audio at 16kHz sample rate, mono channel.

#### Protocol Overview

1. Client connects to `ws://host/v1/realtime`
2. Server sends `session.created` event
3. Client optionally sends `session.update` with model/params
4. Client sends `input_audio_buffer.commit` when ready
5. Client sends `input_audio_buffer.append` events with base64 PCM16 chunks
6. Server sends `transcription.delta` events with incremental text
7. Server sends `transcription.done` with final text + usage
8. Repeat from step 5 for next utterance
9. Optionally, client sends input_audio_buffer.commit with final=True
    to signal audio input is finished. Useful when streaming audio files

#### Client → Server Events

| Event | Description |
| ----- | ----------- |
| `input_audio_buffer.append` | Send base64-encoded audio chunk: `{"type": "input_audio_buffer.append", "audio": "<base64>"}` |
| `input_audio_buffer.commit` | Trigger transcription processing or end: `{"type": "input_audio_buffer.commit", "final": bool}` |
| `session.update` | Configure session: `{"type": "session.update", "model": "model-name"}` |

#### Server → Client Events

| Event | Description |
| ----- | ----------- |
| `session.created` | Connection established with session ID and timestamp |
| `transcription.delta` | Incremental transcription text: `{"type": "transcription.delta", "delta": "text"}` |
| `transcription.done` | Final transcription with usage stats |
| `error` | Error notification with message and optional code |

#### Example Clients

- [openai_realtime_client.py](https://github.com/vllm-project/vllm/tree/main/examples/online_serving/openai_realtime_client.py) - Upload and transcribe an audio file
- [openai_realtime_microphone_client.py](https://github.com/vllm-project/vllm/tree/main/examples/online_serving/openai_realtime_microphone_client.py) - Gradio demo for live microphone transcription

### Tokenizer API

Our Tokenizer API is a simple wrapper over [HuggingFace-style tokenizers](https://huggingface.co/docs/transformers/en/main_classes/tokenizer).
It consists of two endpoints:

- `/tokenize` corresponds to calling `tokenizer.encode()`.
- `/detokenize` corresponds to calling `tokenizer.decode()`.

### Score API

#### Score Template

Some scoring models require a specific prompt format to work correctly. You can specify a custom score template using the `--chat-template` parameter (see [Chat Template](#chat-template)).

Score templates are supported for **cross-encoder** models only. If you are using an **embedding** model for scoring, vLLM does not apply a score template.

Like chat templates, the score template receives a `messages` list. For scoring, each message has a `role` attribute—either `"query"` or `"document"`. For the usual kind of point-wise cross-encoder, you can expect exactly two messages: one query and one document. To access the query and document content, use Jinja's `selectattr` filter:

- **Query**: `{{ (messages | selectattr("role", "eq", "query") | first).content }}`
- **Document**: `{{ (messages | selectattr("role", "eq", "document") | first).content }}`

This approach is more robust than index-based access (`messages[0]`, `messages[1]`) because it selects messages by their semantic role. It also avoids assumptions about message ordering if additional message types are added to `messages` in the future.

Example template file: [examples/pooling/score/template/nemotron-rerank.jinja](../../examples/pooling/score/template/nemotron-rerank.jinja)

### Generative Scoring API

The `/generative_scoring` endpoint uses a CausalLM model (e.g., Llama, Qwen, Mistral) to compute the probability of specified token IDs appearing as the next token. Each item (document) is concatenated with the query to form a prompt, and the model predicts how likely each label token is as the next token after that prompt. This lets you score items against a query — for example, asking "Is this the capital of France?" and scoring each city by how likely the model is to answer "Yes".

This endpoint is automatically available when the server is started with a generative model (task `"generate"`). It is separate from the pooling-based [Score API](#score-api), which uses cross-encoder, bi-encoder, or late-interaction models.

**Requirements:**

- The `label_token_ids` parameter is **required** and must contain **at least 1 token ID**.
- When 2 label tokens are provided, the score equals `P(label_token_ids[0]) / (P(label_token_ids[0]) + P(label_token_ids[1]))` (softmax over the two labels).
- When more labels are provided, the score is the softmax-normalized probability of the first label token across all label tokens.

#### Example

```bash
curl -X POST http://localhost:8000/generative_scoring \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "query": "Is this city the capital of France?",
    "items": ["Paris", "London", "Berlin"],
    "label_token_ids": [9454, 2753]
  }'
```

Here, each item is appended to the query to form prompts like `"Is this city the capital of France? Paris"`, `"... London"`, etc. The model then predicts the next token, and the score reflects the probability of "Yes" (token 9454) vs "No" (token 2753).

??? console "Response"

    ```json
    {
      "id": "generative-scoring-abc123",
      "object": "list",
      "created": 1234567890,
      "model": "Qwen/Qwen3-0.6B",
      "data": [
        {"index": 0, "object": "score", "score": 0.95},
        {"index": 1, "object": "score", "score": 0.12},
        {"index": 2, "object": "score", "score": 0.08}
      ],
      "usage": {"prompt_tokens": 45, "total_tokens": 48, "completion_tokens": 3}
    }
    ```

#### How it works

1. **Prompt Construction**: For each item, builds `prompt = query + item` (or `item + query` if `item_first=true`)
2. **Forward Pass**: Runs the model on each prompt to get next-token logits
3. **Probability Extraction**: Extracts logprobs for the specified `label_token_ids`
4. **Softmax Normalization**: Applies softmax over only the label tokens (when `apply_softmax=true`)
5. **Score**: Returns the normalized probability of the first label token

#### Finding Token IDs

To find the token IDs for your labels, use the tokenizer:

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
yes_id = tokenizer.encode("Yes", add_special_tokens=False)[0]
no_id = tokenizer.encode("No", add_special_tokens=False)[0]
print(f"Yes: {yes_id}, No: {no_id}")
```

## Ray Serve LLM

Ray Serve LLM enables scalable, production-grade serving of the vLLM engine. It integrates tightly with vLLM and extends it with features such as auto-scaling, load balancing, and back-pressure.

Key capabilities:

- Exposes an OpenAI-compatible HTTP API as well as a Pythonic API.
- Scales from a single GPU to a multi-node cluster without code changes.
- Provides observability and autoscaling policies through Ray dashboards and metrics.

The following example shows how to deploy a large model like DeepSeek R1 with Ray Serve LLM: [examples/online_serving/ray_serve_deepseek.py](../../examples/online_serving/ray_serve_deepseek.py).

Learn more about Ray Serve LLM with the official [Ray Serve LLM documentation](https://docs.ray.io/en/latest/serve/llm/index.html).
