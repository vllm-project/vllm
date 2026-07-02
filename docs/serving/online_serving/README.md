# Online Serving

vLLM provides an HTTP server that is compatible with many interfaces!

## OpenAI-Compatible Server

We currently support the following OpenAI APIs:

- [Completions API](./openai_compatible_server.md#completions-api) (`/v1/completions`)
    - Only applicable to [text generation models](../../models/generative_models.md).
    - *Note: `suffix` parameter is not supported.*
- [Chat Completions API](./openai_compatible_server.md#chat-api) (`/v1/chat/completions`)
    - Only applicable to [text generation models](../../models/generative_models.md) with a [chat template](./openai_compatible_server.md#chat-template).
    - *Note: `user` parameter is ignored.*
    - *Note:* Setting the `parallel_tool_calls` parameter to `false` ensures vLLM only returns zero or one tool call per request. Setting it to `true` (the default) allows returning more than one tool call per request. There is no guarantee more than one tool call will be returned if this is set to `true`, as that behavior is model dependent and not all models are designed to support parallel tool calls.
- [Chat Completions batch API](./openai_compatible_server.md#chat-api) (`/v1/chat/completions/batch`)
- [Responses API](./openai_compatible_server.md#responses-api) (`/v1/responses`, `/v1/responses/{response_id}`, `/v1/responses/{response_id}/cancel`)
    - Only applicable to [text generation models](../../models/generative_models.md).
- [Embeddings API](../../models/pooling_models/embed.md#openai-compatible-embeddings-api) (`/v1/embeddings`)
    - Only applicable to [embedding models](../../models/pooling_models/embed.md).
- [Transcriptions API](./speech_to_text.md#transcriptions-api) (`/v1/audio/transcriptions`)
    - Only applicable to [Automatic Speech Recognition (ASR) models](../../models/supported_models.md#transcription).
- [Translation API](./speech_to_text.md#translations-api) (`/v1/audio/translations`)
    - Only applicable to [Automatic Speech Recognition (ASR) models](../../models/supported_models.md#transcription).

## Anthropic APIs

- Anthropic messages API (`/v1/messages`, `/v1/messages/count_tokens`)

## Cohere APIs

- [Cohere Embed API](../../models/pooling_models/embed.md#cohere-embed-api) (`/v2/embed`)
    - Compatible with [Cohere's Embed API](https://docs.cohere.com/reference/embed)
    - Works with any [embedding model](../../models/pooling_models/embed.md#supported-models), including multimodal models.
- [Cohere Rerank API](../../models/pooling_models/scoring.md#rerank-api) (`/rerank`, `/v1/rerank`, `/v2/rerank`)
    - Implements [Jina AI's v1 rerank API](https://jina.ai/reranker/)
    - compatible with [Cohere's v1 & v2 rerank APIs](https://docs.cohere.com/v2/reference/rerank)

## Pooling APIs

For further details on pooling models, please refer to [this page](../../models/pooling_models/README.md).

- [Classification Usages](../../models/pooling_models/classify.md)
    - [Classification API](../../models/pooling_models/classify.md#online-serving) (`/classify`)
    - Only applicable to [classification models](../../models/pooling_models/classify.md).
- [Embedding Usages](../../models/pooling_models/embed.md)
    - [Cohere Embed API](../../models/pooling_models/embed.md#cohere-embed-api) (`/v2/embed`)
    - [OpenAI-compatible Embeddings API](../../models/pooling_models/embed.md#openai-compatible-embeddings-api) (`/v1/embeddings`)
    - Only applicable to [embedding models](../../models/pooling_models/embed.md).
- [Scoring Usages](../../models/pooling_models/scoring.md)
    - [Score API](../../models/pooling_models/scoring.md#score-api) (`/score`, `/v1/score`)
    - [Cohere Rerank API](../../models/pooling_models/scoring.md#rerank-api) (`/rerank`, `/v1/rerank`, `/v2/rerank`)
    - Applicable to [score models](../../models/pooling_models/scoring.md) (cross-encoder, bi-encoder, late-interaction).
- [Pooling API](../../models/pooling_models/README.md#pooling-api) (`/pooling`)
    - Applicable to all [pooling models](../../models/pooling_models/README.md).

## Speech to Text APIs

For further details on speech to text, please refer to [this page](speech_to_text.md).

- [Transcriptions API](./speech_to_text.md#transcriptions-api) (`/v1/audio/transcriptions`)
    - Only applicable to [Automatic Speech Recognition (ASR) models](../../models/supported_models.md#transcription).
- [Translation API](./speech_to_text.md#translations-api) (`/v1/audio/translations`)
    - Only applicable to [Automatic Speech Recognition (ASR) models](../../models/supported_models.md#transcription).
- [Realtime API](./speech_to_text.md#realtime-api) (`/v1/realtime`)
    - Only applicable to [Automatic Speech Recognition (ASR) models](../../models/supported_models.md#realtime-transcription).

## Custom APIs

- [Classification API](../../models/pooling_models/classify.md#classification-api) (`/classify`)
    - Only applicable to [classification models](../../models/pooling_models/classify.md).
- [Score API](../../models/pooling_models/scoring.md#score-api) (`/score`, `/v1/score`)
    - Applicable to [score models](../../models/pooling_models/scoring.md) (cross-encoder, bi-encoder, late-interaction).
- [Pooling API](../../models/pooling_models/README.md#pooling-api) (`/pooling`)
    - Applicable to all [pooling models](../../models/pooling_models/README.md).
- [Generative Scoring API](generative_scoring.md#generative-scoring-api) (`/generative_scoring`)
    - Applicable to [CausalLM models](../../models/generative_models.md) (task `"generate"`).
    - Computes next-token probabilities for specified `label_token_ids`.

## Instrumentator APIs

### Basic APIs

- `/version` - Version information
- `/load` - Server load metrics
- `/v1/models` - List available models
- `/health` - Health check

### Metrics APIs

For further details on metrics, please refer to [this page](../../design/metrics.md).

- `/metrics` - Prometheus-compatible metrics HTTP endpoint

### Offline API Documentation

The FastAPI `/docs` endpoint requires an internet connection by default. To enable offline access in air-gapped environments, use the `--enable-offline-docs` flag:

```bash
vllm serve NousResearch/Meta-Llama-3-8B-Instruct --enable-offline-docs
```

### LoRA dynamic loading

LoRA dynamic loading & unloading is enabled in the API server. This should ONLY be used for local development!

- `/v1/load_lora_adapter` - LoRA dynamic loading
- `/v1/unload_lora_adapter` - LoRA dynamic unloading

### Profiling APIs

For further details on profiling vLLM, please refer to [this page](../../contributing/profiling.md).

- `/start_profile` - Start PyTorch profiler
- `/stop_profile` - Stop PyTorch profiler

### SageMaker APIs

- `/ping` - SageMaker health check
- `/invocations` - SageMaker-compatible endpoint (routes to the same inference functions as `/v1` endpoints)

## Scale-Out APIs

### Tokens IN <> Tokens OUT APIs

- `/inference/v1/generate` - Generate completions
- `/abort_requests` - Abort in-flight requests (only when `--tokens-only` is also set)

### Renderer APIs

For further details on renderer APIs, please refer to [this page](renderer.md).

- [Completions Render API](renderer.md) (`/v1/completions/render`)
    - Render completion requests
- [Chat Completions Render API](renderer.md) (`/v1/chat/completions/render`)
    - Render chat completions

### Derenderer APIs

- `/v1/completions/derender` - Derenderer completion requests
- `/v1/chat/completions/derender` - Derenderer chat completion requests

## Tokenize APIs

- `/tokenize` - Tokenize text
- `/detokenize` - Detokenize tokens
- `/tokenizer_info` - Get comprehensive tokenizer information including chat templates and configuration

## Elastic Expert Parallelism (EEP)

- `/scale_elastic_ep` - Trigger scaling operations
- `/is_scaling_elastic_ep` - Check if scaling is in progress

## Server in development mode

When using the flag VLLM_SERVER_DEV_MODE=1, you enable development endpoints.

**SECURITY WARNING: These endpoints should NOT be used in production!**

### Cache Management APIs

- `/reset_prefix_cache` - Reset prefix cache (can disrupt service)
- `/reset_mm_cache` - Reset multimodal cache (can disrupt service)
- `/reset_encoder_cache` - Reset encoder cache (can disrupt service)

### Weight Transfer APIs (RL Training)

For further details on Weight Transfer, please refer to [this page](../../training/weight_transfer/README.md).

- `/pause` - Pause generation (causes denial of service)
- `/resume` - Resume generation
- `/is_paused` - Check if generation is paused
- `/abort_requests` - Abort in-flight requests (all in-flight, or the given `request_ids`) without pausing the scheduler
- `/init_weight_transfer_engine` - Initialize weight transfer engine for RLHF
- `/start_weight_update` - Prepares the inference engine for a weight update.
- `/update_weights` - Update model weights (can alter model behavior)
- `/finish_weight_update` - Finalizes the weight update
- `/get_world_size` - Get distributed world size

### Collective RPC

- `/collective_rpc` - Execute arbitrary RPC methods on the engine (extremely dangerous)

### Server info

- `/server_info` - Get detailed server configuration

### Sleep Mode APIs

For further details on sleep mode, please refer to [this page](../../features/sleep_mode.md).

- `/sleep` - Put engine to sleep (causes denial of service)
- `/wake_up` - Wake engine from sleep
- `/is_sleeping` - Check if engine is sleeping

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

vLLM community provides a set of chat templates for popular models. You can find them under the [examples](../../../examples) directory.

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

## Ray Serve LLM

Ray Serve LLM enables scalable, production-grade serving of the vLLM engine. It integrates tightly with vLLM and extends it with features such as auto-scaling, load balancing, and back-pressure.

Key capabilities:

- Exposes an OpenAI-compatible HTTP API as well as a Pythonic API.
- Scales from a single GPU to a multi-node cluster without code changes.
- Provides observability and autoscaling policies through Ray dashboards and metrics.

The following example shows how to deploy a large model like DeepSeek R1 with Ray Serve LLM: [examples/ray_serving/ray_serve_deepseek.py](../../../examples/ray_serving/ray_serve_deepseek.py).

Learn more about Ray Serve LLM with the official [Ray Serve LLM documentation](https://docs.ray.io/en/latest/serve/llm/index.html).
