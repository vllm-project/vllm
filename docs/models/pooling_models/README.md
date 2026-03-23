# Pooling Models

!!! note
    We currently support pooling models primarily for convenience. This is not guaranteed to provide any performance improvements over using Hugging Face Transformers or Sentence Transformers directly.

    We plan to optimize pooling models in vLLM. Please comment on <https://github.com/vllm-project/vllm/issues/21796> if you have any suggestions!

## What are pooling models?

Natural Language Processing (NLP) can be primarily divided into the following two types of tasks:

- Natural Language Understanding (NLU)
- Natural Language Generation (NLG)

The generative models supported by vLLM cover a variety of task types, such as the large language models (LLMs) we are familiar with, multimodal models (VLM) that handle multimodal inputs like images, videos, and audio, speech-to-text transcription models, and real-time models that support streaming input. Their common feature is the ability to generate text. Taking it a step further, vLLM-Omni supports the generation of multimodal content, including images, videos, and audio.

As the capabilities of generative models continue to improve, the boundaries of these models are also constantly expanding. However, certain application scenarios still require specialized small language models to efficiently complete specific tasks. These models typically have the following characteristics:

- They do not require content generation.
- They only need to perform very limited functions, without requiring strong generalization, creativity, or high intelligence.
- They demand extremely low latency and may operate on cost-constrained hardware.
- Text-only models typically have fewer than 1 billion parameters, while multimodal models generally have fewer than 10 billion parameters.

Although these models are relatively small in scale, they are still based on the Transformer architecture, similar or even identical to the most advanced large language models today. Many recently released pooling models are also fine-tuned from large language models, allowing them to benefit from the continuous improvements in large models. This architecture similarity enables them to reuse much of vLLM’s infrastructure. If compatible, we would be happy to help them leverage the latest features of vLLM as well.

### Sequence-wise Task and Token-wise Task

The key distinction between sequence-wise task and token-wise task lies in their output granularity: sequence-wise task produces a single result for an entire input sequence, whereas token-wise task yields a result for each individual token within the sequence.

Of course, we also have "plugin" tasks that allow users to customize input and output processors. For more information, please refer to [IO Processor Plugins](../../design/io_processor_plugins.md).

### Pooling Tasks

| Pooling Tasks         | Granularity   | Outputs                                         |
|-----------------------|---------------|-------------------------------------------------|
| `classify` (see note) | Sequence-wise | probability vector of classes for each sequence |
| `embed`               | Sequence-wise | vector representations for each sequence        |
| `token_classify`      | Token-wise    | probability vector of classes for each token    |
| `token_embed`         | Token-wise    | vector representations for each token           |

!!! note
    Within classification tasks, there is a specialized subcategory: Cross-encoder (aka reranker) models. These models are a subset of classification models that accept two prompts as input and output num_labels equal to 1.

### Score Types

The scoring models is designed to compute similarity scores between two input prompts. It supports three model types (aka `score_type`): `cross-encoder`, `late-interaction`, and `bi-encoder`.

| Pooling Tasks         | Granularity   | Outputs                                      | Score Types        | scoring function         |
|-----------------------|---------------|----------------------------------------------|--------------------|--------------------------|
| `classify` (see note) | Sequence-wise | reranker score for each sequence             | `cross-encoder`    | linear classifier        |
| `embed`               | Sequence-wise | vector representations for each sequence     | `bi-encoder`       | cosine similarity        |
| `token_classify`      | Token-wise    | probability vector of classes for each token | nan                | nan                      |
| `token_embed`         | Token-wise    | vector representations for each token        | `late-interaction` | late interaction(MaxSim) |

!!! note
    Only when a classification model outputs num_labels equal to 1 can it be used as a scoring model and have its scoring API enabled.

### Pooling Usages

| Pooling Usages              | Description                                                                                                                                             |
|-----------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------|
| Classification Usages       | Predicting which predefined category, class, or label best corresponds to a given input.                                                                |
| Embedding Usages            | Converts unstructured data (text, images, audio, etc.) into structured numerical vectors (embeddings).                                                  |
| Token Classification Usages | Token-wise classification                                                                                                                               |
| Token Embedding Usages      | Token-wise embedding                                                                                                                                    |
| Scoring Usages              | Computes similarity scores between two inputs. It supports three model types (aka `score_type`): `cross-encoder`, `late-interaction`, and `bi-encoder`. |
| Reward Usages               | Evaluates the quality of outputs generated by a language model, acting as a proxy for human preferences.                                                |

We also have some special models that support multiple pooling tasks, or have specific usage scenarios, or support special inputs and outputs.

For more detailed information, please refer to the link below.

- [Classification Usages](classify.md)
- [Embedding Usages](embed.md)
- [Reward Usages](reward.md)
- [Token Classification Usages](token_classify.md)
- [Token Embedding Usages](token_embed.md)
- [Scoring Usages](scoring.md)
- [Specific Model Examples](specific_models.md)

## Offline Inference

Each pooling model in vLLM supports one or more of these tasks according to
[Pooler.get_supported_tasks][vllm.model_executor.layers.pooler.Pooler.get_supported_tasks],
enabling the corresponding APIs.

### Offline APIs corresponding to pooling tasks

| Task             | APIs                                                                                  |
|------------------|---------------------------------------------------------------------------------------|
| `embed`          | `LLM.embed(...)`, `LLM.encode(..., pooling_task="embed")`, `LLM.score(...)`(see note) |
| `classify`       | `LLM.classify(...)`, `LLM.encode(..., pooling_task="classify")`, `LLM.score(...)`     |
| `token_classify` | `LLM.reward(...)`, `LLM.encode(..., pooling_task="token_classify")`                   |
| `token_embed`    | `LLM.encode(..., pooling_task="token_embed")`, `LLM.score(...)`                       |
| `plugin`         | `LLM.encode(..., pooling_task="plugin")`                                              |

!!! note
    Only when a classification model outputs num_labels equal to 1 can it be used as a scoring model and have its scoring API enabled.

### `LLM.classify`

The [classify][vllm.LLM.classify] method outputs a probability vector for each prompt.
It is primarily designed for [classification models](classify.md).
For more information about `LLM.embed`, see [this page](classify.md#offline-inference).

### `LLM.embed`

The [embed][vllm.LLM.embed] method outputs an embedding vector for each prompt.
It is primarily designed for [embedding models](embed.md).
For more information about `LLM.embed`, see [this page](embed.md#offline-inference).

### `LLM.score`

The [score][vllm.LLM.score] method outputs similarity scores between sentence pairs.
It is primarily designed for [score models](scoring.md).

### `LLM.encode`

The [encode][vllm.LLM.encode] method is available to all pooling models in vLLM.

Please use one of the more specific methods or set the task directly when using `LLM.encode`, refer to the [table above](#offline-apis-corresponding-to-pooling-tasks).

### Examples

```python
from vllm import LLM

llm = LLM(model="intfloat/e5-small", runner="pooling")
(output,) = llm.encode("Hello, my name is", pooling_task="embed")

data = output.outputs.data
print(f"Data: {data!r}")
```

## Online Serving

Our online Server provides endpoints that correspond to the offline APIs:

- Corresponding to `LLM.embed`:
    - [Cohere Embed API](embed.md#cohere-embed-api) (`/v2/embed`)
    - [Openai-compatible Embeddings API](embed.md#openai-compatible-embeddings-api) (`/v1/embeddings`)
- Corresponding to `LLM.classify`:
    - [Classification API](classify.md#online-serving)(`/classify`)
- Corresponding to `LLM.score`:
    - [Score API](scoring.md#score-api)(`/score`)
    - [Rerank API](scoring.md#rerank-api) (`/rerank`, `/v1/rerank`, `/v2/rerank`)
- Pooling API (`/pooling`) is similar to `LLM.encode`, being applicable to all types of pooling models.

The following introduces the Pooling API. For other APIs, please refer to the link above.

### Pooling API

Our Pooling API (`/pooling`) is similar to `LLM.encode`, being applicable to all types of pooling models.

The input format is the same as [Embeddings API](embed.md#openai-compatible-embeddings-api), but the output data can contain an arbitrary nested list, not just a 1-D list of floats.

Please use one of the more specific APIs or set the task directly when using the Pooling API, refer to the [table above](#offline-apis-corresponding-to-pooling-tasks).

Code example: [examples/pooling/pooling/pooling_online.py](../../../examples/pooling/pooling/pooling_online.py)

### Examples

```python
# start a supported embeddings model server with `vllm serve`, e.g.
# vllm serve intfloat/e5-small
import requests

host = "localhost"
port = "8000"
model_name = "intfloat/e5-small"

api_url = f"http://{host}:{port}/pooling"

prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
prompt = {"model": model_name, "input": prompts, "task": "embed"}

response = requests.post(api_url, json=prompt)

for output in response.json()["data"]:
    data = output["data"]
    print(f"Data: {data!r} (size={len(data)})")
```

## Configuration

In vLLM, pooling models implement the [VllmModelForPooling][vllm.model_executor.models.VllmModelForPooling] interface.
These models use a [Pooler][vllm.model_executor.layers.pooler.Pooler] to extract the final hidden states of the input
before returning them.

### Model Runner

Run a model in pooling mode via the option `--runner pooling`.

!!! tip
    There is no need to set this option in the vast majority of cases as vLLM can automatically
    detect the appropriate model runner via `--runner auto`.

### Model Conversion

vLLM can adapt models for various pooling tasks via the option `--convert <type>`.

If `--runner pooling` has been set (manually or automatically) but the model does not implement the
[VllmModelForPooling][vllm.model_executor.models.VllmModelForPooling] interface,
vLLM will attempt to automatically convert the model according to the architecture names
shown in the table below.

| Architecture                                    | `--convert` | Supported pooling tasks      |
|-------------------------------------------------|-------------|------------------------------|
| `*ForTextEncoding`, `*EmbeddingModel`, `*Model` | `embed`     | `token_embed`, `embed`       |
| `*ForRewardModeling`, `*RewardModel`            | `embed`     | `token_embed`, `embed`       |
| `*For*Classification`, `*ClassificationModel`   | `classify`  | `token_classify`, `classify` |

!!! tip
    You can explicitly set `--convert <type>` to specify how to convert the model.

### Pooler Configuration

#### Predefined models

If the [Pooler][vllm.model_executor.layers.pooler.Pooler] defined by the model accepts `pooler_config`,
you can override some of its attributes via the `--pooler-config` option.

#### Converted models

If the model has been converted via `--convert` (see above),
the pooler assigned to each task has the following attributes by default:

| Task       | Pooling Type | Normalization | Softmax |
| ---------- | ------------ | ------------- | ------- |
| `embed`    | `LAST`       | ✅︎            | ❌      |
| `classify` | `LAST`       | ❌            | ✅︎      |

When loading [Sentence Transformers](https://huggingface.co/sentence-transformers) models,
its Sentence Transformers configuration file (`modules.json`) takes priority over the model's defaults.

You can further customize this via the `--pooler-config` option,
which takes priority over both the model's and Sentence Transformers' defaults.

## Removed Features

### Encode task

We have split the `encode` task into two more specific token-wise tasks: `token_embed` and `token_classify`:

- `token_embed` is the same as `embed`, using normalization as the activation.
- `token_classify` is the same as `classify`, by default using softmax as the activation.

Pooling models now default support all pooling, you can use it without any settings.

- Extracting hidden states prefers using `token_embed` task.
- Named Entity Recognition (NER) and reward models prefers using `token_classify` task.

### Score task

`score` task is deprecated and will be removed in v0.20. Please use `classify` instead. Only when a classification model outputs num_labels equal to 1 can it be used as a scoring model and have its scoring API enabled.
