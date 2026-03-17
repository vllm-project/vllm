# Score Models

The Score models is designed to compute similarity scores between two input prompts. It supports three model types (aka `score_type`): `cross-encoder`, `late-interaction`, and `bi-encoder`.

This functionality is supported through the offline `LLM.score(...)` API, along with several online APIs: the `/score` API and the rerank APIs available at `/rerank`, `/v1/rerank`, and `/v2/rerank`.

!!! note
    vLLM handles only the model inference component of RAG pipelines (such as embedding generation and reranking). For higher-level RAG orchestration, you should leverage integration frameworks like [LangChain](https://github.com/langchain-ai/langchain).

## Supported Models

### Cross-encoder models

[Cross-encoder](https://www.sbert.net/examples/applications/cross-encoder/README.html) (aka reranker) models are a subset of classification models that accept two prompts as input and output num_labels equal to 1.

- Text only models

| Architecture | Models | Example HF Models | Score template (see note) | [LoRA](../../features/lora.md) | [PP](../../serving/parallelism_scaling.md) |
| ------------ | ------ | ----------------- | ------------------------- | --------------------------- | --------------------------------------- |
| `BertForSequenceClassification` | BERT-based | `cross-encoder/ms-marco-MiniLM-L-6-v2`, etc. | N/A | | |
| `GemmaForSequenceClassification` | Gemma-based | `BAAI/bge-reranker-v2-gemma`(see note), etc. | [bge-reranker-v2-gemma.jinja](../../../examples/pooling/score/template/bge-reranker-v2-gemma.jinja) | ✅︎ | ✅︎ |
| `GteNewForSequenceClassification` | mGTE-TRM (see note) | `Alibaba-NLP/gte-multilingual-reranker-base`, etc. | N/A | | |
| `LlamaBidirectionalForSequenceClassification`<sup>C</sup> | Llama-based with bidirectional attention | `nvidia/llama-nemotron-rerank-1b-v2`, etc. | [nemotron-rerank.jinja](../../../examples/pooling/score/template/nemotron-rerank.jinja) | ✅︎ | ✅︎ |
| `Qwen2ForSequenceClassification`<sup>C</sup> | Qwen2-based | `mixedbread-ai/mxbai-rerank-base-v2`(see note), etc. | [mxbai_rerank_v2.jinja](../../../examples/pooling/score/template/mxbai_rerank_v2.jinja) | ✅︎ | ✅︎ |
| `Qwen3ForSequenceClassification`<sup>C</sup> | Qwen3-based | `tomaarsen/Qwen3-Reranker-0.6B-seq-cls`, `Qwen/Qwen3-Reranker-0.6B`(see note), etc. | [qwen3_reranker.jinja](../../../examples/pooling/score/template/qwen3_reranker.jinja) | ✅︎ | ✅︎ |
| `RobertaForSequenceClassification` | RoBERTa-based | `cross-encoder/quora-roberta-base`, etc. | N/A | | |
| `XLMRobertaForSequenceClassification` | XLM-RoBERTa-based | `BAAI/bge-reranker-v2-m3`, etc. | N/A | | |
| `*Model`<sup>C</sup>, `*ForCausalLM`<sup>C</sup>, etc. | Generative models | N/A | N/A | \* | \* |

<sup>C</sup> Automatically converted into a classification model via `--convert classify`. ([details](./README.md#model-conversion))  
\* Feature support is the same as that of the original model.

!!! note
    Some models require a specific prompt format to work correctly.

    You can find Example HF Models's corresponding score template in [examples/pooling/score/template/](../../../examples/pooling/score/template)

    Examples : [examples/pooling/score/using_template_offline.py](../../../examples/pooling/score/using_template_offline.py) [examples/pooling/score/using_template_online.py](../../../examples/pooling/score/using_template_online.py)

!!! note
    Load the official original `BAAI/bge-reranker-v2-gemma` by using the following command.

    ```bash
    vllm serve BAAI/bge-reranker-v2-gemma --hf_overrides '{"architectures": ["GemmaForSequenceClassification"],"classifier_from_token": ["Yes"],"method": "no_post_processing"}'
    ```

!!! note
    The second-generation GTE model (mGTE-TRM) is named `NewForSequenceClassification`. The name `NewForSequenceClassification` is too generic, you should set `--hf-overrides '{"architectures": ["GteNewForSequenceClassification"]}'` to specify the use of the `GteNewForSequenceClassification` architecture.

!!! note
    Load the official original `mxbai-rerank-v2` by using the following command.

    ```bash
    vllm serve mixedbread-ai/mxbai-rerank-base-v2 --hf_overrides '{"architectures": ["Qwen2ForSequenceClassification"],"classifier_from_token": ["0", "1"], "method": "from_2_way_softmax"}'
    ```

!!! note
    Load the official original `Qwen3 Reranker` by using the following command. More information can be found at: [examples/pooling/score/qwen3_reranker_offline.py](../../../examples/pooling/score/qwen3_reranker_offline.py) [examples/pooling/score/qwen3_reranker_online.py](../../../examples/pooling/score/qwen3_reranker_online.py).

    ```bash
    vllm serve Qwen/Qwen3-Reranker-0.6B --hf_overrides '{"architectures": ["Qwen3ForSequenceClassification"],"classifier_from_token": ["no", "yes"],"is_original_qwen3_reranker": true}'
    ```

- Multimodal Models

!!! note
    For more information about multimodal models inputs, see [this page](../supported_models.md#list-of-multimodal-language-models).

| Architecture | Models | Inputs | Example HF Models | [LoRA](../../features/lora.md) | [PP](../../serving/parallelism_scaling.md) |
| ------------ | ------ | ------ | ----------------- | ------------------------------ | ------------------------------------------ |
| `JinaVLForSequenceClassification` | JinaVL-based | T + I<sup>E+</sup> | `jinaai/jina-reranker-m0`, etc. | ✅︎ | ✅︎ |
| `LlamaNemotronVLForSequenceClassification` | Llama Nemotron Reranker + SigLIP | T + I<sup>E+</sup> | `nvidia/llama-nemotron-rerank-vl-1b-v2` | | |
| `Qwen3VLForSequenceClassification` | Qwen3-VL-Reranker | T + I<sup>E+</sup> + V<sup>E+</sup> | `Qwen/Qwen3-VL-Reranker-2B`(see note), etc. | ✅︎ | ✅︎ |

<sup>C</sup> Automatically converted into a classification model via `--convert classify`. ([details](README.md#model-conversion))  
\* Feature support is the same as that of the original model.

!!! note
    Similar to Qwen3-Reranker, you need to use the following `--hf_overrides` to load the official original `Qwen3-VL-Reranker`.

    ```bash
    vllm serve Qwen/Qwen3-VL-Reranker-2B --hf_overrides '{"architectures": ["Qwen3VLForSequenceClassification"],"classifier_from_token": ["no", "yes"],"is_original_qwen3_reranker": true}'
    ```

### Late-interaction models

All models that support token embedding task also support using the score API to compute similarity scores by calculating the late interaction of two input prompts. See [this page](token_embed.md) for more information about token embedding models.

### Bi-encoder

All models that support embedding task also support using the score API to compute similarity scores by calculating the cosine similarity of two input prompt's embeddings. See [this page](embed.md) for more information about embedding models.

## Offline Inference

### Pooling Parameters

The following [pooling parameters][vllm.PoolingParams] are only supported by cross-encoder models and do not work for late-interaction and bi-encoder models.

```python
--8<-- "vllm/pooling_params.py:common-pooling-params"
--8<-- "vllm/pooling_params.py:classify-pooling-params"
```

### `LLM.score`

The [score][vllm.LLM.score] method outputs similarity scores between sentence pairs.

```python
from vllm import LLM

llm = LLM(model="BAAI/bge-reranker-v2-m3", runner="pooling")
(output,) = llm.score(
    "What is the capital of France?",
    "The capital of Brazil is Brasilia.",
)

score = output.outputs.score
print(f"Score: {score}")
```

A code example can be found here: [examples/basic/offline_inference/score.py](../../../examples/basic/offline_inference/score.py)

## Online Serving

### Score API

Our Score API (`/score`) is similar to `LLM.score`, compute similarity scores between two input prompts.

#### Parameters

The following Score API parameters are supported:

```python
--8<-- "vllm/entrypoints/pooling/base/protocol.py:pooling-common-params"
--8<-- "vllm/entrypoints/pooling/base/protocol.py:pooling-common-extra-params"
--8<-- "vllm/entrypoints/pooling/base/protocol.py:classify-extra-params"
```

#### Examples

##### Single inference

You can pass a string to both `queries` and `documents`, forming a single sentence pair.

```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/score' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "model": "BAAI/bge-reranker-v2-m3",
  "encoding_format": "float",
  "queries": "What is the capital of France?",
  "documents": "The capital of France is Paris."
}'
```

??? console "Response"

    ```json
    {
      "id": "score-request-id",
      "object": "list",
      "created": 693447,
      "model": "BAAI/bge-reranker-v2-m3",
      "data": [
        {
          "index": 0,
          "object": "score",
          "score": 1
        }
      ],
      "usage": {}
    }
    ```

##### Batch inference

You can pass a string to `queries` and a list to `documents`, forming multiple sentence pairs
where each pair is built from `queries` and a string in `documents`.
The total number of pairs is `len(documents)`.

??? console "Request"

    ```bash
    curl -X 'POST' \
      'http://127.0.0.1:8000/score' \
      -H 'accept: application/json' \
      -H 'Content-Type: application/json' \
      -d '{
      "model": "BAAI/bge-reranker-v2-m3",
      "queries": "What is the capital of France?",
      "documents": [
        "The capital of Brazil is Brasilia.",
        "The capital of France is Paris."
      ]
    }'
    ```

??? console "Response"

    ```json
    {
      "id": "score-request-id",
      "object": "list",
      "created": 693570,
      "model": "BAAI/bge-reranker-v2-m3",
      "data": [
        {
          "index": 0,
          "object": "score",
          "score": 0.001094818115234375
        },
        {
          "index": 1,
          "object": "score",
          "score": 1
        }
      ],
      "usage": {}
    }
    ```

You can pass a list to both `queries` and `documents`, forming multiple sentence pairs
where each pair is built from a string in `queries` and the corresponding string in `documents` (similar to `zip()`).
The total number of pairs is `len(documents)`.

??? console "Request"

    ```bash
    curl -X 'POST' \
      'http://127.0.0.1:8000/score' \
      -H 'accept: application/json' \
      -H 'Content-Type: application/json' \
      -d '{
      "model": "BAAI/bge-reranker-v2-m3",
      "encoding_format": "float",
      "queries": [
        "What is the capital of Brazil?",
        "What is the capital of France?"
      ],
      "documents": [
        "The capital of Brazil is Brasilia.",
        "The capital of France is Paris."
      ]
    }'
    ```

??? console "Response"

    ```json
    {
      "id": "score-request-id",
      "object": "list",
      "created": 693447,
      "model": "BAAI/bge-reranker-v2-m3",
      "data": [
        {
          "index": 0,
          "object": "score",
          "score": 1
        },
        {
          "index": 1,
          "object": "score",
          "score": 1
        }
      ],
      "usage": {}
    }
    ```

##### Multi-modal inputs

You can pass multi-modal inputs to scoring models by passing `content` including a list of multi-modal input (image, etc.) in the request. Refer to the examples below for illustration.

=== "JinaVL-Reranker"

    To serve the model:

    ```bash
    vllm serve jinaai/jina-reranker-m0
    ```

    Since the request schema is not defined by OpenAI client, we post a request to the server using the lower-level `requests` library:

    ??? Code

        ```python
        import requests
        
        response = requests.post(
            "http://localhost:8000/v1/score",
            json={
                "model": "jinaai/jina-reranker-m0",
                "queries": "slm markdown",
                "documents": [
                    {
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": "https://raw.githubusercontent.com/jina-ai/multimodal-reranker-test/main/handelsblatt-preview.png"
                                },
                            }
                        ],
                    },
                    {
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": "https://raw.githubusercontent.com/jina-ai/multimodal-reranker-test/main/handelsblatt-preview.png"
                                },
                            }
                        ]
                    },
                ],
            },
        )
        response.raise_for_status()
        response_json = response.json()
        print("Scoring output:", response_json["data"][0]["score"])
        print("Scoring output:", response_json["data"][1]["score"])
        ```
Full example:

- [examples/pooling/score/vision_score_api_online.py](../../../examples/pooling/score/vision_score_api_online.py)
- [examples/pooling/score/vision_rerank_api_online.py](../../../examples/pooling/score/vision_rerank_api_online.py)

### Rerank API

`/rerank`, `/v1/rerank`, and `/v2/rerank` APIs are compatible with both [Jina AI's rerank API interface](https://jina.ai/reranker/) and
[Cohere's rerank API interface](https://docs.cohere.com/v2/reference/rerank) to ensure compatibility with
popular open-source tools.

Code example: [examples/pooling/score/rerank_api_online.py](../../../examples/pooling/score/rerank_api_online.py)

#### Parameters

The following rerank api parameters are supported:

```python
--8<-- "vllm/entrypoints/pooling/base/protocol.py:pooling-common-params"
--8<-- "vllm/entrypoints/pooling/base/protocol.py:pooling-common-extra-params"
--8<-- "vllm/entrypoints/pooling/base/protocol.py:classify-extra-params"
```

#### Examples

Note that the `top_n` request parameter is optional and will default to the length of the `documents` field.
Result documents will be sorted by relevance, and the `index` property can be used to determine original order.

??? console "Request"

    ```bash
    curl -X 'POST' \
      'http://127.0.0.1:8000/v1/rerank' \
      -H 'accept: application/json' \
      -H 'Content-Type: application/json' \
      -d '{
      "model": "BAAI/bge-reranker-base",
      "query": "What is the capital of France?",
      "documents": [
        "The capital of Brazil is Brasilia.",
        "The capital of France is Paris.",
        "Horses and cows are both animals"
      ]
    }'
    ```

??? console "Response"

    ```json
    {
      "id": "rerank-fae51b2b664d4ed38f5969b612edff77",
      "model": "BAAI/bge-reranker-base",
      "usage": {
        "total_tokens": 56
      },
      "results": [
        {
          "index": 1,
          "document": {
            "text": "The capital of France is Paris."
          },
          "relevance_score": 0.99853515625
        },
        {
          "index": 0,
          "document": {
            "text": "The capital of Brazil is Brasilia."
          },
          "relevance_score": 0.0005860328674316406
        }
      ]
    }
    ```

## More examples

More examples can be found here: [examples/pooling/score](../../../examples/pooling/score)

## Features

AS cross-encoder models are a subset of classification models that accept two prompts as input and output num_labels equal to 1, cross-encoder features should be consistent with (sequence) classification. For more information, see [this page](classify.md#features).

### Score Template

Score Template is only supported for cross-encoder models.

Some scoring models require a specific prompt format to work correctly. You can specify a custom score template using the `--chat-template` parameter (see [Chat Template](../../serving/openai_compatible_server.md#chat-template)).

Score templates are supported for **cross-encoder** models only. If you are using an **embedding** model for scoring, vLLM does not apply a score template.

Like chat templates, the score template receives a `messages` list. For scoring, each message has a `role` attribute—either `"query"` or `"document"`. For the usual kind of point-wise cross-encoder, you can expect exactly two messages: one query and one document. To access the query and document content, use Jinja's `selectattr` filter:

- **Query**: `{{ (messages | selectattr("role", "eq", "query") | first).content }}`
- **Document**: `{{ (messages | selectattr("role", "eq", "document") | first).content }}`

This approach is more robust than index-based access (`messages[0]`, `messages[1]`) because it selects messages by their semantic role. It also avoids assumptions about message ordering if additional message types are added to `messages` in the future.

Example template file: [examples/pooling/score/template/nemotron-rerank.jinja](../../../examples/pooling/score/template/nemotron-rerank.jinja)

### Enable/disable activation

You can enable or disable activation via `use_activation` only works for cross-encoder models.
